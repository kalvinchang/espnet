import math
from typing import Dict, Iterator, List, Optional, Tuple, Union
import unicodedata
from typeguard import typechecked
import logging
import re
import json
import contextlib

import torch
from torch import Tensor, LongTensor
from torch import nn
from allophant.phonetic_features import (
    LanguageAllophoneMappings,
    PhoneticAttributeIndexer,
    FeatureSet,
)

from espnet2.asr.encoder.abs_encoder import AbsEncoder


# TODO: Move layers to layers/ module?
_PAD_VALUE = torch.finfo(torch.float32).min


@torch.jit.script
def _multiply_allophone_matrix(
    phone_logits: Tensor, matrix: Tensor, mask: Tensor, mask_value: float = _PAD_VALUE
) -> Tensor:
    return (
        (phone_logits * matrix.unsqueeze(0))
        .masked_fill_(
            mask.unsqueeze(0),
            mask_value,
        )
        .max(1)
        .values
    )


class AllophoneMapping(nn.Module):
    """
    Allophone layer derived from the Allosaurus architecture (Li et al., 2020).

    .. references::
        Li, Xinjian, Siddharth Dalmia, Juncheng Li, Matthew Lee, Patrick
        Littell, Jiali Yao, Antonios Anastasopoulos, et al. “Universal Phone
        Recognition with a Multilingual Allophone System.” In ICASSP 2020 -
        2020 IEEE International Conference on Acoustics, Speech and Signal
        Processing (ICASSP), 8249–53. Barcelona, Spain: IEEE, 2020.
        https://doi.org/10.1109/ICASSP40776.2020.9054362.
    """

    _allophone_mask: Tensor

    def __init__(
        self,
        shared_phone_count: int,
        phoneme_count: int,
        language_allophones: LanguageAllophoneMappings,
        blank_offset: int = 1,
    ) -> None:
        super().__init__()
        allophones = language_allophones.allophones
        languages = language_allophones.languages
        num_languages = len(languages)
        # Maps language codes to dense indices in the matrix
        self._index_map = {}

        allophone_matrix = torch.zeros(num_languages, shared_phone_count, phoneme_count)
        for dense_index, (language_index, allophone_mapping) in enumerate(
            allophones.items()
        ):
            language_allophone_matrix = allophone_matrix[dense_index]
            # Constructs identity mappings for BLANK in the diagonal
            language_allophone_matrix[range(blank_offset), range(blank_offset)] = 1

            self._index_map[languages[language_index]] = dense_index
            for phoneme, allophones in allophone_mapping.items():
                # Add allophone mappings with the blank offset
                language_allophone_matrix[
                    torch.tensor(allophones) + blank_offset, phoneme + blank_offset
                ] = 1

        # Shared allophone_matrix
        self._allophone_matrices = nn.Parameter(allophone_matrix)

        # Initialization for the l2 penalty according to "Universal Phone Recognition With a Multilingual Allophone Systems" by Li et al.
        self.register_buffer(
            "_initialization", allophone_matrix.clone(), persistent=False
        )
        # Inverse mask for language specific phoneme inventories
        self.register_buffer(
            "_allophone_mask", ~allophone_matrix.bool(), persistent=False
        )

    @property
    def index_map(self) -> Dict[str, int]:
        return self._index_map

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        # Batch size x Shared Phoneme Inventory Size
        batch_matrices = torch.empty(
            *phone_logits.shape[:2],
            self._allophone_matrices.shape[2],
            device=phone_logits.device,
        )
        for index, language_id in enumerate(map(int, language_ids)):
            logits = phone_logits[index].unsqueeze(-1)
            # Max pooling after multiplying with the allophone matrix
            # Replace masked positions with negative infinity
            # since this results in zero probabilities after softmax for phone and
            # phoneme combinations that don't occur in the allophone mappings
            batch_matrices[index] = _multiply_allophone_matrix(
                logits,
                self._allophone_matrices[language_id],
                self._allophone_mask[language_id],
            )

        return batch_matrices

    def forward(
        self, phone_logits: Tensor, language_ids: Tensor, predict: bool = False
    ) -> Tensor:
        # Note that while predictions on the training corpus could still be valid with the allophone layer enabled,
        # language IDs are different for other corpora and therefore not supported
        if predict:
            return phone_logits
        return self.map_allophones(phone_logits, language_ids)

    def l2_penalty(self) -> Tensor:
        """
        Computes the l2 penalty for the allophone layer to be added to the loss

        :return: The l2 penalty for the allophone layer given the current
            weights or 0 if no allophone layer is present in the architecture
        """
        # Calculates the matrix L2 (Frobenius) norm for each allophone matrix and then sums the norms over languages
        return torch.norm_except_dim(
            self._allophone_matrices - self._initialization, dim=0
        ).sum()


class EmbeddingCompositionLayer(nn.Module):
    """
    Embedding composition layer derived from the compositional phone embedding
    layer by Li et al, (2021)

    .. references::
        Li, Xinjian, Juncheng Li, Florian Metze and Alan W. Black.
        “Hierarchical Phone Recognition with Compositional Phonetics.”
        Interspeech (2021).
    """

    _feature_table: Tensor
    _category_offsets: Tensor

    def __init__(self, embedding_size: int, feature_table: Tensor, blank_offset: int = 1, additional_special_tokens: int = 0) -> None:
        super().__init__()

        assert blank_offset > 0, "Blank offset must be at least 1"

        additional_tokens = blank_offset + additional_special_tokens
        self._output_size = feature_table.shape[0] + additional_tokens
        self._blank_offset = blank_offset
        self._additional_special_tokens = additional_special_tokens

        # Add blank embedding + embeddings for every other special symbol if blank_offset > 1 or additional_special_tokens > 0
        num_categories = torch.cat((LongTensor([0] * additional_tokens), feature_table.max(0).values)) + 1
        unused_categories = torch.cat(
            (
                torch.tensor([False] * additional_tokens),
                torch.cat([row.bincount() for row in feature_table.T]) == 0,
            )
        )

        unused_count = unused_categories.sum().item()
        if unused_count:
            logging.info(f"{unused_count} unused feature embeddings")

        # Offsets and one additional entry for the special blank feature
        category_offsets = num_categories.cumsum(0)[additional_tokens - 1:-1].unsqueeze(0)
        feature_table += category_offsets
        self._attribute_embeddings = nn.EmbeddingBag(
            int(num_categories.sum()), embedding_size, mode="sum"
        )
        self._embedding_size = embedding_size

        # Set unused attribute embedding weights to 0
        with torch.no_grad():
            self._attribute_embeddings.weight[unused_categories] = 0

        self.register_buffer("_feature_table", feature_table, persistent=False)
        self.register_buffer("_category_offsets", category_offsets, persistent=False)
        # Scale factor for the dot product with feature embeddings as in Li et al. (2021)
        self.register_buffer(
            "_scale_factor", torch.tensor(math.sqrt(embedding_size)), persistent=False
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self, inputs: Tensor, target_feature_indices: Tensor | None = None
    ) -> Tensor:
        if target_feature_indices is None:
            target_feature_indices = self._feature_table
        else:
            target_feature_indices = target_feature_indices + self._category_offsets

        composed = [
            # Blank + special embeddings (Using the index batch sequences equivalent to [[0], [1], ...])
            self._attribute_embeddings(
                torch.arange(
                    self._blank_offset, dtype=target_feature_indices.dtype, device=inputs.device
                ).unsqueeze(1)
            ),
            # Phonemes
            self._attribute_embeddings(target_feature_indices),
        ]

        if self._additional_special_tokens:
            composed.append(
                self._attribute_embeddings(
                    torch.arange(self._blank_offset, self._blank_offset + self._additional_special_tokens, dtype=target_feature_indices.dtype, device=inputs.device).unsqueeze(1)
                ),
            )

        composed_embeddings = torch.cat(composed).T

        return (inputs @ composed_embeddings) / self._scale_factor


def _normalize_phoneme(feature_type: str, phoneme: str) -> str:
    if feature_type == "phoible":
        # Special case for PHOIBLE
        return unicodedata.normalize("NFD", phoneme).replace("ç", "ç")
    else:
        return unicodedata.normalize("NFD", phoneme)


def _allophone_mapping_with_fallback(
    indexer: PhoneticAttributeIndexer,
    phone_inventory: List[str],
    phoneme_inventory: List[str],
    allophone_languages: List[str],
    composition_inventories: Optional[Dict[str, List[str]]] = None,
    allophone_identity_placeholders: Optional[List[str]] = None,
) -> Tuple[LanguageAllophoneMappings, List[str]]:
    # Separate indices for input phones and output phonemes
    phone_indices = {phone: index for index, phone in enumerate(phone_inventory)}
    phoneme_indices = {phoneme: index for index, phoneme in enumerate(phoneme_inventory)}
    available_phonemes = set(phoneme_indices)
    # Retrieve allophone data from Allophoible
    allophone_data = indexer.allophone_data
    if allophone_data is None:
        raise ValueError("No allophone data is available in the indexer")
    allophone_inventories = allophone_data.inventories
    allophones = {}

    # Retrieve phoneme to allophone indices in the shared phoneme and phone vocabularies respectively
    for language_id, language in enumerate(allophone_languages):
        allophone_inventory = (
            allophone_inventories.loc[allophone_inventories.ISO6393 == language, "Allophones"]
            .str.split(" ")
            .to_dict()
        )
        allophones[language_id] = {
            phoneme_indices[phoneme]: [phone_indices[allophone] for allophone in allophones]
            for phoneme, allophones in allophone_inventory.items()
            # TODO: More accurate to use mapping file for this to filter for each language independently
            # Filter phonemes that don't exist in the training data
            if phoneme in available_phonemes
        }

    # Retrieve phoneme to phoneme identity indices in the shared phoneme and phone vocabularies respectively
    if allophone_identity_placeholders:
        if composition_inventories is None:
            raise ValueError(
                "composition_inventories is undefined but required for constructing identity mappings"
                " for languages in allophone_identity_placeholders."
            )

        for language_id, language in enumerate(allophone_identity_placeholders, len(allophone_languages)):
            # Extend phone inventory with any phones that don't occur in the languages with allophone data
            for phoneme in composition_inventories[language]:
                if phoneme not in phone_indices:
                    phone_indices[phoneme] = len(phone_indices)

            allophones[language_id] = {
                phoneme_indices[phoneme]: [phone_indices[phoneme]]
                for phoneme in composition_inventories[language]
            }

        all_languages = allophone_languages + allophone_identity_placeholders
    else:
        all_languages = allophone_languages

    return LanguageAllophoneMappings(
        allophones,
        all_languages,
        phone_inventory,
    ), list(phone_indices)


EncoderOutputs = Tuple[Tensor, List[Tuple[int, Tensor]]]


class AllophantLayers(AbsEncoder):
    @typechecked
    def __init__(
        self,
        input_size: int,
        phoneme_embedding_size: int,
        composition_features: List[str],
        phoneme_inventory_file: str,
        allophone_languages: Optional[List[str]] = None,
        allophone_identity_placeholders: Optional[List[str]] = None,
        language_id_mapping_path: Optional[str] = None,
        phoible_path: Union[str, None] = None,
        blank_offset: int = 1,
        additional_special_tokens: int = 0,
        use_allophone_layer: bool = True,
        utt_id_separator: str = "_",
        feature_type: str = "phoible",
        composition_inventories_file: Optional[str] = None,
    ):
        super().__init__()

        # Assumes the phoneme inventory is in the form generated from stage 5
        with open(phoneme_inventory_file, "r", encoding="utf-8") as file:
            # Normalize vocabulary for feature compatibility and filter special tokens
            phoneme_inventory = [
                _normalize_phoneme(feature_type, line) for line in map(str.strip, file)
                # Ignore special tokens
                if not (line.startswith("<") and line.endswith(">"))
            ]

        # Used by ESPnetAsrModel to identify encoders that take `utt_id` as an input
        # Required if `use_allophone_layer` is `True`
        # TODO: Disable if use_allophone_layer is False to reduce overhead?
        self.requires_utt_id = True

        self._separator = re.compile(utt_id_separator)

        if language_id_mapping_path is None:
            self._language_map = None
            reverse_map = None
        else:
            with open(language_id_mapping_path, "r", encoding="utf-8") as file:
                self._language_map = json.load(file)
                reverse_map = {
                    normalized: original
                    for original, normalized in self._language_map.items()
                }

        match feature_type:
            case "phoible":
                features = FeatureSet.PHOIBLE
            case "panphon":
                if use_allophone_layer:
                    raise ValueError(
                        "The allophone layer currently only supports PHOIBLE features"
                    )
                features = FeatureSet.PANPHON
            case unsupported:
                raise ValueError(f"Feature set {unsupported} if unsupported")

        with contextlib.nullcontext() if phoible_path is None else open(phoible_path, "r", encoding="utf-8") as phoible_file:
            self._indexer = PhoneticAttributeIndexer(
                features,
                phoible_file,
                composition_features,
                language_inventories=allophone_languages,
                allophones_from_allophoible=use_allophone_layer,
            )

        if composition_inventories_file is not None:
            with open(composition_inventories_file, "r", encoding="utf-8") as file:
                self._composition_inventories = json.load(file)
                self._composition_features = {
                    language: self._indexer.full_attributes.subset(
                        [_normalize_phoneme(feature_type, phoneme) for phoneme in inventory],
                        self._indexer.composition_features.copy(),
                    ).dense_feature_table.long() for language, inventory in self._composition_inventories.items()
                }
        else:
            self._composition_inventories = None
            self._composition_features = None

        if use_allophone_layer:
            if self._indexer.allophone_data is None:
                raise ValueError(
                    "Model configuration using attribute embedding composition and an allophone layer"
                    " requires allophone data in the attribute indexer with but got `None`"
                )
            if allophone_languages is None:
                raise ValueError(
                    "When the allophone layer is enabled `allophone_languages` has to be specified"
                    " for initializing allophone matrices"
                )

            phone_inventory = self._indexer.allophone_data.shared_phone_indexer.phonemes.tolist()
            language_allophones, composition_inventory = _allophone_mapping_with_fallback(
                self._indexer,
                phone_inventory,
                phoneme_inventory,
                allophone_languages,
                self._composition_inventories,
                allophone_identity_placeholders
            )
        else:
            composition_inventory = phoneme_inventory
            # Use the phoneme subset from the attribute indexer with all features for constructing the embeddings
            language_allophones = None

        # Use either the phone or the phoneme subset from the attribute indexer with all features for constructing the embeddings
        training_features = self._indexer.full_attributes.subset(
            composition_inventory,
            self._indexer.composition_features.copy(),
        ).dense_feature_table.long()

        # Projection from hidden size to the phoneme embedding size
        self._projection_layer = nn.Linear(input_size, phoneme_embedding_size)
        self._input_size = input_size

        # Phonetic feature embedding composition for computing phoneme logits
        self._phoneme_composition_layer = EmbeddingCompositionLayer(
            phoneme_embedding_size, training_features, blank_offset, additional_special_tokens
        )

        # Optional per-language allophone layer
        if language_allophones is None:
            self._output_size = self._phoneme_composition_layer.output_size()
            self._allophone_layer = None
        elif reverse_map is None:
            raise ValueError("language_id_mapping_path must be specified when the allophone layer is enabled")
        else:
            self._output_size = len(phoneme_inventory) + blank_offset + additional_special_tokens
            self._allophone_layer = AllophoneMapping(
                self._phoneme_composition_layer.output_size(),
                self._output_size,
                language_allophones,
                blank_offset,
            )

    # TODO: Only return phoneme classifier output size here?
    def output_size(self) -> int:
        return self._output_size

    def _utterance_langcodes(self, utt_id_batch: List[str]) -> Iterator[str]:
        assert self._language_map is not None
        for i in utt_id_batch:
            yield self._language_map[self._separator.split(i, 2)[1]]

    def _inference_with_inventory(self, inputs: Tensor, features: Tensor, pad_to_output_size: bool = True) -> Tensor:
        composition_output = self._phoneme_composition_layer(
            inputs,
            features.to(inputs.device)
        )

        if not pad_to_output_size:
            return composition_output

        pad_size = self._output_size - composition_output.shape[-1]
        tail = self._phoneme_composition_layer._additional_special_tokens
        # Pad with the smallest bfloat16 value between the phonemes and special symbol
        return torch.cat((
            composition_output[..., :-tail],
            torch.full(
                (*composition_output.shape[:2], pad_size),
                torch.finfo(torch.bfloat16).min,
                dtype=composition_output.dtype,
                device=composition_output.device
            ),
            composition_output[..., -tail:],
        ), dim=2)

    def forward(
        self,
        xs_pad: Tensor,
        ilens: Tensor,
        prev_states: Optional[Tensor] = None,
        utt_id: Optional[List[str]] = None,
        target_feature_indices: Optional[Tensor] = None,
        use_language_vocabulary: bool = False,
    ) -> Tuple[Union[EncoderOutputs, Dict[str, Union[Tensor, EncoderOutputs]]], Tensor, Optional[Tensor]]:
        if utt_id is None:
            raise ValueError(
                f"{self.__class__.__name__}.encode was called without utt_id, "
                "which is required by the encoder"
            )

        output = self._projection_layer(xs_pad)

        if use_language_vocabulary and target_feature_indices is None and self._composition_features is not None:
            language_codes = list(self._utterance_langcodes(utt_id))
            first_code = language_codes[0]

            if all(code == first_code for code in language_codes):
                # Fast path if the batch consists only of utterances in a single language
                output = self._inference_with_inventory(
                    output, self._composition_features[first_code]
                )
            else:
                output = torch.cat([
                    self._inference_with_inventory(
                        output[None, i], self._composition_features[code]
                    ) for i, code in enumerate(language_codes)
                ])
        else:
            output = self._phoneme_composition_layer(
                output, target_feature_indices
            )

        # Only use the allophone layer unless another inventory is specified
        if self._allophone_layer is not None and not use_language_vocabulary and target_feature_indices is None:
            language_ids = torch.tensor(
                [self._allophone_layer.index_map[i] for i in self._utterance_langcodes(utt_id)],
                dtype=torch.int64,
            )
            # Assume that the allophone layer should not be used when a custom phoneme inventory is provided
            output = self._allophone_layer(
                output, language_ids, target_feature_indices is not None
            )

            # Return l2_penalty during training to keep the allophone matrix
            # close to its initialization following Li et al., 2020
            if self.training:
                return {
                    "l2_penalty": self._allophone_layer.l2_penalty(),
                    "output": (output, [(0, xs_pad), (1, output)]),
                }, ilens, None

        # Return frontend output as an intermediate output for auxiliary CTC
        return (output, [(0, xs_pad), (1, output)]), ilens, None

    def intermediate_size(self, index_key: str) -> int:
        if index_key == "0":
            return self._input_size
        elif index_key == "1":
            return self._output_size

        raise Exception(f"{index_key!r} is an invalid intermediate layer index for Allophant")
