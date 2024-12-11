import math
from typing import Dict, List, Optional, Tuple, Union
from typeguard import typechecked
import logging
import re
import json

import torch
from torch import Tensor, LongTensor
from torch import nn
from allophant.phonetic_features import LanguageAllophoneMappings, PhoneticAttributeIndexer, FeatureSet

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
        for dense_index, (language_index, allophone_mapping) in enumerate(allophones.items()):
            language_allophone_matrix = allophone_matrix[dense_index]
            # Constructs identity mappings for BLANK in the diagonal
            language_allophone_matrix[range(blank_offset), range(blank_offset)] = 1

            self._index_map[languages[language_index]] = dense_index
            for phoneme, allophones in allophone_mapping.items():
                # Add allophone mappings with the blank offset
                language_allophone_matrix[torch.tensor(allophones) + blank_offset, phoneme + blank_offset] = 1

        # Shared allophone_matrix
        self._allophone_matrices = nn.Parameter(allophone_matrix)

        # Initialization for the l2 penalty according to "Universal Phone Recognition With a Multilingual Allophone Systems" by Li et al.
        self.register_buffer("_initialization", allophone_matrix.clone(), persistent=False)
        # Inverse mask for language specific phoneme inventories
        self.register_buffer("_allophone_mask", ~allophone_matrix.bool(), persistent=False)

    @property
    def index_map(self) -> Dict[str, int]:
        return self._index_map

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        # Batch size x Shared Phoneme Inventory Size
        batch_matrices = torch.empty(
            *phone_logits.shape[:2], self._allophone_matrices.shape[2], device=phone_logits.device
        )
        for index, language_id in enumerate(map(int, language_ids)):
            logits = phone_logits[:, index].unsqueeze(-1)
            # Max pooling after multiplying with the allophone matrix
            # Replace masked positions with negative infinity
            # since this results in zero probabilities after softmax for phone and
            # phoneme combinations that don't occur in the allophone mappings
            batch_matrices[:, index] = _multiply_allophone_matrix(
                logits,
                self._allophone_matrices[language_id],
                self._allophone_mask[language_id],
            )

        return batch_matrices

    def forward(self, phone_logits: Tensor, language_ids: Tensor, predict: bool = False) -> Tensor:
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
        return torch.norm_except_dim(self._allophone_matrices - self._initialization, dim=0).sum()


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

    def __init__(self, embedding_size: int, feature_table: Tensor) -> None:
        super().__init__()

        self._output_size = feature_table.shape[0] + 1

        # Add Single blank embedding
        num_categories = torch.cat((LongTensor([0]), feature_table.max(0).values)) + 1
        unused_categories = torch.cat(
            (torch.tensor([False]), torch.cat([row.bincount() for row in feature_table.T]) == 0)
        )

        unused_count = unused_categories.sum().item()
        if unused_count:
            logging.info(f"{unused_count} unused feature embeddings")

        # Offsets and one additional entry for the special blank feature
        category_offsets = num_categories.cumsum(0)[:-1].unsqueeze(0)
        feature_table += category_offsets
        self._attribute_embeddings = nn.EmbeddingBag(int(num_categories.sum()), embedding_size, mode="sum")

        # Set unused attribute embedding weights to 0
        with torch.no_grad():
            self._attribute_embeddings.weight[unused_categories] = 0

        self.register_buffer("_feature_table", feature_table, persistent=False)
        self.register_buffer("_category_offsets", category_offsets, persistent=False)
        # Scale factor for the dot product with feature embeddings as in Li et al. (2021)
        self.register_buffer("_scale_factor", torch.tensor(math.sqrt(embedding_size)), persistent=False)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, inputs: Tensor, target_feature_indices: Tensor | None = None) -> Tensor:
        if target_feature_indices is None:
            target_feature_indices = self._feature_table
        else:
            target_feature_indices = target_feature_indices + self._category_offsets

        composed_embeddings = torch.cat(
            (
                # Blank embedding (Using the index batch sequence equivalent to [[0]])
                self._attribute_embeddings(torch.zeros(1, 1, dtype=target_feature_indices.dtype, device=inputs.device)),
                # Phonemes
                self._attribute_embeddings(target_feature_indices),
            )
        ).T

        return (inputs @ composed_embeddings) / self._scale_factor


class AllophantLayers(AbsEncoder):

    @typechecked
    def __init__(
        self,
        input_size: int,
        phoneme_embedding_size: int,
        composition_features: List[str],
        allophone_languages: List[str],
        language_id_mapping_path: str,
        phoible_path: Union[str, None] = None,
        blank_offset: int = 1,
        use_allophone_layer: bool = True,
        utt_id_separator: str = "_",
    ):
        super().__init__()

        # Used by ESPnetAsrModel to identify encoders that take `utt_id` as an input
        # Required if `use_allophone_layer` is `True`
        # TODO: Disable if use_allophone_layer is False to reduce overhead?
        self.requires_utt_id = True

        self._separator = re.compile(utt_id_separator)

        with open(language_id_mapping_path, "r", encoding="utf-8") as file:
            self._language_map = json.load(file)
            reverse_map = {normalized: original for original, normalized in self._language_map.items()}

        indexer = PhoneticAttributeIndexer(
            FeatureSet.PHOIBLE,
            phoible_path,
            composition_features,
            language_inventories=allophone_languages,
            allophones_from_allophoible=True,
        )
        if use_allophone_layer:
            if indexer.allophone_data is None:
                raise ValueError(
                    "Model configuration using attribute embedding composition and an allophone layer"
                    " requires allophone data in the attribute indexer with but got `None`"
                )

            training_attributes = indexer.allophone_data.shared_phone_indexer
            language_allophones = LanguageAllophoneMappings.from_allophone_data(
                indexer,
                allophone_languages,
            )
        else:
            # Use the phoneme subset from the attribute indexer with all features for constructing the embeddings
            training_attributes = indexer.full_attributes.subset(
                indexer.phonemes.tolist(),
                indexer.composition_features.copy(),
            )
            language_allophones = None

        # Use either the phone or the phoneme subset from the attribute indexer with all features for constructing the embeddings
        training_features = indexer.full_attributes.subset(
            training_attributes.phonemes.tolist(),
            indexer.composition_features.copy(),
        ).dense_feature_table.long()

        # Projection from hidden size to the phoneme embedding size
        self._projection_layer = nn.Linear(input_size, phoneme_embedding_size)
        # Phonetic feature embedding composition for computing phoneme logits
        self._phoneme_composition_layer = EmbeddingCompositionLayer(phoneme_embedding_size, training_features)

        # Optional per-language allophone layer
        if language_allophones is None:
            self._output_size = self._phoneme_composition_layer.output_size()
            self._allophone_layer = None
            self._language_ids = {}
        else:
            self._language_ids = {reverse_map[language]: i for i, language in enumerate(language_allophones.languages)}
            self._output_size = indexer.phonemes.size + blank_offset
            self._allophone_layer = AllophoneMapping(
                self._phoneme_composition_layer.output_size(),
                self._output_size,
                language_allophones,
                blank_offset
            )

    # TODO: Only return phoneme classifier output size here?
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: Tensor,
        ilens: Tensor,
        prev_states: Optional[Tensor] = None,
        utt_id: Optional[List[str]] = None,
        target_feature_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if utt_id is None:
            raise ValueError(
                f"{self.__class__.__name__}.encode was called without utt_id, "
                "which is required by the encoder"
            )

        output = self._phoneme_composition_layer(self._projection_layer(xs_pad), target_feature_indices)
        if self._allophone_layer is not None:
            language_ids = torch.tensor([self._language_ids[self._separator.split(i, 2)[1]] for i in utt_id], dtype=torch.int64)
            # Assume that the allophone layer should not be used when a custom phoneme inventory is provided
            output = self._allophone_layer(output, language_ids, target_feature_indices is not None)

        return output, ilens, None
