#!/usr/bin/env python
import itertools
from pathlib import Path
import shutil
from typing import Dict, List, Sequence
from argparse import ArgumentParser
import pandas as pd
import re
import json
from tqdm import tqdm
import re
import unicodedata

from phonepiece.ipa import read_ipa
from allophant.phonemes import IpaSegmenter
from allophant.phonetic_features import PhoneticAttributeIndexer, FeatureSet

# from allophant import phoneme_segmentation
from allophant import language_codes


# Mappings for unknown/incomplete glottocodes in DoReCo
_DORECO_MAPPINGS = {"ana1239": "anal1239", "trin178": "trin1274"}

_TIE = "͡"
_LOWER_TIE = "͜"


def get_iso6393(
    language_code: str,
    dataset: str,
    glottomap_codes: Dict[str, str],
    glottomap_closest: Dict[str, str],
) -> str | None:
    match dataset:
        case "doreco":
            try:
                code = glottomap_codes.get(_DORECO_MAPPINGS.get(language_code, language_code))
                if code is None:
                    code = glottomap_closest[language_code]
                    tqdm.write(
                        f"INFO: No direct code for {language_code}, "
                        f"falling back to closest code in Glottolog: {code}"
                    )
                return code
            except KeyError:
                tqdm.write(f"WARNING: No ISO639-3 code found for {language_code}")
                return None
        case "fleurs" | "mswc":
            # Handle non-standard code for simplified script
            if language_code == "zho-s":
                language_code = "zho-Hans"
            else:
                # Removes non-standard Babel suffix (-bab)
                language_code = language_code.removesuffix("-bab")
            return language_codes.standardize_to_iso6393(language_code)

    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _normalize_phoneme(phoneme: str) -> str:
    # Handle special case of ç in PHOIBLE not being in NFD
    return unicodedata.normalize("NFD", phoneme) if "ç" not in phoneme else phoneme


def _is_mark(character: str) -> bool:
    category = unicodedata.category(character)
    return category.endswith("m") or category == "Sk" or category.startswith("M")


def extract_vocabulary(
    train_text_path: Path,
    dump_dir: Path,
    phoneme_indexer: PhoneticAttributeIndexer,
    glottomap: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    delimiters = re.compile("[ ˈ]")
    inventory = set()
    inventories: dict[str, set[str]] = {}
    glottomap_codes = glottomap["code"]
    glottomap_closest = glottomap["closest"]

    phoneme_list = phoneme_indexer.full_attributes.phonemes.tolist()
    language_mappings = {}
    supported_phonemes = set(phoneme_list)
    missing_phonemes = set()
    tokenizer = IpaSegmenter(phoneme_list)
    phonepiece = read_ipa()

    with (train_text_path).open("r", encoding="utf-8") as file:
        last_language = ""
        language = ""

        for line in tqdm(file):
            try:
                utt_id, transcription = line.strip().split(" ", 1)
            except ValueError:
                tqdm.write(line.strip())
                continue
            # Workaround for inconsistent DoReCo subset naming
            _, language_code, dataset, _ = re.split(
                r"_daaki_|_light_warlpiri_|_mojeno_trinitario_|_north_alta_|_",
                utt_id,
                3,
            )

            if language_code != last_language:
                last_language = language_code
                try:
                    language = (
                        get_iso6393(
                            language_code, dataset, glottomap_codes, glottomap_closest
                        )
                        or "MIS-" + language_code
                    )
                except Exception:
                    print(utt_id, language_code, dataset)
                    raise

                language_mappings[language_code] = language
                inventory = inventories.get(language)
                if inventory is None:
                    inventories[language] = inventory = set()

            phonemes = [
                tokenizer.segment(
                    phoneme.replace(_TIE, "").replace(_LOWER_TIE, ""), True
                )
                for word in delimiters.split(transcription)
                for phoneme in phonepiece.tokenize(word)
            ]
            for phoneme in phonemes:
                subsegment = _normalize_phoneme(phoneme[0])
                if len(phoneme) != 1 or subsegment not in supported_phonemes:
                    # Handle rare cases in IPAPack where diacritic ordering
                    # doesn't correspond to the canonical ordering in PHOIBLE
                    reordered = subsegment[:-2] + subsegment[-2::-1]
                    if (
                        all(map(_is_mark, subsegment[-2::-1]))
                        and reordered not in supported_phonemes
                    ):
                        unsplit = "".join(map(_normalize_phoneme, phoneme))
                        missing_phonemes.add((tuple(phoneme), unsplit))
                        # inventory.add(unsplit)
                        continue

                    subsegment = reordered

                inventory.add(subsegment)

    with (dump_dir / "inventories.json").open("w", encoding="utf-8") as file:
        # Sort inventories for consistency
        inventory_lists = {
            language: sorted(phonemes) for language, phonemes in inventories.items()
        }
        if missing_phonemes:
            inventory_lists["MISSING"] = sorted(missing_phonemes)

        json.dump(inventory_lists, file)

    with (dump_dir / "supported_languages.json").open("w", encoding="utf-8") as file:
        json.dump(language_mappings, file)

    print("Phoneme inventories and supported languages extracted")

    return inventory_lists


def collect_inventories(
    dump_dir: Path,
    data_dir: Path,
    glottolog_languages: Path,
    skip_vocab_generation: bool = False,
) -> None:
    with open("local/allophoible/allophoible_v2.csv", "r", encoding="utf-8") as file:
        phoneme_indexer = PhoneticAttributeIndexer(
            FeatureSet.PHOIBLE, file, allophones_from_allophoible=True
        )

    glottomap = pd.read_csv(glottolog_languages)
    glottomap = glottomap[["ID", "ISO639P3code", "Closest_ISO369P3code"]].set_index(
        "ID"
    )
    glottomap = {
        "code": glottomap["ISO639P3code"].dropna().to_dict(),
        "closest": glottomap["Closest_ISO369P3code"].dropna().to_dict(),
    }

    train_text_path = dump_dir / "train/text"

    if skip_vocab_generation:
        with (dump_dir / "inventories.json").open("r", encoding="utf-8") as file:
            inventory_lists = json.load(file)
    else:
        inventory_lists = extract_vocabulary(
            train_text_path, dump_dir, phoneme_indexer, glottomap
        )

    if missing_phonemes := inventory_lists.get("MISSING"):
        raise ValueError(
            f"Found {len(missing_phonemes)} phonemes missing from the database"
        )

    words_dir = data_dir / "token_list/word"
    token_path = words_dir / "tokens.txt"

    # Backup original token inventory
    shutil.copy2(token_path, words_dir / "tokens.txt.original")

    with token_path.open("r", encoding="utf-8") as file:
        clusters = []
        current = []
        end_special_tokens = []

        last_special = True

        for token in map(str.strip, file):
            if token[0] != "<" or token[-1] != ">":
                last_special = False
                continue

            if not last_special:
                clusters.append(current)
                current = []

            last_special = True
            current.append(token)

        if current:
            if last_special:
                end_special_tokens = current
            else:
                clusters.append(current)

        # Appends any special tokens appearing after the first cluster to the start
        start_special_tokens = sum(clusters, [])

    with open("local/allophoible/allophoible_v2.csv", "r", encoding="utf-8") as file:
        # Resolve allophone inventories for all languages in the data
        phoneme_indexer = PhoneticAttributeIndexer(
            FeatureSet.PHOIBLE,
            file,
            language_inventories=list(inventory_lists),
            allophones_from_allophoible=True,
        )

    # Map to PHOIBLE inventories with allophone information
    phoneme_mappings = {}
    for language, inventory in tqdm(inventory_lists.items()):
        if phoneme_indexer.allophone_inventory(language).size == 0:
            tqdm.write(
                f"WARNING: No allophone inventory found for {language}, skipping..."
            )
            # Default to empty mapping
            phoneme_mappings[language] = {}
            continue

        phoneme_mappings[language] = phoneme_indexer.map_language_inventory(
            [inventory], language
        )[0]

    with (dump_dir / "mappings.json").open("w", encoding="utf-8") as file:
        json.dump(phoneme_mappings, file)

    with token_path.open("w", encoding="utf-8") as file:
        shared_mapped_inventory = {
            phoneme
            for target_phonemes in phoneme_mappings.values()
            for phoneme in target_phonemes
        }
        # Replace the original with the mapped shared phoneme inventory
        # while preserving special token positions at the start and end
        file.writelines(
            token + "\n"
            for token in itertools.chain(
                start_special_tokens, shared_mapped_inventory, end_special_tokens
            )
        )

    print(f"Wrote new shared inventory to {token_path}")
    return

    # Backup original transcriptions
    shutil.copy2(train_text_path, dump_dir / "train/text.original")
    dev_text_path = dump_dir / "dev/text"
    shutil.copy2(dev_text_path, dump_dir / "dev/text.original")

    last_language = ""
    language = ""
    mapping = {}

    print(f"Remapping {train_text_path}...")

    with (train_text_path).open("r", encoding="utf-8") as file:
        for line in tqdm(file):
            utt_id, *transcription = (
                line.strip().replace(_TIE, "").replace(_LOWER_TIE, "").split(" ")
            )
            # Workaround for inconsistent DORECO subset naming
            _, language_code, dataset, _ = re.split(
                r"_light_warlpiri_|_daaki_|_", utt_id, 3
            )

            if language_code != last_language:
                last_language = language_code
                language = get_iso6393(language_code, dataset, glottomap)
                mapping = {} if language is None else phoneme_mappings[language]

            remapped = [
                remapped
                for phoneme in transcription
                for remapped in mapping.get(phoneme, phoneme)
            ]
            # print(len(transcription), len(remapped), transcription, remapped)


def main(args: Sequence[str]) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dump_dir",
        default="dump/raw/",
        type=Path,
        help="directory containing asr data from stage 1",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        type=Path,
        help="directory containing processed asr data from stage 4",
    )
    parser.add_argument(
        "--glottolog_languages",
        default="local/glottolog_language_data/languages.csv",
        type=Path,
        help="path to a glottolog languages.csv file for handling glottocodes in DORECO",
    )
    parser.add_argument(
        "--skip_vocab_generation",
        action="store_true",
        help="Skips directly to phoneme remapping using previously extracted phoneme vocabularies",
    )

    arguments = parser.parse_args(args)
    collect_inventories(
        arguments.dump_dir,
        arguments.data_dir,
        arguments.glottolog_languages,
        arguments.skip_vocab_generation,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
