#!/usr/bin/env python
import itertools
import json
from pathlib import Path
import re
import shutil
from typing import Dict, Iterator, List, Sequence, Tuple
from argparse import ArgumentParser
import unicodedata

import yaml


def _normalize_phoneme(feature_type: str, phoneme: str) -> str:
    if feature_type == "phoible":
        # Special case for PHOIBLE
        return unicodedata.normalize("NFD", phoneme).replace("ç", "ç")
    else:
        return unicodedata.normalize("NFD", phoneme)


def _map_indices(
    token_int_path: Path,
    inventories: Dict[str, Dict[int, str]],
    utt_id_seperator: str
) -> Iterator[Tuple[str, List[str]]]:
    with token_int_path.open("r", encoding="utf-8") as file:
        for line in file:
            utt_id, indices = line.strip().split(" ", 1)
            inventory = inventories[re.split(utt_id_seperator, utt_id)[1]]
            yield utt_id, [inventory[index] for index in map(int, indices.split(" "))]


def main(args: Sequence[str]) -> None:
    parser = ArgumentParser()
    parser.add_argument("--decoded_dir", type=Path, required=True)
    parser.add_argument("--decode_config_path", type=Path, required=True)
    parser.add_argument("--model_config_path", type=Path, required=True)

    arguments = parser.parse_args(args)

    with arguments.decode_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        # Don't perform mapping if the full vocabulary was used and it is not required
        if not config.get("use_language_vocabulary", False):
            return

    with arguments.model_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        encoder_conf = config["encoder_conf"]
        inventories_path = encoder_conf.get("composition_inventories_file")
        language_id_map_path = encoder_conf.get("language_id_mapping_path")
        feature_type = encoder_conf.get("feature_type", "panphon")
        utt_id_seperator = encoder_conf.get("utt_id_seperator", "_")
        phoneme_inventory_path = encoder_conf.get("phoneme_inventory_file")

        if not inventories_path or not language_id_map_path or not phoneme_inventory_path:
            raise ValueError(
                "Cannot map token IDs for a model without a configured"
                " composition_inventories_file, language_id_map_path, and phoneme_inventory_path"
            )

    with open(language_id_map_path, "r", encoding="utf-8") as file:
        language_id_map = json.load(file)
        reverse_map = {code: original for original, code in language_id_map.items()}

    with open(phoneme_inventory_path) as file:
        special_symbols = []
        for line in file:
            line = line.strip()
            if not (line.startswith("<") and line.endswith(">")):
                break
            special_symbols.append(line)

    with open(inventories_path, "r", encoding="utf-8") as file:
        inventories = json.load(file)
        composition_inventories = {
            reverse_map[code]: {
                index: _normalize_phoneme(feature_type, phoneme)
                for index, phoneme in enumerate(
                    itertools.chain(special_symbols, inventory)
                )
            } for code, inventory in inventories.items()
        }

    # TODO: map_to_phoible style backup
    data_set_dir: Path = arguments.decoded_dir

    backup_path = data_set_dir / "token.backup"
    if not backup_path.exists():
        shutil.copy2(data_set_dir / "token", backup_path)

    backup_path = data_set_dir / "text.backup"
    if not backup_path.exists():
        shutil.copy2(data_set_dir / "text", backup_path)

    with (
        (data_set_dir / "token").open("w", encoding="utf-8") as token_file,
        (data_set_dir / "text").open("w", encoding="utf-8") as text_file
    ):
        for utt_id, phonemes in _map_indices(
            data_set_dir / "token_int", composition_inventories, utt_id_seperator
        ):
            line = f"{utt_id} {' '.join(phonemes)}\n"
            token_file.write(line)
            text_file.write(line)



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
