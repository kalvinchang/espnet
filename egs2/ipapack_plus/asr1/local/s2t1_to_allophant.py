#!/usr/bin/env python
from pathlib import Path
from typing import Sequence
from argparse import ArgumentParser
import json
import os

from tqdm import tqdm


def _extract_phoneme_data(data_path: Path, mapping_out: Path, train_path: Path) -> None:
    # Create train path if it doesn't exist yet
    os.makedirs(train_path, exist_ok=True)

    supported_languages = {}
    mapping = {}
    # Assumes that text files are both ordered the same way but applies sanity checks
    with (
        (data_path / "text").open("r", encoding="utf-8") as txt,
        (data_path / "text.ctc").open("r", encoding="utf-8") as ctc
    ):
        for text_line, ctc_line in tqdm(zip(txt, ctc)):
            text_uttid, prompt = text_line.strip().split(" ", 1)

            if not text_uttid.endswith("_pr"):
                continue

            uttid, _ = ctc_line.split(" ", 1)
            if uttid != text_uttid:
                raise ValueError(
                    "text_file and textctc_file are either not sorted in the same order or a transcription is missing:"
                    f" {text_uttid!r} != {uttid!r}"
                )

            # Extract ISO639-3 code from language token
            lang_code = prompt.split(">", 1)[0][1:]
            # Assume every code is already in ISO639-3
            supported_languages[lang_code] = lang_code

            prefix, suffix = uttid.removesuffix("_pr").split("_", 1)
            mapping[text_uttid] = f"{prefix}_{lang_code}_{suffix}"

    with mapping_out.open("x", encoding="utf-8") as mapping_file:
        json.dump(mapping, mapping_file)

    with (train_path / "supported_languages.json").open("w", encoding="utf-8") as language_file:
        json.dump(supported_languages, language_file)


def main(args: Sequence[str]) -> None:
    parser = ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--mapping_out", type=Path)
    parser.add_argument("--train_path", type=Path)

    arguments = parser.parse_args(args)
    _extract_phoneme_data(arguments.data_path, arguments.mapping_out, arguments.train_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
