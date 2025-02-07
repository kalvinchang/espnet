#!/usr/bin/env python
import os
from typing import Dict, List, Sequence, Set
from argparse import ArgumentParser
from pathlib import Path
import json
import warnings
import shutil

from tqdm import tqdm


_FILES = {
    "orthography": 0,
    "spk2utt": slice(1, None),
    "text": 0,
    "text.ctc": 0,
    "utt2spk": 0,
    "uttid_map": 1,
    "utt2num_samples": 0,
    "wav.scp": 0,
}

_TO_COPY = {"audio_format", "feats_type"}
_IGNORED_FOLDERS = {"data", "logs"}


def create_data_subset(source_dir: Path, target_dir: Path, uttid_transform: Path | Dict[str, str] | List[str], auxiliary_data_tags: Set[str] | None = None, ctc_to_text: bool = False) -> None:
    if not isinstance(uttid_transform, (dict, list)):
        with uttid_transform.open("r", encoding="utf-8") as file:
            uttid_transform = json.load(file)
            if not isinstance(uttid_transform, (dict, list)):
                raise ValueError(f"Unexpected JSON root type: {uttid_transform}")

    os.makedirs(target_dir, exist_ok=True)

    # Treat lists as simple filters with identity transform
    if isinstance(uttid_transform, list):
        uttid_transform = {uttid: uttid for uttid in uttid_transform}

    if auxiliary_data_tags is None:
        auxiliary_data_tags = set()

    files = {}
    unknown = []
    for file in source_dir.iterdir():
        name = file.name
        # Detect any spurious directories
        if file.is_dir():
            if name in _IGNORED_FOLDERS:
                continue
            unknown.append(file)
        elif (uttid_column := _FILES.get(name)) is not None:
            files[name] = uttid_column
        elif name in auxiliary_data_tags:
            files[name] = 0
        elif name in _TO_COPY:
            shutil.copy2(file, target_dir / name)
        else:
            unknown.append(file)

    if unknown:
        warnings.warn(f"The following files are not known and will be ignored while creating the subset: {unknown}")

    for filename, uttid_column in tqdm(files.items()):
        phoneme_cleanup = False
        target_filename = filename

        if ctc_to_text:
            if filename == "text":
                continue
            elif filename == "text.ctc":
                target_filename = "text"
                phoneme_cleanup = True

        with (source_dir / filename).open("r", encoding="utf-8") as source, (target_dir / target_filename).open("x", encoding="utf-8") as target:
            for line in tqdm(source, leave=False, position=1):
                columns = line.split()
                if isinstance(uttid_column, int):
                    uttid = columns[uttid_column]
                    # Filter rows
                    if uttid not in uttid_transform:
                        continue
                    # Transform the uttid
                    columns[uttid_column] = uttid_transform[uttid]

                    # Assumes this can only be the when utt_ids are in column 0
                    if phoneme_cleanup:
                        columns[1] = " ".join(columns[1].removeprefix("/").removesuffix("/").split("//"))
                else:
                    previously_empty = not columns[uttid_column]
                    # Filter and transform all uttids
                    transformed_columns = [uttid_transform[uttid] for uttid in columns[uttid_column] if uttid in uttid_transform]
                    # Remove rows which weren't already empty and all uttids have been filtered from
                    if not previously_empty and not transformed_columns:
                        continue

                    # Replace uttid_columns with filtered and/or transformed ones
                    columns[uttid_column] = transformed_columns

                target.write(" ".join(columns) + "\n")



def main(args: Sequence[str]) -> None:
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("-m", "--uttid_transform", type=Path)
    parser.add_argument("-a", "--auxiliary_data_tags", type=str.split)
    parser.add_argument("-c", "--ctc_to_text", action="store_true")

    arguments = parser.parse_args(args)
    create_data_subset(
        arguments.source_dir,
        arguments.target_dir,
        arguments.uttid_transform,
        set(arguments.auxiliary_data_tags) if arguments.auxiliary_data_tags is not None else None,
        arguments.ctc_to_text,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
