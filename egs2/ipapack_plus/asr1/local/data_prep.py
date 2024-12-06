import argparse
from collections import defaultdict
import glob
import logging
import os
from pathlib import Path
import sys

from ipatok import tokenise
from glob import glob
from lhotse import CutSet
import pandas as pd
from scipy.io import wavfile
from tarfile import ReadError
from tqdm import tqdm
import webdataset as wds


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert downloaded data to Kaldi format"
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("downloads"),
        required=True
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=Path("data"),
        required=True
    )
    parser.add_argument(
        "--min_wav_length",
        type=float,
        default=0.5,
    )
    return parser


def get_split(source_dir, dataset, orig_split):
    TEST_SETS = {
        "librispeech", "mls", "aishell"
    }

    # use the splits Jian already created
    if dataset == "doreco":
        # all of DoReCo is a test set
        return "test_doreco"
    elif "test" in orig_split:
        return f"test_{dataset}"
    elif "dev" in orig_split:
        return "dev"
    else:
        # may not always contain train/dev/test
        # e.g. kazakh2_shar/audio2
        return "train"


def generate_train_dev_test_splits(source_dir, dataset_shards):
    train_dev_test_splits = {}

    # the subdirectories of shard_name are the original splits from Jian
    splits = defaultdict(list) # split -> dataset
    for shard in dataset_shards:
        shard_name = shard.stem
        dataset = shard_name.replace("_shar", "")
        for orig_split in shard.iterdir():
            orig_split_name = orig_split.stem
            split = get_split(source_dir, dataset, orig_split_name)
            splits[split].append(orig_split)

    return splits


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y/%b/%d %H:%M:%S",
        stream=sys.stdout
    )

    SAMPLING_RATE = 16000

    parser = get_parser()
    args = parser.parse_args()
    min_wav_length = args.min_wav_length
    source_dir = args.source_dir
    data_dir = args.target_dir

    # get list of datasets in IPAPack++
    dataset_shards = source_dir.glob('*_shar')
    # ex: downloads/aishell_shar -> aishell_shar
    datasets = [d.parts[1] for d in dataset_shards]
    datasets = list(set(datasets))
    # set and glob are non-deterministic, so sort
    datasets = sorted(datasets)
    logging.info(f"{len(datasets)} speech train data files found: {datasets}")

    logging.info("Beginning processing dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = generate_train_dev_test_splits(source_dir, dataset_shards)
    for split, split_datasets in splits.items():
        for i, dataset in tqdm(enumerate(split_datasets)):
            logging.info("Processing %s" % data_path)

            # glob is non-deterministic -> sort after globbing
            supervision = sorted(data_path.glob('*/cuts*'))
            supervision = [str(f) for f in supervision]
            recording = sorted(data_path.glob('*/recording*'))
            recording = [str(f) for f in recording]
            assert len(supervision) == len(recording)

            logging.info("%s shards found" % len(supervision))

            cuts = CutSet.from_shar(
                        {
                            "cuts": supervision,
                            "recording": recording
                        }
                    )

            for cut in tqdm(cuts):
                writer.write(cut)

            logging.info(f"Processing done! {len(datasets)-i-1}" +
                          "datasets remaining.")
