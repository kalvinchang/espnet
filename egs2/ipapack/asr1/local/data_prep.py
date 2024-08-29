import argparse
import glob
from pathlib import Path

import webdataset as wds
from scipy.io import wavfile
import pandas as pd

# TODO: remove
from tqdm import tqdm

# adapted from https://github.com/juice500ml/espnet/blob/wav2gloss/egs2/wav2gloss/asr1/local/data_prep.py


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
    return parser

def normalize_text():
    # TODO: remove spaces and re-introduce spaces
    # TODO: phoneme tokenizer
    pass

def get_dataset_name(tar_path):
    # get original dataset name
    if tar_path.name.startswith('mswc'):
        # MSWC: mswc-{train,dev,test}
        return 'mswc'
    elif tar_path.stem.endswith(('train','dev','test')):
        # FLEURS: -{train,dev,test}.tar
        return 'fleurs'
    else:
        return 'doreco'


if __name__ == "__main__":
    SAMPLING_RATE = 16000

    parser = get_parser()
    args = parser.parse_args()

    for dataset in ['mswc', 'fleurs', 'doreco']:
        print("making directory", args.source_dir.joinpath(dataset))
        args.source_dir.joinpath(dataset).mkdir(parents=True, exist_ok=True)
    print("making directory", args.target_dir)
    args.target_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in tqdm(args.source_dir.glob("*.tar")):
        split = path.stem
        original_dataset = get_dataset_name(path)

        path = str(path)
        ds = wds.WebDataset(path).decode()
        ds = ds.to_tuple("__key__", "__url__", "npy", "__local_path__", "txt")

        for utt_id, _, audio, _, ipa in ds:
            new_path = (f'{args.source_dir}/{original_dataset}/'
                        f'{split}/{utt_id}.wav')
            Path(new_path).parent.mkdir(parents=True, exist_ok=True)
            # audio is just samples. SAMPLING_RATE determines how many samples per sec.
            wavfile.write(new_path, SAMPLING_RATE, audio)
            # TODO: remove the path

            rows.append((utt_id, original_dataset, new_path, ipa))

        print('finished', path)
    
    pd.DataFrame(rows).to_csv(args.source_dir.joinpath('transcript.csv'), index=False)

    # TODO: kaldi format
