import argparse
import glob
from pathlib import Path

import webdataset as wds
from scipy.io import wavfile
import pandas as pd
from tarfile import ReadError
from phonepiece.ipa import read_ipa

# TODO: remove
from tqdm import tqdm

# adapted from https://github.com/juice500ml/espnet/blob/wav2gloss/egs2/
#       wav2gloss/asr1/local/data_prep.py


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


def generate_train_dev_test_splits(original_dataset, split, utt_id,
                                   doreco_splits):
    # DoReCo: language code (ex: ana1239)
        # use their splits (Table 11) - train/test
        # they use glottocode https://doreco.huma-num.fr/languages
    # MSWC: split + batch? (ex: mswc-dev-000001)
        # use their splits
    # FLEURS: language code + split (ex: af_za-test)
    if original_dataset == 'doreco':
        # 0148_DoReCo_doreco_anal1239_anm_20152111_Ngahring_PO_56_283.wav
        return (doreco_splits[doreco_splits['glottocode_or_prefix'] == split]
                .iloc[0]['split'])
    elif original_dataset == 'mswc':
        return split.split('-')[1]
    elif original_dataset == 'fleurs':
        return split.split('-')[1]


def normalize_text(ipa, ipa_tokenizer):
    # remove whitespace
    ipa = "".join(ipa.split())
    # phone(me) tokenizer (phonepiece)
    ipa_tokens = ipa_tokenizer.tokenize(ipa)
    # re-introduce spaces (like TIMIT)
    return " ".join(ipa_tokens)


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

    ipa_tokenizer = read_ipa()

    rows = []
    # glob is non-deterministic -> sort after globbing
    for i, path in tqdm(enumerate(sorted(args.source_dir.glob("*.tar")))):
        split = path.stem
        original_dataset = get_dataset_name(path)

        path = str(path)
        ds = wds.WebDataset(path).decode()
        ds = ds.to_tuple("__key__", "__url__", "npy", "__local_path__", "txt")

        try:
            for utt_id, _, audio, _, ipa in ds:
                new_path = (f'{args.source_dir}/{original_dataset}/'
                            f'{split}/{utt_id}.wav')
                Path(new_path).parent.mkdir(parents=True, exist_ok=True)
                # audio is just a list of samples.
                # SAMPLING_RATE determines how many samples per sec.
                # enforce min_wav_length
                if len(audio) / SAMPLING_RATE < min_wav_length:
                    continue
                wavfile.write(new_path, SAMPLING_RATE, audio)
                ipa = normalize_text(ipa, ipa_tokenizer)
                rows.append((utt_id, original_dataset, new_path, ipa))
        except ReadError as e:
            print('failed to untar', path, e)
        print('\nfinished', path)

    df = pd.DataFrame(rows)
    df.to_csv(args.source_dir.joinpath(f'{args.source_dir}/transcript.csv'), index=False,
        headers=['utt_id', 'dataset', 'path', 'ipa'])
    

    # TODO: kaldi format
    
