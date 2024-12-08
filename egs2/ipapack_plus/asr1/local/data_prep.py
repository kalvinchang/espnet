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
    splits = defaultdict(list) # split -> dataset name
    for shard in dataset_shards:
        shard_name = shard.stem
        dataset = shard_name.replace("_shar", "")
        for orig_split in shard.iterdir():
            orig_split_name = orig_split.stem
            split = get_split(source_dir, dataset, orig_split_name)
            splits[split].append((dataset, orig_split_name))

    return splits


def get_utt_id(dataset, split, count):
    return f"aaaaa_{dataset}_{split}_{count:025d}"


def generate_df(source_dir, data_dir):
    # get list of datasets in IPAPack++
    dataset_shards = list(source_dir.glob('*_shar'))
    # ex: downloads/aishell_shar -> aishell_shar
    datasets = [d.parts[1] for d in dataset_shards]
    datasets = list(set(datasets))
    # set and glob are non-deterministic, so sort
    datasets = sorted(datasets)
    logging.info(f"{len(datasets)} speech train data files found: {datasets}")

    rows, utt_count = [], 1

    logging.info("Starting to process dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = generate_train_dev_test_splits(source_dir, dataset_shards)

    for split, split_datasets in splits.items():
        for i, (dataset, orig_split_name) in tqdm(enumerate(split_datasets)):
            # ex: downloads/mls_portuguese/test - untarred files
            dataset_path = source_dir / dataset / orig_split_name
            # ex: downloads/mls_portuguese_shar/test - the cuts are here
            shar_folder = dataset + '_shar'
            shar_path = source_dir / shar_folder / orig_split_name
            logging.info("Processing %s" % dataset)

            # glob is non-deterministic -> sort after globbing
            #   order is important
            #   b/c CutSet assumes cuts is in the same order as recording
            supervision = sorted(shar_path.glob('cuts*'))
            supervision = [str(f) for f in supervision]
            recording = sorted(shar_path.glob('recording*'))
            recording = [str(f) for f in recording]
            assert len(supervision) == len(recording)

            logging.info(f"{len(supervision)} shards found")

            # load from the downloaded shard
            cuts = CutSet.from_shar(
                        {
                            "cuts": supervision,
                            "recording": recording
                        }
                    )
            # each cut is like an utterance
            for cut in tqdm(cuts, miniters=1000):
                metadata = cut.supervisions
                if len(metadata) == 0:
                    logging.error('metadata list length 0')
                    continue
                elif len(metadata) != 1:
                    logging.error('metadata list longer than 1')
                metadata = metadata[0]

                # utterance level information
                old_utt_id = metadata.recording_id
                utt_id = get_utt_id(dataset, split, utt_count)
                utt_count += 1
                duration = metadata.duration

                lang = metadata.language
                speaker = metadata.speaker
                # transcript
                text = ''
                if 'orthographic' in metadata.custom:
                    text = metadata.custom['orthographic']
                ipa_original = ''
                if 'original' in metadata.custom:
                    ipa_original = metadata.custom['original']
                elif 'phones' in metadata.custom:
                    ipa_original = metadata.custom['phones']
                ipa_clean = metadata.text
                shard = ''
                if 'shard_origin' in cut.custom:
                    shard = cut.custom['shard_origin']
                # path to audio
                path = str((dataset_path / utt_id).with_suffix('.wav'))
                rows.append((utt_id, old_utt_id, dataset, split, shard, duration, lang, speaker, text, ipa_original, ipa_clean, path))

            logging.info(f"{dataset} done! {len(split_datasets)-i-1}" +
                          "datasets remaining for the split.")

    columns = [
        'utt_id', 'old_utt_id', 'dataset', 'split', 'shard', 'duration',
        'lang', 'speaker',
        'text', 'ipa_original', 'ipa_clean',
        'path'
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(source_dir / 'transcript.csv', index=False)
    logging.info("saved transcripts and metadata to downloads/transcript.csv")
    return df


def normalize_phones(transcription):
    # remove long vowels
    # use IPA ɡ
    transcription = transcription.replace("ː", "").replace("g", "ɡ")

    # remove whitespace
    ipa = "".join(transcription.split())
    # phone(me) tokenizer (ipatok)
    ipa_tokens = tokenise(ipa)
    # re-introduce spaces (like TIMIT)
    return " ".join(ipa_tokens)



def df_to_kaldi(df, source_dir, data_dir):
    # preprocess into kaldi format
    ipa_tokenizer = read_ipa()

    # kaldi format
    for split, split_df in df.groupby('split'):
        print("testing", split)

        # to save time
        if "test" not in split:
            continue

        split_dir = args.target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_dir(args.source_dir, split_dir, split_df)


# adapted from https://github.com/juice500ml/espnet/blob/wav2gloss/egs2/
#       wav2gloss/asr1/local/data_prep.py
def write_dir(source_dir, target_dir, transcripts):
    wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
    text = open(target_dir / "text", "w", encoding="utf-8")
    utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
    utt_id_mapping = open(source_dir / "uttid_map", "w", encoding="utf-8")

    for _, row in transcripts.iterrows():
        utt_id, path, dataset, ipa = (row['utt_id'], row['path'],
                                                  row['dataset'],
                                                  row['ipa'])

        old_utt_id = row['old_utt_id']
        # map original utt_id to new utt_id (note: not required by kaldi)
        utt_id_mapping.write(f"{old_utt_id} {utt_id}\n")

        wavscp.write(f"{utt_id} {path}\n")
        text.write(f"{utt_id} {ipa}\n")
        # ESPnet does not use speaker info for ASR anymore
        utt2spk.write(f"{utt_id} aaaaa\n")

    wavscp.close()
    text.close()
    utt2spk.close()
    utt_id_mapping.close()

    logging.info(f"{target_dir}: {len(transcripts)} lines
        written to {str(target_dir)}.")


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

    output = Path(source_dir / 'transcript.csv')
    if output.exists():
        logging.info(f"loading transcripts and metadata from {str(output)}")
        df = pd.load_csv(output)
    else:
        df = generate_df(source_dir, target_dir)
    
    # exclude the following langs
        # from FLEURS: 'ga_ie', 'sd_in', 'ar_eg', 'ml_in', 'lo_la', 'da_dk', 'ko_kr', 'ny_mw', 'mn_mn', 'so_so', 'my_mm'
        # Samir et al 2024 found that the data available for
        #   these languages unfortunately have low quality transcriptions.
    FLEURS_EXCLUDE = {
        'ga_ie', 'sd_in', 'ar_eg', 'ml_in', 'lo_la', 'da_dk', 'ko_kr',
        'ny_mw', 'mn_mn', 'so_so', 'my_mm'
    }
    df = df[~df['split'].isin(FLEURS_EXCLUDE)]

    # normalize phones
    df['split'] = df.apply(lambda row: normalize_phones(row['text']), axis=1)
    df.to_csv(source_dir / 'transcript_normalized.csv', index=False)

    df_to_kaldi(df, source_dir, target_dir)
