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
                #   {recording_id}-{idx}-{channel}.flac
                old_utt_id = cut.id
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
                # do not use .with_suffix('.flac') b/c kazakh2 old_utt_id's
                # look like dataset_audio2_21_538_1.wav-0.flac
                #   which suggests the dataset was accidentally unpacked twice
                path = f"{str(dataset_path)}/{old_utt_id}.flac"
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
        # affricates without tie bar (e.g. ts) are broken into 2 segments: ts -> t s
    ipa_tokens = tokenise(ipa)
    # reduce phonemes
        # remove vowel and consonant length b/c it's inconsistently marked
        # not all keys will be used
    mapping = {'p̃': 'p', 'l̴̩': 'l̴', 'iʲ': 'i', 'ɨʲ': 'ɨ', 'ɔʲ': 'ɔ', 'ʲ': '', 'ʔˤ': 'ʔ', 'nʼ': 'n', 'eʼ': 'e', 'ŋ̪': 'ŋ', 'jj': 'j', 'l̤': 'l', 'd͡zʷ': 'd͡z', 'bˤː': 'b', 'o̞': 'o', 'ɲ̤': 'ɲ', 'rˤː': 'r', 'ʒʲː': 'ʒ', 'tʼː': 't', 'ʃʰ': 'ʃ', 'sʲː': 's', 'ɖʰ': 'ɖ', 'i̤': 'i', 'sˠ': 's', 'ɔ̃ː': 'ɔ', 'i̝': 'i', 't̪ˠ': 't', 'e̤': 'e', 't̪͡s̪': 't͡s', 'ŋː': 'ŋ', 'ɪˤː': 'ɪ', 't͡ʃʼː': 't͡ʃ', 'ɽ̤': 'ɽ', 'n̪ː': 'n', 'ɔ̤': 'ɔ', 'ʑː': 'ʑ', 'ʂː': 'ʂ', 'lʲː': 'l', 'β̞': 'β', 'u̝': 'u', 't͡ʃʲː': 't͡ʃ', 'pʲː': 'p', 'd͡ʒˤː': 'd͡ʒ', 'd͡ʒʲ': 'd͡ʒ', 't͡sʰ': 't͡s', 'n̤': 'n', 'ɜː': 'ə', 'rʲː': 'r', 'əː': 'ə', 'tʰː': 't', 'sˤː': 's', 'ʋʲː': 'ʋ', 'ʔʲ': 'ʔ', 'ɑ̯': 'ɑ', 'ũː': 'u', 'ĩː': 'i', 'xʰ': 'x', 'dˤː': 'd', 'ɟː': 'ɟ', 'ɨ̃': 'ɨ', 't̪͡s̪ʲ': 't͡s', 'w̤': 'w', 'lˤː': 'l', 'ʊˤː': 'ʊ', 'æ̃': 'æ', 'hː': 'h', 'ɱ̩': 'ɱ', 'vʲː': 'v', 'mʲː': 'm', 'æ̯': 'æ', 'e̞': 'e', 'xʷ': 'x', 'fˤː': 'f', 'h̩': 'h', 'o̤': 'o', 'ķ': 'k', 'œːˀ': 'œː', 'ɪ́': 'ɪ', 'ɑː̃': 'ɑː', 'ɲ̊': 'ɲ', 'j̩̩̩': 'j', 'r̂': 'r', 'ɛ̯ˑ': 'ɛ', 'çː': 'ʝ', 'ãː': 'a', 't̩̩̩̩̩̩̩̩': 't', 'x⁽ʲ⁾': 'x', 'uː́': 'uː', 'mʰ': 'm', 'gʲ̩ʲ': 'ɡʲ', 'r͈ʲ': 'rʲ', 'ɑː̆': 'ɑː', 'j̩̩̩̩': 'j', 'ĩ̯': 'i', 'e͜oː': 'eoː', 'ŋ̍': 'ŋ', 'ɑ̃̃': 'ɑ̃', 'd̚': 'd', 'ʋʰ': 'ʋ', 'ɛ̃̃': 'ɛ̃', 'ð̠ˠ': 'ð', 'u͡ə': 'uə', 'ʼə': 'ə', 'i˔': 'i', 'ʼa': 'a', 'ʲa': 'ja', 'u̩': 'u', 'ɪ̯̌ː': 'ɪ̯', 'ä̂': 'a', 'ə͡ʊ': 'əʊ', 'ĩ̃': 'i', 'ɛ̝ː': 'ɛ', 'iː́': 'iː', 'â͡l': 'al', 'r̂ˑ': 'r', 'b̥ˀ': 'b', 'ŋ̪ˠ': 'ŋ', 'vʲ̩': 'vʲ', 'ʒ͡ʲ': 'ʒ', 'ğ': 'ɡ', 'ɡ̆': 'ɡ', 'ó': 'o', 'ş': 's', 't͡s̪': 't͡s', 'i͜y': 'iy', 't͡ş': 't͡ʃ', 'd̤͡z̤': 'd͡z', 'ɒ͜ú': 'ɒu', '˧ˀ': '', 'ɒ͜úˑ': 'ɒ', 'ʉː́': 'ʉː', 'k̩̩̩': 'k', 'l̪ᵊ': 'l̪', 'û': 'u', 'ð̠ˠˀ': 'ð', 'i͜yː': 'i', 'l˔': 'l', 'jʲ': 'j', 'ʃːˀ': 'ʃː', 'âː': 'aː', 'æ͜ɑː': 'æ', 's˔': 's', 'ŋ́ˑ': 'ŋ', 'n̪ˑ': 'n̪', 'ɑː́': 'ɑː', 'iˑ': 'i', 'uː̃': 'uː', 'kˠ': 'k', 'ž': 'ʒ', '˩˩': '', 'ɫ̪': 'ɫ', 'd̩': 'd', 'mᵑ': 'mŋ', 'zʰ': 'z', 'o̞ː': 'o', 'ɡ̥': 'ɡ', 'kʲ̩̩ʲ': 'kʲ', 'æ͡iː': 'æi', 't͡': 't', '‘': '', 'œ̞ː': 'œ', 't⁽ʲ⁾': 't', 'ɪ̯ˀ': 'ɪ̯', 'ɾ̪ː': 'ɾ', 'aːˀ': 'aː', 'äˑ': 'a', 'âˑ': 'a', 'ɖ̚': 'ɖ', 'ú': 'u', 'ø͡i': 'øi', 'ñ': 'n', 'ṳ': 'u', 'uˑ': 'u', 'œ̞ːˀ': 'œ', 'æ͜ɑ': 'æ', 'gː': 'ɡː', 'b̩': 'b', 'ʊ̯ˀ': 'ʊ̯', 'é': 'e', 'ɾ̪ˠ': 'ɾ', 'ûː': 'uː', 'ɛ̂': 'ɛ', 'æ̝ː': 'æ', 'iˀː': 'i', 'p̪': 'p', 'ðˠˀ': 'ð', 'mᵊ': 'mə', 'o̩': 'o', 'kʲ̩ʲ': 'kʲ', 'ũ̯': 'u', 'ʋᵊ': 'ʋə', 'ʈ̚': 'ʈ', 'ü': 'a', 'ǒː': 'oː', 'ěː': 'eː', '˨˩˦': '', 'p̩': 'p', 'b̚': 'b', 'dˤˤ': 'd', 'êː': 'eː', 'ʌʲ': 'ʌj', 'š': 'ʃ', 'p⁽ʲ⁾': 'p', 'aˑ': 'a', 'y͡i': 'yi', 'îː': 'iː', 'f⁽ʲ⁾': 'f', 'æː͡ɪ': 'æɪ', 'ǁ̰': 'ǁ', 'n⁽ʲ⁾': 'n', 'wᵊ': 'wə', 'ǔː': 'uː', 'ʲu': 'ju', 'ǐː': 'iː', 'ɔ̩': 'ɔ', 'v⁽ʲ⁾': 'v', 'ɔˑ': 'ɔ', 'ì': 'i', 'ʈːʰ': 'ʈː', 'ɫ̩ː': 'ɫ', 'r̩ː': 'r̩', 'æ̌ː': 'æ', 'j̊': 'j', 'n̪ᵊ': 'n̪ə', 'ɪ̂': 'ɪ', 'gʲ̩̩': 'ɡʲ', 'ᵐv': 'mv', '”': '', 'ɔːˀ': 'ɔː', 'ʷo': 'wo', 'k̩': 'k', 'ⁿz': 'nz', 'ã': 'ã', 'aː̃': 'aː', 'a̠ː': 'a̠', 'z⁽ʲ⁾': 'z', 'ðˤˤ': 'ðˤ', 't̩': 't', 'gʲ̩': 'ɡʲ', 'oᵐ': 'om', 'ʧ': 't͡ʃ', 'î': 'i', 's⁽ʲ⁾ː': 's', 't͡ʃːʰ': 't͡ʃː', 'd̪̚': 'd̪', 'ʲɛ': 'jɛ', 'ä̂ˑ': 'ä', 'ʔˤː': 'ʔ', 'øːˀ': 'øː', 'ɪ̂ˑ': 'ɪ', 'ɛʲ': 'ɛj', 'ǀ̰': 'ǀ', 'ɛːˀ': 'ɛː', 'í': 'i', 'm⁽ʲ⁾': 'm', 'ᵑg': 'ŋɡ', 'tˤˤ': 'tˤ', 'ʲj͡aʲ': 'ja', 'gʰ': 'ɡʰ', 'ũː': 'ũ', 'kʲ̩̩': 'kʲ', 'oⁿ': 'on', 'iᵐ': 'im', 'sˤˤ': 'sˤ', 'ɕ͈': 'ɕ', 'ɪ̌͡ə': 'ɪə', 'ŋ̊': 'ŋ', 'ǃ̰': 'ǃ', 'ɽʱ': 'ɽɦ', 'yːˀ': 'yː', 'o͡ʊ': 'oʊ', 'oʲ': 'oj', 'eᵐ': 'em', 't̪ːʰ': 't̪', 't̪̚': 't̪', 'd͡ʒʱ': 'd͡ʒɦ', 'ɾᵊ': 'ɾə', 'ä̌ː': 'aː', 'd͡ʐ': 'ɖ͡ʐ', 'j͡aʲ': 'ja', 'p͈': 'p', 'oᵑ': 'oŋ', 'uᵐ': 'um', 'ɒːˀ': 'ɒː', 'ᵐɓ': 'mɓ', 'iⁿ': 'in', 'ʲɔ': 'jɔ', 'ɑːˀ': 'ɑː', 'ə̀': 'ə', 'uⁿ': 'un', '̤ɡ̤': 'ɡ̤', 'ôː': 'oː', 'jᵊ': 'jə', 'ɛ́ː': 'ɛ', 'ǁ̤': 'ǁ', 'uᵑ': 'uŋ', 'ɔ́ː': 'ɔ', 'ʊ̯ˑ': 'ʊ̯', 'æ͡i': 'æi', 'ɪ̯ˑ': 'ɪ̯', 'ʲɪ': 'jɪ', 'uːˀ': 'uː', 'kːʰ': 'kː', 'ɛ̀ː': 'ɛ', 'eⁿ': 'en', 'kʲ̩': 'kʲ', 'ɔ̀ː': 'ɔ', 'gʷ': 'ɡʷ', 'ⁿɗ': 'nɗ', 's⁽ʲ⁾': 's', 'eᵑ': 'eŋ', 'ə́': 'ə', 'ĩː': 'ĩ', 'oːˀ': 'oː', 't͡ɕ͈': 't͡ɕ', 'òː': 'oː', 'ɡʱ': 'ɡɦ', 'ǃ̤': 'ǃ', 'ǀ̤': 'ǀ', 'ĭ': 'i', 'aᵑ': 'aŋ', 'ùː': 'uː', 'aⁿ': 'an', 'uʲ': 'uj', 's͈': 's', 'iːˀ': 'iː', 'a͡ʊ': 'aʊ', 'aʲ': 'aj', 'k͈': 'k', 'iᵑ': 'iŋ', 'aᵐ': 'am', 'ɐ̯ˀ': 'ɐ̯', 't̻͡͡sʲ': 'ts', 'ɔ́': 'ɔ', 'ʲj͡a': 'ja', 't͈': 't', 'á': 'a', 'ᶑ': 'd', 'úː': 'uː', 'ŏ': 'ö', 'ˤː': '', 'ä̃ː': 'aː', 'ɑʲ': 'ɑj', 'ʁ̥': 'ʁ', 'l̠ʲ': 'l̠', 'ɖʱ': 'ɖɦ', 'õː': 'õ', 'æːˀ': 'æː', 'r̝̊': 'r̝', 'j͡a': 'ja', 'a͡ɪ': 'aɪ', 'èː': 'eː', 'n̠ʲ': 'n̠', 'd̪ʱ': 'd̪', 'eːˀ': 'eː', 'ẽː': 'ẽ', 'ĕ': 'e', 'ɛ́': 'ɛ', 'ɰᵝ': 'ɰβ', 't͡ʂ': 'ʈ͡ʂ', 'õ': 'õ', 'íː': 'iː', 'bʱ': 'bɦ', 'àː': 'aː', '˩˧': '', 'ṯ': 't', 'éː': 'eː', '˧ˀ˥': '', 'ʈ͡͡ʂ': 'ʈ͡ʂ', 'p̚': 'p', 'ìː': 'iː', 'óː': 'oː', 'ĩ': 'ĩ', 'ă': 'a', 'ŋ͡m': 'ŋm', 'tˢ': 'ts', 'ũ': 'ũ', 'b̥': 'b', 'ɝ': 'ɜ˞', 'áː': 'aː', 'ẽ': 'ẽ', 'ɡ̊': 'ɡ', 'k̚': 'k', '˩˩˦': '', '˧˩˨': '', 't̻͡͡s': 't͡s', 't̚': 't', 'ɾ̪': 'ɾ', 'ʌ̹': 'ʌ', '˨˦': '', 'ç': 'ç', '˨ˀ˩': '', '˧˨': '', 'äː': 'aː', '˧˩˧': '', 'd̥': 'd', '˦˥': '', '˧˧': '', 'g': 'ɡ', '˨˩': '', '˧˥': '', '˥˩': '', '˥': '', '˧': '', '˦': '', '˨': '', '˩': ''}
    ipa_tokens = [mapping.get(token, token) for token in ipa_tokens]
    # re-introduce spaces (like TIMIT)
    ipa_tokens = " ".join(ipa_tokens).replace("  ", " ")
    # split dipthongs
    for dipthong in ['ɑj', 'aj', 'ɛj', 'oj', 'uj', 'ʌj']:
        ipa_tokens = ipa_tokens.replace(dipthong, dipthong[0] +' '+ dipthong[1])
    return ipa_tokens


def text_normalization(orthography):
    # most of the text normalization seems to have done
    #   in the creation of IPAPack++
    # we just need to remove punctuation and symbols
    # see local/all_symbols to see all symbols
    # see local/bad_symbols for which are removed by this regex
    return re.sub(r'\p{P}|\p{S}', '', orthography)


def df_to_kaldi(df, source_dir, data_dir):
    # kaldi format
    for split, split_df in tqdm(df.groupby('split')):
        logging.info(f"processing {split}")
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_dir(source_dir, split_dir, split_df)


# adapted from https://github.com/juice500ml/espnet/blob/wav2gloss/egs2/
#       wav2gloss/asr1/local/data_prep.py
def write_dir(source_dir, target_dir, transcripts):
    # note: The "text" file is used to store phonemes,
    #       while the orthography is stored in "orthography."
    #       What might be confusing is that the "text" column in the df
    #       stores the orthography.
    wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
    text = open(target_dir / "text", "w", encoding="utf-8")
    utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
    utt_id_mapping = open(target_dir / "uttid_map", "w", encoding="utf-8")
    prompt = open(target_dir / "orthography", "w", encoding="utf-8")

    for _, row in transcripts.iterrows():
        utt_id, path, dataset, ipa, orthography = (row['utt_id'], row['path'],
            row['dataset'], row['ipa_clean'], row['text'])

        old_utt_id = row['old_utt_id']
        # map original utt_id to new utt_id (note: not required by kaldi)
        utt_id_mapping.write(f"{old_utt_id} {utt_id}\n")
        split = row['split']

        # {source_dir}/{dataset}/{split}/{old_utt_id}.flac
        wavscp.write(f"{utt_id} {path}\n")
        text.write(f"{utt_id} {ipa}\n")
        # ESPnet does not use speaker info for ASR anymore
        utt2spk.write(f"{utt_id} aaaaa\n")

        if pd.isna(orthography):
            orthography = ''
        prompt.write(f"{utt_id} {orthography}\n")

    wavscp.close()
    text.close()
    utt2spk.close()
    utt_id_mapping.close()
    prompt.close()

    logging.info(f"{target_dir}: {len(transcripts)} lines" +
        f"written to {str(target_dir)}.")


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
        df = pd.read_csv(output)
        logging.info(f"finished loading transcripts and metadata from {str(output)}")
    else:
        df = generate_df(source_dir, data_dir)
    
    # exclude the following langs
        # from FLEURS: 'ga_ie', 'sd_in', 'ar_eg', 'ml_in', 'lo_la', 'da_dk', 'ko_kr', 'ny_mw', 'mn_mn', 'so_so', 'my_mm'
        # Samir et al 2024 found that the data available for
        #   these languages unfortunately have low quality transcriptions.
    FLEURS_EXCLUDE = {
        'ga_ie', 'sd_in', 'ar_eg', 'ml_in', 'lo_la', 'da_dk', 'ko_kr',
        'ny_mw', 'mn_mn', 'so_so', 'my_mm'
    }
    df = df[~df['split'].isin(FLEURS_EXCLUDE)]
    REMOVE_LANGS = {
        'ia'  # Interlingua
    }
    df = df[~df['lang'].isin(REMOVE_LANGS)]
    logging.info("finished removing languages")

    # drop empty rows
    df = df.dropna(subset=["ipa_clean"])

    # normalize phones
    df['ipa_clean'] = df.apply(lambda row: normalize_phones(row['ipa_clean']), axis=1)
    # normalize text
    df['text'] = df.apply(lambda row: text_normalization(row['text']), axis=1)

    logging.info("finished text normalization")
    df.to_csv(source_dir / 'transcript_normalized.csv', index=False)

    df_to_kaldi(df, source_dir, data_dir)
    logging.info("finished converting to kaldi format")
