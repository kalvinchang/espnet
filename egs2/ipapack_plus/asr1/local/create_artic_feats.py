import argparse
import json
from multiprocessing import Pool, Manager
import multiprocessing

from panphon import FeatureTable
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
    parser.add_argument(
        "--data_dir",
        default="dump/raw/train",
        type=str,
        help="directory containing asr data",
    )
    parser.add_argument(
        "--feature_type",
        default="panphon",
        choices={"panphon", "allophoible"},
        help="feature set to extract",
    )
    parser.add_argument(
        "--write_vocabulary",
        action="store_true",
        help="Writes a vocabulary file with the possible values for each feature, including <blank>",
    )
    parser.add_argument(
        "--dont_overwrite",
        action="store_true",
        help="will raise an error if the feature text files already exist",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
        help="Number of jobs for extracting articulatory features",
    )

    args = parser.parse_args()

    if args.feature_type == "panphon":
        ft = FeatureTable()
        artic_feats = ft.names
    else:
        from allophant.phonetic_features import PhoneticAttributeIndexer, FeatureSet
        with open("local/allophoible/allophoible_v2.csv", "r", encoding="utf-8") as file:
            ft = PhoneticAttributeIndexer(FeatureSet.PHOIBLE, file).full_attributes
        artic_feats = ft.feature_names

    feat_vocabularies = {feat: set() for feat in artic_feats}

    with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file:
        utts = in_file.readlines()

        artic_feat_files = {}
        for feat in artic_feats:
            artic_feat_files[feat] = open(f"{args.data_dir}/{feat}", "x" if args.dont_overwrite else "w", encoding="utf-8")

        print("generating articulatory features")
        artic_feat_lists = { feat:[] for feat in artic_feats }

        def process_utterance(utt):
            oov_phonemes, utt = utt

            utt_split = utt.split()
            utt_id = utt_split[0]
            phonemes = utt_split[1:]

            # map utt -> feat -> list of values
            utt_featlist = { feat:[] for feat in artic_feats }

            if isinstance(ft, FeatureTable):
                for phoneme in phonemes:
                    phoneme = _normalize_phoneme_phoible(phoneme)
                    # Ignore unknown tokens from the validation set
                    if phoneme == "<unk>":
                        continue
                    fts = ft.word_fts(phoneme)
                    if len(fts) == 0:
                        if phoneme not in oov_phonemes:
                            tqdm.write(f"Found OOV phoneme: {phoneme!r}")
                        oov_phonemes[phoneme] = None
                        continue

                    for feature, value in zip(artic_feats, fts[0].strings()):
                        utt_featlist[feature].append(value)
            else:
                for phoneme in phonemes:
                    # Ignore unknown tokens from the validation set
                    if phoneme == "<unk>":
                        continue
                    # Get Phoible feature contours for each phoneme
                    try:
                        fts = ft.feature_vector(phoneme)
                    except KeyError:
                        oov_phonemes[phoneme] = None
                        continue
                    for feature, values in zip(artic_feats, fts):
                        utt_featlist[feature].extend(ft.feature_values(feature, values))

            return utt_id, utt_featlist

        with Manager() as manager, Pool(args.nj) as pool:
            oov_phonemes = manager.dict()
            for utt_id, utt_featlist in tqdm(pool.imap(process_utterance, ((oov_phonemes, utterance) for utterance in utts), chunksize=1000), total=len(utts)):
                # use a list instead of map in case utt_id is not alphabetical
                for feat in artic_feats:
                    feats = utt_featlist[feat]
                    feat_vocabularies[feat].update(feats)
                    artic_feat_files[feat].write(f"{utt_id} {' '.join(feats)}\n")

            if len(oov_phonemes) > 0:
                print(len(oov_phonemes), "OOV phonemes not covered by panphon:")
                print("\n".join(oov_phonemes.keys()))
                raise Exception("OOV phoneme. Please fix in preprocessing")

        # write in batches to make it less I/O intensive
        print("writing")
        for feat in tqdm(artic_feats):
            writer = artic_feat_files[feat]
            writer.close()

        if args.write_vocabulary:
            with open(f"{args.data_dir}/feature_values.json", "x" if args.dont_overwrite else "w", encoding="utf-8") as file:
                # Sort vocabularies for consistency
                json.dump({feat: ["<blank>"] + sorted(vocabulary) for feat, vocabulary in feat_vocabularies.items()}, file)
