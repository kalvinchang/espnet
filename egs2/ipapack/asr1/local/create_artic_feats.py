import argparse
import json

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
        "--dont_overwrite",
        action="store_true",
        help="will raise an error if the feature text files already exist",
    )

    args = parser.parse_args()
    ft = FeatureTable()
    artic_feats = ft.names
    feat_vocabularies = {feat: set() for feat in artic_feats}

    with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file:
        utts = in_file.readlines()

        artic_feat_files = {}
        for feat in artic_feats:
            artic_feat_files[feat] = open(f"{args.data_dir}/{feat}", "x" if args.dont_overwrite else "w", encoding="utf-8")

        oov_phonemes = set()

        print("generating articulatory features")
        artic_feat_lists = { feat:[] for feat in artic_feats }
        for utt in tqdm(utts):
            utt_split = utt.split()
            utt_id = utt_split[0]
            phonemes = utt_split[1:]

            # map utt -> feat -> list of values
            utt_featlist = { feat:[] for feat in artic_feats }
            for phoneme in phonemes:
                fts = ft.word_fts(phoneme)
                if len(fts) == 0:
                    oov_phonemes.add(fts)
                    continue

                segment = fts[0]
                for feature, value in zip(artic_feats, segment.strings()):
                    utt_featlist[feature].append(value)

            # use a list instead of map in case utt_id is not alphabetical
            for feat in artic_feats:
                feats = utt_featlist[feat]
                feat_vocabularies[feat].update(feats)
                artic_feat_files[feat].write(f"{utt_id} {' '.join(feats)}\n")

        # write in batches to make it less I/O intensive
        print("writing")
        for feat in tqdm(artic_feats):
            writer = artic_feat_files[feat]
            writer.close()

        with open(f"{args.data_dir}/feature_values.json", "x" if args.dont_overwrite else "w", encoding="utf-8") as file:
            # Sort vocabularies for consistency
            json.dump({feat: sorted(vocabulary) for feat, vocabulary in feat_vocabularies.items()}, file)

        if len(oov_phonemes) > 0:
            print(len(oov_phonemes), "OOV phonemes not covered by panphon:")
            print("\n".join(oov_phonemes))
            raise Exception("OOV phoneme. Please fix in preprocessing")
