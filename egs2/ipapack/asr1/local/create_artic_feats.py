import argparse
from panphon import FeatureTable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
    parser.add_argument(
        "--data_dir",
        default="dump/raw/train",
        type=str,
        help="directory containing asr data",
    )

    args = parser.parse_args()
    ft = FeatureTable()
    artic_feats = ft.names

    with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file:
        utts = in_file.readlines()

        artic_feat_files = {}
        for feat in artic_feats:
            artic_feat_files[feat] = open(f"{args.data_dir}/{feat}", "w", encoding="utf-8")

        oov_phonemes = set()

        artic_feat_lists = { feat:[] for feat in artic_feats }
        for utt in utts:
            utt_split = utt.split()
            utt_id = utt_split[0]
            phonemes = utt_split[1:]

            # map utt -> feat -> list of values
            utt_featlist = { feat:[] for feat in artic_feats }
            for phoneme in phonemes:
                fts = ft.word_fts(phoneme)
                if len(fts) == 0:
                    oov_phonemes.add(fts)
                assert len(fts) == 1
                # TODO: remove

                segment = fts[0]
                assert len(artic_feats) == len(segment.strings()) # TODO: remove
                for feature, value in zip(artic_feats, segment.strings()):
                    utt_featlist[feature].append(value)

            # use a list instead of map in case utt_id is not alphabetical
            for feat in artic_feats:
                artic_feat_lists[feat].append((utt_id, utt_featlist[feat]))

        # write in batches to make it less I/O intensive
        for feat in artic_feats:
            writer = artic_feat_files[feat]
            for utt_id, feat_list in artic_feat_lists[feat]:
                writer.write(f"{utt_id} {' '.join(artic_feat_lists)}\n")

            writer.close()

        print(len(oov_phonemes), "OOV phonemes not covered by panphon:")
        print("\n".join(oov_phonemes))
