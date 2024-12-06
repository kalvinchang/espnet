import os
import sys
from tqdm import tqdm
from ipatok import tokenise
from glob import glob
from lhotse import CutSet
from lhotse.shar.writers import SharWriter
from pathlib import Path
import logging


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
    datasets = source_dir.glob('*_shar')
    datasets = [d.parts[1] for d in datasets]
    datasets = list(set(datasets))
    # set is non-deterministic
    datasets = sorted(datasets)
    logging.info(f"{len(datasets)} speech train data files found: {datasets}")

    logging.info("Beginning processing dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    with SharWriter(
        data_dir, fields={"recording": "flac"}, shard_size=20000
    ) as writer:
        for i, dataset in tqdm(enumerate(datasets)):
            data_path = source_dir / dataset
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
