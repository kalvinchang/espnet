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

    inpath = args.source_dir
    outpath = args.target_dir

    # get list of datasets in IPAPack+
    filelist = glob(inpath + '/**/*.jsonl.gz', recursive=True)
    datasets = [file.replace(inpath, '') for file in filelist]
    datasets = [file.replace(os.path.basename(file), '') for file in datasets]
    datasets = list(set(datasets))
    datasets = [file for file in datasets \
        if 'dev' not in file and 'test' not in file and 'doreco' not in file]
    print(datasets)
    logging.info("%s speech train data files found!" % len(datasets))
    logging.info("Beginning processing dataset")
    data_dir = Path(outpath)
    data_dir.mkdir(parents=True, exist_ok=True)

    with SharWriter(
        data_dir, fields={"recording": "flac"}, shard_size=20000
    ) as writer:
        for i, dataset in enumerate(datasets):
            data_path = inpath / dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Processing %s" % data_path)

            supervision = sorted(glob(os.path.join(data_path, 'cuts*')))
            recording = sorted(glob(os.path.join(data_path, 'recording*')))
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

            logging.info("Processing done! %s datasets remaining."
                % (len(datasets)-i-1))

