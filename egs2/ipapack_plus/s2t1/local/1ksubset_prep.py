import os
import kaldiio
from tqdm import tqdm

"""
Preparation
1. Build a new directory for running the filtered data (e.g., ipapack_fleco/asr1)
2. Make a directory to store the train/dev/test dump directory (e.g., ipapack_fleco/asr1/dump/raw)
    - in this case, "basedir" is "dump/raw"
3. Run this script in the directory to get correct path for wav.scp files
Takes about 20 min to run
"""

basedir = "dump" # path to the new train/dev/test directory
plusdir = "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/asr1/dump/raw" # path to ipapack_plus's dump train/dev/test dump directory
textdir = "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/s2t1/dump"

for dataset in ["train", "dev"]:
    print(dataset[0].upper()+dataset[1:], "set:")
    print("-> Subample to 1/80...")
    currentdir = f"{basedir}/{dataset}_1000"
    os.system(f"mkdir -p {currentdir}/texts")
    
    # 1. direct copy: feats_type & audio_format
    os.system(f"cp {plusdir}/{dataset}/feats_type {currentdir}/feats_type")
    os.system(f"echo flac.ark > {currentdir}/audio_format")

    # 2. subsampling: retain 1 per 80 lines
    for file in ["spk2utt", "utt2spk", "wav.scp", "utt2num_samples"]:
        os.system(f"awk 'NR % 80 == 1' {plusdir}/{dataset}/{file} > {currentdir}/{file}")
        # repeat file for 4 times
        for i in range(4):
            os.system(f"cat {currentdir}/{file} >> {currentdir}/{file}.tmp")
        os.system(f"mv {currentdir}/{file}.tmp {currentdir}/{file}")
    for file in ["text.asr", "text.ctc", "text.g2p_prev", "text.p2g_prev", "text", "text.asr_ctc", "text.g2p", "text.p2g", "text.prev"]:
        os.system(f"awk 'NR % 80 == 1' {textdir}/{dataset}/{file} > {currentdir}/texts/{file}")
    os.system(f"cp {currentdir}/texts/text {currentdir}/texts/text.na")
    # remove the text and replace with <na> since no previous text for ASR, phoneme recognition
    os.system(f"sed -i 's/ .*/ <na>/' {currentdir}/texts/text.na")
    
    # 3. combine text files
    for file in ["text", "text.asr", "text.g2p", "text.p2g"]:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text")
    for file in ["text.na", "text.na", "text.g2p_prev", "text.p2g_prev"]:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text.prev")
    # note: for G2P, the ctc text comes from phoneme recognition
    #       for P2G, the ctc text comes from ASR
    for file in ["text.ctc", "text.asr_ctc", "text.ctc", "text.asr_ctc"]:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text.ctc")
    
    """ Uncomment the 3 steps below to split wav.scp into n parts in data/
    # Setup splits
    print("-> Setting up splits...")
    os.system(f"mkdir {currentdir}/data")
    nsplit = 32

    # 1. form n splits
    os.system(f"split -d -n l/{nsplit} {currentdir}/wav.scp {currentdir}/data/wav.")
    os.system(f"split -d -n l/{nsplit} {currentdir}/utt2num_samples {currentdir}/data/samples.")
    for i in range(nsplit):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"mkdir {formatdir}")
        os.system(f"mv {currentdir}/data/wav.{i:02d} {formatdir}/wav.scp")
        os.system(f"mv {currentdir}/data/samples.{i:02d} {formatdir}/utt2num_samples")
    
    # 2. In each split, get new data_wav.ark and wav.scp
    for i in tqdm(range(nsplit)):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"sed 's|dump/raw|{plusdir}|g' {formatdir}/wav.scp > {formatdir}/wav.scp.tmp")
        os.system(f"mv {formatdir}/wav.scp.tmp {formatdir}/wav.scp")
        d = kaldiio.load_scp(f'{formatdir}/wav.scp')
        kaldiio.save_ark(f'{formatdir}/data_wav.ark', d, 
            write_function="soundfile", scp=f'{formatdir}/wav.scp')
        
    # 3. Update wav.scp by combining wav.scp in data/format.1~nsplit
    os.system(f"rm {currentdir}/wav.scp")
    for i in range(nsplit):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"cat {formatdir}/wav.scp >> {currentdir}/wav.scp")
    """

    print(f"Finished!")