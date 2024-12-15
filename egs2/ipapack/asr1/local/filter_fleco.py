import os
import kaldiio
from tqdm import tqdm

"""
Preparation
1. Build a new base directory for running the filtered data (e.g., ipapack_fleco/asr1)
2. Make a directory to store the train/dev/test dump directory (e.g., ipapack_fleco/asr1/dump/raw)
3. Run this script in the base directory to get correct path for wav.scp files
Takes about 20 min to run
"""

basedir = f"dump/raw" # path to the new train/dev/test directory
plusdir = f"../../ipapack_plus/asr1/dump/raw" # path to ipapack_plus's dump train/dev/test dump directory

os.system(f"mkdir -p {basedir}/train/data {basedir}/dev/data")

for dataset in ["train", "dev"]:
    print(dataset[0].upper()+dataset[1:], "set:")
    print("-> Copying and finding lines with 'fleurs'...")
    currentdir = f"{basedir}/{dataset}"
    # direct copy: feats_type & audio_format
    os.system(f"cp {plusdir}/{dataset}/feats_type {currentdir}/feats_type")
    os.system(f"echo flac.ark > {currentdir}/audio_format")

    # filtering: find lines with "fleurs"
    for file in ["spk2utt", "text", "utt2spk", "wav.scp", "utt2num_samples"]:
        os.system(f"grep 'fleurs' {plusdir}/{dataset}/{file} > {currentdir}/{file}")
    
    # Setup splits
    print("-> Setting up splits...")
    # 1. form 32 splits
    os.system(f"split -d -n l/32 {currentdir}/wav.scp {currentdir}/data/wav.")
    os.system(f"split -d -n l/32 {currentdir}/utt2num_samples {currentdir}/data/samples.")
    for i in range(32):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"mkdir {formatdir}")
        os.system(f"mv {currentdir}/data/wav.{i:02d} {formatdir}/wav.scp")
        os.system(f"mv {currentdir}/data/samples.{i:02d} {formatdir}/utt2num_samples")
    
    # 2. In each split, get new data_wav.ark and wav.scp
    for i in tqdm(range(32)):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"sed 's|dump/raw|{plusdir}|g' {formatdir}/wav.scp > {formatdir}/wav.scp.tmp")
        os.system(f"mv {formatdir}/wav.scp.tmp {formatdir}/wav.scp")
        d = kaldiio.load_scp(f'{formatdir}/wav.scp')
        kaldiio.save_ark(f'{formatdir}/data_wav.ark', d, 
            write_function="soundfile", scp=f'{formatdir}/wav.scp')
        
    # 3. Update wav.scp by combining wav.scp in data/format.1~32
    os.system(f"rm {currentdir}/wav.scp")
    for i in range(32):
        formatdir = f"{currentdir}/data/format.{i+1}"
        os.system(f"cat {formatdir}/wav.scp >> {currentdir}/wav.scp")
    
    print(f"Finished!")


# test: copy from test_doreco
print("Test set:")
os.system(f"cp -r {plusdir}/test_doreco {basedir}/test")
# wav.scp: replace "test_doreco/" with "test/"
print("-> Replacing 'test_doreco/' with 'test/' in wav.scp...")
os.system(f"sed 's|test_doreco|test|g' {basedir}/test/wav.scp > {basedir}/test/wav.scp.tmp")
os.system(f"mv {basedir}/test/wav.scp.tmp {basedir}/test/wav.scp")
for i in range(32):
    formatdir = f"{basedir}/test/data/format.{i+1}"
    os.system(f"sed 's|test_doreco|test|g' {formatdir}/wav.scp > {formatdir}/wav.scp.tmp")
    os.system(f"mv {formatdir}/wav.scp.tmp {formatdir}/wav.scp")
print("Finished!")