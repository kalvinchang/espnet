import sys, os
import kaldiio
import numpy as np
from tqdm import tqdm

def check_scp(path):
    #d = kaldiio.load_scp_sequential(path)
    #for key, numpy_array in d:
    #   audio = numpy_array[1]
    d = kaldiio.load_scp(path)
    for uttid in tqdm(d):
        rate, audio = d[uttid]
        if audio.ndim != 1:
            print(key, audio.ndim)
    print(f"{path} checked.")

def correct_scp(path):
    d = kaldiio.load_scp(path)
    for uttid in tqdm(d):
        rate, audio = d[uttid]
        if audio.ndim != 1:
            # for multi-channel audio, average the audio across the channels
            audio = np.mean(audio, axis=0)
        # ref: pyscripts/audio/format_wav.scp.py line 369
        kaldiio.save_ark('new_data_wav.ark', {uttid: (audio, rate)}, append=True, 
            write_function="soundfile", scp='new_wav.scp',
            write_kwargs={"format": "flac", "subtype": None})

def correct_ark(path, pre='nnew'):
    d = kaldiio.load_ark(path)
    for uttid, numpy_array in tqdm(d):
        rate, wave = numpy_array
        if wave.ndim != 1:
            wave = np.mean(wave, axis=0)
        # ref: pyscripts/audio/format_wav.scp.py line 369
        kaldiio.save_ark(f'{pre}_data_wav.ark', {uttid: (wave, rate)}, append=True, 
            write_function="soundfile", scp=f'{pre}_wav.scp',
            write_kwargs={"format": "flac", "subtype": None})

def remove_lines(path):
    with open("diff_train.txt", "r") as f:
        diffs = f.readlines()
    lines = []
    for l in diffs:
        if l[0]=='<': continue
        rm = l.split('d')[0]
        if ',' in rm:
            start, end = rm.split(',')
            for i in range(int(start), int(end)+1):
                lines.append(str(i))
        else:
            lines.append(rm)
    splitsize = 2000
    command = []
    for i in range(0, len(lines), splitsize):
        command.append(f"sed -i.bak '{'d;'.join(lines[i:i+splitsize])+'d'}' {path}")
    #os.system(f"awk 'NR!~/^({lines})$/' {path} > tmp")
    #os.system(f"sed -i.bak '{lines}' {path}")
    command.reverse()
    for c in command:
        os.system(c)
            

if __name__ == '__main__':
    #correct_ark(f'dump/raw/org/train/data/format.24/oldwav/data_wav.ark')
    #check_scp('new_wav.scp')
    remove_lines('wav.scp') # dump/raw/train/

    """
    # check for multichannel data
    split = int(sys.argv[1])
    for i in range(8*split + 1, 8*(split+1)+1):
        print(f"format.{i}")
        check_ark(f'dump/raw/org/train/data/format.{i}/wav.scp')
    """