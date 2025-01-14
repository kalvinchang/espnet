import os
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import langcodes
from langcodes import tag_is_valid
import argparse

SOP = "<SOP>"
SOS = "<SOS>"
EOS = "<EOS>"
ASR = "<asr>"
PR = "<pr>"
G2P = "<g2p>"
P2G = "<p2g>"
NO_TIME = "<notimestamps>"
SAMPLE_RATE = 16000
LANG = "<LANG>"  # Should be mapping from utt_id to language code
UNK_LANG = "<UNK_LANG>"
remove_space_lang = ['<cmn>', '<jpn>', '<kor>', '<vie>']
copy_files = ["feats_type", "spk2utt", "utt2num_samples", "utt2spk", "wav.scp"]


# TODO: unhardcode when we prepare final code

def draw_figure(lang2dur, subset_name, image_dir):
    # Sort by count and keep only the top 10 languages
    sorted_langs = sorted(lang2dur.items(), key=lambda item: item[1], reverse=True)[:10]
    
    for lang, duration in sorted_langs:
        plt.bar(lang, duration / 3600)

    plt.xlabel('Language')
    plt.ylabel('Hours')
    plt.title(f'Language Distribution in {subset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"{subset_name}_language_distribution.png"))
    plt.close()
    plt.clf()


def main(root_dir, output_dir, lang_dist_json, draw_only=False):
    ROOT_DUMP_DIR = os.path.join(root_dir, "dump/raw")
    ROOT_DATA_DIR = os.path.join(root_dir, "data")
    ROOT_DF_DIR = os.path.join(root_dir, "downloads")

    columns = ["utt_id", "lang"]
    normalize_df = pd.read_csv(
        os.path.join(ROOT_DF_DIR, "transcript_normalized.csv"), usecols=columns
    )
    doreco_df = pd.read_csv(
        os.path.join(ROOT_DF_DIR, "transcript_doreco.csv"), usecols=columns
    )
    combined_df = pd.concat([normalize_df, doreco_df])
    utt2lang = {row["utt_id"]: row["lang"] for _, row in combined_df.iterrows()}


    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    all_dump_dirs = sorted(os.listdir(ROOT_DUMP_DIR))
    all_data_dirs = sorted(os.listdir(ROOT_DATA_DIR))
    if draw_only:
        for data_dir in tqdm(all_data_dirs):
            process_dir = os.path.join(output_dir, data_dir)
            with open(os.path.join(process_dir, lang_dist_json), "r") as f:
                lang2dur = json.load(f)
            draw_figure(lang2dur, data_dir, image_dir)
    else:
        # Process each data directory
        for data_dir in tqdm(all_data_dirs):
            print(f"Processing {data_dir}")

            dump_dir = os.path.join(ROOT_DUMP_DIR, data_dir)
            data_dir_path = os.path.join(ROOT_DATA_DIR, data_dir)
            process_dir = os.path.join(output_dir, data_dir)
            
            os.makedirs(process_dir, exist_ok=True)
            
            # Copy necessary files from dump_dir to process_dir
            for file_name in copy_files:
                src_file = os.path.join(dump_dir, file_name)
                dst_file = os.path.join(process_dir, file_name)
                if os.path.exists(src_file):
                    with open(src_file, "r") as src, open(dst_file, "w") as dst:
                        dst.write(src.read())
                        
            # Read orthography and phoneme sequences
            orthography = open(os.path.join(data_dir_path, "orthography"), "r").readlines()
            phoneme_seq = open(os.path.join(dump_dir, "text"), "r").readlines()
            
            unk_language_set = set()

            # Create mappings
            utt2orthography = {
                o.strip().split(maxsplit=1)[0]: (
                    o.strip().split(maxsplit=1)[1]
                    if len(o.strip().split(maxsplit=1)) > 1
                    else ""
                )
                for o in orthography
            }
            utt2phoneme_seq = {
                p.strip().split(maxsplit=1)[0]: p.strip().split(maxsplit=1)[1]
                for p in phoneme_seq
            }

            # Write text file
            with open(os.path.join(process_dir, "text"), "w") as pr_text, \
                open(os.path.join(process_dir, "text.prev"), "w") as prev_text, \
                open(os.path.join(process_dir, "text.ctc"), "w") as text_ctc, \
                open(os.path.join(process_dir, "text.asr"), "w") as asr_text, \
                open(os.path.join(process_dir, "text.asr_ctc"), "w") as asr_text_ctc, \
                open(os.path.join(process_dir, "text.g2p"), "w") as g2p_text, \
                open(os.path.join(process_dir, "text.g2p_prev"), "w") as prev_g2p_text, \
                open(os.path.join(process_dir, "text.p2g"), "w") as p2g_text, \
                open(os.path.join(process_dir, "text.p2g_prev"), "w") as prev_p2g_text :

                for utt_id, p in utt2phoneme_seq.items():
                    p = "".join([f"/{char}/" for char in p.split()])
                    o = utt2orthography[utt_id]
                    lang = utt2lang[utt_id]
                    try:
                        if tag_is_valid(lang):
                            l = langcodes.get(lang).to_alpha3()
                        else:
                            l = langcodes.find(lang).to_alpha3()
                        if l == "zho":
                            LANG = "<cmn>"
                        else:
                            LANG = f"<{l}>"
                            
                        
                    except:
                        unk_language_set.add(lang)
                        LANG = UNK_LANG
                        
                    
                    if LANG in remove_space_lang:
                        o = o.replace(" ", "")

                    utt2lang[utt_id] = LANG
                    
                    pr_text.write(f"{utt_id} {LANG}{PR}{NO_TIME}{p}\n")
                    prev_text.write(f"{utt_id} {o}\n")
                    text_ctc.write(f"{utt_id} {p}\n")
                    asr_text.write(f"{utt_id} {LANG}{ASR}{NO_TIME}{o}\n")
                    asr_text_ctc.write(f"{utt_id} {o}\n")
                    g2p_text.write(f"{utt_id} {LANG}{G2P}{NO_TIME}{p}\n")
                    prev_g2p_text.write(f"{utt_id} {o}\n")
                    p2g_text.write(f"{utt_id} {LANG}{P2G}{NO_TIME}{o}\n")
                    prev_p2g_text.write(f"{utt_id} {p}\n")

            
            lang2dur = {} # lang -> duration (sec)
            
            utt2num_samples_path = os.path.join(process_dir, "utt2num_samples")
            with open(utt2num_samples_path, "r") as f:
                for line in f:
                    utt_id, num_samples = line.strip().split(maxsplit=1)
                    lang = utt2lang.get(utt_id, UNK_LANG)
                    lang2dur[lang] = lang2dur.get(lang, 0) + float(num_samples) / SAMPLE_RATE
            
            json.dump(lang2dur, open(os.path.join(process_dir, lang_dist_json), "w"))
            # Plot language distribution
            
            draw_figure(lang2dur, os.path.basename(data_dir_path), image_dir)
            
            
            with open(os.path.join(process_dir, "unk_languages.txt"), "w") as f:
                for lang in unk_language_set:
                    f.write(f"{lang}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IPA Pack data")
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/asr1", help="Root directory")
    parser.add_argument("--output_dir", type=str, default="OWSM_format", help="Output directory")
    parser.add_argument("--lang_dist_json", type=str, default="language_distribution.json", help="Language distribution JSON filename")
    parser.add_argument("--draw_only", action="store_true", help="Only draw the figures")
    args = parser.parse_args()
    
    main(**vars(args))
