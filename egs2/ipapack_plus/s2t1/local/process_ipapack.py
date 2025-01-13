import os
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import langcodes
from langcodes import tag_is_valid
import argparse


# TODO: un-hardcode when we prepare the final code

def draw_figure(language_set, subset_name):
    # Sort by count and keep only the top 5 languages
    sorted_langs = sorted(language_set.items(), key=lambda item: item[1], reverse=True)[:10]
    
    for lang, count in sorted_langs:
        plt.bar(lang, count)

    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.title(f'Language Distribution in {subset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"{subset_name}_language_distribution.png"))
    plt.close()
    plt.clf()
# Constants
ROOT_DUMP_DIR = (
    "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/asr1/dump/raw"
)
ROOT_DATA_DIR = "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/asr1/data"
ROOT_DF_DIR = (
    "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack_plus/asr1/downloads"
)

columns = ["utt_id", "lang"]
normalize_df = pd.read_csv(
    os.path.join(ROOT_DF_DIR, "transcript_normalized.csv"), usecols=columns
)
doreco_df = pd.read_csv(
    os.path.join(ROOT_DF_DIR, "transcript_doreco.csv"), usecols=columns
)
combined_df = pd.concat([normalize_df, doreco_df])
utt2lang = {row["utt_id"]: row["lang"] for _, row in combined_df.iterrows()}


SOP = "<SOP>"
SOS = "<SOS>"
EOS = "<EOS>"
ASR = "<asr>"
PR = "<pr>"
G2P = "<g2p>"
P2G = "<p2g>"
NO_TIME = "<notimestamps>"
LANG = "<LANG>"  # Should be mapping from utt_id to language code
UNK_LANG = "<UNK_LANG>"
remove_space_lang = ['<cmn>', '<jpn>', '<kor>', '<vie>']
# Create output directory
os.makedirs("OWSM_format", exist_ok=True)
image_dir = os.path.join("OWSM_format", "images")
os.makedirs(image_dir, exist_ok=True)
# Get sorted list of directories
def main(draw_only=False):
    
        
    all_dump_dirs = sorted(os.listdir(ROOT_DUMP_DIR))
    all_data_dirs = sorted(os.listdir(ROOT_DATA_DIR))
    if draw_only:
        for data_dir in tqdm(all_data_dirs):
            process_dir = os.path.join("OWSM_format", data_dir)
            with open(os.path.join(process_dir, "language_distribution.json"), "r") as f:
                language_set = json.load(f)
            draw_figure(language_set, data_dir)
    else:
        # Process each data directory
        for data_dir in tqdm(all_data_dirs):
            print(f"Processing {data_dir}")

            dump_dir = os.path.join(ROOT_DUMP_DIR, data_dir)
            data_dir_path = os.path.join(ROOT_DATA_DIR, data_dir)
            process_dir = os.path.join("OWSM_format", data_dir)
            
            os.makedirs(process_dir, exist_ok=True)

            # Read orthography and phoneme sequences
            orthography = open(os.path.join(data_dir_path, "orthography"), "r").readlines()
            phoneme_seq = open(os.path.join(dump_dir, "text"), "r").readlines()
            
            language_set, unk_language_set = {}, set()

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
                        
                    if LANG not in language_set:
                        language_set[LANG] = 0
                    language_set[LANG] += 1
                    
                    if LANG in remove_space_lang:
                        o = o.replace(" ", "")


                    pr_text.write(f"{utt_id} {LANG}{PR}{NO_TIME}{p}\n")
                    prev_text.write(f"{utt_id} {o}\n")
                    text_ctc.write(f"{utt_id} {p}\n")
                    asr_text.write(f"{utt_id} {LANG}{ASR}{NO_TIME}{o}\n")
                    asr_text_ctc.write(f"{utt_id} {o}\n")
                    g2p_text.write(f"{utt_id} {LANG}{G2P}{NO_TIME}{p}\n")
                    prev_g2p_text.write(f"{utt_id} {o}\n")
                    p2g_text.write(f"{utt_id} {LANG}{P2G}{NO_TIME}{o}\n")
                    prev_p2g_text.write(f"{utt_id} {p}\n")

            
            json.dump(language_set, open(os.path.join(process_dir, "language_distribution.json"), "w"))
            # Plot language distribution
            
            draw_figure(language_set, os.path.basename(data_dir_path))
            
            
            with open(os.path.join(process_dir, "unk_languages.txt"), "w") as f:
                for lang in unk_language_set:
                    f.write(f"{lang}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IPA Pack data")
    parser.add_argument("--draw_only", action="store_true", help="Only draw the figures")
    args = parser.parse_args()
    main(draw_only=args.draw_only)
