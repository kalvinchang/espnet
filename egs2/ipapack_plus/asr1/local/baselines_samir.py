import numpy as np
import jiwer
import panphon.distance
import loguru
import click
import ipdb
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from ipatok import tokenise
from pathlib import Path
from tqdm import tqdm
import kaldiio
import pandas as pd
import torch
from allosaurus.app import read_recognizer
import soundfile as sf
from dotenv import dotenv_values

config = dotenv_values(".env")
HF_CACHE_DIR = config['HF_CACHE_DIR']

logger = loguru.logger

def run_g2p(wav_path):
    data = []
    model = Wav2Vec2ForCTC.from_pretrained("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", cache_dir=HF_CACHE_DIR).to('cuda')
    processor = Wav2Vec2Processor.from_pretrained("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", cache_dir=HF_CACHE_DIR)
    with open(wav_path, "r") as f:
        for line in tqdm(f, desc="Loading SCP"):
            utt_id, path = line.strip().split(None, 1)
            _, wav = kaldiio.load_mat(path)
            # tokenize
            input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to('cuda')
            
            # retrieve logits
            with torch.no_grad():
                logits = model(input_values).logits
            
            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            final = " ".join(transcription)
            data.append([utt_id, final])
    df = pd.DataFrame(data, columns=["utt_id", "hyp"])
    df.to_csv("wav2vec2phoneme_hyp.csv", index=False)
    return df

def read_txt(text_path):
    data = []
    seen_utt_ids = set()
    with open(text_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(" ", 1)
            utt_id, text = parts
            if utt_id not in seen_utt_ids:
                data.append([utt_id, text])
                seen_utt_ids.add(utt_id)
    df = pd.DataFrame(data, columns=["utt_id", "ref"])
    print(len(df))
    df.to_csv("wav2vec2phoneme_ref.csv", index=False)
    return df

@click.group()
def main():
    pass

@click.command()
def generate_w2v2_predictions():
    wav_scp = Path("/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data/test_wav.scp")
    text_path = Path("/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data/test_text")
    
    print("Loading text data...")
    ref_df = read_txt(text_path)
    print(len(ref_df))

    print("generating g2p...")
    hyp_df = run_g2p(wav_scp)
    print(len(ref_df))
    total_df = pd.merge(ref_df, hyp_df, on="utt_id")

def run_allosaurus(wav_path):
    model = read_recognizer()
    data = []
    with open(wav_path, "r") as f:
        for line in tqdm(f, desc="Loading SCP"):
            utt_id, path = line.strip().split(None, 1)
            _, wav = kaldiio.load_mat(path)
            audio_path = "temp_allosaurus.wav"
            # tokenize
            sf.write(audio_path, wav, 16000)
            # retrieve logits
            
            # take argmax and decode
            prediction = model.recognize(audio_path)
            final = " ".join(prediction)
            data.append([utt_id, final])
    df = pd.DataFrame(data, columns=["utt_id", "hyp"])
    df.to_csv("wav2vec2phoneme_allosaurus_hyp.csv", index=False)

@click.command()
def generate_allosaurus_predictions():
    wav_scp = Path("/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data/test_wav.scp")
    text_path = Path("/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data/test_text")

    ref_df = read_txt(text_path)
    logger.info(f"Loaded hypotheses")
    run_allosaurus(wav_scp)

@click.command()
@click.argument('ref_path')
@click.argument('hyp_path')
@click.argument('baseline_name')
def compute_eval_scores(ref_path, hyp_path, baseline_name):
    ref_df = pd.read_csv(ref_path)
    hyp_df = pd.read_csv(hyp_path)
    df = ref_df.join(hyp_df, rsuffix='_right')
    assert len(ref_df) == len(df)

    # Validate required columns
    if "ref" not in df.columns or "hyp" not in df.columns:
        print("Error: The CSV must contain 'ref' and 'hyp' columns.")
        return

    # Handle missing values
    df["hyp"] = df["hyp"].fillna("")

    # Preprocess 'hyp' column: Remove numbers and dots
    df["hyp"] = df["hyp"].str.replace(r'\d+', '', regex=True).str.replace('.', '', regex=False)

    dst = panphon.distance.Distance()
    # Calculate PER and CER for each row
    per_list = []
    cer_list = []
    fer_list = []
    for _, row in tqdm(df.iterrows()):
        ref = row["ref"].replace(" ", "")  # Remove spaces in 'ref'
        hyp = row["hyp"].replace(" ", "")  # Remove spaces in 'hyp'
        if " " not in row['hyp']:
            with_space_hyp = tokenise(row['hyp'])
            with_space_hyp = ' '.join(with_space_hyp)
        else:
            with_space_hyp = row['hyp']

        # Compute PER and CER for the row
        per = jiwer.wer(row["ref"], with_space_hyp)  # Use original strings for PER
        cer = jiwer.cer(ref, hyp)  # Use space-removed strings for CER
        fer = dst.feature_error_rate(with_space_hyp, row["ref"])

        per_list.append(per)
        cer_list.append(cer)
        fer_list.append(fer)

    # Add PER and CER as new columns
    df["PER"] = per_list
    df["CER"] = cer_list
    df["FER"] = fer_list

    print(f"Average PER: {np.mean(per_list):.4f}")
    print(f"Average CER: {np.mean(cer_list):.4f}")
    print(f"Average FER: {np.mean(fer_list):.4f}")

    # Print or save the updated DataFrame
    print(df[["ref", "hyp", "PER", "CER", "FER"]])
    output_path = f"{baseline_name}_with_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed CSV")
    pass

main.add_command(generate_w2v2_predictions)
main.add_command(generate_allosaurus_predictions)
main.add_command(compute_eval_scores)

if __name__ == "__main__":
    main()
    

