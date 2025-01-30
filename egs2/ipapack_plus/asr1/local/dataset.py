import os
import json
import torch
import numpy as np
import soundfile as sf
from typing import Dict, Tuple, List
from tqdm import tqdm
import kaldiio

class IPAPack():
    def __init__(self, data_path, scp_file, text_file, utt2dur_file, max_audio_duration, vocab_path, debug=False, debug_size=None):
        self.max_audio_duration = max_audio_duration
        self.data = {}
        self.data_path = data_path

        # Step 1: Filter utterances by duration
        print(f"Filtering utterances based on duration <= {self.max_audio_duration} seconds.")
        self.utt_ids = self._filter_by_duration_utt2dur(utt2dur_file)

        # Step 2: Apply debug filtering immediately
        if debug and debug_size:
            self.set_debug(debug_size)

        print(f"Loading data...")
        # Step 3: Load data (text and scp)
        self._load_text(text_file)
        self._load_scp(scp_file)
        self._load_vocab(vocab_path)

    def _filter_by_duration_utt2dur(self, utt2dur_file: str) -> List[str]:
        utt2dur_path = os.path.join(self.data_path, utt2dur_file)
        filtered_utt_ids = []
        tot_utt_ids = []
        with open(utt2dur_path, "r") as f:
            for line in f:
                utt_id, duration = line.split()
                duration = float(duration)
                tot_utt_ids.append(utt_id)
                if 0.1 <= duration <= self.max_audio_duration:
                    filtered_utt_ids.append(utt_id)
        if "test" in utt2dur_file:
            print(f"Returning all utterances for {utt2dur_file} as it includes 'test'.")
            return tot_utt_ids
        else:
            print(f"Filtered {len(filtered_utt_ids)} utterances based on duration <= {self.max_audio_duration} seconds.")
            return filtered_utt_ids

    def set_debug(self, size):
        """Limit dataset size for debugging."""
        size = min(size, len(self.utt_ids))
        self.utt_ids = self.utt_ids[:size]  # Filter the utt_ids list
        print(f"Debug mode: Dataset reduced to {len(self.utt_ids)} utterances.")

    def _load_scp(self, scp_file):
        """Load SCP data only for filtered utt_ids, line-by-line."""
        scp_path = os.path.join(self.data_path, scp_file)
        if not os.path.exists(scp_path):
            raise FileNotFoundError(f"SCP file not found: {scp_path}")
        print("loading scp file")
        with open(scp_path, "r") as f:
            for line in tqdm(f, desc="Loading SCP"):
                utt_id, path = line.strip().split(None, 1)
                if utt_id in self.utt_ids:
                    if utt_id not in self.data:
                        self.data[utt_id] = {}
                    self.data[utt_id]["scp"] = path

    def _load_text(self, text_file: str):
        """Load text data only for filtered utt_ids."""
        text_path = os.path.join(self.data_path, text_file)
        print("loading text file")
        with open(text_path, "r") as f:
            for line in tqdm(f):
                parts = line.strip().split(" ", 1)
                utt_id, text = parts
                if utt_id in self.utt_ids:
                    if utt_id not in self.data:
                        self.data[utt_id] = {}
                    self.data[utt_id]["text"] = text
        print(f"Loaded text for {len(self.data)} utterances.")

    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from a JSON file."""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        with open(vocab_path, "r") as vocab_file:
            self.char_to_int = json.load(vocab_file)

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        entry = self.data[utt_id]
        try:
            _, wav = kaldiio.load_mat(entry["scp"])
        except Exception as e:
            print(f"Error reading wav for utt_id: {utt_id}, error: {e}")
            return None
            
        # text_int = torch.tensor([self.char_to_int[p] for p in entry["text"].split(" ")])
        text_int = torch.tensor([self.char_to_int.get(p, self.char_to_int.get(" ")) for p in entry["text"].split(" ")])
        return utt_id, {
            "speech": wav, 
            "text": text_int  
        }
