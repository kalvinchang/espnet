import os
import json
import torch
import numpy as np
import soundfile as sf
from espnet2.train.dataset import ESPnetDataset
from typing import Dict, Tuple, List
from tqdm import tqdm
import kaldiio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer

class IPAPack(ESPnetDataset):
    def __init__(
        self,
        scp_file: str,
        text_file: str,
        utt2dur_file: str,
        max_audio_duration: float,
        vocab_path: str = None,
        debug: bool = False,
        debug_size: int = None,
    ):
        self.max_audio_duration = max_audio_duration  # Maximum allowed duration in seconds
        self.data: Dict[str, Dict[str, str]] = {}
        self.data_path = "/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local"
        
        # Load features, text, and vocabulary
        self._load_scp(scp_file)
        self._load_text(text_file)
        self._load_vocab(vocab_path)

        self._filter_by_duration_utt2dur(utt2dur_file)
        
        self.utt_ids = sorted(self.data.keys())

        if debug and debug_size is not None:
            self.set_debug(debug_size)

        print(f"Loaded {len(self.utt_ids)} utterances.")

    def _load_scp(self, scp_file: str):
        scp_file = kaldiio.load_scp(scp_file)
        for i, (utt_id, mat) in tqdm(enumerate(scp_file.items())):
            sr, wav = mat
            self.data[utt_id] = {"array": wav, "sampling_rate": sr}
            if i == 100:
                break

    def _load_text(self, text_file: str):
        text_file = os.path.join(self.data_path, text_file)
        with open(text_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                if utt_id in self.data:
                    self.data[utt_id]["text"] = text
                # else:
                    # print(f"Warning: {utt_id} not found in SCP data.")

        self.data = {utt_id: entry for utt_id, entry in self.data.items() if "text" in entry}

    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from a JSON file."""
        with open(vocab_path, "r") as vocab_file:
            self.char_to_int = json.load(vocab_file)
        self.vocab_size = len(self.char_to_int)
        return self.vocab_size

    def _filter_by_duration_utt2dur(self, utt2dur_file: str):
        """Filter utterances exceeding the maximum duration."""
        utt2dur_file = os.path.join(
           self.data_path, utt2dur_file,
        )
        # Skip filtering if the file is for the test set
        if "test" in utt2dur_file:
            print("Skipping filtering for test set.")
        utt2dur = {}
        with open(utt2dur_file, "r") as f:
            for line in f:
                utt_id, dur = line.strip().split(" ")
                utt2dur[utt_id] = float(dur)

        filtered_data = {}
        for utt_id, entry in self.data.items():
            duration = utt2dur.get(utt_id, None)
            if duration is None:
                # print(f"Warning: Duration for {utt_id} not found in utt2dur. Skipping.")
                continue
            if duration <= self.max_audio_duration:
                filtered_data[utt_id] = entry
            else:
                # print(f"Skipping {utt_id}: duration {duration:.2f}s exceeds {self.max_audio_duration}s")
                continue
        self.data = filtered_data

    def load_processor(self, vocab_path: str):
        tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        return processor
        
    def set_debug(self, size: int = 100):
        """Reduce dataset size for debugging."""
        size = min(size, len(self.data))  # Ensure size doesn't exceed dataset size
        self.data = dict(list(self.data.items())[:size])  # Reduce data
        self.utt_ids = sorted(self.data.keys())  # Update utt_ids
        print(f"Debug mode: Dataset reduced to {len(self.utt_ids)} utterances.")

    def __len__(self) -> int:
        print(f"Dataset size (utt_ids): {len(self.utt_ids)}")
        return len(self.utt_ids)

    def __getitem__(self, idx):
        if idx >= len(self.utt_ids):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.utt_ids)} items.")
        utt_id = self.utt_ids[idx]
        entry = self.data[utt_id]
        wavs = entry["array"]
        sr = entry["sampling_rate"]

        processor = self.load_processor("./vocab.json")
        input_values = processor(wavs, sampling_rate=sr).input_values[0]

        text_int = torch.tensor([self.char_to_int[p] for p in entry["text"].split(" ")])
        # with processor.as_target_processor():
        #     text_int = processor(entry["text"]).input_ids

        # print(f"utt_id: {utt_id}, input_values: {input_values.shape}, speech_lengths: {input_values.shape}, labels: {text_int}, text_lengths: {len(text_int)}")
        return utt_id, {"speech": input_values, "text": text_int}
 