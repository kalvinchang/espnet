import os
import json
import torch
import numpy as np
import soundfile as sf
from espnet2.train.dataset import ESPnetDataset
from typing import Dict, Tuple, List


class IPAPack(ESPnetDataset):
    """
    Custom dataset class for IPA-based speech data.
    """

    def __init__(
        self,
        scp_file: str,
        text_file: str,
        utt2dur_file: str,
        max_audio_duration: float,
        base_path: str = "/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack/asr1",
        vocab_path: str = None,
        debug: bool = False,
        debug_size: int = None,
    ):
        self.base_path = base_path
        self.max_audio_duration = max_audio_duration  # Maximum allowed duration in seconds
        self.data: Dict[str, Dict[str, str]] = {}

        # Load features, text, and vocabulary
        self._load_scp(scp_file)
        self._load_text(text_file)

        if vocab_path:
            self._load_vocab(vocab_path)
        else:
            self._build_and_save_vocab(
                os.path.join(self.base_path, "data/train/text"), "./vocab.json"
            )

        self._filter_by_duration_utt2dur(utt2dur_file)
        self.utt_ids = sorted(self.data.keys())

        # Apply debug size if specified and debug is True
        if debug and debug_size is not None:
            self.set_debug(debug_size)

        # Debug: Print the number of utterances loaded
        print(f"Loaded {len(self.utt_ids)} utterances.")

    def _load_scp(self, scp_file: str):
        scp_file = os.path.join(self.base_path, scp_file)
        with open(scp_file, "r") as f:
            for line in f:
                utt_id, path = line.strip().split(" ", 1)
                self.data[utt_id] = {"feats_path": path}

    def _load_text(self, text_file: str):
        text_file = os.path.join(self.base_path, text_file)
        with open(text_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                if utt_id in self.data:
                    self.data[utt_id]["text"] = text
                else:
                    print(f"Warning: {utt_id} not found in SCP data.")

        # Remove entries without valid text
        self.data = {utt_id: entry for utt_id, entry in self.data.items() if "text" in entry}


    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from a JSON file."""
        with open(vocab_path, "r") as vocab_file:
            self.int_to_char = json.load(vocab_file)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}
        self.vocab_size = len(self.int_to_char)

    def _build_and_save_vocab(self, text_file: str, vocab_file_path: str):
        """Build vocabulary from text and save it as JSON."""
        phoneme_set = self._compute_vocab(text_file)
        self.int_to_char = dict(enumerate(phoneme_set, start=1))
        self.char_to_int = {v: k for k, v in self.int_to_char.items()}
        self.vocab_size = len(self.int_to_char)

        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            json.dump(self.int_to_char, vocab_file, ensure_ascii=False, indent=4)

    def _compute_vocab(self, text_file: str) -> List[str]:
        """Extract unique phonemes from text data."""
        phoneme_set = set()
        with open(text_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    _, text = parts
                    phoneme_set.update(text.split())
        return sorted(phoneme_set)

    def _filter_by_duration_utt2dur(self, utt2dur_file: str):
        """Filter utterances exceeding the maximum duration."""
        utt2dur_file = os.path.join(
            "/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data",
            utt2dur_file,
        )
        utt2dur = {}
        with open(utt2dur_file, "r") as f:
            for line in f:
                utt_id, dur = line.strip().split()
                utt2dur[utt_id] = float(dur)

        filtered_data = {}
        for utt_id, entry in self.data.items():
            duration = utt2dur.get(utt_id, None)
            if duration is None:
                print(f"Warning: Duration for {utt_id} not found in utt2dur. Skipping.")
                continue
            if duration <= self.max_audio_duration:
                filtered_data[utt_id] = entry
            else:
                continue
                # print(f"Skipping {utt_id}: duration {duration:.2f}s exceeds {self.max_audio_duration}s")
        self.data = filtered_data

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
        wav_path = os.path.join(self.base_path, entry["feats_path"])
        wavs, sampling_rate = sf.read(wav_path)
        text_int = torch.tensor([self.char_to_int.get(p, 0) for p in entry["text"].split(" ")])

        text = [p for p in entry["text"].split(" ")]
        print(utt_id, text)
        return utt_id, {"speech": wavs, "text": text_int}
                        
                        