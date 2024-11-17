import kaldiio
import numpy as np
from espnet2.train.dataset import ESPnetDataset
import os

class IPAPack(ESPnetDataset):
    def __init__(self, scp_file, text_file, utt2dur_file, max_audio_duration,
        base_path="/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack/asr1"):
        self.data = {}
        self.base_path = base_path
        self.max_audio_duration = max_audio_duration  # Maximum allowed duration in seconds

        # Load feature paths
        self._load_scp(scp_file)

        # Load text data
        self._load_text(text_file)

        # Map characters to integer IDs for encoding, ignoring spaces
        unique_tokens = sorted(self.compute_vocab_size(text_file))
        self.int_to_char = dict(enumerate(unique_tokens, start=1))
        self.char_to_int = {v: k for k, v in self.int_to_char.items()}
        self.vocab_size = len(unique_tokens)
        print(unique_tokens)

        # Filter long audio utterances)
        self._filter_by_duration_utt2dur(utt2dur_file)

        # Final list of utterance IDs
        self.utt_ids = sorted(self.data.keys())

    def _load_scp(self, scp_file):
        with open(scp_file, 'r') as f:
            for line in f:
                utt_id, path = line.strip().split(' ', 1)
                self.data[utt_id] = {"feats_path": path}

    def _load_text(self, text_file):
        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                if utt_id in self.data:
                    self.data[utt_id]["text"] = text
                else:
                    print(f"Warning: Text for {utt_id} not found in filtered features.")

        # Remove entries without text (empty texts)
        self.data = {utt_id: entry for utt_id, entry in self.data.items() if "text" in entry}

    def compute_vocab_size(self, text_file):
        """Compute the set of unique phonemes from the text file."""
        with open(text_file, 'r') as f:
            phoneme_set = set()
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    _, text = parts
                    phonemes = text.split(" ")
                    phoneme_set.update(phonemes)
        return phoneme_set

    def _filter_by_duration_utt2dur(self, utt2dur_file):
        """Filter utterances based on durations from utt2dur file."""
        if 'dev' in utt2dur_file or 'test' in utt2dur_file:
            print(f"Skipping filtering for {utt2dur_file}")
            return self.data

        utt2dur = {}
        fpath = '/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data/' + utt2dur_file
        with open(fpath, 'r') as f:
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

    def set_debug(self, size=100):
        print(f"Reducing dataset size to {size} for debugging")
        self.data = {k: v for i, (k, v) in enumerate(self.data.items()) if i < size}
        self.utt_ids = sorted(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        entry = self.data[utt_id]
        feat_path = os.path.join(self.base_path, entry['feats_path'])

        try:
            features = kaldiio.load_mat(feat_path)
        except Exception as e:
            print(f"Error loading features for {utt_id}: {e}")
            raise e

        # Convert text to integer IDs, ignoring spaces
        texts = np.array([self.char_to_int[char] for char in entry["text"].split(' ')], dtype=np.int32)
        return utt_id, {
            "speech": features,
            "text": texts,
        }

    def get_vocab_size(self):
        return self.vocab_size
