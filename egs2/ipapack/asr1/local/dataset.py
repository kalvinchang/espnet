import kaldiio
import numpy as np
from espnet2.train.dataset import ESPnetDataset
import os

class IPAPack(ESPnetDataset):
    def __init__(
        self, 
        scp_file, 
        text_file, 
        utt2dur_file, 
        base_path="/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack/asr1", 
        frame_shift=0.01,  # in seconds
        max_duration=20.0  # in seconds
    ):
        self.data = {}
        self.base_path = base_path
        self.frame_shift = frame_shift  # Frame shift in seconds
        self.max_duration = max_duration  # Maximum allowed duration in seconds

        # Load feature paths
        self._load_scp(scp_file)

        # Load text data
        self._load_text(text_file)

        # Map characters to integer IDs for encoding, ignoring spaces
        unique_tokens = self.compute_vocab_size(text_file)
        sorted_unique_tokens = sorted(unique_tokens)
        self.char_to_int = {token: idx + 1 for idx, token in enumerate(sorted_unique_tokens)}  # Start indexing from 1 for padding
        self.int_to_char = {idx: token for token, idx in self.char_to_int.items()}  # Reverse mapping for decoding
        self.vocab_size = len(unique_tokens)

        # Filter out entries with audio longer than max_duration
        if utt2dur_file:
            self._filter_by_duration_utt2dur(utt2dur_file)
        else:
            self._filter_by_duration_features()

        # Final list of utterance IDs
        self.utt_ids = sorted(self.data.keys())

    def _load_scp(self, scp_file):
        """Load feature paths from scp file."""
        with open(scp_file, 'r') as f:
            for line in f:
                utt_id, path = line.strip().split(' ', 1)
                self.data[utt_id] = {"feats_path": path}

    def _load_text(self, text_file):
        """Load text data and associate it with utterance IDs."""
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

        # Remove entries without text
        self.data = {utt_id: entry for utt_id, entry in self.data.items() if "text" in entry}

    def compute_vocab_size(self, text_file):
        """Compute the set of unique tokens (phonemes) from the text file."""
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
            if duration <= self.max_duration:
                filtered_data[utt_id] = entry
            else:
                print(f"Skipping {utt_id}: duration {duration:.2f}s exceeds {self.max_duration}s")
        
        self.data = filtered_data

    def _filter_by_duration_features(self):
        """Filter utterances based on durations computed from feature files."""
        filtered_data = {}
        for utt_id, entry in self.data.items():
            feat_path = os.path.join(self.base_path, entry['feats_path'])

            try:
                features = kaldiio.load_mat(feat_path)
                num_frames = features.shape[0]
                duration = num_frames * self.frame_shift  # duration in seconds
                if duration <= self.max_duration:
                    filtered_data[utt_id] = entry
                else:
                    print(f"Skipping {utt_id}: duration {duration:.2f}s exceeds {self.max_duration}s")
            except Exception as e:
                print(f"Error loading features for {utt_id}: {e}")
                continue

        self.data = filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        entry = self.data[utt_id]
        feat_path = os.path.join(self.base_path, entry['feats_path'])

        # Load features with error handling
        try:
            features = kaldiio.load_mat(feat_path)
        except Exception as e:
            print(f"Error loading features for {utt_id}: {e}")
            # It's better to raise an exception to avoid unexpected None values
            raise e

        # Convert text to integer IDs, ignoring spaces
        texts = np.array([self.char_to_int.get(char, 0) for char in entry["text"] if char != ' '], dtype=np.int32)
        return utt_id, {
            "speech": features,
            "text": texts,
        }

    def get_vocab_size(self):
        return self.vocab_size
