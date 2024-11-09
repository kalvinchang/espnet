import torch
from torch.utils.data import Dataset, DataLoader
import kaldiio
import torchaudio
import numpy as np
from espnet2.train.dataset import ESPnetDataset


class IPAPack(ESPnetDataset):
    def __init__(self, scp_file, text_file, base_path="/ocean/projects/cis210027p/kchang1/espnet/egs2/ipapack/asr1/"):
        self.data = {}
        self.base_path = base_path
        with open(scp_file, 'r') as f:
            for line in f:
                utt_id, path = line.strip().split(' ', 1)
                self.data[utt_id] = {"feats_path": path}

        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue  # Skip lines if they don't contain phoneme sequence
                utt_id = parts[0]
                text = parts[1]
                if utt_id in self.data:
                    self.data[utt_id]["text"] = text
                else:
                    raise ValueError(f"Text file contains an `utt_id` not found in the features file: {utt_id}")

        # Remove entries without text
        self.data = {utt_id: entry for utt_id, entry in self.data.items() if "text" in entry}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the `utt_id` by index from the dictionary's keys
        utt_id = list(self.data.keys())[idx]
        entry = self.data[utt_id]
        feat_path = f"{self.base_path}/{entry['feats_path']}"
        features = kaldiio.load_mat(feat_path)

        texts = np.array([ord(char) for char in entry["text"] if char != ' ' and ord(char) != 0], dtype=np.int32)
        return utt_id, {
            "speech": features, ## num_frames x feature_dim
            "text": texts,
        }        
