import torch
from torch.utils.data import Dataset, DataLoader
import kaldiio

from espnet2.train.dataset import ESPnetDataset


class IPAPack(ESPnetDataset):
    def __init__(self, scp_file, text_file):
        self.data, self.text = [], []
        # assumes feats.scp and text are in the same utterance order
          # TODO: verify if this is okay assumption
        # ex: data/train/feats.scp
        with open(scp_file, 'r') as f:
            for line in f:
                # <utt_id> <path_to_ark>:<index>
                utt_id, path = line.strip().split(' ', 1)
                self.data.append((utt_id, path))
        # ex: data/train/text
        with open(text_file, 'r') as f:
            for line in f:
                # <utt_id> <phonemic transcription>
                utt_id, text = line.strip().split(' ', 1)
                self.text.append((utt_id, text))
        assert len(self.data) == len(self.text), "Speech features and text of different lengths"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id, path = self.data[idx]
        _, text = self.text[idx]

        features = kaldiio.load_mat(path)
        return {
            "utt_id": utt_id,
            "speech": features,
            "text": text
        }
