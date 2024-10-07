from local.dataset import IPAPack
from espnet2.train.collate_fn import common_collate_fn

import torch.nn


class FrozenFeatsSingleCTC(nn.Module):
    def __init__(self, xeus_model_dim, vocab_dim):
        super().__init__()
        # takes in XEUS features and adds single CTC head
        self.linear = nn.Linear(model_dim, vocab_dim)

    def forward(self, x):
        # TODO: check if this matches with ESPnet's input format
        return self.linear(x)


if __name__ == "__main__":
    # TODO: argparse
    train_dset = IPAPack(scp_file="data/train/feats.scp", text_file="data/train/text")
    # TODO: is the collcate_fn necessary if it's already an ESPnet dataset?
    # TODO: batch_bins ?
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=common_collate_fn)
    # TODO: train loop
    for utt_ids, batch in train_loader:
        # TODO: may need to do this? https://espnet.github.io/espnet/guide/espnet2/train/common_collate_fn.html

        # utt_ids: list of utt id's
        # batch: {"speech": ..., "speech_lengths": ..., "text": ..., "text_lengths": ...}
        pass

    model(**batch)
