import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import IPAPack
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.train.collate_fn import common_collate_fn
from espnet2.tasks.ssl import SSLTask
from s3prl.nn import Featurizer

def get_vocab_size(text_file):
    with open(text_file, 'r') as f:
        phoneme_set = set()
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                _, text = parts
                phonemes = text.split(" ")
                phoneme_set.update(phonemes)
    return len(phoneme_set)

def get_parser():
    parser = argparse.ArgumentParser(description="Phoneme recognition baselines")
    parser.add_argument("--articulatory_losses", type=bool, default=False, required=False)
    parser.add_argument("--train_dir", type=Path, default=Path("data/train"), required=True)
    parser.add_argument("--dev_dir", type=Path, default=Path("data/dev"), required=True)
    parser.add_argument("--epochs", type=int, default=50, required=True)
    parser.add_argument("--lr", type=float, default=0.001, required=True)
    return parser

class XEUSEncoder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        xeus_model, train_args = SSLTask.build_model_from_file(
            config_file=None,
            model_file=checkpoint_path, 
            device=device
        )
        self.model = xeus_model
        self.model.frontend = None
        self.model.preencoder = None
        if not hasattr(self.model, 'hidden_sizes'):
            self.model.hidden_sizes = [self.model.encoder.output_size]
        if not hasattr(self.model, 'downsample_rates'):
            self.model.downsample_rates = [1] ## no downsampling

    def forward(self, speech, speech_lengths):
        batch_size, time, feature_dim = speech.shape
        feats = self.model.encode(speech=speech, speech_lengths=speech_lengths, use_mask=False, use_final_output=False)[0]
        return feats

class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path, device, vocab_size):
        super().__init__()
        self.xeus = XEUSEncoder(checkpoint_path, device)
        self.xeus.model.num_layers = 19
        self.weighted_sum = Featurizer(upstream=self.xeus.model)
        self.encoder = LinearEncoder(input_size=1024, output_size=256, dropout_rate=0.1, input_layer="linear")
        self.ctc = CTC(odim=vocab_size + 1, encoder_output_size=256)

    def forward(self, speech, speech_lengths):
        feats = self.xeus(speech, speech_lengths)

        all_lens = [speech_lengths for _ in feats]
        weighted_sum_feats, hs_len = self.weighted_sum(feats, all_lens)
        
        logits = self.encoder(weighted_sum_feats, hs_len)[0]
        # hs_len = torch.tensor(hs_len, dtype=torch.int32).to(speech.device)
        
        return logits, hs_len

    def compute_loss(self, logits, labels, input_lengths, label_lengths):
        log_probs = logits.permute(1, 0, 2)
        # ESPnet does the log_softmax for you
        return self.ctc.loss_fn(log_probs, labels, input_lengths, label_lengths)

def train_step(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        utt_ids, data = batch
        speech = data["speech"].to(device)
        speech_lengths = data["speech_lengths"].to(device)
        text = data["text"].to(device)
        text_lengths = data["text_lengths"].to(device)
        
        logits, input_lengths = model(speech, speech_lengths)
        
        loss = model.compute_loss(logits, text, input_lengths, text_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def dev_step(model, dev_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            utt_ids, data = batch  # Separate utt_ids and data dictionary
            speech = data["speech"].to(device)
            speech_lengths = data["speech_lengths"].to(device)
            text = data["text"].to(device)
            text_lengths = data["text_lengths"].to(device)

            logits, speech_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(logits, text, speech_lengths, text_lengths)
            total_loss += loss.item()

    return total_loss / len(dev_loader)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    train_dset = IPAPack(scp_file=f"{args.train_dir}/feats.scp", text_file=f"{args.train_dir}/text")
    dev_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text")

    vocab_size = get_vocab_size(text_file=f"{args.train_dir}/text")
    print(f"Vocabulary size: {vocab_size}")

    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size).to(device)
    writer = SummaryWriter(log_dir="runs/fine_tuning_xeus_phoneme_ctc")

    # Adjust the collate_fn usage to ensure data format compatibility
    train_loader = DataLoader(train_dset, batch_size=4, shuffle=True, collate_fn=common_collate_fn, num_workers=2)
    dev_loader = DataLoader(dev_dset, batch_size=4, collate_fn=common_collate_fn, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_step(model, optimizer, train_loader, device)
        dev_loss = dev_step(model, dev_loader, device)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", dev_loss, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")

    writer.close()
