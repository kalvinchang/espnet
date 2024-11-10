import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from panphon import FeatureTable
from tqdm import tqdm

import jiwer
from dataset import IPAPack
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.train.collate_fn import common_collate_fn
from espnet2.tasks.ssl import SSLTask
from s3prl.nn import Featurizer

def get_parser():
    parser = argparse.ArgumentParser(description="Phoneme recognition baselines")
    parser.add_argument("--articulatory_losses",type=bool, default=False, required=False)
    parser.add_argument("--train_dir", type=Path, default=Path("data/train"), required=True)
    parser.add_argument("--dev_dir", type=Path, default=Path("data/dev"), required=True)
    parser.add_argument("--epochs", type=int, default=50, required=True)
    parser.add_argument("--lr", type=float, default=0.001, required=True)
    return parser

class XEUSEncoder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        xeus_model, _ = SSLTask.build_model_from_file(
            config_file=None,
            model_file=checkpoint_path, 
            device=device
        )
        self.model = xeus_model
        self.model.frontend = None ## skip frontend because we already have features extracted (feats.scp)
        self.model.preencoder = None ## skip preencoder because we already have features extracted (feats.scp)
        if not hasattr(self.model, 'hidden_sizes'):
            self.model.hidden_sizes = [self.model.encoder.output_size]
        if not hasattr(self.model, 'downsample_rates'):
            self.model.downsample_rates = [1] ## no downsampling

    def forward(self, speech, speech_lengths):
        ## need to fix: use_maks=True
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
        return logits, hs_len

    def compute_loss(self, logits, labels, input_lengths, label_lengths):
        log_probs = logits.permute(1, 0, 2)
        # ESPnet does the log_softmax for you
        return self.ctc.loss_fn(log_probs, labels, input_lengths, label_lengths)

    def decode(self, logits, input_lengths):
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        preds = torch.argmax(log_probs, dim=-1).cpu().numpy()  # Shape: [batch_size, max_time]

        sequences = []
        blank_id = 0
        for i, pred in enumerate(preds):
            pred = pred[:input_lengths[i]]
            seq = []
            prev_token = None
            for token in pred:
                if token != blank_id and token != prev_token:
                    seq.append(token)
                prev_token = token
            sequences.append(seq)
        return sequences

def train_step(model, optimizer, train_loader, artic_feats, device):
    accumulation_steps = 4
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

        # articulatory_ctc_losses = None
        # if args.articulatory_losses == True:
        #     articulatory_ctc_losses = [CTC(odim=3, encoder_output_size=256).loss_fn for _ in artic_feats]
        #     for aux_loss in articulatory_ctc_losses:
        #         loss += aux_loss(logits)
        
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear CUDA cache periodically

        total_loss += loss.item() * accumulation_steps  # Reverse the earlier division for accuracy
    return total_loss / len(train_loader)

    #     optimizer.step()
    #     total_loss += loss.item()
    # return total_loss / len(train_loader)

def dev_step(model, dev_loader, device):
    model.eval()
    total_loss = 0
    total_wer = 0
    total_samples = 0
    padding_value = -32768
    with torch.no_grad():
        for batch in dev_loader:
            utt_ids, data = batch
            speech = data["speech"].to(device)
            speech_lengths = data["speech_lengths"].to(device)
            text = data["text"].to(device)
            text_lengths = data["text_lengths"].to(device)

            logits, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(logits, text, input_lengths, text_lengths)
            total_loss += loss.item()
            
            # Decode the logits to get predicted text sequences
            predicted_sequences = model.decode(logits, input_lengths)
            
            # Convert target sequences to text for WER calculation
            target_sequences = []
            for idx_seq in text.cpu().numpy():
                seq = [train_loader.dataset.int_to_char[idx] for idx in idx_seq if idx != 0 and idx != padding_value]
                target_sequences.append(" ".join(seq))

            # Calculate WER for each prediction
            for pred, target in zip(predicted_sequences, target_sequences):
                pred_text = " ".join([train_loader.dataset.int_to_char[idx] for idx in pred])
                total_wer += jiwer.wer(target, pred_text)
                total_samples += 1
    
    avg_loss = total_loss / len(dev_loader)
    avg_wer = total_wer / total_samples
    print(avg_loss, avg_wer)
    return avg_loss, avg_wer

    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    train_dset = IPAPack(scp_file=f"{args.train_dir}/feats.scp", text_file=f"{args.train_dir}/text", utt2dur_file="train_utt2dur")
    dev_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text", utt2dur_file="dev_utt2dur")

    vocab_size = train_dset.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size).to(device)
    writer = SummaryWriter(log_dir="runs/fine_tuning_xeus_phoneme_ctc")

    # Adjust the collate_fn usage to ensure data format compatibility
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, collate_fn=common_collate_fn, num_workers=4)
    dev_loader = DataLoader(dev_dset, batch_size=1, collate_fn=common_collate_fn, num_workers=4)

    ft = FeatureTable()
    artic_feats = ft.names
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_dev_wer = float('inf')  # Initialize with a large value to track the best WER
    for epoch in range(args.epochs):
        train_loss = train_step(model, optimizer, train_loader, artic_feats, device)
        dev_loss, dev_wer = dev_step(model, dev_loader, device)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", dev_loss, epoch)
        writer.add_scalar("WER/Validation", dev_wer, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev WER: {dev_wer:.4f}")

        # Check if the current model is the best so far
        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Dev WER: {best_dev_wer:.4f}")

    writer.close()
