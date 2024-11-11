import os
import glob 
import argparse
from pathlib import Path

from dataset import IPAPack
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.train.collate_fn import common_collate_fn
from espnet2.tasks.ssl import SSLTask

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from s3prl.nn import Featurizer
from panphon import FeatureTable
from tqdm import tqdm
import jiwer

# TODO: if time, could make this into a Frontend (multilayer_feature)
class XEUSEncoder(nn.Module):
    # meant to be finetuned
    def __init__(self, checkpoint_path, device):
        super().__init__()
        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        # source: https://www.wavlab.org/activities/2024/xeus/
        # (xs_pad, intermediate_outs), olens, None
        # we recommend use_mask=True during fine-tuning
        # TODO: use_mask=True (EJ)
        feats = self.model.encode(speech=speech, speech_lengths=speech_lengths, use_mask=False, use_final_output=False)[0]
        # ex: List of [batch, frames, model_dim] tensors
        return feats

class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path, device, vocab_size):
        super().__init__()
        self.xeus = XEUSEncoder(checkpoint_path, device)
        # TODO: load from file
        self.xeus.model.num_layers = 19
        # TODO: may need to cast model into S3PRLUpstream
        self.weighted_sum = Featurizer(upstream=self.xeus.model)
        # don't downsample to preserve as much temporal resolution as possible
        self.encoder = LinearEncoder(input_size=1024, output_size=256, dropout_rate=0.1, input_layer="linear", normalize_before = True)
        self.ctc = CTC(odim=vocab_size + 1, encoder_output_size=256) # +1 for blank token

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        feats = self.xeus(speech, speech_lengths)
        all_lens = [speech_lengths for _ in feats]
        weighted_sum_feats, hs_len = self.weighted_sum(feats, all_lens)
        logits = self.encoder(weighted_sum_feats, hs_len)[0]        
        return logits, hs_len
    
    def compute_loss(self, logits, labels, input_lengths, label_lengths):
        log_probs = logits.permute(1, 0, 2) 
        phoneme_ctc_loss = self.ctc.loss_fn(log_probs, labels, input_lengths, label_lengths)
        return phoneme_ctc_loss
    
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

def get_parser():
    parser = argparse.ArgumentParser(
        description="Phoneme recognition baselines"
    )
    parser.add_argument(
        "--articulatory_losses",
        default=False,
        required=False
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=Path("data/train"),
        required=True
    )
    parser.add_argument(
        "--dev_dir",
        type=Path,
        default=Path("data/dev"),
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        required=True
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        required=True
    )
    parser.add_argument(
        "--max_audio_duration",
        type=float,
        default=20,
        required=True
    )

    return parser

def train_step(model, optimizer, train_loader, articulatory_ctc, device, epoch, iteration, accumulation_steps=4):
    model.train()
    total_loss = 0
    checkpoint_files = []
    print(iteration)

        # Calculate the starting batch index for this epoch
    batches_per_epoch = len(train_loader)
    start_batch = start_iteration // batches_per_epoch

    for i, batch in tqdm(enumerate(train_loader)):
        # Skip to start batch if resuming mid-epoch
        if i < start_batch:
            continue

        iteration += 1
        optimizer.zero_grad()
        utt_ids, data = batch        
        speech = data["speech"].to(device)
        speech_lengths = data["speech_lengths"].to(device)
        text = data["text"].to(device)
        text_lengths = data["text_lengths"].to(device)
        
        logits, input_lengths = model(speech, speech_lengths)
        loss = model.compute_loss(logits, text, input_lengths, text_lengths)

        if articulatory_ctc == True:
            ft = FeatureTable()
            artic_feats = ft.names
            articulatory_ctc_losses = [CTC(odim=3, encoder_output_size=256).loss_fn for _ in artic_feats]
            for aux_loss in articulatory_ctc_losses:
                aux_loss_val = aux_loss(logits, text, input_lengths, text_lengths)
                loss += aux_loss_val  

        # Accumulate gradients
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Save checkpoint every 1000 iterations
        if iteration % 1000 == 0:
            checkpoint_path = f"baseline_checkpoints/model_checkpoint_{args.max_audio_duration}_epoch{epoch}_iter{iteration}.pth"
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            checkpoint_files.append(checkpoint_path)

            print(f"Checkpoint saved at iteration {iteration} (epoch {epoch}).")
            torch.cuda.empty_cache()  # Clear GPU cache to free up memory

            # Remove old checkpoints if limit is exceeded
            if len(checkpoint_files) > 3:
                oldest_checkpoint = checkpoint_files.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)

    return total_loss / len(train_loader), iteration


def dev_step(model, dev_loader, device):
    model.eval()
    total_loss = 0
    total_wer = 0
    total_samples = 0
    padding_value = -32768 ## based on common_collate_fn
    with torch.no_grad():
        for batch in tqdm(dev_loader):
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

            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dev_loader)
    avg_wer = total_wer / total_samples
    return avg_loss, avg_wer

def load_checkpoint(model, optimizer, device):
    checkpoint_files = glob.glob(f"baseline_checkpoints/model_checkpoint_{args.max_audio_duration}_epoch*_iter*.pth")
    if not checkpoint_files:
        return 0, 0  # No checkpoint found, start from scratch
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_iter')[-1].split('.')[0]))
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    
    print(f"Resuming from checkpoint {latest_checkpoint}: epoch {epoch}, iteration {iteration}")
    return epoch, iteration

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    train_dset = IPAPack(scp_file=f"{args.train_dir}/feats.scp", text_file=f"{args.train_dir}/text", utt2dur_file="train_utt2dur",  max_audio_duration=args.max_audio_duration)
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True, collate_fn=common_collate_fn, num_workers=4)
    dev_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text", utt2dur_file="dev_utt2dur")
    dev_loader = DataLoader(dev_dset, batch_size=2, collate_fn=common_collate_fn, num_workers=4)

    vocab_size = train_dset.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size).to(device)
    writer = SummaryWriter(log_dir="runs/fine_tuning_xeus_phoneme_ctc")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # Load the latest checkpoint if available
    start_epoch, start_iteration = load_checkpoint(model, optimizer, device)
    best_dev_wer = float('inf') 
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, start_iteration = train_step(model, optimizer, train_loader, args.articulatory_losses, device, epoch, start_iteration)
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
