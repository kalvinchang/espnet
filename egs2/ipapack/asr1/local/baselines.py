import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import glob 
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import LambdaLR
from transformers import Wav2Vec2Model
from torch.amp import autocast, GradScaler
from torchaudio.models.decoder import ctc_decoder

from torch.utils.tensorboard import SummaryWriter
import jiwer
from dataset import IPAPack  # Replace with your dataset file
from collate_fn import common_collate_fn  # Replace with your collate_fn file

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(model, optimizer, scheduler, scaler, device, args):
    # Look for all available subepoch checkpoints
    checkpoint_files = glob.glob(f"{args.checkpoint_dir}/model_checkpoint_epoch*_subepoch*.pth")

    if not checkpoint_files:
        print("No checkpoints found. Starting training from scratch.")
        return 1, 0  # Start from epoch 1, subepoch 0

    # Extract epoch and subepoch from the checkpoint filenames
    def extract_epoch_subepoch(checkpoint):
        match = re.search(r'epoch(\d+)_subepoch(\d+)', checkpoint)
        if match:
            epoch, subepoch = match.groups()
            return int(epoch), int(subepoch)
        else:
            logging.warning(f"Checkpoint '{checkpoint}' missing 'epoch' or 'subepoch' information. Starting fresh.")
            return 1, 0

    # Sort checkpoints by epoch and subepoch in descending order
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda x: extract_epoch_subepoch(x),
        reverse=True
    )
    # Select the latest checkpoint
    latest_checkpoint = checkpoint_files[0]
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint['epoch']
        subepoch = checkpoint['subepoch']

        learning_rate = checkpoint.get('learning_rate', None)  # Retrieve learning rate if available
        if learning_rate:
            print(f"Resumed with learning rate: {learning_rate}")

        print(f"Resuming training from Epoch {epoch}, Subepoch {subepoch}: {latest_checkpoint}")
        return epoch, subepoch

    except Exception as e:
        print(f"Error loading checkpoint '{latest_checkpoint}': {e}")
        print("Starting training from scratch.")
        return 1, 0  # Start fresh if loading fails

def partition_indices(dataset, num_parts):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # Shuffle indices to ensure randomness

    part_size = dataset_size // num_parts
    partitions = [
        indices[i * part_size: (i + 1) * part_size] for i in range(num_parts)
    ]
    if dataset_size % num_parts != 0:
        partitions[-1].extend(indices[num_parts * part_size:])  # Handle leftover indices
    return partitions


class SSLWithTransformers(nn.Module):
    def __init__(self, ssl_model_name, num_transformer_layers, vocab_path, num_classes, beam_width=10, beam_threshold=50.0):
        super(SSLWithTransformers, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.feature_dim = self.model.config.hidden_size

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dim_feedforward=self.feature_dim)
        self.transformers = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # CTC Decoder
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.blank_idx = 0
        self.ignore_id = -32768
        self.CTCDecoder = ctc_decoder(
            lexicon=None, 
            tokens=list(self.vocab.values()), 
            lm=None,  
            nbest=1,
            beam_size=beam_width,
            beam_threshold=beam_threshold,
            blank_token=self.vocab["0"], 
            sil_token=self.vocab[str(len(self.vocab)-1)]
        )

    def forward(self, input_values):
        with torch.no_grad():
            ssl_outputs = self.model(input_values)
            features = ssl_outputs.last_hidden_state  # Shape: (batch_size, seq_len, feature_dim)
            # print(f"features: {features.shape}")

        transformer_out = self.transformers(features)  # Shape: (batch_size, seq_len, feature_dim)        
        logits = self.fc(transformer_out)  # Shape: (batch_size, seq_len, num_classes)
        # print(f"transformer_out: {transformer_out.shape}")
        # print(f"logits: {logits.shape}")
        return logits

    def greedy_decode(self, log_probs):
        predictions = torch.clamp(torch.argmax(log_probs, dim=-1), min=0, max=len(self.vocab) - 1)
        return [
            " ".join(self.vocab[str(idx.item())] for j, idx in enumerate(pred) if idx != self.blank_idx and (j == 0 or idx != pred[j-1])) for pred in predictions
            ]

    def beam_search_decode(self, log_probs):
        if not log_probs.is_cpu:
            log_probs = log_probs.cpu()
        decoded = self.CTCDecoder(log_probs)
        decoded_texts = []
        for sample in decoded:
            for hyp in sample:
                text = " ".join(self.vocab[str(idx.item())] for idx in hyp.tokens)
                decoded_texts.append(text)
        return decoded_texts

## schedulaer 
def get_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Warmup and decay scheduler."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler, num_warmup_steps

# Training Step
def train_step(model, optimizer, scheduler, train_loader, device, epoch, accumulation_steps, scaler, num_batches, writer, global_step):
    model.train()
    total_loss = 0
    ctc_loss = CTCLoss(blank=0)  # Assume blank token index is 0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", total=num_batches)):
        utt_id, data = batch
        speech = data["speech"].to(device)
        speech_lengths = data["speech_lengths"].to(device)
        text = data["text"].to(device)
        text_lengths = data["text_lengths"].to(device)

        with autocast("cuda"):
            logits = model(speech)
            log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
            # print(f"log_probs: {log_probs.shape}")
            speech_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long, device=device)
            # print(f"speech_lengths: {speech_lengths}")
            loss = ctc_loss(log_probs, text, speech_lengths, text_lengths) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()
        global_step += 1
        if global_step % 1000 == 0:
            avg_loss = total_loss / (i + 1)
            writer.add_scalar("Loss/Train (batch)", avg_loss, global_step)

    return total_loss / len(train_loader)

# Validation Step
def dev_step(model, dev_loader, device, int_to_char, num_batches, writer, global_step):
    model.eval()
    total_loss = 0
    total_per_greedy = 0
    total_cer_greedy = 0
    total_per_beam = 0
    total_cer_beam = 0
    total_samples = 0

    ctc_loss = CTCLoss(blank=0)  # Assume blank token index is 0
    with torch.no_grad(), autocast(device_type="cuda"):
        for i, batch in enumerate(tqdm(dev_loader, desc="Validation", unit="batch", total=num_batches)):
            utt_id, data = batch
            speech = data["speech"].to(device)
            speech_lengths = data["speech_lengths"].to(device)
            text = data["text"].to(device)
            text_lengths = data["text_lengths"].to(device)

            logits = model(speech)
            log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
            speech_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, text, speech_lengths, text_lengths)
            total_loss += loss.item()

            # Decode logits using both methods
            decoded_sequences_greedy = model.greedy_decode(log_probs)
            decoded_sequences_beam = model.beam_search_decode(log_probs)

            # Calculate PER and CER
            for pred, target in zip(decoded_sequences_greedy, text.cpu().numpy()):
                # print(target)
                target_text = " ".join([int_to_char.get(idx, "<UNK>") for idx in target if idx != model.ignore_id])
                total_per_greedy += jiwer.wer(target_text, pred)
                total_cer_greedy += jiwer.cer(target_text, pred)
                print(f"target: {target_text}")
                print(f"pred(greedy): {pred}")
                total_samples += 1
                print("")

            for pred, target in zip(decoded_sequences_beam, text.cpu().numpy()):
                target_text = " ".join([int_to_char.get(idx, "<UNK>") for idx in target if idx != model.ignore_id])
                pred = pred.replace("|", "")
                print(f"target: {target_text}")
                print(f"pred(beam): {pred}")
                total_per_beam += jiwer.wer(target_text, pred)
                total_cer_beam += jiwer.cer(target_text, pred)
                print("")

            ## Log batch of the validation set 
            global_step += 1
            if global_step % 1000 == 0:
                avg_loss = total_loss / (i + 1)
                avg_per_greedy = total_per_greedy / total_samples
                avg_cer_greedy = total_cer_greedy / total_samples
                avg_per_beam = total_per_beam / total_samples
                avg_cer_beam = total_cer_beam / total_samples

                writer.add_scalar('Loss/Dev total loss (batch)', avg_loss, global_step)
                writer.add_scalar('PER/greedy (batch)', avg_per_greedy, global_step)
                writer.add_scalar('CER/greedy (batch)', avg_cer_greedy, global_step)
                writer.add_scalar('PER/beam (batch)', avg_per_beam, global_step)
                writer.add_scalar('CER/beam (batch)', avg_cer_beam, global_step)

    avg_loss = total_loss / num_batches
    avg_per_greedy = total_per_greedy / total_samples
    avg_cer_greedy = total_cer_greedy / total_samples
    avg_per_beam = total_per_beam / total_samples
    avg_cer_beam = total_cer_beam / total_samples

    return avg_loss, avg_per_greedy, avg_cer_greedy, avg_per_beam, avg_cer_beam

# Main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="SSLWithTransformers Training")
    parser.add_argument("--base_model", type=str, default=16, help="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab JSON")
    parser.add_argument("--beam_size_test", type=int, default=10, help="Beam size for testing")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("baseline_checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with smaller dataset")
    parser.add_argument("--max_audio_duration", type=float, default=20, required=True, help="Max audio duration in seconds")
    args = parser.parse_args()

    print(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    train_dset = IPAPack(
        scp_file="data/train_filtered_wav.scp",
        text_file="data/train_filtered_text",
        utt2dur_file="data/train_utt2dur",
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,
        debug=args.debug,
        debug_size=1000,
    )
    dev_dset = IPAPack(
        scp_file="data/dev_filtered_wav.scp",
        text_file="data/dev_filtered_text",
        utt2dur_file="data/dev_utt2dur",
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path, 
        debug=args.debug,
        debug_size=100,
    )
    test_dset = IPAPack(
        scp_file="data/test_doreco_filtered_wav.scp",
        text_file="data/test_doreco_filtered_text",
        utt2dur_file="data/test_doreco_utt2dur",
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,  # Load existing vocabulary
        debug=args.debug,
        debug_size=100,
    )

    # Load vocabulary from vocab.json
    if args.vocab_path and os.path.exists(args.vocab_path):
        with open(args.vocab_path, "r") as f:
            vocab = json.load(f)
        int_to_char = {int(idx): char for idx, char in vocab.items()}
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")
    
    num_subepochs = 3 
    train_partitions = partition_indices(train_dset, num_subepochs)  # Partition dataset into subepochs

    # Model
    model = SSLWithTransformers(
        ssl_model_name=args.base_model,
        num_transformer_layers=12,
        vocab_path=args.vocab_path,
        num_classes=len(vocab),
        beam_width=args.beam_size_test,
        beam_threshold=50.0
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # Optimizer and Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda")

    num_training_steps = args.epochs * (len(train_dset) // args.batch_size)
    scheduler, warmup_steps = get_scheduler_with_warmup(optimizer, num_training_steps * 0.1, num_training_steps)

    # Training
    base_model = args.base_model.split("/")[-1]
    writer = SummaryWriter(log_dir=f"runs/fine_tuning_xeus_phoneme_ctc_{base_model}_{args.max_audio_duration}_{args.batch_size}_{args.lr}")

    best_dev_loss = float("inf")  # Initialize best development loss

    # Load checkpoint if available
    try:
        start_epoch, start_subepoch = load_checkpoint(model, optimizer, scheduler, scaler, device, args)
    except Exception as e:
        print(f"Failed to load checkpoint due to error: {e}")
        start_epoch, start_subepoch = 1, 0  # Start fresh

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch: {epoch}/{args.epochs} ===")
        for subepoch, subset_indices in enumerate(train_partitions, start=1):
            if epoch == start_epoch and subepoch < start_subepoch:
                continue  # Skip completed subepochs if resuming

            print(f"\n--- Subepoch: {subepoch}/{num_subepochs} ---")

            train_subset = Subset(train_dset, subset_indices)
            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                collate_fn=common_collate_fn,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
            )
            dev_loader = DataLoader(
                dev_dset,
                batch_size=args.batch_size,
                collate_fn=common_collate_fn,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
            )
            test_loader = DataLoader(
                test_dset,
                batch_size=args.batch_size,
                collate_fn=common_collate_fn,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
            )

            train_loss = train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
                accumulation_steps=args.accumulation_steps,
                scaler=scaler,
                num_batches=len(train_loader),
                writer=writer,
                global_step=global_step,
            )

            # Subepoch-level logging
            subepoch_step = epoch + (subepoch / num_subepochs)
            writer.add_scalar('Loss/Train total loss (epoch)', train_loss, subepoch_step)

            # Save checkpoint after validation
            
            checkpoint_path = f"{args.checkpoint_dir}/model_checkpoint_epoch{epoch}_subepoch{subepoch}_{base_model}.pth"
            torch.save({
                'epoch': epoch,
                'subepoch': subepoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                # 'dev_loss': dev_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            dev_loss, dev_per_greedy, dev_cer_greedy, dev_per_beam, dev_cer_beam = dev_step(
                    model=model,
                    dev_loader=dev_loader,
                    device=device,
                    int_to_char=int_to_char,
                    num_batches=len(dev_loader),
                    writer=writer,
                    global_step=global_step,
                )
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Dev Loss: {dev_loss:.4f}, PER (Greedy): {dev_per_greedy:.4f}, CER (Greedy): {dev_cer_greedy:.4f},  PER (beam): {dev_per_beam:.4f}, CER (beam): {dev_cer_beam:.4f}")

            # Save checkpoint after validation
            checkpoint_path = f"{args.checkpoint_dir}/model_checkpoint_epoch{epoch}_subepoch{subepoch}_{base_model}.pth"
            torch.save({
                'epoch': epoch,
                'subepoch': subepoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'dev_loss': dev_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            writer.add_scalar('Loss/Dev total loss (epoch)', dev_loss, subepoch_step)

            # Log validation metrics
            writer.add_scalar('PER/greedy (epoch)', dev_per_greedy, subepoch_step)
            writer.add_scalar('CER/greedy (epoch)', dev_cer_greedy, subepoch_step)
            writer.add_scalar('PER/beam (epoch)', dev_per_beam, subepoch_step)
            writer.add_scalar('CER/beam (epoch)', dev_cer_beam, subepoch_step)

            # Update best model
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_model_path = f"{args.checkpoint_dir}/best_model_epoch{epoch}_subepoch{subepoch}_{base_model}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Dev Loss: {best_dev_loss:.4f}")

    writer.close()
    print("Training completed.")
