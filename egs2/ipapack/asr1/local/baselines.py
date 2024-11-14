import glob
import os
import argparse
from panphon import FeatureTable
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
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import jiwer
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModel
from types import SimpleNamespace

# Define XEUS Encoder
class XEUSEncoder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        xeus_model, _ = SSLTask.build_model_from_file(config_file=None, model_file=checkpoint_path, device=device)
        self.model = xeus_model
        self.model.frontend = None
        self.model.preencoder = None
        if not hasattr(self.model, 'hidden_sizes'):
            self.model.hidden_sizes = [self.model.encoder.output_size]
        if not hasattr(self.model, 'downsample_rates'):
            self.model.downsample_rates = [1] ## no downsampling
    def forward(self, speech, speech_lengths):
        feats = self.model.encode(speech=speech, speech_lengths=speech_lengths, use_mask=False, use_final_output=False)[0]
        return feats

# Define Finetuning Model
class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path, device, vocab_size, lora_config=None):
        super().__init__()
        self.xeus = XEUSEncoder(checkpoint_path, device)
        self.xeus.model.num_layers = 19
        self.weighted_sum = Featurizer(upstream=self.xeus.model)
        self.encoder = LinearEncoder(input_size=1024, output_size=256, dropout_rate=0.1, input_layer="linear", normalize_before=True)
        self.ctc = CTC(odim=vocab_size + 1, encoder_output_size=256)
        self.config = SimpleNamespace(use_return_dict=True)
        if lora_config:
            # Apply LoRA to the XEUS model
            self.xeus.model = get_peft_model(self.xeus.model, lora_config)
            print("LoRA applied to the XEUS model.")

    def forward(self, speech, speech_lengths):
        feats = self.xeus(speech, speech_lengths)
        weighted_feats, hs_len = self.weighted_sum(feats, [speech_lengths] * len(feats))
        logits = self.encoder(weighted_feats, hs_len)[0]
        return logits, hs_len

    def compute_loss(self, logits, labels, input_lengths, label_lengths):
        log_probs = logits.permute(1, 0, 2)  # [B, T, C] -> [T, B, C] for CTC loss 
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

def get_parser():
    parser = argparse.ArgumentParser(description="Phoneme recognition baselines with LoRA")
    parser.add_argument("--articulatory_losses", default=False, required=False)
    parser.add_argument("--train_dir", type=Path, default=Path("data/train"), required=True, help="Training data directory")
    parser.add_argument("--dev_dir", type=Path, default=Path("data/dev"), required=True, help="Development data directory")
    parser.add_argument("--epochs", type=int, default=50, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, required=True, help="Learning rate")
    parser.add_argument("--max_audio_duration", type=float, default=20, required=True, help="Max audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=4, required=True, help="Batch size")
    # parser.add_argument("--checkpoint_dir", type=Path, default=Path("baseline_checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=Path, default=Path("runs/fine_tuning_xeus_phoneme_ctc"), help="TensorBoard log directory")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    return parser

# Training step with mixed precision and gradient accumulation
def train_step(model, optimizer, train_loader, articulatory_ctc, device, epoch, iteration, accumulation_steps=4):
    model.train()
    total_loss = 0
    checkpoint_files = []
    batches_per_epoch = len(train_loader)
    start_batch = iteration // batches_per_epoch

    for i, batch in enumerate(tqdm(train_loader)):
        if i < start_batch:
            continue
        iteration += 1  # Increment global iteration count
        _, data = batch
        speech = data["speech"].to(device, non_blocking=True, dtype=torch.float32)
        speech_lengths = data["speech_lengths"].to(device, non_blocking=True)
        text = data["text"].to(device, non_blocking=True)
        text_lengths = data["text_lengths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            logits, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(logits, text, input_lengths, text_lengths)
            loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            print(f"Non-finite loss at iteration {iteration}. Skipping batch.")
            continue

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() 

        # Save checkpoint every 1000 iterations
        if iteration % 1000 == 0:  # Use global iteration count
            checkpoint_path = (
                f"baseline_checkpoints/model_checkpoint_{args.max_audio_duration}_"
                f"{args.batch_size}_epoch{epoch}_iter{iteration}.pth"
            )
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            checkpoint_files.append(checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration} (epoch {epoch}).")

            # Remove old checkpoints if limit is exceeded
            if len(checkpoint_files) > 3:
                oldest_checkpoint = checkpoint_files.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)

    avg_loss = total_loss / len(train_loader)
    return avg_loss, iteration  # Return the updated iteration count



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

def test_step(model, test_loader, device):
    model.eval()
    total_wer = 0
    total_samples = 0
    padding_value = -32768  # based on common_collate_fn

    with torch.no_grad():
        for batch in tqdm(test_loader):
            utt_ids, data = batch
            speech = data["speech"].to(device)
            speech_lengths = data["speech_lengths"].to(device)
            text = data["text"].to(device)
            text_lengths = data["text_lengths"].to(device)

            logits, input_lengths = model(speech, speech_lengths)
            
            # Decode the logits to get predicted text sequences
            predicted_sequences = model.decode(logits, input_lengths)

            # Convert target sequences to text for WER calculation
            target_sequences = []
            for idx_seq in text.cpu().numpy():
                seq = [test_loader.dataset.int_to_char[idx] for idx in idx_seq if idx != 0 and idx != padding_value]
                target_sequences.append(" ".join(seq))

            # Calculate WER for each prediction
            for pred, target in zip(predicted_sequences, target_sequences):
                pred_text = " ".join([test_loader.dataset.int_to_char[idx] for idx in pred])
                total_wer += jiwer.wer(target, pred_text)
                total_samples += 1

            torch.cuda.empty_cache()

    avg_wer = total_wer / total_samples
    return avg_wer

def load_checkpoint(model, optimizer, device):
    checkpoint_files = glob.glob(f"baseline_checkpoints/model_checkpoint_{args.max_audio_duration}_{args.batch_size}_epoch*_iter*.pth")
    if not checkpoint_files:
        return 0, 0  # No checkpoint found, start from scratch
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_iter')[-1].split('.')[0]))
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    
    print(f"Resuming from checkpoint {latest_checkpoint}: epoch {epoch}, iteration {iteration}")
    return epoch, iteration

# Main training loop
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    train_dset = IPAPack(f"{args.train_dir}/feats.scp", f"{args.train_dir}/text", utt2dur_file="train_utt2dur", max_audio_duration=args.max_audio_duration)
    dev_dset = IPAPack(f"{args.dev_dir}/feats.scp", f"{args.dev_dir}/text", utt2dur_file="dev_utt2dur", max_audio_duration=args.max_audio_duration)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=common_collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dset, batch_size=args.batch_size, collate_fn=common_collate_fn, num_workers=4, pin_memory=True)

    vocab_size = train_dset.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Define LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules = [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_out"
        ]
    )
    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size, lora_config=lora_config).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = GradScaler("cuda")
    writer = SummaryWriter(log_dir="runs/fine_tuning")

    start_epoch, start_iteration = load_checkpoint(model, optimizer, device)
    best_dev_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss, start_iteration = train_step(model, optimizer, train_loader, args.articulatory_losses, device, epoch, start_iteration)
        dev_loss = dev_step(model, dev_loader, device)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Dev", dev_loss, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Dev Loss: {best_dev_loss:.4f}")

    writer.close()

    ### Evaluate ## 
    best_model_path = "best_model.pth"
    test_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text", utt2dur_file="test_utt2dur", max_audio_duration=args.max_audio_duration)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, collate_fn=common_collate_fn, num_workers=4)

    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(best_model_path))

    test_wer = test_step(model, test_loader, device)
    print(f"Test WER: {test_wer:.4f}")