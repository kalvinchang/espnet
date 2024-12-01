import os
import json
import logging
import random
import argparse
import numpy as np
import pickle
import re
from panphon import FeatureTable
from pathlib import Path
from tqdm import tqdm
import glob
import math
import itertools  # Import here to avoid potential issues
import jiwer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CTCLoss
from torchaudio.models.decoder import ctc_decoder
from torch.utils.tensorboard import SummaryWriter


from peft import AdaLoraConfig, get_peft_model, TaskType
from dataset import IPAPack
from collate_fn import common_collate_fn
from s3prl.nn import Featurizer
from espnet2.tasks.ssl import SSLTask

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# XEUS Encoder definition
class XEUSEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device, normalize: bool = False):
        super().__init__()
        xeus_model, _ = SSLTask.build_model_from_file(
            config_file=None, model_file=checkpoint_path, device=device
        )
        self.model = xeus_model
        
        if not hasattr(self.model, 'hidden_sizes'):
            if hasattr(self.model.encoder, 'encoders'):
                hidden_size = self.model.encoder.encoders[0].attn.linear_q.in_features
                num_layers = len(self.model.encoder.encoders)
                self.model.hidden_sizes = [hidden_size] * num_layers
            else:
                raise AttributeError("XEUS model missing 'encoders' attribute.")
        self.model.hidden_size = hidden_size
        self.model.num_layers = num_layers
        self.model.downsample_rates = [32] * num_layers

    def forward(self, speech: torch.FloatTensor, speech_lengths: torch.LongTensor):
        # Encode the speech inputs
        encoded_outputs = self.model.encode(
            speech=speech, speech_lengths=speech_lengths, use_mask=True, use_final_output=False
        )
        hidden_states = encoded_outputs[0]  # List of tensors, one per layer (B, T, D)

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = [hidden_states]

        # Dynamically compute encoded lengths for all 19 layers
        batch_size = speech_lengths.size(0)
        num_layers = len(hidden_states)
        encoded_lengths = [
            torch.tensor(
                [hidden_states[0].size(1)] * batch_size,  # Use the time dimension from the first layer
                device=speech_lengths.device
            )
            for _ in range(num_layers)
        ]

        # Return hidden states and encoded lengths for all layers
        return hidden_states, encoded_lengths

class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path, vocab_path, device, adalora_config=None, beam_width=10, beam_threshold=50.0):
        super().__init__()
        # Initialize XEUS encoder
        self.xeus = XEUSEncoder(checkpoint_path, device)
        self.hidden_size = self.xeus.model.hidden_size
        self.xeus.model.num_layers = 19
        self.weighted_sum = Featurizer(
            upstream=self.xeus.model,
            layer_selections=None,
            normalize=False
        )

        # Load vocab
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.num_classes = len(self.vocab)  # Include blank and silence tokens
        self.blank_idx = 0  # Assumes "<blank>" is mapped to 0 in vocab
        self.sil_token_idx = max(map(int, self.vocab.keys()))  # Assumes "<sil>" is the last token
        self.ignore_id = -32768  

        # CTC Loss
        self.ctc_loss = CTCLoss(blank=self.blank_idx, zero_infinity=False)
        self.aux_ctc = CTCLoss(blank=self.blank_idx, zero_infinity=False)

        # Linear classifier to project XEUS outputs to vocab size
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, 5) for _ in range(24) # 24 articulatory features
        ])  # 3 articulatory features + blank + sil

        # CTC Decoder
        self.CTCDecoder = ctc_decoder(
            lexicon=None,  # Lexicon-free decoding
            tokens=list(self.vocab.values()), 
            lm=None,  
            nbest=1,
            beam_size=beam_width,
            beam_threshold=beam_threshold,
            blank_token=self.vocab["0"], 
            sil_token=self.vocab[str(self.sil_token_idx)],
        )

        if adalora_config:
            self.xeus.model = get_peft_model(self.xeus.model, adalora_config)
            self.xeus.model.print_trainable_parameters()  

    def forward(self, speech, speech_lengths):
        all_hs, all_hs_lens = self.xeus(speech, speech_lengths)
        hs, hs_len = self.weighted_sum(all_hs, all_hs_lens)
        return hs, hs_len

    def compute_loss(self, hs, targets, input_lengths, target_lengths):
        logits = self.classifier(hs)
        # print(f"Debug: logits min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
        # print(f"Debug: speech shape={hs.shape}, speech_lengths={input_lengths}")
        # print(f"Debug: targets shape={targets.shape}, target_lengths={target_lengths}")

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)
        
        # Validate input lengths
        max_input_length = input_lengths.max().item()
        assert max_input_length <= log_probs.size(0), (
            f"Input length ({max_input_length}) exceeds log_probs time dimension ({log_probs.size(0)})."
        )

        mask = targets != self.ignore_id
        label_lengths = mask.sum(dim=1)  # Update label lengths based on valid labels
        targets = torch.where(mask, targets, self.blank_idx)

        # print(f"target: {targets.size()}, {targets}")
        # print(f"Debug: log_probs shape={log_probs.shape}, input_lengths={input_lengths}, label_lengths={label_lengths}")

        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, label_lengths)

        print(f"CTC loss: {loss.item()}")
        # Auxiliary articulatory losses (only during training)
        if self.training and getattr(args, 'articulatory_losses', False):
            # Initialize Panphon FeatureTable
            ft = FeatureTable()
            artic_feats = ft.names  # List of articulatory feature names (e.g., 24 features)

            # Create ground-truth articulatory features for the batch
            gt_features = []
            for target in targets:
                word_array = ft.word_array(
                    ft_names=artic_feats,
                    word=" ".join(self.vocab[str(idx.item())] for idx in target if idx.item() != self.blank_idx),
                )

                # Convert word_array to tensor and remap values (-1, 0, 1) -> (1, 2, 3)
                word_array = torch.tensor(word_array, dtype=torch.float32).T  # [num_feats, seq_len]
                word_array = (word_array + 2).clamp(min=1)  # Shift values: -1 -> 1, 0 -> 2, 1 -> 3

                # Explicitly handle blank tokens (set to 0)
                word_array = torch.where(word_array == self.blank_idx, torch.tensor(0, dtype=torch.float32), word_array)

                gt_features.append(word_array)

            # Pad ground-truth features
            max_seq_len = max(feat.size(1) for feat in gt_features)
            gt_features = torch.stack([
                torch.nn.functional.pad(feat, (0, max_seq_len - feat.size(1)), value=self.ignore_id)
                for feat in gt_features
            ])  # [B, num_feats, max_seq_len]

            # Transpose to match expected shape: [B, max_seq_len, num_feats]
            gt_features = gt_features.permute(0, 2, 1)  # [B, max_seq_len, num_feats]
            # print(f"Debug: gt_features shape={gt_features.shape}")  # [B, max_seq_len, num_feats]

            # Auxiliary loss computation for each feature
            aux_losses = []
            for i, feature_name in enumerate(artic_feats):
                aux_logits = self.aux_classifiers[i](hs)  # [B, T, 5]
                aux_log_probs = torch.nn.functional.log_softmax(aux_logits, dim=-1).transpose(0, 1)  # [T, B, 5]
                # print(f"Debug: aux_logits shape={aux_logits.shape}")

                # Extract ground truth for the i-th feature
                gt_feature_i = gt_features[:, :, i]  # [B, max_seq_len]
                # print(f"Debug: gt_feature_{feature_name} shape={gt_feature_i.shape}")

                # Compute auxiliary loss for the i-th feature
                aux_loss = self.aux_ctc(aux_log_probs, gt_feature_i, input_lengths, label_lengths)
                aux_losses.append(aux_loss)
                # print(f"Auxiliary loss for {feature_name}: {aux_loss.item()}")

            # Combine auxiliary losses
            total_aux_loss = sum(aux_losses)
            aux_loss_weight = getattr(args, 'aux_loss_weight', 1.0)  # Default weight is 1.0
            loss += total_aux_loss * aux_loss_weight

            # print(f"Debug: Total auxiliary loss={total_aux_loss.item()}, scaled={total_aux_loss.item() * aux_loss_weight}")

        return loss

    def greedy_decode(self, hs):
        logits = self.classifier(hs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        predictions = torch.clamp(torch.argmax(log_probs, dim=-1), min=0, max=len(self.vocab) - 1)

        # Convert predictions to text using vocab
        return [
            " ".join(self.vocab[str(idx.item())] for j, idx in enumerate(pred) if idx != self.blank_idx and (j == 0 or idx != pred[j-1])) for pred in predictions
            ]

    def beam_search_decode(self, hs):
        logits = self.classifier(hs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Ensure logits are on CPU
        if not logits.is_cpu:
            logits = logits.cpu()

        # Decode using the CTC decoder
        decoded = self.CTCDecoder(log_probs.cpu())
        # print(decoded)

        # Convert decoded tokens into text, including blanks and silence
        decoded_texts = []
        for sample in decoded:
            for hyp in sample:
                # print(hyp.tokens)
                text = " ".join(self.vocab[str(idx.item())] for idx in hyp.tokens)
                decoded_texts.append(text)

        return decoded_texts

def get_parser():
    parser = argparse.ArgumentParser(description="Phoneme recognition with AdaLoRA and BeamSearch")
    parser.add_argument("--use_adalora", action='store_true', help="Enable AdaLoRA for model fine-tuning")
    parser.add_argument("--articulatory_losses", action='store_true', help="Enable articulatory losses")
    parser.add_argument("--aux_loss_weight", default=1, required=False, help="aritculatory losses weight")
    parser.add_argument("--epochs", type=int, default=50, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, required=True, help="Initial learning rate")
    parser.add_argument("--max_audio_duration", type=float, default=20, required=True, help="Max audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=4, required=True, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("baseline_checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--beam_size_test", type=int, default=10, help="Beam size for testing")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with smaller dataset")

    # Vocabulary path argument
    parser.add_argument("--vocab_path", type=Path, default=Path("./vocab.json"), help="Path to the saved vocabulary JSON file")

    return parser


# Training step with mixed precision and gradient accumulation
def train_step(model, optimizer, scheduler, train_loader, device, epoch, accumulation_steps, scaler, writer, args, num_batches):
    model.train()
    total_loss = 0    
    print(f"Starting Epoch {epoch}")

    # Initialize tqdm with the total number of remaining batches
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", total=num_batches, dynamic_ncols=True)

    for i, batch in enumerate(progress_bar, start=1):
        _, data = batch
        speech = data["speech"].to(device, non_blocking=True, dtype=torch.float32)
        speech_lengths = data["speech_lengths"].to(device, non_blocking=True)
        text = data["text"].to(device, non_blocking=True)
        text_lengths = data["text_lengths"].to(device, non_blocking=True)

        with autocast("cuda"):
            logits, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(logits, text, input_lengths, text_lengths)
            loss = loss / accumulation_steps  # Normalize loss for accumulation
        
        scaler.scale(loss).backward()

        # Gradient accumulation
        if i % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            # Scheduler stepping is handled after epoch
            scheduler.step()

        total_loss += loss.item()

        # Save checkpoint every 1000 iterations
        if i % 1000 == 0:
            writer.add_scalar(f"Loss/Train", loss.item(), epoch * num_batches + i)
 
    avg_loss = total_loss / num_batches
    return avg_loss


# Validation step
def dev_step(model, dev_loader, device, int_to_char, num_batches):
    model.eval()
    total_loss = 0
    total_per_greedy = 0
    total_cer_greedy = 0
    total_per_beam = 0
    total_cer_beam = 0
    total_samples = 0
    num_batches = len(dev_loader)
    print(f"\nValidation... Total batches in dev_loader: {num_batches}")

    processed_samples = 0

    with torch.no_grad(), autocast("cuda"):
        for i, batch in enumerate(tqdm(dev_loader, desc="Validation", unit="batch", total=num_batches)):
            utt_ids, data = batch
            batch_size = data["speech"].size(0)
            processed_samples += batch_size

            speech = data["speech"].to(device, non_blocking=True, dtype=torch.float32)
            speech_lengths = data["speech_lengths"].to(device, non_blocking=True)
            text = data["text"].to(device, non_blocking=True)
            text_lengths = data["text_lengths"].to(device, non_blocking=True)

            # Forward pass
            feats, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(feats, text, input_lengths, text_lengths)
            total_loss += loss.item()

            # Decode logits using both methods
            decoded_sequences_greedy = model.greedy_decode(feats)
            decoded_sequences_beam = model.beam_search_decode(feats)

            # Calculate PER and CER
            for pred, target in zip(decoded_sequences_greedy, text.cpu().numpy()):
                target_text = " ".join([int_to_char.get(idx, "<UNK>") for idx in target if idx != model.ignore_id])
                total_per_greedy += jiwer.wer(target_text, pred)
                total_cer_greedy += jiwer.cer(target_text, pred)
                print(f"target:{target_text}")
                print(f"pred (greedy):{pred}")
                total_samples += 1

            for pred, target in zip(decoded_sequences_beam, text.cpu().numpy()):
                target_text = " ".join([int_to_char.get(idx, "<UNK>") for idx in target if idx != model.ignore_id])
                pred = pred.replace("|", "")
                print(f"target:{target_text}")
                print(f"pred (beam):{pred}")
                total_per_beam += jiwer.wer(target_text, pred)
                total_cer_beam += jiwer.cer(target_text, pred)

    print(f"Total samples processed in validation: {processed_samples}")
    print(f"Total samples in dev_dset: {len(dev_loader.dataset)}")

    avg_loss = total_loss / len(dev_loader)
    avg_per_greedy = total_per_greedy / total_samples
    avg_cer_greedy = total_cer_greedy / total_samples
    avg_per_beam = total_per_beam / total_samples
    avg_cer_beam = total_cer_beam / total_samples

    return avg_loss, avg_per_greedy, avg_cer_greedy, avg_per_beam, avg_cer_beam


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
            return 1, 0  # Default to starting fresh

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

        # Check and reinitialize classifier if output size doesn't match vocab size
        vocab_size = len(model.vocab)
        classifier_output_size = model.classifier.out_features
        if classifier_output_size != vocab_size:
            print(f"Classifier output size ({classifier_output_size}) does not match vocab size ({vocab_size}). Reinitializing classifier.")
            model.classifier = nn.Linear(model.hidden_size, vocab_size).to(device)

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

# Define a worker initialization function for reproducibility
def worker_init_fn(worker_id, random_seed):
    # Each worker has a different seed based on the base random_seed and worker_id
    seed = random_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)

# Main training loop
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'  # Update as needed

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load vocabulary from vocab.json
    if args.vocab_path and os.path.exists(args.vocab_path):
        with open(args.vocab_path, "r") as f:
            vocab = json.load(f)
        # Assuming vocab is a dictionary mapping string indices to phonemes
        int_to_char = {int(idx): char for idx, char in vocab.items()}
        vocab_size = len(vocab)
        vocab_list = list(vocab.values())
        print(f"Vocabulary loaded from {args.vocab_path} with size {len(vocab_list)}.")
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")

    # Initialize datasets
    train_dset = IPAPack(
        scp_file="data/train/wav.scp",
        text_file="data/train/text",
        utt2dur_file="train_utt2dur",
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,  # Load existing vocabulary
        debug=args.debug,
        debug_size=500
    )
    dev_dset = IPAPack(
        scp_file="data/dev/wav.scp",
        text_file="data/dev/text",
        utt2dur_file="dev_utt2dur",
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,  # Load existing vocabulary
        debug=args.debug,
        debug_size=10
    )

    num_subepochs = 3  # Define the number of subepochs per epoch
    train_partitions = partition_indices(train_dset, num_subepochs)  # Partition dataset into subepochs

    # Define AdaLoRA Configuration if enabled
    adalora_config = None
    if args.use_adalora:
        adalora_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["linear_q", "linear_k", "linear_v", "linear_out"]
        )

    # Initialize model
    model = FinetuneXEUSPhonemeCTC(
        checkpoint_path=checkpoint_path,
        device=device,
        vocab_path="./vocab.json",
        adalora_config=adalora_config,
        beam_width=args.beam_size_test,
        beam_threshold=50.0
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda")

    # Scheduler with warmup
    num_training_steps = math.ceil(len(train_dset) / (args.batch_size * args.accumulation_steps)) * args.epochs
    scheduler, warmup_steps = get_scheduler_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps)

    # TensorBoard writer
    adalora_state_str = "adalora" if args.use_adalora else "no_adalora"
    articulatory_state_str = "art" if args.articulatory_losses else "no_art"
    writer = SummaryWriter(log_dir=f"runs2/fine_tuning_xeus_phoneme_ctc_{adalora_state_str}_{args.max_audio_duration}_{args.batch_size}_{args.lr}_{articulatory_state_str}_{args.aux_loss_weight}")

    best_dev_loss = float("inf")  # Initialize best development loss

    # Load checkpoint if available
    try:
        start_epoch, start_subepoch = load_checkpoint(model, optimizer, scheduler, scaler, device, args)
    except Exception as e:
        print(f"Failed to load checkpoint due to error: {e}")
        start_epoch, start_subepoch = 1, 0  # Start fresh

    # Main training loop
 # Main training loop
for epoch in range(start_epoch, args.epochs + 1):
    print(f"\n=== Epoch: {epoch}/{args.epochs} ===")
    for subepoch, subset_indices in enumerate(train_partitions, start=1):
        if epoch == start_epoch and subepoch < start_subepoch:
            continue  # Skip completed subepochs if resuming

        print(f"\n--- Subepoch: {subepoch}/{num_subepochs} ---")

        # Prepare DataLoader for current subepoch
        train_subset = Subset(train_dset, subset_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            collate_fn=common_collate_fn,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, random_seed)
        )
        dev_loader = DataLoader(
            dev_dset,
            batch_size=512,
            collate_fn=common_collate_fn,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, random_seed)
        )

        # Training step
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            scaler=scaler,
            writer=writer,
            args=args,
            num_batches=len(train_loader),
        )
        
        # Log training loss and learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train_loss', train_loss, epoch + (subepoch / num_subepochs))
        writer.add_scalar('Learning_Rate', current_lr, epoch + (subepoch / num_subepochs))

        # Save checkpoint after training
        training_checkpoint_path = f"{args.checkpoint_dir}/model_checkpoint_training_epoch{epoch}_subepoch{subepoch}_{articulatory_state_str}.pth"
        torch.save({
            'epoch': epoch,
            'subepoch': subepoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'learning_rate': current_lr,
        }, training_checkpoint_path)
        print(f"Training checkpoint saved: {training_checkpoint_path}")

        # Validation step
        dev_loss, dev_per1, dev_cer1, dev_per2, dev_cer2 = dev_step(model, dev_loader, device, int_to_char, len(dev_loader))
        print(f"Dev Loss: {dev_loss:.4f}, PER (Greedy): {dev_per1:.4f}, CER (Greedy): {dev_cer1:.4f},  PER (beam): {dev_per2:.4f}, CER (beam): {dev_cer2:.4f}")

        # Log validation metrics
        writer.add_scalar('Loss/dev_loss', dev_loss, epoch + (subepoch / num_subepochs))
        writer.add_scalar('PER/avg_per_greedy', dev_per1, epoch + (subepoch / num_subepochs))
        writer.add_scalar('CER/avg_cer_greedy', dev_cer1, epoch + (subepoch / num_subepochs))
        writer.add_scalar('PER/avg_per_beam', dev_per2, epoch + (subepoch / num_subepochs))
        writer.add_scalar('CER/avg_cer_beam', dev_cer2, epoch + (subepoch / num_subepochs))

        # Save checkpoint after validation
        validation_checkpoint_path = f"{args.checkpoint_dir}/model_checkpoint_validation_epoch{epoch}_subepoch{subepoch}_{articulatory_state_str}.pth"
        torch.save({
            'epoch': epoch,
            'subepoch': subepoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'learning_rate': current_lr,
        }, validation_checkpoint_path)
        print(f"Validation checkpoint saved: {validation_checkpoint_path}")

        # Update best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model_path = f"{args.checkpoint_dir}/best_model_epoch{epoch}_subepoch{subepoch}_{articulatory_state_str}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Dev Loss: {best_dev_loss:.4f}")

writer.close()
print("Training completed.")
