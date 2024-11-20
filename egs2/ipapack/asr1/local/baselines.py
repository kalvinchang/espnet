import glob
import os
import argparse
from panphon import FeatureTable
from pathlib import Path
from dataset import IPAPack
from espnet.nets.beam_search import BeamSearch
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.asr.ctc import CTC
from espnet.nets.scorers.ctc import CTCPrefixScorer
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
        xeus_model, _ = SSLTask.build_model_from_file(
            config_file=None, model_file=checkpoint_path, device=device)
        self.model = xeus_model
        self.model.frontend = None
        self.model.preencoder = None
        if not hasattr(self.model, 'hidden_sizes'):
            self.model.hidden_sizes = [self.model.encoder.output_size]
        if not hasattr(self.model, 'downsample_rates'):
            self.model.downsample_rates = [1] ## no downsampling

    def forward(self, speech, speech_lengths):
        feats = self.model.encode(
            speech=speech,
            speech_lengths=speech_lengths,
            use_mask=False,
            use_final_output=False,
        )[0]
        return feats


# Define Finetuning Model
class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path, device, vocab_size, lora_config=None, beam_size=1):
        super().__init__()
        self.xeus = XEUSEncoder(checkpoint_path, device)
        self.xeus.model.num_layers = 19
        self.weighted_sum = Featurizer(upstream=self.xeus.model)
        self.ctc = CTC(odim=vocab_size + 1, encoder_output_size=1024)
        self.aux_ctc = CTC(odim=3, encoder_output_size=1024) # (+, 0, -)
        if lora_config:
            # Apply LoRA to the XEUS model
            self.xeus.model = get_peft_model(self.xeus.model, lora_config)
            print("LoRA applied to the XEUS model.")

        self.sos = self.eos = vocab_size + 1
        self.ignore_id = -32768
        self.beam_size = beam_size
        self.beam_search = BeamSearch(
            scorers={"ctc": CTCPrefixScorer(self.ctc, eos=self.eos)},
            weights={"ctc": 1.0},
            beam_size=beam_size,
            vocab_size=vocab_size + 1,
            sos=self.sos,
            eos=self.eos,
            return_hs=False,
            normalize_length=False,
        )

    def forward(self, speech, speech_lengths):
        feats = self.xeus(speech, speech_lengths)
        feats, hs_len = self.weighted_sum(feats, [speech_lengths] * len(feats))
        # feats = self.encoder(weighted_feats, hs_len)[0]
        return feats, hs_len

    def compute_loss(self, feats, labels, feat_lengths, label_lengths):
        loss = self.ctc(feats, feat_lengths, labels, label_lengths)
        if args.articulatory_losses is True:
            ft = FeatureTable()
            artic_feats = ft.names
            aux_losses = [self.aux_ctc for _ in artic_feats]
            gt = []
            for label in labels: gt.append(ft.word_array(ft_names=ft.names, word=label) +1) # (-1,0,1)->(0,1,2)
            gt = np.array(gt).transpose(2,0,1) # [B, seq, artic_feat] -> [artic_feat, B, seq]
            for i, aux_loss in enumerate(aux_losses):
                al = aux_loss(feats, feat_lengths, gt[i], label_lengths)
                loss += 0.05 * al # so that all of them add up to ~1?
        return loss

    def decode(self, feats, feat_lengths):
        seqs = []
        for feat, feat_length in zip(feats, feat_lengths):
            best_hyp = self.beam_search(feat[:feat_length])[0]
            seq = best_hyp.yseq.tolist()
            seq = list(filter(lambda x: x not in (0, self.eos, self.sos), seq))
            seqs.append(seq)
        return seqs

def get_parser():
    parser = argparse.ArgumentParser(description="Phoneme recognition baselines with LoRA")
    parser.add_argument("--articulatory_losses", default=False, required=False)
    parser.add_argument("--train_dir", type=Path, default=Path("data/train"), required=True, help="Training data directory")
    parser.add_argument("--dev_dir", type=Path, default=Path("data/dev"), required=True, help="Development data directory")
    parser.add_argument("--epochs", type=int, default=50, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, required=True, help="Learning rate")
    parser.add_argument("--max_audio_duration", type=float, default=20, required=True, help="Max audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=4, required=True, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("baseline_checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--accumulation_steps", type=float, default=4, help="accumulation steps")
    parser.add_argument("--beam_size_test", type=int, default=10, help="beam size for testing")
    parser.add_argument("--debug", action='store_true', help="debug mode")
    return parser

# Training step with mixed precision and gradient accumulation
def train_step(model, optimizer, train_loader, articulatory_ctc, device, epoch, iteration, accumulation_steps):
    model.train()
    total_loss = 0
    checkpoint_files = []
    batches_per_epoch = len(train_loader)

    print(f"Resuming training from Epoch {epoch}, Iteration {iteration}")
    for i, batch in enumerate(tqdm(train_loader, initial=iteration, total=len(train_loader))):
        iteration += 1  
        _, data = batch
        speech = data["speech"].to(device, non_blocking=True, dtype=torch.float32)
        speech_lengths = data["speech_lengths"].to(device, non_blocking=True)
        text = data["text"].to(device, non_blocking=True)
        text_lengths = data["text_lengths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            feats, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(feats, text, input_lengths, text_lengths)
            loss = loss / accumulation_steps

        # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
        # backward() before step() for each loss
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() 

        # Save checkpoint every 1000 iterations
        if iteration % 1000 == 0: 
            checkpoint_path = (
                f"{args.checkpoint_dir}/model_checkpoint_{args.max_audio_duration}_"
                f"{args.batch_size}_{args.lr}_epoch{epoch}_iter{iteration}.pth"
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

            writer.add_scalar("Loss (iter)", loss.item(), iteration)

            # Remove old checkpoints if limit is exceeded
            if len(checkpoint_files) > 2:
                oldest_checkpoint = checkpoint_files.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)

    avg_loss = total_loss / len(train_loader)
    return avg_loss, iteration  # Return the updated iteration count


def dev_step(model, dev_loader, device):
    model.eval()
    total_loss = 0
    total_wer, total_per = 0, 0
    total_samples = 0
    padding_value = -32768 ## based on common_collate_fn
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            utt_ids, data = batch
            speech = data["speech"].to(device)
            speech_lengths = data["speech_lengths"].to(device)
            text = data["text"].to(device)
            text_lengths = data["text_lengths"].to(device)

            feats, input_lengths = model(speech, speech_lengths)
            loss = model.compute_loss(feats, text, input_lengths, text_lengths)
            total_loss += loss.item()

            # Decode the feats to get predicted text sequences
            predicted_sequences = model.decode(feats, input_lengths)

            # Convert target sequences to text for WER calculation
            target_sequences = []
            for idx_seq in text.cpu().numpy():
                seq = [train_loader.dataset.int_to_char[idx] for idx in idx_seq if idx != -32768]
                target_sequences.append(" ".join(seq))

            # Calculate WER for each prediction
            for pred, target in zip(predicted_sequences, target_sequences):
                pred_text = " ".join([train_loader.dataset.int_to_char[idx] for idx in pred])
                total_wer += jiwer.wer(target, pred_text)
                total_per += jiwer.cer(target, pred_text)
                total_samples += 1

            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dev_loader)
    avg_wer = total_wer / total_samples
    avg_per = total_per / total_samples
    return avg_loss, avg_wer, avg_per


def load_checkpoint(model, optimizer, device):
    checkpoint_files = glob.glob(f"{args.checkpoint_dir}/model_checkpoint_{args.max_audio_duration}_{args.batch_size}_{args.lr}_epoch*_iter*.pth")
    if not checkpoint_files:
        print("No checkpoints found. Starting training from scratch.")
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
    test_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text", utt2dur_file="test_utt2dur", max_audio_duration=args.max_audio_duration)

    if args.debug:
        train_dset.set_debug()
        dev_dset.set_debug(size=10)
        test_dset.set_debug(size=10)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=common_collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dset, batch_size=args.batch_size, collate_fn=common_collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, collate_fn=common_collate_fn, num_workers=4)

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
    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size, lora_config=lora_config, beam_size=1).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = GradScaler("cuda")
    writer = SummaryWriter(log_dir=f"runs/fine_tuning_xeus_phoneme_ctc_{args.max_audio_duration}_{args.batch_size}_{args.lr}")

    start_epoch, start_iteration = load_checkpoint(model, optimizer, device)
    best_dev_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss, start_iteration = train_step(model, optimizer, train_loader, args.articulatory_losses, device, epoch, start_iteration, args.accumulation_steps)
        dev_loss, dev_wer = dev_step(model, dev_loader, device)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Dev", dev_loss, epoch)
        writer.add_scalar("WER/Dev", dev_wer, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), f"{args.checkpoint_dir}/model_checkpoint_{args.max_audio_duration}_"
                f"{args.batch_size}_{args.lr}_best_model.pth")
            print(f"New best model saved with Dev Loss: {best_dev_loss:.4f}")

    ### Evaluate ## 
    best_model_path = f"{args.checkpoint_dir}/model_checkpoint_{args.max_audio_duration}_{args.batch_size}_{args.lr}_best_model.pth"
    model = FinetuneXEUSPhonemeCTC(checkpoint_path=checkpoint_path, device=device, vocab_size=vocab_size, beam_size=args.beam_size_test).to(device)
    model.load_state_dict(torch.load(best_model_path))

    _, test_wer, test_per = dev_step(model, test_loader, device)
    print(f"Test WER: {test_wer:.4f}; Test PER: {test_per:.4f}")
    writer.add_scalar("WER/Test", test_wer, epoch)
    writer.add_scalar("PER/Test", test_per, epoch)
    writer.close()
