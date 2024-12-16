import os
import json
import argparse
import logging

import torch
from torch import nn
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from transformers import Wav2Vec2Model
from transformers import get_linear_schedule_with_warmup

import jiwer
from dataset import IPAPack
from collate_fn import common_collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_learnable_parameters(model):
    """Calculate the total number of learnable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

class SSLWithTransformersModel(LightningModule):
    def __init__(self, ssl_model_name, num_classes, lr, vocab_path, warmup_fraction=0.05):
        super(SSLWithTransformersModel, self).__init__()
        self.save_hyperparameters()
        self.model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        for param in self.model.parameters():  # Freeze Wav2Vec2 layers
            param.requires_grad = False

        feature_dim = self.model.config.hidden_size
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim
        )
        self.transformers = nn.TransformerEncoder(transformer_layer, num_layers=12)
        self.fc = nn.Linear(feature_dim, num_classes)

        self.ctc_loss = CTCLoss(blank=0)  # Blank token index is 0
        self.lr = lr
        self.warmup_fraction = warmup_fraction  # Fraction of warmup steps

        # Load vocabulary
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.int_to_char = {v: k for k, v in self.vocab.items()}
        vocab_list = list(self.vocab.keys())

        self.ctc_decoder = ctc_decoder(
            lexicon=None,  
            tokens=vocab_list, 
            lm=None, 
            nbest=1,  
            beam_size=args.beam_size,  
            blank_token="[PAD]",
            sil_token=" ", 
            unk_word="[UNK]",
        )

    def forward(self, x):
        with torch.no_grad():
            ssl_outputs = self.model(x)
            features = ssl_outputs.last_hidden_state

        transformer_out = self.transformers(features)
        logits = self.fc(transformer_out)
        return logits

    def greedy_decode(self, log_probs):
        """Greedy decoding: Argmax followed by collapsing repeated indices and removing blanks."""
        # Predictions: (B, T) where each value is the index of the most probable class
        predictions = torch.argmax(log_probs, dim=-1)  # (B, T)

        decoded_texts = []
        for pred in predictions:
            decoded_text = []
            last_idx = None
            for idx in pred:
                idx = idx.item()
                if idx != 0 and idx != last_idx:  # Skip blank (index 0) and repeated tokens
                    decoded_text.append(self.int_to_char[idx])
                last_idx = idx
            decoded_texts.append(" ".join(decoded_text))
        return decoded_texts

    def beam_search_decode(self, log_probs):
        """Beam search decoding using CTCDecoder."""
        if log_probs.device.type != "cpu":
            log_probs = log_probs.cpu().contiguous()

        # Decode using the CTCDecoder
        beam_results = self.ctc_decoder(log_probs)

        decoded_texts = []
        for sample in beam_results:
            # Process n-best results for each sample
            for hyp in sample:
                decoded_text = " ".join(self.int_to_char[idx.item()] for idx in hyp.tokens if idx.item() != 0)
                decoded_texts.append(decoded_text)
        return decoded_texts

    def training_step(self, batch, batch_idx):
        # batch is (utt_ids, dict_of_tensors)
        utt_ids, inputs = batch
        speech = inputs["speech"]                # (B, T_speech)
        speech_lengths = inputs["speech_lengths"]# (B,)
        text = inputs["text"]                    # (B, T_label)
        text_lengths = inputs["text_lengths"]    # (B,)

        logits = self.forward(speech)
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
        output_lengths = torch.tensor([logits.size(1)] * logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.ctc_loss(log_probs, text, output_lengths, text_lengths)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        utt_ids, inputs = batch
        speech = inputs["speech"]                # (B, T)
        raw_speech_lengths = inputs["speech_lengths"]  # Actual sequence lengths (B,)
        text = inputs["text"]
        text_lengths = inputs["text_lengths"]

        # Forward pass
        logits = self.forward(speech)  # (B, T_out, C)
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
        output_lengths = torch.tensor([logits.size(1)] * logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.ctc_loss(log_probs, text, output_lengths, text_lengths)

        pred_texts_greedy = self.greedy_decode(log_probs)
        pred_texts_beam = self.beam_search_decode(log_probs)

        # Calculate PER and CER
        for pred, target in zip(pred_texts_greedy, text.cpu().numpy()):
            # print(target)
            target_text = " ".join([self.int_to_char.get(idx, "[UNK]") for idx in target if idx != -32768])
            per_greedy = jiwer.wer(target_text, pred)
            cer_greedy = jiwer.cer(target_text, pred)
            # print(f"target: {target_text}")
            # print(f"pred(greedy): {pred}")
            # print("")

        for pred, target in zip(pred_texts_beam, text.cpu().numpy()):
            target_text = " ".join([self.int_to_char.get(idx, "[UNK]") for idx in target if idx != -32768])
            pred = pred.replace("|", "")
            # print(f"target: {target_text}")
            # print(f"pred(beam): {pred}")
            per_beam = jiwer.wer(target_text, pred)
            cer_beam = jiwer.cer(target_text, pred)
            # print("")

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_per_greedy", per_greedy, prog_bar=True, logger=True)
        self.log("val_per_beam", per_beam, prog_bar=True, logger=True)
        self.log("val_cer_greedy", cer_greedy, prog_bar=True, logger=True)
        self.log("val_cer_beam", cer_beam, prog_bar=True, logger=True)
        print(f"val_loss: {loss}, PER (greedy): {per_greedy}, PER (beam): {per_beam}")
        return {"val_loss": loss, "val_wer_greedy": per_greedy, "val_wer_beam": per_beam, "val_cer_greedy": cer_greedy, "val_cer_beam": cer_beam}


    def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=self.lr)
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(self.warmup_fraction * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            print(warmup_steps)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True,  default="/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--ssl_model_name", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--max_audio_duration", type=float, default=15.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_size", type=int, default=100)
    args = parser.parse_args()

    seed_everything(42)

    train_dataset = IPAPack(
        scp_file=os.path.join(args.base_path, "data/train_wav.scp"),
        text_file=os.path.join(args.base_path, "data/train_text"),
        utt2dur_file=os.path.join(args.base_path, "data/train_utt2dur"),
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,
        debug=args.debug,
        debug_size=args.debug_size,
    )
    dev_dataset = IPAPack(
        scp_file=os.path.join(args.base_path, "data/dev_wav.scp"),
        text_file=os.path.join(args.base_path, "data/dev_text"),
        utt2dur_file=os.path.join(args.base_path, "data/dev_utt2dur"),
        max_audio_duration=args.max_audio_duration,
        vocab_path=args.vocab_path,
        debug=args.debug,
        debug_size=args.debug_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=common_collate_fn,
    )
    val_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=common_collate_fn,
    )

    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    num_class = len(list(vocab.keys()))

    # Model
    model = SSLWithTransformersModel(
        ssl_model_name=args.ssl_model_name,
        num_classes=num_class,
        lr=args.lr,
        vocab_path=args.vocab_path,
        warmup_fraction=0.05 
    )

    learnable_params = count_learnable_parameters(model)
    print(f"Total Learnable Parameters: {learnable_params}")

    # Logging and Checkpoints
    logger = TensorBoardLogger("logs", name="fine_tune_xlsr_final_layer")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=3, mode="min", filename="checkpoint-{epoch:02d}-{val_loss:.2f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accumulate_grad_batches=args.accumulation_steps,
        devices=1,
        accelerator="gpu",
    )

    trainer.fit(model, train_loader, val_loader)