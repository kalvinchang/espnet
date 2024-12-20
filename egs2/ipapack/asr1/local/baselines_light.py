import os
import json
import argparse
import logging
import torch

from torch import nn
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torchmetrics.text import WordErrorRate, CharErrorRate
from torchmetrics.aggregation import MeanMetric

from transformers import AutoModel, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import IPAPack


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_learnable_parameters(model):
    """Calculate the total number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_vocab(vocab_path):
    """Load vocabulary mapping from a JSON file."""
    with open(vocab_path, "r") as vocab_file:
        return json.load(vocab_file)

def collate_fn(batch, ssl_model_name):
    """Collate function for DataLoader to process batch data."""
    utt_ids, speech_list, text_list = [], [], []
    for utt_id, sample in batch:
        utt_ids.append(utt_id)
        speech_list.append(sample["speech"])
        text_list.append(sample["text"])

    processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)
    input_values = processor(speech_list, return_tensors="pt", sampling_rate=16000, padding=True, return_attention_mask=True)

    padded_text = pad_sequence(text_list, batch_first=True, padding_value=-100)
    text_lengths = torch.tensor([len(t) for t in text_list], dtype=torch.long)

    return utt_ids, {
        "input_values": input_values.input_values.squeeze(0),
        "input_lengths": input_values.attention_mask.sum(dim=1),
        "target": padded_text,
        "target_lengths": text_lengths,
    }

class SSLWithTransformersModel(LightningModule):
    def __init__(
            self,
            ssl_model_name: str,
            num_layers: int,
            lr: float,
            weight_decay: float,
            vocab_path: str,
            warmup_fraction: float,
            attach_tf: bool,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = load_vocab(vocab_path)
        self.int_to_char = {int(idx): char for char, idx in self.vocab.items()}
        self.model = AutoModel.from_pretrained(ssl_model_name)

        self.blank = 0
        assert self.blank not in self.vocab.keys()

        feature_dim = self.model.config.hidden_size
        self.prediction_head = nn.Linear(feature_dim, len(self.vocab))

        if attach_tf:
            for param in self.model.parameters():
                param.requires_grad = False
            self.transformers = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=feature_dim, nhead=16, dim_feedforward=4096), ## same with xls-r-300m
                num_layers=num_layers)
        else:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False  
            for param in self.model.encoder.parameters():
                param.requires_grad = True  
            self.transformers = lambda x: x

        self.lr = lr
        self.wd = weight_decay
        self.warmup_fraction = warmup_fraction

        self.per = WordErrorRate()
        self.cer = CharErrorRate()
        self.loss_acc = MeanMetric()

    def forward(self, x):
        ssl_outputs = self.model(x).last_hidden_state
        return self.prediction_head(self.transformers(ssl_outputs))

    def calculate_loss(self, input_values, speech_lengths, text, text_lengths):
        logits = self(input_values)
        log_probs = log_softmax(logits, dim=-1).permute(1, 0, 2)
        input_lengths = self.model._get_feat_extract_output_lengths(speech_lengths)
        input_lengths = input_lengths.to(torch.long)

        loss = ctc_loss(
            log_probs, text, input_lengths, text_lengths,
            reduction="none", blank=self.blank,
        ) / text_lengths

        return log_probs, input_lengths, loss

    def training_step(self, batch):
        utt_id, inputs = batch
        _, _, loss = self.calculate_loss(
            inputs["input_values"], inputs["input_lengths"],
            inputs["target"], inputs["target_lengths"]
        )
        mean_loss = loss.mean()

        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
            print(f"Skipping loss for utt_id: {utt_id}")
            return None
        
        self.log("train_loss", mean_loss, on_step=True, prog_bar=True)
        return mean_loss

    def _no_decode(self, tokens: torch.LongTensor):
        return " ".join([
            self.int_to_char[token.item()] if token.item() != self.blank else "_" 
            for token in tokens
        ])

    def _greedy_decode(self, tokens: torch.LongTensor):
        tokens = torch.unique_consecutive(tokens)
        return " ".join([
            self.int_to_char[token.item()]
            for token in tokens if token.item() != self.blank
        ])

    def decode(self, indices, lengths, method="greedy"):
        _decode = {"greedy": self._greedy_decode, "no_decode": self._no_decode}[method]
        return [
            _decode(tokens[:length])
            for tokens, length in zip(indices, lengths)
        ]

    def validation_step(self, batch):
        _, inputs = batch
        log_probs, log_prob_length, loss = self.calculate_loss(
            inputs["input_values"], inputs["input_lengths"],
            inputs["target"], inputs["target_lengths"]
        )

        # log_probs: (T, B, C), log_prob_length: (B, )
        pred_ids = log_probs.argmax(dim=-1).permute(1, 0).detach().cpu()
        pred_texts = self.decode(pred_ids, log_prob_length)
        target_texts = self.decode(inputs["target"], inputs["target_lengths"])

        pred_texts_cer = [text.replace(" ", "") for text in pred_texts]
        target_texts_cer = [text.replace(" ", "") for text in target_texts]

        self.per(pred_texts, target_texts)
        self.cer(pred_texts_cer, target_texts_cer)
        self.log("val_per", self.per, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_cer", self.cer, prog_bar=True, on_step=False, on_epoch=True)
        self.loss_acc(loss.detach())
        self.log("val_loss", self.loss_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch):
        utt_id, inputs = batch
        log_probs, log_prob_length, loss = self.calculate_loss(
            inputs["input_values"], inputs["input_lengths"],
            inputs["target"], inputs["target_lengths"]
        )

        pred_ids = log_probs.argmax(dim=-1).permute(1, 0).detach().cpu()
        pred_texts = self.decode(pred_ids, log_prob_length)
        target_texts = self.decode(inputs["target"], inputs["target_lengths"])

        pred_texts_cer = [text.replace(" ", "") for text in pred_texts]
        target_texts_cer = [text.replace(" ", "") for text in target_texts]

        self.per(pred_texts, target_texts)
        self.cer(pred_texts_cer, target_texts_cer)
        self.log("test_per", self.per, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_cer", self.cer, prog_bar=True, on_step=False, on_epoch=True)

        with open(f"test_output.txt", "a") as file:
            for idx, utt in enumerate(utt_id):
                # Calculate individual CER and PER per utterance
                per_score = self.per(pred_texts[idx], target_texts[idx])
                cer_score = self.cer(pred_texts_cer[idx], target_texts_cer[idx])

                output_line = (
                    f"utt_id: {utt} | target: {target_texts[idx]}\n"
                    f"utt_id: {utt} | pred: {pred_texts[idx]}\n" 
                    f"utt_id: {utt} | PER: {per_score:.4f} | CER: {cer_score:.4f}\n"
                )
                print(output_line.strip())  # Print to console
                file.write(output_line)     # Write to file
            file.write("\n")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        warmup_steps = int(self.warmup_fraction * self.trainer.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--ssl_model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_fraction", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--attach_tf", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_audio_duration", type=float, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_size", type=int)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    print(args)

    seed_everything(42)

    if not args.inference:
        train_dataset = IPAPack(
            data_path = args.base_path,
            scp_file=os.path.join(args.base_path, "data/train_filtered_wav.scp"),
            text_file=os.path.join(args.base_path, "data/train_filtered_text"),
            utt2dur_file=os.path.join(args.base_path, "data/train_filtered_utt2dur"),
            max_audio_duration=args.max_audio_duration,
            vocab_path=args.vocab_path,
            # debug=args.debug,
            # debug_size=args.debug_size,
        )
        dev_dataset = IPAPack(
            data_path = args.base_path,
            scp_file=os.path.join(args.base_path, "data/dev_filtered_wav.scp"),
            text_file=os.path.join(args.base_path, "data/dev_filtered_text"),
            utt2dur_file=os.path.join(args.base_path, "data/dev_filtered_utt2dur"),
            max_audio_duration=args.max_audio_duration,
            vocab_path=args.vocab_path,
            # debug=args.debug,
            # debug_size=args.debug_size,
        )
        test_dataset = IPAPack(
            data_path = args.base_path,
            scp_file=os.path.join(args.base_path, "data/test_wav.scp"),
            text_file=os.path.join(args.base_path, "data/test_text"),
            utt2dur_file=os.path.join(args.base_path, "data/test_utt2dur"),
            max_audio_duration=args.max_audio_duration,
            vocab_path=args.vocab_path,
            # debug=args.debug,
            # debug_size=args.debug_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, args.ssl_model_name)
        )
        val_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, args.ssl_model_name)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, args.ssl_model_name)
        )

    model = SSLWithTransformersModel(args.ssl_model_name, args.num_layers, args.lr, args.weight_decay, args.vocab_path, args.warmup_fraction, args.attach_tf)
    logger = TensorBoardLogger("logs", name=f"{args.ssl_model_name.split('/')[-1]}_{args.num_layers}")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, save_on_train_epoch_end=True, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    trainer = Trainer(
        max_epochs=args.max_epochs, logger=logger, callbacks=[checkpoint_callback, early_stopping, lr_monitor], devices=1, accelerator="gpu",
        accumulate_grad_batches=args.accumulation_steps, gradient_clip_val=1.0
    )

    print(f"Total Learnable Parameters: {count_learnable_parameters(model)}")
    
    if not args.inference:
        # trainer.fit(model, train_loader, val_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
        trainer.test(model=model, dataloaders=test_loader, ckpt_path = checkpoint_callback.best_model_path)

    if args.inference:
        test_dataset = IPAPack(
            data_path = args.base_path,
            scp_file=os.path.join(args.base_path, "data/test_wav.scp"),
            text_file=os.path.join(args.base_path, "data/test_text"),
            utt2dur_file=os.path.join(args.base_path, "data/test_utt2dur"),
            max_audio_duration=args.max_audio_duration,
            vocab_path=args.vocab_path,
            debug=args.debug,
            debug_size=args.debug_size,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, args.ssl_model_name)
        )
        
        trainer.test(model=model, dataloaders=test_loader, ckpt_path = args.checkpoint)

if __name__ == "__main__":
    main()