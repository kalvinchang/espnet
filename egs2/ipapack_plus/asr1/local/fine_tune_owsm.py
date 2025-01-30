import os
from glob import glob
import json
import csv

import numpy as np
import librosa
import kaldiio

import torch
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter
import espnetez as ez

# Define hyper parameters
CSV_DIR = f"./csv"
EXP_DIR = f"./exp/finetune"
STATS_DIR = f"./exp/stats_finetune"

FINETUNE_MODEL = "espnet/owsm_v3"
# 
LORA_TARGET = [
    "w_1", "w_2", "merge_proj"
]

pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    beam_size=10,
    device="cuda"
) # Load model to extract configs.

tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter

with open("vocab.json", "r") as f:
    vocab = json.load(f)
    special_tokens = vocab
    special_tokens.append("<pr>")

print(special_tokens)

tokenizer, converter, _ = ez.preprocess.add_special_tokens(
    tokenizer, converter, pretrained_model.s2t_model.decoder.embed[0],
    special_tokens=special_tokens
)

pretrain_config = vars(pretrained_model.s2t_train_args)
del pretrained_model

# For the configuration, please refer to the last cell in this notebook.
finetune_config = ez.config.update_finetune_config(
	's2t',
	pretrain_config,
	f"config/finetune.yaml"
)

# define model loading function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False

def build_model_fn(args):
    pretrained_model = Speech2Text.from_pretrained(
        FINETUNE_MODEL,
        beam_size=10,
    )
    model = pretrained_model.s2t_model
    model.train()
    print(f'Trainable parameters: {count_parameters(model)}')
    freeze_parameters(model)

    # apply lora
    create_lora_adapter(model, target_modules=LORA_TARGET)
    print(f'Trainable parameters after LORA: {count_parameters(model)}')
    return model

def parse_csv(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)
    return data_list

# custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        # data_list is a list of tuples (audio_path, text, text_ctc)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._parse_single_data(self.data[idx])

    def _parse_single_data(self, d):
        return {
            "audio_path": d["speech"],
            "text": d["text"],
            "text_prev": "<na>",
            "text_ctc": d['text_ctc'],
        }

train_data_list = parse_csv("csv/train.csv")
dev_data_list = parse_csv("csv/dev.csv")


train_dataset = CustomDataset(train_data_list)
valid_dataset = CustomDataset(dev_data_list)

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

# The output of CustomDatasetInstance[idx] will converted to np.array
# with the functions defined in the data_info dictionary.
data_info = {
    "speech": lambda d: kaldiio.load_mat(d["audio_path"])[1].astype(np.float32),
    "text": lambda d: tokenize(d["text"]),
    "text_prev": lambda d: tokenize(d["text_prev"]),
    "text_ctc": lambda d: tokenize(d["text_ctc"]),
}

# Convert into ESPnet-EZ dataset format
train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)

trainer = ez.Trainer(
    task='s2t',
    train_config=finetune_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    build_model_fn=build_model_fn, # provide the pre-trained model
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1
)
trainer.collect_stats()
trainer.train()