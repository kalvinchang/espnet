from pathlib import Path

from local.dataset import IPAPack
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.train.collate_fn import common_collate_fn
from espnet2.tasks.ssl import SSLTask

import torch.nn
from s3prl.nn import Featurizer
from panphon import FeatureTable
from tqdm import tqdm


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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        wavs = speech
        # source: https://www.wavlab.org/activities/2024/xeus/
        # (xs_pad, intermediate_outs), olens, None
        # we recommend use_mask=True during fine-tuning
        # TODO: double check the list format
        final_feats, feats = self.model.encode(wavs, speech_lengths, use_mask=True, use_final_output=False)[0]
        # ex: List of [batch, frames, model_dim] tensors

        # based on https://github.com/pytorch/audio/blob/ba696ea3dfec4cbe693bf06a84c75dc196077f5b/src/torchaudio/models/wav2vec2/model.py#L85
            # just return the length of the original data
            # the # frames of each item pre-padding
        return feats, speech_lengths


class FinetuneXEUSPhonemeCTC(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.xeus = XEUSEncoder(checkpoint_path)
        # TODO: load from file
        self.xeus.model.num_layers = 19
        # TODO: may need to cast model into S3PRLUpstream
        self.weighted_sum = Featurizer(upstream=self.xeus.model)
        # don't downsample to preserve as much temporal resolution as possible
        self.encoder = LinearEncoder(input_size=1024, output_size=256, dropout_rate=0.1, input_layer="linear", normalize_before: bool = True)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
    ):
        return self.encoder(self.weighted_sum(self.xeus(speech, speech_lengths)))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Phoneme recognition baselines"
    )
    parser.add_argument(
        "--articulatory_losses",
        type=bool,
        default=False,
        required=True
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
    return parser


def dev_step(model, optimizer, dev_loader, phoneme_ctc):
    model.eval()
    total_loss = 0
    for i, (utt_ids, batch) in tqdm(enumerate(dev_loader)):
        speech_lens = batch["speech_lengths"]
        phonemes, phoneme_len = batch["text"], batch["text_lengths"]

        logits = model(**batch)
        loss = phoneme_ctc(logits, phonemes, speech_lens, phoneme_len)
        # we don't care about articulatory loss as a metric b/c it's just an auxiliary loss
        total_loss += loss.item()
        num_updates += 1

    return total_loss / num_updates


def train_step(model, optimizer, train_loader, phoneme_ctc, articulatory_ctc=None):
    model.train()
    total_loss, num_updates = 0, 0
    for i, (utt_ids, batch) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        # TODO: do the .to(device) here?

        # TODO: may need to do this? https://espnet.github.io/espnet/guide/espnet2/train/common_collate_fn.html
        # utt_ids: list of utt id's
        # batch: {"speech": ..., "speech_lengths": ..., "text": ..., "text_lengths": ...}
        logits = model(**batch)

        speech_lens = batch["speech_lengths"]
        phonemes, phoneme_len = batch["text"], batch["text_lengths"]
        # TODO: should the model return speech_lens?
        # ESPnet does the log_softmax for you
        loss = phoneme_ctc(logits, phonemes, speech_lens, phoneme_len)
        if articulatory_ctc:
            for aux_loss in articulatory_ctc:
                loss += aux_loss(logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_updates += 1

        if i % 1000 === 999:
            # TODO: plot the loss
            pass

    return total_loss / num_updates


def train(model, optimizer, train_loader, dev_loader, phoneme_ctc, articulatory_ctc=None, artic_feats, epochs, device):
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, phoneme_ctc, articulatory_ctc, device)
        # TODO: save the best model(?)
        # TODO: model averaging?
        # TODO: learning rate scheduler
        dev_loss = dev_step(model, dev_loader, phoneme_ctc, device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'
    model = FinetuneXEUSPhonemeCTC(checkpoint_path, device)

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    train_dset = IPAPack(scp_file=f"{args.train_dir}/feats.scp", text_file=f"{args.train_dir}/text")
    # TODO: is the collate_fn necessary if it's already an ESPnet dataset?
    # TODO: batch_bins ?
    train_loader = DataLoader(train_dset, batch_size=4, collate_fn=common_collate_fn)
    dev_dset = IPAPack(scp_file=f"{args.dev_dir}/feats.scp", text_file=f"{args.dev_dir}/text")
    dev_loader = DataLoader(dev_dset, batch_size=4, collate_fn=common_collate_fn)

    # TODO: how does ESPnet infer the vocab_size? from a token_list ?
    phoneme_ctc = CTC(odim=vocab_size, encoder_output_size=xeus_model_dim)
    phoneme_ctc_loss = ctc.loss_fn

    articulatory_ctc_losses = None
    if args.articulatory_losses:
        articulatory_ctc_losses = [CTC(odim=3, encoder_output_size=xeus_model_dim).loss_fn for _ in artic_feats]

    ft = FeatureTable()
    artic_feats = ft.names

    # TODO: take in beta from config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    train(model, optimizer, train_loader, dev_loader, phoneme_ctc_loss, articulatory_ctc_losses, artic_feats, args.epochs, device)
