from local.dataset import IPAPack
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.train.collate_fn import common_collate_fn
from espnet2.tasks.ssl import SSLTask

import torch.nn
from s3prl.nn import Featurizer
from panphon import FeatureTable


# TODO: if time, could make this into a Frontend (multilayer_feature)
class XEUSEncoder(nn.Module):
    # meant to be finetuned
    def __init__(self, checkpoint_path):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
            config_file=None,
            model_file=checkpoint_path,
            device=self.device
        )
        self.model = xeus_model

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        wavs = speech.to(self.device)
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


def train(artic_feats):
    # takes in XEUS features and adds single CTC head
    # TODO: how does ESPnet infer the vocab_size? from a token_list ?
    phoneme_ctc = CTC(odim=vocab_size, encoder_output_size=xeus_model_dim)
    phoneme_ctc_loss = ctc.loss_fn

    articulatory_ctc_losses = [CTC(odim=3, encoder_output_size=xeus_model_dim) for _ in artic_feats]
    pass


if __name__ == "__main__":
    checkpoint_path = '/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'
    model = FinetuneXEUSPhonemeCTC(checkpoint_path)
    
    # TODO: argparse
    train_dset = IPAPack(scp_file="data/train/feats.scp", text_file="data/train/text")
    # TODO: is the collcate_fn necessary if it's already an ESPnet dataset?
    # TODO: batch_bins ?
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=common_collate_fn)

    ft = FeatureTable()
    artic_feats = ft.names
    
    # TODO: train loop
    for utt_ids, batch in train_loader:
        # TODO: may need to do this? https://espnet.github.io/espnet/guide/espnet2/train/common_collate_fn.html

        # utt_ids: list of utt id's
        # batch: {"speech": ..., "speech_lengths": ..., "text": ..., "text_lengths": ...}
        pass

    model(**batch)
