from torch.nn.utils.rnn import pad_sequence
from espnet2.tasks.ssl import SSLTask
import soundfile as sf
import torch


def create_dataset():
    # ESPnet-EZ format
    pass



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    wavs, sampling_rate = sf.read('downloads/fleurs/af_za-train/1_8887709998161918393.wav') # sampling rate should be 16000
    wav_lengths = torch.LongTensor([len(wav) for wav in [wavs]]).to(device)
    # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:274.)
    wavs = pad_sequence(torch.Tensor([wavs]), batch_first=True).to(device) 

    # TODO: load IPAPack (dataloader)

    
    xeus_model, xeus_train_args = SSLTask.build_model_from_file(
        config_file=None,
        model_file='/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth',
        device=device
    )

    # we recommend use_mask=True during fine-tuning
    # take the output of the last layer -> batch_size x seq_len x hdim
    feats = xeus_model.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0][-1]
    # ex: [1, 1097, 1024] for 20 s file


# TODO: extract articulatory features from panphon

# TODO: finetune XEUS

