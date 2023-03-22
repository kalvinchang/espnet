#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000  # data_prep.pl already downsampled
n_fft=1024
n_shift=256
win_length=null

opts=
if [ "${fs}" -eq 16000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_config=conf/train.yaml
inference_config=conf/decode.yaml

lang=zh_TW
train_set=train_$lang
valid_set=dev_$lang
test_sets="dev_$lang test_$lang"

./tts.sh \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type char \
    --cleaner none \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_xvector true \
    ${opts} "$@"
