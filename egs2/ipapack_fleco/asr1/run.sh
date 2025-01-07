#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test"
# TODO: rename to test_doreco

encoder=transformer
frontend=
if [[ -n "$frontend" ]]; then
    asr_config=conf/tuning/train_asr_${frontend}_${encoder}.yaml
    feats_type=extracted
else
    # TODO: add the config for XEUS (conv2d1 or conv2d2)
    asr_config=conf/tuning/train_asr_${encoder}.yaml
    feats_type=raw
    audio_format=flac.ark
fi
inference_config=conf/decode_asr.yaml

# 99.5% of data is <= 20 sec long
max_wav_duration=20

# TODO: remove stop_stage
./asr.sh \
    --stage 10 \
    --stop_stage 11 \
    --ngpu 4 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --token_type word \
    --max_wav_duration $max_wav_duration \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type "${feats_type}" \
    --audio_format "${audio_format}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"
