#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test"

encoder=transformer
frontend=xeus_frozen
asr_config=conf/tuning/train_asr_${frontend}_${encoder}.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --stage 11 \
    --ngpu 4 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --token_type word \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type extracted \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"
