#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test_fleurs test_doreco"
unseen_test_sets="test_doreco"

encoder=allophant
asr_config=conf/tuning/train_asr_${encoder}_shared_panphon.yaml
inference_config=conf/decode_allophant.yaml
aux_ctc="syl son cons cont delrel lat nas strid voi sg cg ant cor distr lab hi lo back round velaric tense"


./asr.sh \
    --asr_tag "debug_shared_2" \
    --auxiliary_data_tags "${aux_ctc}" \
    --post_process_local_data_opts "--stage 2 --unseen_test_sets \"${unseen_test_sets}\"" \
    --stage 11 \
    --stop_stage 11 \
    --ngpu 2 \
    --nj 2 \
    --gpu_inference true \
    --inference_nj 4 \
    --token_type word \
    --max_wav_duration 30 \
    --audio_format flac.ark \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"
