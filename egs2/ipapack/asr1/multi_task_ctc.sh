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
frontend=xeus
asr_config=conf/tuning/train_asr_${frontend}_multitask_${encoder}.yaml
inference_config=conf/decode_asr.yaml
feats_type=extracted
aux_ctc="syl son cons cont delrel lat nas strid voi sg cg ant cor distr lab hi lo back round velaric tense long "


./asr.sh \
    --auxiliary_data_tags "${aux_ctc}" \
    --post_process_local_data_opts "--stage 2" \
    --stage 3 \
    --ngpu 4 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --token_type word \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type "${feats_type}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"

    # TODO: idk if the config'll work or not

    # TODO: BPE tokenization
