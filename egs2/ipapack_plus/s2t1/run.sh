#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=dev

nbpe=50000
s2t_config=conf/tuning/train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn.yaml
inference_config=conf/decode_s2t.yaml


# adapted from https://github.com/espnet/espnet/blob/master/egs2/owsm_v3.1/s2t1/run.sh

# TODO: remove stop_stage
# TODO: num_nodes

./s2t.sh \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 16 \
    --ngpu 4 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 64 \
    --num_splits_s2t 12 \
    --max_wav_duration 20 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/bpe_nlsyms.txt \
    --nlsyms_txt data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
