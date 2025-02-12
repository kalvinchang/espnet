#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train_allophant_mapped"
valid_set="dev_allophant_mapped"
test_sets="test_reduced_allophant"
# unseen_test_sets="test_doreco"
unseen_test_sets=

encoder=allophant
asr_config=conf/tuning/train_asr_${encoder}_allophones_phoible.yaml
inference_config=conf/decode_allophant.yaml
# All PHOIBLE features except for tone
aux_ctc="advancedTongueRoot anterior approximant back click consonantal constrictedGlottis continuant coronal delayedRelease distributed dorsal epilaryngealSource fortis front high labial labiodental lateral low loweredLarynxImplosive nasal periodicGlottalSource raisedLarynxEjective retractedTongueRoot round short sonorant spreadGlottis stress strident syllabic tap tense trill long"

#  --unseen_test_sets \"${unseen_test_sets}\"
./asr.sh \
    --asr_tag "allophant" \
    --wordtoken_list "data/token_list/word/tokens_mapped.txt" \
    --auxiliary_data_tags "${aux_ctc}" \
    --post_process_local_data_opts "--stage 2 --feature_type allophoible" \
    --stage 5 \
    --stop_stage 5 \
    --ngpu 0 \
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
    --inference_asr_model "valid.loss_ctc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"
