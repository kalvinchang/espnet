#!/usr/bin/env bash

# adapted from egs2/wav2gloss/asr1/local/data.sh (Kwanghee Choi)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=100
min_wav_duration=0.5
SECONDS=0

train_data_dir=
valid_data_dir=
unseen_test_sets=
feature_type=panphon


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./utils/parse_options.sh


log "data preparation started"

if [ -z "${IPAPACK_PLUS}" ]; then
    log "Fill the value of 'IPAPACK_PLUS' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${IPAPACK_PLUS}"

    mkdir -p ${IPAPACK_PLUS}
    local/download.sh ${IPAPACK_PLUS}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for IPAPack+"

    python3 local/data_prep.py --source_dir ${IPAPACK_PLUS} --target_dir data --min_wav_length ${min_wav_duration}

    for dir in data/train data/dev data/test_fleurs data/test_doreco; do
        utils/fix_data_dir.sh "$dir"
        utils/validate_data_dir.sh --no-feats "$dir" || exit 1
    done
fi

if [ ${stage} -eq 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "data prep stage 2: Additional data processing - This should only be called after ASR stage 4"
    # create file of articulatory features for auxiliary CTC
    python local/create_artic_feats.py --feature_type "${feature_type}" --data_dir "${train_data_dir}" --write_vocabulary
    python local/create_artic_feats.py --feature_type "${feature_type}" --data_dir "${valid_data_dir}"
    python local/map_to_phoible.py --skip_mapping --unseen_test_sets "${unseen_test_sets}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
