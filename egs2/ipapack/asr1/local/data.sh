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

if [ -z "${IPAPACK}" ]; then
    log "Fill the value of 'IPAPACK' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${IPAPACK}"

    mkdir ${IPAPACK}
    pip3 install -r local/requirements.txt

    python3 local/download.py
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for IPAPack"

    # untarring happens here
    python3 local/data_prep.py --source_dir ${IPAPACK} --target_dir data --min_wav_length ${min_wav_duration}

    for dir in data/train data/dev data/test; do
        utils/fix_data_dir.sh $dir
        utils/validate_data_dir.sh --no-feats $dir || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
