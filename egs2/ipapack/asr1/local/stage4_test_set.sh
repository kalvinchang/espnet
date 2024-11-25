#!/usr/bin/env bash

# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_stages=            # Spicify the stage to be skipped
test_sets="test_doreco test_mswc test_fleurs"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

    # asr.sh intentionally avoids skipping Stage 4 for test data
    # However, when extracting SSL features (e.g. XEUS),
        # you will want to remove long utterances to avoid out of memory errors
    for dset in ${test_sets}; do
        # Copy data dir
        # for test set, no need to copy
        _suf="/org"
        if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/${_suf}/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}${_suf}/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
        else
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/${dset}" "${data_feats}${_suf}/${dset}"
            cp "${data_feats}/${dset}/feats_type" "${data_feats}${_suf}/${dset}/feats_type"
            cp "${data_feats}/${dset}/audio_format" "${data_feats}${_suf}/${dset}/audio_format"
            cp "${data_feats}/${dset}/utt2num_samples" "${data_feats}${_suf}/${dset}/utt2num_samples"
        fi


        # Remove short utterances
        _feats_type="$(<${data_feats}/${dset}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}${_suf}/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}${_suf}/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"
        else
            # Get frame shift in ms from conf/fbank.conf
            _frame_shift=
            if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                # Assume using conf/fbank.conf for feature extraction
                _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
            fi
            if [ -z "${_frame_shift}" ]; then
                # If not existing, use the default number in Kaldi (=10ms).
                # If you are using different number, you have to change the following value manually.
                _frame_shift=10
            fi

            _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

            cp "${data_feats}${_suf}/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
            <"${data_feats}${_suf}/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                    >"${data_feats}/${dset}/feats_shape"
            <"${data_feats}${_suf}/${dset}/feats.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                >"${data_feats}/${dset}/feats.scp"
        fi

        # Remove empty text
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            <"${data_feats}${_suf}/${dset}/${ref_txt}" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/${ref_txt}"
        done

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh \
            ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
            "${data_feats}/${dset}"
    done

    if [ -n "${post_process_local_data_opts}" ]; then
        # Do any additional local data post-processing here
        local/data.sh ${post_process_local_data_opts} --asr_data_dir "${data_feats}/${train_set}"
    fi

    # shellcheck disable=SC2002,SC2068,SC2005
    for lm_txt in ${lm_train_text[@]}; do
        suffix=$(echo "$(basename ${lm_txt})" | sed 's/text//')
        <${lm_txt} awk -v suffix=${suffix} ' { if( NF != 1 ) {$1=$1 suffix; print $0; }} '
    done > "${data_feats}/lm_train.txt"
fi
