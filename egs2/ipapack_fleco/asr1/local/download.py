from huggingface_hub import snapshot_download


if __name__ == '__main__':
    # currently 6 partitions (12/5/2024) but could change
    # Only download partition 2 which contains doreco and fleurs
    partition = 2

    # keep trying until it succeeds
    while True:
        try:
            print(f"downloading anyspeech/ipapack_plus_{partition}")
            snapshot_download(
                repo_id=f"anyspeech/ipapack_plus_{partition}",
                repo_type="dataset",
                local_dir="downloads",
                allow_patterns=["fleurs_shar/*/*", "doreco_shar/*/*"],
                local_dir_use_symlinks=False,
                resume_download=False,
                max_workers=4
            )
            print(f"finished anyspeech/ipapack_plus_{partition}")
            break
        except Exception as e:
            print(f"Retrying: {e}")
            continue
    print("All done")
