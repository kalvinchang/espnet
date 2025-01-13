from utils import (
    LANGUAGES, SYMBOL_NA, SYMBOL_NOSPEECH, SYMBOLS_TIME, TASK_TOKENS,
    PHONEME_VOCABULARY
)


# source: https://github.com/espnet/espnet/blob/master/
#         egs2/owsm_v1/s2t1/local/generate_nlsyms.py
if __name__ == "__main__":
    out = "data/nlsyms.txt"

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"{s}" for s in LANGUAGES],
        *TASK_TOKENS,
        *SYMBOLS_TIME,
    ]

    with open(out, "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        for phoneme in PHONEME_VOCABULARY:
            fp.write(f"/{phoneme}/\n")
