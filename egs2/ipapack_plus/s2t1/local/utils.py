from typing import Dict, List, Optional, Tuple, Union


# adapted from https://github.com/espnet/espnet/blob/master/egs2/
#              owsm_v1/s2t1/local/utils.py
############################################
# Definitions of shared symbols and values #
############################################
SYMBOL_NA: str = "<na>"  # symbol denoting text is not available
SYMBOL_NOSPEECH: str = "<nospeech>"  # symbol denoting non-speech audio
SPEECH_MAX_LEN: float = 20  # max speech length in seconds
SPEECH_RESOLUTION: float = 0.04  # resolution in seconds
# all timestamp symbols
SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
]
TASK_TOKENS: List[str] = [
    "<asr>", "<pr>", "<g2p>", "<p2g>"
]
# TODO:
LANGUAGES = {
}
