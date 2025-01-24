from pathlib import Path
import json

import langcodes


# identify unseen langs
unseen_langs = set()

with open('dump/train/language_distribution.json', 'r') as f:
    data = json.load(f)
    seen_langs = set(data.keys())

for split in Path('dump').iterdir():
    # remove DORECO
    if split == 'train' or 'doreco' in str(split) or not (split / 'language_distribution.json').exists():
        continue
    with open(split / 'language_distribution.json', 'r') as f:
        data = json.load(f)
        unseen_langs |= (set(list(data.keys())) - seen_langs)


# for each language, return its English name
print("seen_langs", len(seen_langs))
for lang_token in sorted(seen_langs):
    lang = lang_token.replace("<", "").replace(">", "")
    print(f"\"{lang_token}\": \"{langcodes.get(lang).display_name()}\",")
print()
print("unseen_langs", len(unseen_langs))
for lang_token in sorted(unseen_langs):
    lang = lang_token.replace("<", "").replace(">", "")
    print(f"\"{lang_token}\": \"{langcodes.get(lang).display_name()}\",")
