from tqdm import tqdm
import pandas as pd


df = pd.read_csv('downloads/transcript_normalized.csv')
df = df[['split', 'duration', 'lang', 'text']]


seen_langs = set()
unseen_langs = set()

splits = [('full', df), ('train', df[df['split'] == 'train']), ('dev', df[df['split'] == 'dev']), ('test', df[df['split'].str.startswith("test")])]
for split, split_df in tqdm(splits):
    print('split', split)
    print('duration (h)', split_df['duration'].sum() / 3600)
    print('langs', len(split_df['lang'].unique()))
    langs = set(split_df['lang'].unique())
    print('langs', sorted(list(langs)))
    if split == 'train':
        seen_langs |= langs
    elif split == 'test':
        split_unseen_langs = langs - unseen_langs
        print('unseen langs for', split, len(split_unseen_langs), split_unseen_langs)
        unseen_langs |= split_unseen_langs
    
    orthography = split_df[split_df['text'] != '']
    print('duration (h) with text in the split', orthography['duration'].sum() / 3600)

print('unseen langs', len(unseen_langs), unseen_langs)
