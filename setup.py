import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

DATA_DIR = os.path.join(os.environ['DSETS'], 'allstate')
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

mappings = {}; emb_size = {}
for col in tqdm([f'cat{i}' for i in range(1, 117)]):
    vals = sorted(df[col].unique())
    emb_size[col] = len(vals)
    mapping = dict(zip(vals, range(len(vals))))
    mappings[col] = mapping
    df[col] = df[col].apply(lambda x: mapping[x])

np.random.seed = 42
train_ids, val_ids = train_test_split(df.index.values, test_size=0.1)

train_df = df.loc[train_ids].reset_index(drop=True)
val_df = df.loc[val_ids].reset_index(drop=True)

train_df.to_csv('data/traindata.csv', index=False)
val_df.to_csv('data/valdata.csv', index=False)

with open('data/emb_size.json', 'w') as f:
    json.dump(emb_size, f)

with open('data/mappings.json', 'w') as f:
    json.dump(mappings, f)
