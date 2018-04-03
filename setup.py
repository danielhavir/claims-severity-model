import os
import pandas as pd
import json
from tqdm import tqdm

DATA_DIR = os.path.join(os.environ['data'], 'allstate')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Data directory
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the csv files.')

args = parser.parse_args()

train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

mappings = {}; emb_size = {}
for col in tqdm([f'cat{i}' for i in range(1, 117)]):
    vals = sorted(train_df[col].unique())
    emb_size[col] = len(vals)
    mapping = dict(zip(vals, range(1,len(vals)+1)))
    mappings[col] = mapping
    train_df[col] = train_df[col].apply(lambda x: mapping[x])
    test_df[col] = test_df[col].apply(lambda x: mapping[x] if x in mapping else 0)

train_df.to_csv(os.path.join(args.data_dir, 'traindata.csv'), index=False)
test_df.to_csv(os.path.join(args.data_dir, 'testdata.csv'), index=False)

with open('data/emb_size.json', 'w') as f:
    json.dump(emb_size, f)

with open('data/mappings.json', 'w') as f:
    json.dump(mappings, f)
