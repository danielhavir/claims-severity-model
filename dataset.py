import torch.utils.data as thd
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def load_mappings():
    import json
    with open('data/emb_size.json', 'r') as f:
        emb_size = json.load(f)
    with open('data/mappings.json', 'r') as f:
        mappings = json.load(f)
    
    return mappings, emb_size

class AllStateDset(thd.Dataset):
    def __init__(self, path, train=True):
        self.train = train
        self.df = pd.read_csv(path)
        
        self.ids = self.df['id']
        self.df.drop('id', axis=1, inplace=True)
        if train:
            self.y = self.df['loss'].values
            self.df.drop('loss', axis=1, inplace=True)
        else:
            mappings, emb_size = load_mappings()
            for col in tqdm([f'cat{i}' for i in range(1, 117)]):
                mapping = mappings[col]
                rnd_map = np.random.randint(0, emb_size[col]+1)
                self.df[col] = self.df[col].apply(lambda x: mapping[x] if x in mapping else rnd_map)
        
        self.cat_columns = [f'cat{i}' for i in range(1, 117)]
        self.cont_columns = [f'cont{i}' for i in range(1, 15)]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        sample = {'cat': self.df.loc[idx, self.cat_columns].values,
                  'cont': self.df.loc[idx, self.cont_columns].values}
        
        if self.train:
            sample['label'] = self.y[idx]
        
        return sample
