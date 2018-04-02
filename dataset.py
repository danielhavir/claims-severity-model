import torch.utils.data as thd
import pandas as pd

class AllStateDset(thd.Dataset):
    def __init__(self, df, train=True):
        self.train = train
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        
        self.ids = self.df['id']
        self.df.drop('id', axis=1, inplace=True)
        if train:
            self.y = self.df['loss'].values
            self.df.drop('loss', axis=1, inplace=True)
        
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
