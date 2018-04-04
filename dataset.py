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

        self.dim = self.df.shape[1]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.loc[idx].values
        
        if self.train:
            data = (data, self.y[idx])
        
        return data
