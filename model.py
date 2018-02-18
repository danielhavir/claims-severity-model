import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Net(nn.Module):
    def __init__(self, emb_size):
        super(Net, self).__init__()
        self.cat_modules = []
        self.cat_out_dim = 0
        for col in [f'cat{i}' for i in range(1, 117)]:
            emb = nn.Embedding(emb_size[col], min(emb_size[col], 10))
            self.cat_modules.append(emb)
            self.cat_out_dim += min(emb_size[col], 10)
        self.cat_modules = nn.ModuleList(self.cat_modules)
        self.cont1 = nn.Linear(14,32)
        self.fc2 = nn.Linear(32+self.cat_out_dim, 128)
        #self.fc3 = nn.Linear(256, 32)
        self.output = nn.Linear(128, 1)
    
    def forward(self, cat_x, cont_x):
        out = self.cont1(cont_x)
        for i in range(116):
            x = self.cat_modules[i](cat_x[:, i])
            out = torch.cat((out, x), 1)
        
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        """
        out = self.fc3(out)
        out = F.relu(out)
        """
        out = self.output(out)
        return out
