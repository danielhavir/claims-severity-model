import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as thd
import numpy as np
import os
from dataset import AllStateDset
from tqdm import tqdm
import pdb

def main(model, v):
    DATA_DIR = os.path.join(os.environ['DSETS'], 'allstate')
    testset = AllStateDset(os.path.join(DATA_DIR, 'test.csv'), train=False)
    size = len(testset)
    testloader = thd.DataLoader(testset, batch_size=1024, num_workers=4)
    criterion = nn.L1Loss()

    first = True
    model.eval()
    pbar = tqdm(testloader, total=size//1024+1)
    for data in pbar:
        categorical, continuous = data['cat'].cuda().long(), data['cont'].cuda().float()

        categorical, continuous = Variable(categorical), Variable(continuous)

        output = model(categorical, continuous)
        
        if first:
            outputs = output
            first = False

        else:
            outputs = torch.cat((outputs, output), dim=0)

    pbar.close()
        
    print(f'Phase TEST done')

    import pandas as pd
    ids = testset.ids.values
    preds = outputs.view(-1).data.cpu().numpy()
    subm = np.stack([ids, preds], axis=1)
    subm = pd.DataFrame(subm, columns=['id', 'loss'])
    subm['id'] = subm['id'].astype(int)
    pdb.set_trace()
    subm.to_csv(f'data/subm/submission_0{v}.csv', index=False)

if __name__ == '__main__':
    v='01'
    model = torch.load(f'data/checkpoints/model_0{v}.ckpt')
    main()
