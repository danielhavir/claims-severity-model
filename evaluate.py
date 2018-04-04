import torch
from torch.autograd import Variable
import torch.utils.data as thd
import numpy as np
import pandas as pd
import os
from dataset import AllStateDset
from tqdm import tqdm
import argparse

DATA_DIR = os.path.join(os.environ['data'], 'allstate')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Model
parser.add_argument('model', type=int, choices=[1,2,3], help='Which model to choose.')
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=3072, help='Batch size.')
# Data directory
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the csv files.')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

model = torch.load(f'data/checkpoints/model_0{args.model}.ckpt')
if use_gpu:
    model.cuda()

test_df = pd.read_csv(os.path.join(args.data_dir, 'testdata.csv'))

if args.model in [1,2]:
    train_df = pd.read_csv(os.path.join(args.data_dir, 'traindata.csv')).drop('loss', axis=1)
    df = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)
    df = pd.get_dummies(df, columns=[f'cat{i}' for i in range(1, 117)])
    for col in df.columns:
        if col.endswith('_0'):
            df = df.drop(col, axis=1)
    test_df = df.loc[train_df.shape[0]:,:].reset_index(drop=True)
    del df, train_df
    testset = AllStateDset(test_df, train=False, one_hot=True)
else:
    testset = AllStateDset(test_df, train=False)

testloader = thd.DataLoader(testset, batch_size=args.batch_size, num_workers=4)

outputs = None
model.eval()
pbar = tqdm(testloader, total= len(testset)//args.batch_size+1)
for data in pbar:
    if args.model in [1,2]:
        inputs = data['data'].float()
        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs)

    elif args.model == 3:
        categorical, continuous = data['cat'].long(), data['cont'].float()
        if use_gpu:
            categorical, continuous = categorical.cuda(), continuous.cuda()
        categorical, continuous = Variable(categorical), Variable(continuous)
        inputs = (categorical, continuous)

    output = model(inputs)
        
    if outputs is None:
        outputs = output
    else:
        outputs = torch.cat((outputs, output), dim=0)

pbar.close()
        
print(f'Phase TEST done')

ids = testset.ids.values
preds = outputs.view(-1).data.cpu().numpy()
subm = np.stack([ids, preds], axis=1)
subm = pd.DataFrame(subm, columns=['id', 'loss'])
subm['id'] = subm['id'].astype(int)
subm.to_csv(f'data/subm/submission_0{args.model}.csv', index=False)
