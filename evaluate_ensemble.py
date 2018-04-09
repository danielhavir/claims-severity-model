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
# Number of KFold splits
parser.add_argument('-ns', '--n_splits', type=int, default=10, help='Number of KFold splits.')
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=3072, help='Batch size.')
# Data directory
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the csv files.')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

test_df = pd.read_csv(os.path.join(args.data_dir, 'testdata.csv'))

train_df = pd.read_csv(os.path.join(args.data_dir, 'traindata.csv')).drop('loss', axis=1)
df = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)
df = pd.get_dummies(df, columns=[f'cat{i}' for i in range(1, 117)])
for col in df.columns:
    if col.endswith('_0'):
        df = df.drop(col, axis=1)
test_df = df.loc[train_df.shape[0]:,:].reset_index(drop=True)
del df, train_df

testset = AllStateDset(test_df, train=False)
testloader = thd.DataLoader(testset, batch_size=args.batch_size, num_workers=4)

outputs = None
for ii in range(args.n_splits):
    print(2*'#', f'Evaluating model #{ii+1}', 2*'#')
    model_outputs = None
    model = torch.load(os.path.join('data', 'checkpoints', f'ensemble_{args.model}{ii}.ckpt'))
    if use_gpu:
        model.cuda()

    model.eval()
    pbar = tqdm(testloader, total= len(testset)//args.batch_size+1)
    for inputs in pbar:
        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs.float())

        output = model(inputs).data.cpu()
            
        if model_outputs is None:
            model_outputs = output
        else:
            model_outputs = torch.cat((model_outputs, output), dim=0)

    pbar.close()
    model.cpu()
    model_outputs = model_outputs.unsqueeze(0).cpu()

    if outputs is None:
        outputs = model_outputs
    else:
        outputs = torch.cat((outputs, model_outputs), dim=0)
            
print(f'Phase TEST done')
import pdb; pdb.set_trace()
outputs = outputs.mean(dim=0).squeeze(-1)
ids = testset.ids.values
preds = outputs.view(-1).cpu().numpy()
subm = np.stack([ids, preds], axis=1)
subm = pd.DataFrame(subm, columns=['id', 'loss'])
subm['id'] = subm['id'].astype(int)
subm.to_csv(f'data/subm/ensemble_submission_0{args.model}.csv', index=False)
