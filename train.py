import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as thd
import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import AllStateDset
from model import *
from tqdm import tqdm
from time import time
import argparse

DATA_DIR = os.path.join(os.environ['data'], 'allstate')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Model
parser.add_argument('model', type=int, choices=[1,2,3], help='Which model to choose.')
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=1024, help='Batch size.')
# Epochs
parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of epochs.')
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float, default=0.02, help='Learning rate.')
# Gamma learning rate decay
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay.')
# Run on CPU?
parser.add_argument('--cpu', action='store_true', help='Flag whether to run on cpu. (Automatically if no GPU is detected.)')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')
# Prevent from saving checkpoints?
parser.add_argument('--no_checkpoints', action='store_true', help='Flag whether to prevent from checkpoints.')
# Data directory
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the csv files.')
# Random state seed
parser.add_argument('--seed', type=int, default=42, help='Random state, i.e. seed. (Default: 42)')
args = parser.parse_args()

use_gpu = (torch.cuda.is_available() and not args.cpu)

if use_gpu and args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()

with open('data/emb_size.json', 'r') as f:
    emb_size = json.load(f)

df = pd.read_csv(os.path.join(args.data_dir, 'traindata.csv'))
if args.model in [1,2]:
    df = pd.get_dummies(df, columns=emb_size.keys())
    
train_ids, val_ids = train_test_split(df.index.values, test_size=0.1, random_state=args.seed)

train_df = df.loc[train_ids].reset_index(drop=True)
val_df = df.loc[val_ids].reset_index(drop=True)

if args.model == 1:
    trainset = AllStateDset(train_df, one_hot=True)
    validset = AllStateDset(val_df, one_hot=True)
    model = Net1(trainset.dim)

elif args.model == 2:
    trainset = AllStateDset(train_df, one_hot=True)
    validset = AllStateDset(val_df, one_hot=True)
    model = Net2(trainset.dim)

elif args.model == 3:
    trainset = AllStateDset(train_df)
    validset = AllStateDset(val_df)
    model = Net3(emb_size)

print(model)

trainloader = thd.DataLoader(trainset, batch_size=args.batch_size, num_workers=4)
validloader = thd.DataLoader(validset, batch_size=args.batch_size, num_workers=4)
print(4*'#', 'loaders ready'.upper(), 4*'#', end='\n\n')

loaders = {'train': trainloader, 'val': validloader}
sizes = {'train': len(trainset), 'val': len(validset)}

if use_gpu:
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.cuda()

print(4*'#', 'model built'.upper(), 4*'#', end='\n\n')

criterion = nn.L1Loss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.lr_decay)
best_loss = float('inf')

times = [] ##### DELETE

print(4*'#', 'starting training'.upper(), 4*'#', end='\n\n')
for epoch in range(1, args.epochs+1):
    print(2*'#', f'Epoch {epoch}', 2*'#')
    t0 = time()
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0; total = 0
        pbar = tqdm(loaders[phase], total=sizes[phase]//args.batch_size+1)
        for i, data in enumerate(pbar):
            if args.model in [1,2]:
                inputs, labels = data['data'].float(), data['label']
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs = Variable(inputs)
            
            elif args.model == 3:
                categorical, continuous, labels = data['cat'].long(), data['cont'].float(), data['label']
                if use_gpu:
                    categorical, continuous, labels = categorical.cuda(), continuous.cuda(), labels.cuda()
                categorical, continuous = Variable(categorical), Variable(continuous)
                inputs = (categorical, continuous)
            
            labels = Variable(labels.float())

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            running_loss += loss.data[0]
            
            pbar.set_postfix(loss=round(running_loss / (i+1), 3))
        
        pbar.close()
        epoch_loss = running_loss / (i+1)
        
        print(f'Phase {phase.upper()}, Epoch {epoch}, Loss {round(epoch_loss, 3)}',
        f'Time {round(time()-t0, 2)}s', end='\n\n')

        
    if epoch > 1:
        times.append((time()-t0))
    
    if not args.no_checkpoints and epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model, f'data/checkpoints/model_0{args.model}.ckpt')
        print(2*'#', f'Best loss: {best_loss} achieved. Model saved', 2*'#')
    
    scheduler.step()

print(sum(times)/len(times))
