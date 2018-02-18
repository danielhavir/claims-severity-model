import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as thd
import os, json
from dataset import AllStateDset
from model import Net
from tqdm import tqdm

epochs = 3
batch_size = 1024
multi_gpu = False
v = '02'

trainset = AllStateDset('data/traindata.csv')
validset = AllStateDset('data/valdata.csv')

trainloader = thd.DataLoader(trainset, batch_size=batch_size, num_workers=4)
validloader = thd.DataLoader(validset, batch_size=batch_size, num_workers=4)
print(4*'#', 'loaders ready'.upper(), 4*'#', end='\n\n')

with open('data/emb_size.json', 'r') as f:
    emb_size = json.load(f)

loaders = {'train': trainloader, 'val': validloader}
sizes = {'train': len(trainset), 'val': len(validset)}

model = Net(emb_size)

if torch.cuda.is_available():
    model.cuda()

if multi_gpu:
    model = nn.DataParallel(model)

print(4*'#', 'model built'.upper(), 4*'#', end='\n\n')

criterion = nn.L1Loss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=0.01)
best_loss = float('inf')

print(4*'#', 'starting training'.upper(), 4*'#', end='\n\n')
for epoch in range(1, epochs+1):
    print(2*'#', f'Epoch {epoch}', 2*'#')
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0; total = 0
        pbar = tqdm(loaders[phase], total=sizes[phase]//batch_size+1)
        for data in pbar:
            categorical, continuous, labels = data['cat'].cuda().long(), data['cont'].cuda().float(), data['label'].cuda()

            categorical, continuous = Variable(categorical), Variable(continuous)
            labels = Variable(labels.float())

            optimizer.zero_grad()

            output = model(categorical, continuous)
            loss = criterion(output, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            total += labels.size(0)
            running_loss += loss.data[0]
            
            pbar.set_postfix(loss=round(running_loss / total, 3))
            """
            if phase == 'train' and (i+1) % 25 == 0:
                print(f'Epoch {epoch}, Iter {i}, Loss {round(running_loss / total, 3)}')
            """
        
        pbar.close()
        epoch_loss = running_loss / sizes[phase]
        
        print(f'Phase {phase.upper()}, Epoch {epoch}, Loss {round(epoch_loss, 3)}', end='\n\n')

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model, f'data/checkpoints/model_0{v}.ckpt')
    #scheduler.step()

import evaluate
evaluate.main(model, v)
