import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self, emb_size):
		super(Net, self).__init__()
		self.cat_modules = []
		self.cat_out_dim = 0
		for col in [f'cat{i}' for i in range(1, 117)]:
			size = min(emb_size[col]+1, 15)
			emb = nn.Embedding(emb_size[col]+1, size)
			self.cat_modules.append(emb)
			self.cat_out_dim += size
		
		self.cont1 = nn.Linear(14,48)
		self.cat_modules = nn.ModuleList(self.cat_modules)
		self.prelu = nn.PReLU()
		self.drop = nn.Dropout(p=0.3)
		self.hidden = nn.Sequential(
			nn.Linear(48+self.cat_out_dim, 256),
			nn.BatchNorm1d(256),
			nn.PReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(256, 64),
			nn.BatchNorm1d(64),
			nn.PReLU(),
			nn.Dropout(p=0.5)
		)
		self.output = nn.Linear(64, 1)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight.data)
			
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	
	def forward(self, cat_x, cont_x):
		out = self.cont1(cont_x)
		for i in range(116):
			x = self.cat_modules[i](cat_x[:, i])
			out = torch.cat((out, x), 1)
		
		out = self.prelu(out)
		out = self.drop(out)
		out = self.hidden(out)
		out = self.output(out)
		return out
