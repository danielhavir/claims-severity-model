import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
	def __init__(self, in_dim):
		super(Net1, self).__init__()
		self.layer_1 = nn.Linear(in_dim, 64)
		self.output = nn.Linear(64, 1)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight.data)
	
	def forward(self, x):
		out = self.layer_1(x)
		out = F.relu(out)
		out = self.output(out)
		return out

class Net2(nn.Module):
	def __init__(self, in_dim):
		super(Net2, self).__init__()
		self.layer_1 = nn.Linear(in_dim, 384)
		self.layer_2 = nn.Linear(384, 256)
		self.layer_3 = nn.Linear(256, 64)
		self.output = nn.Linear(64, 1)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight.data)
	
	def forward(self, x):
		out = self.layer_1(x)
		out = F.relu(out)
		out = self.layer_2(out)
		out = F.relu(out)
		out = self.layer_3(out)
		out = F.relu(out)
		out = self.output(out)
		return out

class Net3(nn.Module):
	def __init__(self, in_dim):
		super(Net3, self).__init__()
		self.layer_1 = nn.Linear(in_dim, 384)
		self.prelu = nn.PReLU()
		self.drop = nn.Dropout(p=0.5)
		self.hidden = nn.Sequential(
			nn.Linear(384, 256),
			nn.PReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(256, 64),
			nn.PReLU(),
			nn.Dropout(p=0.5)
		)
		self.output = nn.Linear(64, 1)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight.data)
	
	def forward(self, x):
		out = self.layer_1(x)
		out = self.prelu(out)
		out = self.drop(out)
		out = self.hidden(out)
		out = self.output(out)
		return out
