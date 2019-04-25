

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
	    super(Net, self).__init__()
	    self.layer_input = dict()
	    self.fc1 = nn.Linear(784, 512, bias=False)
	    self.fc2 = nn.Linear(512, 10, bias=False)

	def forward(self, x):
	    x = x.view(-1, 784)
	    self.layer_input['fc1'] = x.data
	    x = F.relu(self.fc1(x))
	    self.layer_input['fc2'] = x.data
	    x = F.softmax(self.fc2(x), dim = 1)
	    return x
