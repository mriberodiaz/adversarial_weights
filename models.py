import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import torch.nn as nn
import torch.nn.functional as F

# Custom layer with two sets of weights
class AdversarialLinear(nn.Module):
    # TODO: What is this and is it needed?
    __constants__ = ['bias']
    
    def __init__(self, in_features, out_features, eps=0.1, bias=True):
        super(AdversarialLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # This controls the maximum per-weight relative perturbation
        self.eps = eps
        
        # Primary weights
        self.primary_weight   = Parameter(torch.Tensor(out_features, in_features))
        # Adversarial weights
        self.adversary_weight = Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize primary weights
        init.kaiming_uniform_(self.primary_weight, a=math.sqrt(5), nonlinearity='relu')
        # Initialize adversarial weights - much smaller
        # TODO: This needs to be exactly the epsilon-inf-ball
        # TODO: Epsilon needs to be passed as a parameter
        init.uniform_(self.adversary_weight, a=-0.01, b=0.01)
        # Initialize biases
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        # Hadamard product w(1+a)
        # Tanh implicitly bounds adversary
        return F.linear(input, 
                        self.primary_weight * (1+self.eps*F.tanh(self.adversary_weight)),
                        self.bias)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Regular FF, FC network
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

# Adversarial FF, FC network
class AdversarialNet(nn.Module):
    def __init__(self):
        super(AdversarialNet, self).__init__()
        self.layer_input = dict()
        self.fc1 = AdversarialLinear(784, 512, bias=False)
        self.fc2 = AdversarialLinear(512, 10, bias=False)
        
        # Create iterable collections of parameters
        # TODO: Add bias later, as well as dynamic number of layers
        self.primary_weights   = [self.fc1.primary_weight,
                                  self.fc2.primary_weight]
        self.adversary_weights = [self.fc1.adversary_weight,
                                  self.fc2.adversary_weight]
    
    def forward(self, x):
        x = x.view(-1, 784)
        self.layer_input['fc1'] = x.data
        x = F.relu(self.fc1(x))
        self.layer_input['fc2'] = x.data
        x = F.softmax(self.fc2(x), dim = 1)
        return x