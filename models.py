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
    
    def __init__(self, in_features, out_features, bias=True, eps=0.01):
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
        init.zeros_(self.adversary_weight)
        # Initialize biases
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def reset_adversary(self):
        # Initialize adversarial weights - much smaller
        init.uniform_(self.adversary_weight, a=-1., b=1.)
        # Initialize biases
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def cancel_adversary(self):
        # Force adversary weights to zero - when the primary player begins
        init.zeros_(self.adversary_weight)
            
    # Routine to update the primary weights after the adversary's turn
    # The primary weights become w(1 + eps*tanh(a)), the adversary become zero
    def update_primary(self):
        # Update primary weights
        with torch.no_grad():
            self.primary_weight *= (1 + self.eps*torch.tanh(self.adversary_weight))
        # Reset adversary weights to zero
        init.zeros_(self.adversary_weight)
        
    def forward(self, input):
        # Hadamard product w(1+a)
        # Tanh implicitly bounds adversary
        return F.linear(input, 
                        self.primary_weight * (1+self.eps*torch.tanh(self.adversary_weight)),
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
        
        # Create iterable collection of parameters
        self.primary_weights = [self.fc1.weight, self.fc2.weight]
        
    def forward(self, x):
	    x = x.view(-1, 784)
	    self.layer_input['fc1'] = x.data
	    x = F.relu(self.fc1(x))
	    self.layer_input['fc2'] = x.data
	    x = F.softmax(self.fc2(x), dim = 1)
	    return x

# Adversarial FF, FC network
class AdversarialNet(nn.Module):
    def __init__(self, eps):
        super(AdversarialNet, self).__init__()
        self.layer_input = dict()
        self.fc1 = AdversarialLinear(784, 512, bias=False, eps=eps)
        self.fc2 = AdversarialLinear(512, 10, bias=False, eps=eps)
        
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
    
# Flatten tensor utility
def flat(lista, total_params, requires_grad):
   res = torch.zeros(total_params)
   res.requires_grad = requires_grad
   ind=0
   for g in lista:
       numb = g.numel()
       g_flat = g.view(numb)
       res[ind : ind + numb] = g_flat
       ind += numb
   return res

# Accuracy utility
def accuracy(net, testloader, USE_GPU):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = cudafy(USE_GPU, data)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

# CUDA utility
def cudafy(use_gpu, seq, device=None):
    """ If use_gpu is True, returns cuda version of everything in tuple seq"""
    if use_gpu is False:
        return tuple(_.cpu() for _ in seq)
    else:
        if device != None:
            return tuple(_.to(device) for _ in seq)
        else:
            return tuple(_.cuda() for _ in seq)