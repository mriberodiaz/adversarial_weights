import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import utilities as utils

class PLNN(nn.Module):
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only
    """
    def __init__(self, layer_sizes=None, dtype=torch.FloatTensor):
        super(PLNN, self).__init__()

        if layer_sizes is None:
            layer_sizes = [32, 64, 128, 64, 32, 10]
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.net = self.build_network(layer_sizes)

    def build_network(self, layer_sizes):
        layers = OrderedDict()

        num = 1
        for size_pair in zip(layer_sizes, layer_sizes[1:]):
            size, next_size = size_pair
            layer = nn.Linear(size, next_size, bias=True).type(self.dtype)
            layers[str(num)] = layer
            self.fcs.append(layer)
            num = num + 1
            layers[str(num)] = nn.ReLU()
            num = num + 1

        del layers[str(num-1)]      # No ReLU for the last layer

        net = nn.Sequential(layers).type(self.dtype)
        print('Network Layer Sizes')
        print(self.layer_sizes)

        return net

    def get_parameters(self):
        params = []
        for fc in self.fcs:
            fc_params = [elem for elem in fc.parameters()]
            for param in fc_params:
                params.append(param)

        return params


class PLNN_seq(PLNN):
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only

        built from nn.Sequential
    """

    def __init__(self, sequential, layer_sizes, dtype=torch.FloatTensor):
        super(PLNN_seq, self).__init__(layer_sizes, dtype)
        self.fcs = [layer for layer in sequential if type(layer) == nn.Linear]

        self.net = sequential

