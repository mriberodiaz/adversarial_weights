import torch
import numpy as np

def as_numpy(tensor_or_array):
    """ If given a tensor or numpy array returns that object cast numpy array
    """

    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.cpu().detach().numpy()
    return tensor_or_array

def create_adversarial_bounds(network, matrix_bounds, bias_bounds):
    for layer, fc in enumerate(network.fcs):
        for param in fc.parameters():
            print(param)


def adv_projection_step(network, bound_ratios, norm='l2'):
    """Project current gradient on norm bounded constraints. Bound of each parameter
       given as a fraction of the norm of the original parameter (w.r.t given norm).
       Fraction value given per layer through bound_ratios"""

    if norm == 'l2':
        for fc, ratio in zip(network.fcs, bound_ratios):
            for param in fc.parameters():
                grad = param.grad
                c = ratio*np.linalg.norm(param)/np.linalg.norm(grad)
                grad_hat = grad*c
                print(grad)
                print(grad_hat)
                param.grad = grad_hat
    else:
        raise NotImplementedError

def PLNN_adversarial_bounds(network, matrix_bounds, bias_bounds):
    # Create a nested list of bounds for each parameter in the network
    bounds = []
    for layer, fc in enumerate(network.fcs):
        layer_bounds = []
        for param in fc.parameters():
            if len(np.shape(param)) > 1 :
                # we have a matrix
                bound = matrix_bounds[layer]
            else:
                # we have a vector
                bound = bias_bounds[layer]
            layer_bounds.append(bound)
        bounds.append(layer_bounds)