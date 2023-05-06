""" 
nnFunc Module 

This module contains an explicit construction of several
functions that are useful for building neural networks:
    - Cosine (Annealing) Learning Schedule 
    - Cross Entropy
    - Log Softmax
    - One Hot Encoder
    - ReLU
    - Softmax
"""

# Load packages
import torch, math

# - Cosine Schedule (Learning Rate) -
def cosine_schedule(initial_lr:int, t_global:int, t_max:int):
    """ The final learning rate is about x10 smaller than the initial """
    return (initial_lr/2)*(1 + math.cos((t_global*math.pi)/(4*t_max/3))) 

# - Cross Entropy (Loss) -
def cross_entropy(nn_output, target_prob, reduction='mean', label_smoothing=0):
    # Agrees with Pytorch implementation
    """ nn_output (logits) : (batch_size, num_classes)
        target_prob: (batch_size, num_classes)
        For example: num_classes == vocabulary_size """
    if label_smoothing !=0:
        target_prob = target_prob*(1 - label_smoothing) + label_smoothing/target_prob.shape[-1]
    batch_loss = torch.einsum('ij,ij->i', target_prob, -log_softmax(nn_output, dim=-1))
    if reduction == 'mean':
        return torch.mean(batch_loss)
    elif reduction == 'sum':
        return torch.sum(batch_loss)
    else:
        raise Exception('The "reduction" parameter must be "mean" or "sum".')

# - LogSoftmax -
def log_softmax(input, dim:int):
    # Agrees with Pytorch implementation
    max_regulator = torch.max(input, dim, keepdim=True)[0]
    in_exp = torch.exp(input - max_regulator)
    out = (input - max_regulator) - torch.log(torch.sum(in_exp, dim, keepdim=True))
    return out

# - One Hot (Encoding) -
def one_hot(input, num_classes:int):
    # Agrees with Pytorch implementation
    """ input: (*). output: (*, num_classes)"""
    return torch.zeros(*input.shape, num_classes, device=input.device).scatter_(-1, input.unsqueeze(-1), 1)

# - ReLU -
def relu(input):
    # Agrees with Pytorch implementation
    return (input > 0)*input

# - Sigmoid -
def sigmoid(input):
    return 1/(1 + torch.exp(-input))

# - Softmax -
def softmax(input, dim:int, temp:int=1):
    # Agrees with Pytorch implementation
    max_regulator = torch.max(input, dim, keepdim=True)[0] # Regulates to avoid overflow
    in_exp = torch.exp((input - max_regulator)/temp)
    norm = torch.sum(in_exp, dim, keepdim=True)
    return in_exp/norm