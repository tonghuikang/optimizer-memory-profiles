"""
Variant of airbench94_muon which removes the whiten freezing
and uses the `airbench` dependency.

Ideal for continuing research.

Runs in ~2.8 seconds on a 400W NVIDIA A100
Attains 94.02 mean accuracy (n=900 trials)

---------
Ablations:

Current version -> 94.017 (n=900)

^ init head to zero -> ~93.90?

^ momentum=0.6 for norm_biases -> 94.008 (n=900)
^ optimize norm_biases using optimizer3 -> 93.984 (n=300)
^ nesterov=False for norm_biases SGD -> 93.991 (n=600)
^ artificial Adam for whiten_bias and head -> 94.014 (n=300)
^ artificial Adam with nesterov momentum -> 94.013 (n=300)
^ nesterov=False in Muon -> 93.733 (n=300)

^ bs=1000 -> 93.84 (n=100)
^ bs=1000 beta=0.8 lr=0.5x -> 93.89 (n=50)
^ bs=1000 beta=0.6 lr=0.5x -> 93.885 (n=100)
^ bs=1000 beta=0.8 -> 93.22 (n=50)
^ bs=1000 beta=0.7 lr=0.5x -> 93.95 (n=100)
^ bs=1000 beta=0.7 lr=0.7x -> 93.91 (n=90)

"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

#############################################
#                   Muon                    #
#############################################

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

