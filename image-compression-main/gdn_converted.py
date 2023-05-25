import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from typing import List


class GDN(nn.Module):
    """
    Generalized divisive normalization block.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    TODO: can't convert(because of LowerBound)
    """

    def __init__(
        self,
        ch,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = math.sqrt(self.beta_min + self.reparam_offset**2 + self.pedestal)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        gamma = torch.sqrt(g + self.pedestal)

        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        x = torch.tanh(x)
        beta = self.beta.reshape((1, -1, 1, 1))

        # Apply norm
        outputs = None
        if self.inverse:
            outputs = x * beta
        else:
            outputs = x / beta

        return outputs


class OctaveGDN(GDN):
    def __init__(
        self, 
        ch, inverse=False, beta_min=0.000001, gamma_init=0.1, reparam_offset=2 ** -18, 
        alpha_in: List[float] = [0, 0.25, 1],
        alpha_out: List[float] = [0, 0.25, 1],
    ):
        super().__init__(ch, inverse, beta_min, gamma_init, reparam_offset)

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.in_branch = len(self.alpha_in) - 1
        self.out_branch = len(self.alpha_out) - 1
        self.bals = nn.Parameter(torch.Tensor(self.out_branch, ch))
    
    def forward(self, x):
        x_h, x_l = x
        x_h = torch.tanh(x_h)
        x_l = torch.tanh(x_l)

        _, ch1, _, _ = x_h.shape

        # Beta bound and reparam
        beta = self.beta
        betas = [beta[:ch1], beta[ch1:-1]]

        # Apply norm
        if self.inverse:
            x_h = x_h * betas[0]
            x_l = x_l * betas[1]
        else:
            x_h = x_h / betas[0]
            x_l = x_l / betas[1]

        return x_h, x_l


if __name__ == "__main__":
    model = OctaveGDN(64).cuda()
    dummy_input = [torch.rand((2, 16, 32, 32)).cuda(), torch.rand((2, 48, 32, 32)).cuda()]
    dummy_output = model(dummy_input)
    for i in dummy_output:
        print(i.shape)
