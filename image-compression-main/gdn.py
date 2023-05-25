import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from typing import List


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


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
        unfold = False
        if x.dim() == 5:
            unfold = True
            bs, ch, d, w, h = x.size() 
            x = x.view(bs, ch, d*w, h)

        _, ch, _, _ = x.shape

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)

        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = F.conv2d(x**2, gamma, beta)
        norm_ = torch.sqrt(torch.abs(norm_))
        # assert torch.isnan(norm_).sum() == 0

        # Apply norm
        outputs = None
        if self.inverse:
            outputs = x * norm_
        else:
            outputs = x / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
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
        xset = [x_h, x_l]

        _, ch1, _, _ = x_h.shape
        _, ch2, _, _ = x_l.shape
        ch = ch1 + ch2

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)

        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        yset = []
        ysets = []
        for j in range(self.out_branch):
            ysets.append([])

        for i in range(min(len(xset), self.in_branch)):
            x = xset[i]
            begin_x = int(round(ch * self.alpha_in[i]))
            end_x = int(round(ch * self.alpha_in[i+1]))
            if begin_x == end_x:
                continue
            for j in range(self.out_branch):
                begin_y = int(round(ch * self.alpha_out[j]))
                end_y = int(round(ch * self.alpha_out[j+1]))
                if begin_y == end_y:
                    continue
                scale_factor = 2 ** (i - j)

                this_bias = beta[begin_y:end_y]
                this_weight = gamma[begin_y:end_y, begin_x:end_x, :,:].view(-1, 1, 1, 1)

                y = F.conv2d(x ** 2, this_weight, this_bias)
                ysets[j].append(y)

        for j in range(self.out_branch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
        del ysets

        # Norm pool calc
        yset_norm = []
        for y in yset:
            yset_norm.append(torch.sqrt(torch.abs(y)))

        # Apply norm
        if self.inverse:
            x_h = x_h * yset_norm[0]
            x_l = x_l * yset_norm[1]
        else:
            x_h = x_h / yset_norm[0]
            x_l = x_l / yset_norm[1]

        return x_h, x_l


if __name__ == "__main__":
    model = OctaveGDN(64).cuda()
    dummy_input = [torch.rand((2, 16, 32, 32)).cuda(), torch.rand((2, 48, 32, 32)).cuda()]
    dummy_output = model(dummy_input)
    for i in dummy_output:
        print(i.shape)
