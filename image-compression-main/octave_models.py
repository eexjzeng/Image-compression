import copy
from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *


class OctaveConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        alpha_in: float = 0.5,
        alpha_out: float = 0.5,
        bias: bool = False
    ):
        super().__init__()

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2, mode="nearest")

        self.stride = stride
        self.is_dw = groups == in_channels
        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        in_channels_low = int(in_channels * alpha_in)
        in_channels_high = in_channels - int(in_channels * alpha_in)
        out_channels_low = int(out_channels * alpha_out)
        out_channels_high = out_channels - int(out_channels * alpha_out)

        self.conv_l2l = nn.Conv2d(
            in_channels_low, out_channels_low,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups * alpha_in, bias=bias
        ) if alpha_in != 0 and alpha_out != 0 else None
        self.conv_l2h = nn.Conv2d(
            in_channels_low, out_channels_high,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups * alpha_in, bias=bias
        ) if alpha_in != 0 and alpha_out != 1 else None
        self.conv_h2l = nn.Conv2d(
            in_channels_high, out_channels_low,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups * alpha_in, bias=bias
        ) if alpha_in != 1 and alpha_out != 0 else None
        self.conv_h2h = nn.Conv2d(
            in_channels_high, out_channels_high,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups * alpha_in, bias=bias
        ) if alpha_in != 1 and alpha_out != 1 else None
    
    def forward(self, x: List):
        x_h, x_l = x

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None

        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class GeneralizedOctaveConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        alpha_in: List[float] = [0, 0.5, 1],
        alpha_out: List[float] = [0, 0.5, 1],
        bias: bool = False,
        use_balance: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weights = nn.Parameter(torch.Tensor(
            out_channels, in_channels // self.groups, 
            self.kernel_size[0], self.kernel_size[1]))

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.in_branch = len(self.alpha_in) - 1
        self.out_branch = len(self.alpha_out) - 1

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.use_balance = use_balance
        if self.use_balance:
            self.bals = nn.Parameter(torch.Tensor(self.out_branch, out_channels))

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.bals is not None:
            nn.init.normal_(self.bals, mean=1.0, std=0.05)

    def forward(self, xset):
        yset = []
        ysets = []
        for j in range(self.out_branch):
            ysets.append([])
        if not isinstance(xset, List):
           xset = [xset, ]

        if self.use_balance:
            bals_norm = torch.abs(self.bals) / (torch.abs(self.bals).sum(dim=0) + 1e-14)

        for i in range(min(len(xset), self.in_branch)):
            x = xset[i]
            begin_x = int(round(self.in_channels * self.alpha_in[i] / self.groups))
            end_x = int(round(self.in_channels * self.alpha_in[i+1] / self.groups))
            if begin_x == end_x:
                continue
            for j in range(self.out_branch):
                begin_y = int(round(self.out_channels * self.alpha_out[j]))
                end_y = int(round(self.out_channels * self.alpha_out[j+1]))
                if begin_y == end_y:
                    continue
                
                try:
                    h, w = xset[j].shape[2:4]
                except:
                    h, w = x.shape[2:4]
                if self.stride == 2:
                    this_output_shape = (h // 2, w // 2)
                else:
                    this_output_shape = (h, w)

                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                if self.use_balance:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]
                    this_weight = this_weight*bals_norm[j,begin_y:end_y].view(this_weight.shape[0],1,1,1)
                else:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]

                y = F.conv2d(x, this_weight, this_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                y = F.interpolate(y, size=this_output_shape)

                ysets[j].append(y)

        for j in range(self.out_branch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
        del ysets

        return yset


class GeneralizedOctaveConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        alpha_in: List[float] = [0, 0.25, 1],
        alpha_out: List[float] = [0, 0.25, 1],
        bias: bool = False,
        use_balance: bool = False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.weights = nn.Parameter(torch.Tensor(
            out_channels, in_channels // self.groups, 
            self.kernel_size[0], self.kernel_size[1]))

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.in_branch = len(self.alpha_in) - 1
        self.out_branch = len(self.alpha_out) - 1

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.use_balance = use_balance
        if self.use_balance:
            self.bals = nn.Parameter(torch.Tensor(self.out_branch, out_channels))
        else:
            self.bals = None

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.bals is not None:
            nn.init.normal_(self.bals, mean=1.0, std=0.05)

    def forward(self, xset):
        yset = []
        ysets = []
        for j in range(self.out_branch):
            ysets.append([])

        if not isinstance(xset, List):
           xset = [xset, ]

        if self.use_balance:
            bals_norm = torch.abs(self.bals) / (torch.abs(self.bals).sum(dim=0) + 1e-14)

        for i in range(min(len(xset), self.in_branch)):
            x = xset[i]
            begin_x = int(round(self.in_channels * self.alpha_in[i] / self.groups))
            end_x = int(round(self.in_channels * self.alpha_in[i+1] / self.groups))
            if begin_x == end_x:
                continue
            for j in range(self.out_branch):
                begin_y = int(round(self.out_channels * self.alpha_out[j]))
                end_y = int(round(self.out_channels * self.alpha_out[j+1]))
                if begin_y == end_y:
                    continue

                h, w = xset[j].shape[2:4]
                if self.stride == 2:
                    this_output_shape = (h * 2, w * 2)
                else:
                    this_output_shape = (h, w)

                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                if self.use_balance:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]
                    this_weight = this_weight*bals_norm[j,begin_y:end_y].view(this_weight.shape[0],1,1,1)
                else:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]

                y = F.conv_transpose2d(
                    x, this_weight.permute(1, 0, 2, 3), this_bias,
                    stride=self.stride, padding=self.padding, output_padding=self.output_padding,
                    groups=self.groups, dilation=self.dilation
                )
                y = F.interpolate(y, size=this_output_shape)
                ysets[j].append(y)

        for j in range(self.out_branch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
        del ysets

        return yset


class OctaveGaussianStructureBlock(nn.Module):
    def __init__(self, out_channels_n: int = 192, out_channels_m: int = 324, alpha: List = [0, 1]):
        super().__init__()

        self.scale = len(alpha) - 1

        self.conv1_11 = nn.Sequential(
            GeneralizedOctaveConv(
                out_channels_n, out_channels_m,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.conv1_21 = nn.Sequential(
            GeneralizedOctaveConv(
                out_channels_m, out_channels_m,
                stride=2,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_m, out_channels_m,
                stride=2,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.conv_x3_down = nn.Sequential(
            nn.Conv2d(3, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            GeneralizedOctaveConv(
                out_channels_n, out_channels_m,
                alpha_in=[0, 1], alpha_out=alpha
            )
        )
        self.conv3 = nn.Sequential(
            GeneralizedOctaveConv(
                out_channels_m, out_channels_n * 2,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_n * 2, out_channels_m * self.scale,
                stride=2,
                alpha_in=alpha, alpha_out=[0, 1]
            )
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        mu_convs = []
        for _ in range(self.scale):
            mu_convs.append(nn.Conv2d(out_channels_m, out_channels_m, 3, 1, 1))

        sigma_convs = []
        for _ in range(self.scale):
            sigma_convs.append(nn.Conv2d(out_channels_m, out_channels_m, 3, 1, 1))

        weight_convs = []
        for _ in range(self.scale):
            weight_convs.append(ChannelAttention(out_channels_m, return_weight=True))

        self.mu_conv = nn.ModuleList(mu_convs)
        self.sigma_conv = nn.ModuleList(sigma_convs)
        self.weight_conv = nn.ModuleList(weight_convs)

    def forward(self, xs_1: List, xs_2: List, x3: torch.Tensor):
        """
        inputs:
            xs_1:
            [
                shape: (n, 41, 16, 16),
                shape: (n, 41, 8, 8),
                shape: (n, 41, 4, 4),
                shape: (n, 69, 2, 2),
            ]
            xs_2:
            [
                shape: (n, 69, 64, 64)
                shape: (n, 69, 32, 32)
                shape: (n, 69, 16, 16)
                shape: (n, 117, 8, 8)
            ]
            x3: torch.Tensor(n, 1, 128, 128)

        return: 4 times []
            mus: (n, c, 8, 8)
            sigmas: (n, c, 8, 8)
            weights: (n, c, 1, 1)
        """
        xs_1 = self.conv1_11(xs_1)
        xs_21 = self.conv1_21(xs_2)
        xs3_down = self.conv_x3_down(x3)
        xs = []
        for i, (x_12, x1) in enumerate(zip(xs_21, xs_1)):
            x = F.interpolate(
                xs3_down[i], scale_factor=2**(-1 * i),
                mode="bilinear", align_corners=True, recompute_scale_factor=True
            )
            xs.append(x_12 + x1 + x)

        for i, x in enumerate(xs):
            xs[i] = self.relu(x)
        x = self.conv3(xs)[0]
        xs = x.chunk(self.scale, 1)

        mus = []
        sigmas = []
        weights = []
        for i, _ in enumerate(xs):
            mu_conv = self.mu_conv[i]
            sigma_conv = self.sigma_conv[i]
            weight_conv = self.weight_conv[i]

            mus.append(mu_conv(xs[i]).detach())
            sigmas.append(sigma_conv(xs[i]).detach())
            weights.append(weight_conv(xs[i]).detach())
        return mus, sigmas, weights


class OctaveAttention(nn.Module):
    def __init__(self, channels_n: int = 64, channels_m:int = 64, reduction: int = 8, alpha: List=[0, 1], num_resblock: int = 3):
        super().__init__()
        self.channels = channels_n * 3 + channels_m
        self.reduction = reduction
        self.alpha = alpha

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu0 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = GeneralizedOctaveConv(
            self.channels, self.channels // self.reduction,
            alpha_in=alpha, alpha_out=alpha
        )

        self.res_blocks1 = nn.ModuleList(
            self._res_block() for _ in range(num_resblock)
        )
        self.convs2 = nn.ModuleList(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1) for _ in range(num_resblock)
        )

        self.conv_tail = GeneralizedOctaveConv(
            self.channels // self.reduction, channels_m,
            alpha_in=alpha, alpha_out=alpha
        )

    def forward(self, xs):
        xs = self.conv(xs)
        xs_weight = xs.copy()

        for i, x in enumerate(xs_weight):
            x_avg = torch.mean(x, dim=1, keepdim=True)
            x_max, _ = torch.max(x, dim=1, keepdim=True)
            x_weight = torch.cat([x_avg, x_max], dim=1)
            xs_weight[i] = x_weight

        for res_block in self.res_blocks1:
            identity = xs
            xs = res_block(xs)
            for i, x in enumerate(xs):
                xs[i] = self.relu(x) + identity[i]

        xs_weight_true = []
        for conv in self.convs2:
            for i, x_weight in enumerate(xs_weight):
                xs_weight_true.append(self.relu0(conv(x_weight)))
        del xs_weight

        for i, (x_weight, x) in enumerate(zip(xs_weight_true, xs)):
            xs[i] = torch.sigmoid(x_weight) * x

        xs = self.conv_tail(xs)

        return xs

    def _res_block(self):
        res_block = nn.Sequential(
            GeneralizedOctaveConv(
                self.channels // self.reduction, self.channels // self.reduction,
                kernel_size=1, stride=1, padding=0,
                alpha_in=self.alpha, alpha_out=self.alpha
            ),
            GeneralizedOctaveConv(
                self.channels // self.reduction, self.channels // self.reduction,
                alpha_in=self.alpha, alpha_out=self.alpha
            ),
            GeneralizedOctaveConv(
                self.channels // self.reduction, self.channels // self.reduction,
                kernel_size=1, stride=1, padding=0,
                alpha_in=self.alpha, alpha_out=self.alpha
            ),
        )
        return res_block


class OctaveAnalysisNetOnlyAttention(AnalysisNet):
    def __init__(
        self,
        out_channels_n: int = 192,
        out_channels_m: int = 324,
        embed_dim: int = 192,
        window_size: int = 8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        alpha: List = [0, 1]
    ):
        super().__init__(out_channels_n, out_channels_m, embed_dim, window_size, depths, num_heads)

        self.attention = OctaveAttention(
            channels_n=out_channels_n, channels_m=out_channels_m,
            alpha=alpha,
        )

    def forward(self, x, scale: float = 0.2):
        xs = []
        x = self.conv1(x)
        xs.append(x)
        x = self.forward_swin(F.interpolate(x, scale_factor=0.5), 0, 1) + F.interpolate(x, scale_factor=0.5) * scale
        xs.append(x)
        x = self.forward_swin(F.interpolate(x, scale_factor=0.5), 2, 3) + F.interpolate(x, scale_factor=0.5) * scale
        xs.append(x)
        x = self.forward_swin(F.interpolate(x, scale_factor=0.5), 4, 5) + F.interpolate(x, scale_factor=0.5) * scale
        xs.append(x)
        x = self.attention(xs)
        return x



class OctaveAnalysisNetRestorm(nn.Module):
    def __init__(
        self, out_channels_n: int = 64, out_channels_m: int = 64,
        num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], 
        expansion_factor=2.66,
        alpha: List = [0, 1]
    ):
        super().__init__()
        channels = [out_channels_n, out_channels_n, out_channels_n, out_channels_m]

        self.conv = nn.Conv2d(3, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(*[
                    TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)
                ]) for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
            ]
        )
        self.attention = OctaveAttention(
            channels_n=out_channels_n, channels_m=out_channels_m,
            alpha=alpha,
        )
    
    def forward(self, x):
        xs = []
        x = self.conv(x)
        x = self.encoders[0](x)
        xs.append(x)
        x = self.encoders[1](F.interpolate(x, scale_factor=0.5))
        xs.append(x)
        x = self.encoders[1](F.interpolate(x, scale_factor=0.5))
        xs.append(x)
        x = self.encoders[1](F.interpolate(x, scale_factor=0.5))
        xs.append(x)
        x = self.attention(xs)
        return x



class OctaveAnalysisPriorNet(AnalysisPriorNet):
    def __init__(self, out_channels_n=192, out_channels_m=320, alpha: List = [0, 1]):
        super().__init__(out_channels_n, out_channels_m)
        self.conv1 = GeneralizedOctaveConv(
            out_channels_m, out_channels_n,
            alpha_in=alpha, alpha_out=alpha
        )
        self.conv2 = nn.Sequential(
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n, stride=2,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.conv3 = nn.Sequential(
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n, stride=2,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n * 2,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.attention = OctaveAttention(
            channels_n=out_channels_n // 3, channels_m=out_channels_n,
            alpha=alpha
        )

    def forward(self, xs):
        xs = self.conv1(xs)
        xs = self.conv2(xs)
        xs = self.conv3(xs)
        xs = self.attention(xs)
        return xs


class OctaveSynthesisNet(SynthesisNet):
    def __init__(self, out_channels_n=192, out_channels_m=320, alpha: List = [0, 1]):
        super().__init__(out_channels_n, out_channels_m)
        self.deconv1 = nn.Sequential(
            GeneralizedOctaveConvTranspose(
                out_channels_m, out_channels_n, stride=2, output_padding=1,
                alpha_in=[0, 1], alpha_out=alpha
            ),
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=1, output_padding=0,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.igdn1 = OctaveGDN(out_channels_n, alpha_in=alpha, alpha_out=alpha, inverse=True)

        self.deconv2 = nn.Sequential(
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=2, output_padding=1,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=1, output_padding=0,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.igdn2 = OctaveGDN(out_channels_n, alpha_in=alpha, alpha_out=alpha, inverse=True)

        self.deconv3 = nn.Sequential(
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=2, output_padding=1,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=1, output_padding=0,
                alpha_in=alpha, alpha_out=[0, 1]
            )
        )

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        res = x
        x = self.deconv3(x)[0]
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x, torch.cat(res, dim=1)


class OctaveSynthesisPriorNet(SynthesisPriorNet):
    def __init__(self, out_channels_n=192, out_channels_m=320, alpha: List = [0, 1]):
        super().__init__(out_channels_n, out_channels_m)
        self.deconv1 = nn.Sequential(
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=2, output_padding=1,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.deconv2 = nn.Sequential(
            GeneralizedOctaveConvTranspose(
                out_channels_n, out_channels_n, stride=2, output_padding=1,
                alpha_in=alpha, alpha_out=alpha
            ),
            GeneralizedOctaveConv(
                out_channels_n, out_channels_n,
                alpha_in=alpha, alpha_out=alpha
            )
        )
        self.deconv3 = GeneralizedOctaveConvTranspose(
            out_channels_n, out_channels_m, stride=1, output_padding=0,
            alpha_in=alpha, alpha_out=alpha
        )

    def forward(self, xs):
        xs = self.deconv1(xs)
        for i, x in enumerate(xs):
            xs[i] = self.relu1(x)
        xs = self.deconv2(xs)
        for i, x in enumerate(xs):
            xs[i] = self.relu2(x)
        xs = self.deconv3(xs)
        return xs


class OctaveFuse(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 64, alpha: List = [0, 1]):
        super().__init__()
        self.fuse = GeneralizedOctaveConv(channels, channels, alpha_in=alpha, alpha_out=[0, 1])
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )

    def forward(self, x: List[torch.Tensor]):
        x = self.fuse(x)[0]
        x = self.conv(x) + x  # NOTE: GOOD FOR CTRL BPP
        return x
