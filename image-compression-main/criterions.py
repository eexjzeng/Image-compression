import torch
import torch.nn as nn
import torchvision
from collections import namedtuple, OrderedDict
from typing import Sequence

import utils


class LossSSIM(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs

    def forward(self, x):
        ssim_value = utils.calc_ssim(self.inputs, x)
        return ssim_value


class Dist2LogitLayer(nn.Module):
    def __init__(self, channels: int = 32, use_sigmoid: bool = False):
        super(Dist2LogitLayer, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.module = nn.Sequential(
            nn.Conv2d(5, channels, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 1, stride=1, padding=0, bias=True),
        )

    def forward(self, d0, d1, eps=1e-1):
        return self.module(torch.cat((d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, channels: int = 32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(channels=channels)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net(d0, d1)
        return self.loss(self.logit, per)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.n_channels_list = [64, 128, 256, 512, 512]
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class LearnedPerceptualImagePatchSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VGG16()
        self.lin = LinLayers(self.net.n_channels_list)
        url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
            + f'master/lpips/weights/v0.1/vgg.pth'
        pretrain_linparams = torch.hub.load_state_dict_from_url(
            url, progress=True,
            map_location=None if torch.cuda.is_available() else torch.device('cpu')
        )
        state_dict = OrderedDict()
        for key, val in pretrain_linparams.items():
            new_key = key
            new_key = new_key.replace('lin', '')
            new_key = new_key.replace('model.', '')
            state_dict[new_key] = val
        self.lin.load_state_dict(state_dict)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
