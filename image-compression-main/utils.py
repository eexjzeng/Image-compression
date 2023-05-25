import math
from typing import List
from numbers import Real

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable

from torchvision import transforms


def get_length(generator):
    return sum(1 for _ in generator)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def upsampling(img, x, y):
    func = nn.Upsample(size=[x, y], mode='bilinear', align_corners=True)
    return func(img)


def generate_noise(size, channels=1, type='gaussian', scale=2):
    if type == 'gaussian':
        noise = torch.randn(channels, size[0], round(size[1]/scale), round(size[2]/scale))
        noise = upsampling(noise, size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(channels, size[0], size[1], size[2]) + 5
        noise2 = torch.randn(channels, size[0], size[1], size[2])
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(channels, size[0], size[1], size[2])
    return noise * 10.


def concat_noise(img, *args):
    noise = generate_noise(*args)
    if isinstance(img, torch.Tensor):
        noise = noise.to(img.device)
    else:
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    mixed_img = torch.cat((img, noise), 1)
    return mixed_img


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    cs = torch.flatten(cs_map, 2).mean(-1)
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return cs, ssim_per_channel


def calc_ssim(img1, img2, window_size=11):
   """calculate SSIM"""
   (_, channel, _, _) = img1.size()
   window = create_window(window_size, channel)

   if img1.is_cuda:
       window = window.cuda(img1.get_device())
   window = window.type_as(img1)

   ssim_per_channel, _ = _ssim(img1, img2, window, window_size, channel)

   return ssim_per_channel.mean()


def calc_msssim(img1, img2, window_size=11, weights=None):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(img1.device, dtype=img1.dtype)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
       window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(img1, img2, window, window_size, channel)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = F.avg_pool2d(img1, kernel_size=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    return ms_ssim_val.mean().item()


def calc_psnr(img1, img2):
    """calculate PNSR on cuda and cpu: img1 and img2 have range [0, 1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1 / torch.sqrt(mse).item())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def feature_gaussian_probs_based_sigma(feature, sigma, mu=None):
    if mu is None:
        mu = torch.zeros_like(sigma)
    sigma = sigma.clamp(1e-10, 1e10)
    gaussian = D.Cauchy(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    # loss_prob = torch.sum(-1.0 * gaussian.log_prob(feature) / math.log(2.0))
    return total_bits


def log_prob(comp, x):
    x = comp._pad(x)
    log_prob_x = comp.component_distribution.log_prob(x + 1e-10)
    log_mix_prob = torch.log_softmax(comp.mixture_distribution.logits, dim=-1)
    return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)


def feature_gaussian_mixture_module_probs_based_sigma(
    feature, 
    mus: List[torch.Tensor], sigmas: List[torch.Tensor], weights: List[float], 
    device="cuda"
):
    num_layers = len(mus)
    n, c, h, w = mus[0].shape
    
    scale_weight = []
    for i, weight in enumerate(weights):
        if i == 0:
            scale_weight.append(1)
        else:
            scale_weight.append(weight / weights[i-1])

    mixture_weights = torch.zeros(num_layers).to(device)
    mixture_mus = torch.zeros((num_layers, n, c, h, w)).to(device)
    mixture_sigmas = torch.zeros((num_layers, n, c, h, w)).to(device)

    for i in range(num_layers):
        mixture_mus[i, ...] = mus[i]
        mixture_sigmas[i, ...] = sigmas[i].clamp(1e-10, 1e10)
        if i == 0:
            mixture_weights[i] = 1 / sum(scale_weight)
        else:
            mixture_weights[i] = mixture_weights[0] * scale_weight[i]

    mix = D.Categorical(mixture_weights)
    comp = D.Independent(D.Normal(mixture_mus, torch.abs(mixture_sigmas)), reinterpreted_batch_ndims=4)
    gaussian_mixture_module = D.MixtureSameFamily(mix, comp)

    total_bits = torch.sum(-1.0 * (log_prob(gaussian_mixture_module, feature)) / math.log(2.0))

    return total_bits


def iclr18_estimate_bits(bit_estimator, z, eps=1e-18):
    prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
    total_bits = torch.sum(
        torch.clamp(-1.0 * torch.log(prob + eps) / math.log(2.0), 0, 50)
    )
    return total_bits


def quantize(x, mean=None):
    if mean is not None:
        x = x - mean
        x = torch.round(x)
        x = x + mean
    else:
        x = torch.round(x)
    return x


def quantize_st(x, mean=None):
    if mean is not None:
        x = x - mean
    delta = (torch.round(x) - x).detach()
    x = x + delta
    if mean is not None:
        x = x + mean
    return x
