import math
from turtle import forward
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from restormer import TransformerBlock

from swinir import *
from gdn import GDN, OctaveGDN
# from gdn_converted import GDN, OctaveGDN


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class MaskedConv2d(nn.Module):
    def __init__(self, mask_type: str = "A", *args: Any, **kwargs: Any):
        super().__init__()

        self.conv = nn.Conv2d(*args, **kwargs)

        self.mask = torch.ones_like(self.conv.weight.data)
        _, _, h, w = self.mask.shape
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x):
        self.conv.weight.data *= self.mask.to(x.device)
        return self.conv(x)


class EntropyParameters(nn.Module):
    def __init__(self, channels: int, scale: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * scale, channels, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 128, reduction: int = 8, return_weight: bool = False):
        super(ChannelAttention, self).__init__()
        self.return_weight = return_weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        if not self.return_weight:
            return x * score
        else:
            return score


class RCAB(nn.Module):
    def __init__(self, num_features, reduction: int = 8):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 8, num_resblock: int = 3):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        self.res_blocks1 = nn.ModuleList(
            self._res_block(self.channels, self.channels) for _ in range(num_resblock)
        )
        self.res_blocks2 = self._res_block(2, 1)

    def forward(self, x):
        res = x

        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_weight = torch.cat([x_avg, x_max], dim=1)

        for res_block in self.res_blocks1:
            identity = x
            x = res_block(x)
            x += identity

        x_weight = self.res_blocks2(x_weight)
        x_weight = torch.sigmoid(x_weight)
        x *= x_weight

        return x + res

    def _res_block(self, in_channels, out_channels):
        res_block = nn.Sequential(
            nn.Conv2d(
                in_channels, self.channels // self.reduction,
                kernel_size=1, stride=1, padding=0
            ),
            nn.Conv2d(
                self.channels // self.reduction, self.channels // self.reduction,
                kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                self.channels // self.reduction, out_channels,
                kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU()
        )
        return res_block


class ChannelNorm2d(nn.Module):
    def __init__(self, in_channels: int = 64, momentum=1e-1, affine=True, eps=1e-3):
        super().__init__()
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(
            x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed


class AnalysisNet(nn.Module):
    """AnalysisNet"""
    def __init__(
        self,
        out_channels_n:int = 192,
        out_channels_m: int = 324,
        embed_dim: int = 192,
        window_size: int = 8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
    ):
        super(AnalysisNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels_n, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
        )
        self.gdn1 = GDN(out_channels_n)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
        )
        self.gdn2 = GDN(out_channels_n)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )
        self.gdn3 = GDN(out_channels_n)

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1),
        )

        # for swin-encoder
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(p=0.)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=128, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=128, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=4,
                qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0.,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
                img_size=128,
                patch_size=1,
                resi_connection="1conv"
            )
            self.layers.append(layer)

        self.attention = ChannelAttention(num_features=out_channels_m)

    def forward_swin(self, x, begin: int = 0, end: int = 1):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        layers = []
        for cnt, layer in enumerate(self.layers):
            if begin <= cnt <= end:
                layers.append(layer)
        for layer in layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        x = self.attention(x)
        return x


class AnalysisPriorNet(nn.Module):
    """AnalysisPriorNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(AnalysisPriorNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channels_m, out_channels_n, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )

        self.attention = ChannelAttention(num_features=out_channels_n)

    def forward(self, x):
        # x = torch.abs(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        return x


class SynthesisNet(nn.Module):
    """SynthesisNet"""
    def __init__(self, out_channels_n=192, out_channels_m=324):
        super(SynthesisNet, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_m, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, 3, 1, 1)
        )
        self.igdn1 = GDN(out_channels_n, inverse=True)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, 3, 1, 1)
        )
        self.igdn2 = GDN(out_channels_n, inverse=True)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, 3, 1, 1)
        )
        self.igdn3 = GDN(out_channels_n, inverse=True)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        res = F.interpolate(x, scale_factor=4, mode="nearest")
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x) + res)
        x = self.deconv4(x)
        return x


class SynthesisPriorNet(nn.Module):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(SynthesisPriorNet, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        x = torch.exp(x)
        return x


class FuseFeatures(nn.Module):
    def __init__(self, out_channels_n:int = 64, out_channels_m:int = 64):
        super().__init__()
        self.up_z = nn.Sequential(
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1),
        )

        self.conv = nn.Sequential(
            RCAB(num_features=out_channels_m+out_channels_n, reduction=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels_m+out_channels_n, out_channels_m, 1),
        )

    def forward(self, feature, z):
        # feature: n, out_channels_m, h, w
        # z: n, out_channels_n, h // 4, w // 4
        # output: n, out_channels_m, h, w
        up_z = self.up_z(z)
        feature = self.conv(torch.cat([feature, up_z], dim=1))
        return feature


class SynthesisEnhancement(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 32, num_resblock: int = 3):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)

        self.res_blocks = nn.ModuleList(
            self._res_block() for _ in range(num_resblock)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, in_channels * (4 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x):
        x = self.conv1(x)
        res = x
        for res_block in self.res_blocks:
            identity = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
            x = res_block(x) + identity
        x = x + res
        x = self.conv2(x)
        return x

    def _res_block(self):
        res_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
        )
        return res_block


class FuseFeatureSynthesisEnhancement(SynthesisEnhancement):
    def __init__(self, in_channels: int = 3, channels: int = 32, num_resblock: int = 2, feature_channels: int = 192, alpha: float = 0.25):
        super().__init__(in_channels, channels, num_resblock)

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.conv_fuse = nn.Conv2d(int(feature_channels * alpha), channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels * 2, in_channels * (4 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x, feature_high):
        x = self.conv1(x)
        res = self.conv_fuse(feature_high)
        for res_block in self.res_blocks:
            identity = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
            x = res_block(x) + identity
        x = torch.cat([x, res], dim=1)
        x = self.conv2(x)
        return x


class GaussianStructureBlock(nn.Module):
    def __init__(self, channels: int = 192, num_gaussian: int = 2):
        super().__init__()

        self.scale = 3 * num_gaussian - 1

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 3, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 3, channels * 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, channels * self.scale, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.chunk(self.scale, 1)
        return x
