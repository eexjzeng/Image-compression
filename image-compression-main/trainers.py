import torch
import torch.nn as nn
# from dropblock import DropBlock2D

from models import *
from octave_models import *
from entropy_blocks import EntropyBottleneck, GaussianConditional

import utils


class OctaveEfficientDeepImageCompression(nn.Module):
    def __init__(
            self,
            out_channels_n: int = 192,
            out_channels_m: int = 324,
            embed_dim: int = 192,
            window_size: int = 8,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
    ):
        super().__init__()
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m
        channels = out_channels_n * 3 + out_channels_m
        alpha = [
            0, out_channels_n / channels, out_channels_n * 2 / channels,
               out_channels_n * 3 / channels, 1
        ]
        alpha_fuse = [
            0, out_channels_m / channels, (out_channels_n + out_channels_m) / channels,
               (out_channels_n * 2 + out_channels_m) / channels, 1
        ]
        self.alpha = alpha

        # For encoder-decoder / hyper-encoder / fuse
        self.encoder = OctaveAnalysisNetOnlyAttention(
            out_channels_n, out_channels_m,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            alpha=alpha
        )
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)
        self.encoder_prior = OctaveAnalysisPriorNet(out_channels_n, out_channels_m, alpha)
        self.fuse_feature = OctaveFuse(channels=out_channels_m, alpha=alpha_fuse, out_channels=out_channels_m)
        self.fuse_z = OctaveFuse(channels=out_channels_n, alpha=alpha_fuse, out_channels=out_channels_m)
        # NOTE: recommand: out_channels_n == out_channels_m, for this fuse_feature_z
        self.fuse_feature_z = FuseFeatures(out_channels_n=out_channels_n, out_channels_m=out_channels_m)

        # For gaussian's params block, EntropyBlocks
        self.entropy_bottleneck = EntropyBottleneck(out_channels_n)
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_module = OctaveGaussianStructureBlock(out_channels_n, out_channels_m, alpha)

    # def gen_broken_img(self, x, threshold: float = 0., block_size: int = 7):
    #     drop_block = DropBlock2D(block_size=block_size, drop_prob=threshold * 5)
    #     broken_img = drop_block(x)
    #     return broken_img.detach()

    def get_recon_mu_sigma(self, mus, sigmas, weights, device="cuda"):
        scale_weight = []
        num_layers = len(mus)
        n, c, _, _ = weights[0].shape
        mixture_weights = torch.zeros(num_layers, n, c, 1, 1).to(device)
        for i, weight in enumerate(weights):
            if i == 0:
                scale_weight.append(torch.ones_like(weights[0]).to(device))
            else:
                scale_weight.append(weight / weights[i - 1])
        for i in range(num_layers):
            if i == 0:
                mixture_weights[i, ...] = torch.ones_like(weights[0]).to(device) / sum(scale_weight)
            else:
                mixture_weights[i, ...] = mixture_weights[0] * scale_weight[i]
        recon_mu = torch.zeros_like(mus[0]).to(device)
        recon_sigma_temp = torch.zeros_like(mus[0]).to(device)
        for i in range(len(mus)):
            recon_mu += mus[i] * mixture_weights[i]
            recon_sigma_temp += (torch.pow(mus[i], 2) + sigmas[i]) * mixture_weights[i]
        recon_sigma = torch.sqrt(torch.abs(
            recon_sigma_temp - torch.pow(recon_mu, 2)
        ) + 1e-18)
        return recon_mu, recon_sigma

    def forward_swin(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x, p: float = 0.):
        batch_size = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]
        # step 1: init
        quant_noise_feature = torch.zeros(
            x.size(0), self.out_channels_m, x.size(2) // 16, x.size(3) // 16
        ).to(x.device)
        quant_noise_z = torch.zeros(
            x.size(0), self.out_channels_n, x.size(2) // 64, x.size(3) // 64
        ).to(x.device)

        # step 2: input image into encoder
        # loss_feature = torch.tensor([0.]).to(x.device)
        # broken_x = self.gen_broken_img(x, threshold=p)

        # if self.training:
        #     features = self.encoder(x)
        #     if p != 0:
        #         broken_features = self.encoder(broken_x)
        #         for broken_feature, feature in zip(broken_features, features):
        #             loss_feature += self.criterion(broken_feature, feature)
        #     else:
        #         broken_features = features.copy()
        # else:
        #     features = self.encoder(broken_x)
        #     broken_features = features.copy()
        features = self.encoder(x)

        # step 3: put feature into prior(entropy encoding)
        zs = self.encoder_prior(features)

        # step 4: compress the feature by the prior(entropy decoding and gaussian mixture models)
        # 1. get fused features and fused zs
        compressed_feature = self.fuse_feature(features[::-1])
        compressed_z = self.fuse_z(zs[::-1])
        if self.training:
            # unit_feature = abs(compressed_feature.mean().item() * compressed_feature.shape[1])
            unit_feature = 1.
            # unit_z = abs(compressed_z.mean().item() * compressed_z.shape[1])
            unit_z = 1.
            quant_noise_feature = nn.init.uniform_(
                torch.zeros_like(quant_noise_feature),
                -0.5 * unit_feature, 0.5 * unit_feature
            ).to(x.device)
            quant_noise_z = nn.init.uniform_(
                torch.zeros_like(quant_noise_z),
                -0.5 * unit_z, 0.5 * unit_z
            ).to(x.device)
        if self.training:
            compressed_feature = compressed_feature + quant_noise_feature
            compressed_z = compressed_z + quant_noise_z
        else:
            compressed_feature = torch.round(compressed_feature)
            compressed_z = torch.round(compressed_z)

        # 2. get params of gaussian condition
        mus, sigmas, weights = self.gaussian_module(zs, features, x)
        # mus, sigmas, weights = self.gaussian_module(zs, broken_features, broken_x)
        recon_mu, recon_sigma = self.get_recon_mu_sigma(mus, sigmas, weights, device=x.device)

        # 3. get features' and zs' hat and likelihoods and their bpp
        features_hat, features_likelihoods = self.gaussian_conditional(compressed_feature, recon_sigma, recon_mu)
        zs_hat, zs_likelihoods = self.entropy_bottleneck(compressed_z)

        bpp_feature = -torch.log2(torch.abs(features_likelihoods) + 1e-10).sum() / (batch_size * h * w)
        bpp_z = -torch.log2(torch.abs(zs_likelihoods) + 1e-10).sum() / (batch_size * h * w)
        assert torch.isnan(bpp_z + bpp_feature) == False

        # step 5: get the image from decoder and it's enhancement
        ########################################
        # Decoder, need features_hat ans zs_hat
        ########################################
        features = self.fuse_feature_z(features_hat, torch.exp(zs_hat))
        features = self.forward_swin(compressed_feature) + features
        recon_image = self.decoder(features)
        # clipped_recon_image = recon_image_high.clamp(0, 1)
        # clipped_recon_image = (torch.tanh(recon_image_high) + 1) / 2

        return recon_image, bpp_feature, bpp_z, 0


if __name__ == "__main__":
    model = OctaveEfficientDeepImageCompression().cuda()
    dummy_input = torch.rand((2, 3, 128, 128)).cuda()
    model.train()
    dummy_output = model(dummy_input, p=0.)
    for i in dummy_output:
        print(i)
    model.eval()
    dummy_output = model(dummy_input, p=0.)
    for i in dummy_output:
        print(i)
