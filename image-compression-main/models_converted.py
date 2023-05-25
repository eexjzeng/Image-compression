import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bit_estimator import BitEstimator
import utils


class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 128, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return score, x * score

        
# ---------
# EDIC
# ---------
class AnalysisNet(nn.Module):
    """AnalysisNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisNet, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.in1 = nn.BatchNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.in2 = nn.BatchNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.in3 = nn.BatchNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv2d(out_channels_n, out_channels_m, kernel_size=5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.in1(self.conv1(x))
        x = self.in2(self.conv2(x))
        x = self.in3(self.conv3(x))
        x = self.conv4(x)
        return x


class AnalysisPriorNet(nn.Module):
    """AnalysisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisPriorNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channels_m, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class SynthesisNet(nn.Module):
    """SynthesisNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(SynthesisNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_m, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn1 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.in3 = nn.InstanceNorm2d(out_channels_n, affine=True, track_running_stats=True)

        self.deconv4 = nn.ConvTranspose2d(out_channels_n, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class SynthesisPriorNet(nn.Module):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192, withDLMM: bool = False):
        super(SynthesisPriorNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.LeakyReLU(0.1)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1)

        self.withDLMM = withDLMM
        if withDLMM:
            self.conv_tail = nn.Conv2d(out_channels_m, 12 * out_channels_m, kernel_size=1, stride=1, padding=0)
            # 12 = 3 * 4, DLMM_Channels: Channels * 4 * len(["mu", "scale", "mix"])

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        if self.withDLMM:
            self.conv_tail(x)
        # x = torch.exp(x)
        return x


class SynthesisPriorCANet(SynthesisPriorNet):
    """SynthesisPriorNet with channel attention"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(SynthesisPriorCANet, self).__init__()
        self.ca = ChannelAttention(num_features=out_channels_m)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        mu, x = self.ca(x)
        return mu, x


class EDICImageCompression(nn.Module):
    """EDICImageCompression"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICImageCompression, self).__init__()
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m

        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)

        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_std = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

        self.bit_estimator_z = BitEstimator(out_channels_n)

    def forward(self, x):
        # step 1: init
        quant_noise_feature = torch.zeros(
            x.size(0),
            self.out_channels_m, x.size(2) // 16, x.size(3) // 16
        ).to(x.device)
        quant_noise_z = torch.zeros(
            x.size(0), 
            self.out_channels_n, x.size(2) // 64, x.size(3) // 64
        ).to(x.device)

        quant_noise_feature = nn.init.uniform_(
            torch.zeros_like(quant_noise_feature), 
            -0.5, 0.5
        ).to(x.device)
        quant_noise_z = nn.init.uniform_(
            torch.zeros_like(quant_noise_z), 
            -0.5, 0.5
        ).to(x.device)

        # step 2: input image into encoder
        feature = self.encoder(x)
        batch_size = feature.shape[0]

        # step 3: put feature into prior(entropy encoding)
        z = self.encoder_prior(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        # step 4: compress the feature by the prior(entropy decoding)
        recon_mu = self.decoder_prior_mu(compressed_z)
        recon_sigma = self.decoder_prior_std(compressed_z)
        # recon_mu, recon_sigma = self.decoder_prior(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        compressed_feature_renorm = utils.quantize_st(compressed_feature_renorm, mean=recon_mu)

        # step 5: get the image from decoder
        recon_image = self.decoder(compressed_feature_renorm)
        # clipped_recon_image = recon_image.clamp(0, 255)       # because it's float, clamp it

        total_bits_feature, _ = utils.feature_probs_based_sigma(
            compressed_feature_renorm, recon_sigma
        )
        total_bits_z, _ = utils.iclr18_estimate_bits(self.bit_estimator_z, compressed_z)

        bpp_feature = total_bits_feature / (batch_size * x.shape[2] * x.shape[3])
        bpp_z = total_bits_z / (batch_size * x.shape[2] * x.shape[3])

        return recon_image, bpp_feature, bpp_z


if __name__ == "__main__":
    from torchsummary import summary
    a = EDICImageCompression().cuda()
    summary(a, (1, 64, 64))
    # if you use HiFIC, torchsummary may raise TypeError
