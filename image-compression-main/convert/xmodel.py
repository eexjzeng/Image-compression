"""
use Vitis-ai: 1.4.0
"""
import cv2
import numpy as np
from scipy import sparse
import zlib

import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import sys
sys.path.append("..")

from models_converted import *


class EDICEncoder(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICEncoder, self).__init__()
        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)

    def forward(self, x):
        feature = self.encoder(x)
        z = self.encoder_prior(feature)
        return feature, z


class EDICDecoder(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICDecoder, self).__init__()
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
    
    def forward(self, feature, z):
        recon_mu = self.decoder_prior_mu(z)
        feature = utils.quantize_st(feature, recon_mu)
        recon_image = self.decoder(feature)
        return recon_image


class EDICConvert(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICConvert, self).__init__()
        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)

        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_std = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

        self.bit_estimator_z = BitEstimator(out_channels_n)

    def forward(self, x):
        feature = self.encoder(x)
        recon_image = self.decoder(feature)
        return recon_image


def model_info_read(model):
    for name, buf in model.named_buffers():
        if 'anchor_grid' in name:
            register_buffers = {'anchor_grid': buf}
    return register_buffers


def quantize(img, quant_mode: str = "calib", deploy: bool = False, device=torch.device("cpu")):
    print("\nbegin {}:".format(quant_mode))
    edic_model = EDICConvert()
    edic_params = torch.load("../model/pretrain_converted_low.pth", map_location=device)
    edic_model.load_state_dict(edic_params)

    # need ==> edic_encoder + edic_decoder
    # edic_encoder
    edic_encoder = EDICEncoder()
    encoder_dict =  edic_encoder.state_dict()
    encoder_state_dict = {k: v for k, v in edic_params.items() if k in encoder_dict.keys()}
    encoder_dict.update(encoder_state_dict)
    edic_encoder.load_state_dict(encoder_dict)
    # edic_decoder
    edic_decoder = EDICDecoder()
    decoder_dict =  edic_decoder.state_dict()
    decoder_state_dict = {k: v for k, v in edic_params.items() if k in decoder_dict.keys()}
    decoder_dict.update(decoder_state_dict)
    edic_decoder.load_state_dict(decoder_dict)

    print("============================== PyTorch local ==================================")
    features, z = edic_encoder(img)

    features = torch.round(features)
    z = torch.round(z)
    feature_numpy = features.int().detach().numpy()[0]

    features_sparses = []
    for i in range(feature_numpy.shape[0]):
        s = sparse.csr_matrix(feature_numpy[i, ...])
        feature_str = str(s).replace("\n", "").replace("\t", "").replace(" ", "").replace(":", "") + "\n"
        features_sparses.append(feature_str)
    features_str = "".join(features_sparses)

    with open("features.edic", "wb") as f:
        features_bytes = str.encode(features_str) 
        compress_str = zlib.compress(features_bytes, zlib.Z_BEST_COMPRESSION)
        f.write(compress_str)

    recon_img = edic_decoder(features, z)

    print("PSNR: {}".format(utils.calc_psnr(img, recon_img)))
    print("SSIM: {}".format(utils.calc_msssim(img, recon_img)))
    cv2.imwrite("recon_img_{}.jpg".format(quant_mode), recon_img.squeeze().detach().numpy())

    print("\n================================ Quantizer ====================================")
    quantizer = torch_quantizer(
        quant_mode, edic_model, (img), device=device
    )
    quant_model = quantizer.quant_model
    recon_img_int = quant_model(img)

    print("\nPSNR: {}".format(utils.calc_psnr(img, recon_img_int)))
    print("SSIM: {}".format(utils.calc_msssim(img, recon_img_int)))
    cv2.imwrite("recon_img_int_{}.jpg".format(quant_mode), recon_img_int.squeeze().detach().numpy())

    quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=True)

    # feature = torch.randn([1, 192, 32, 32])
    # z = torch.randn([1, 128, 8, 8])
    # encoder_quantizer = torch_quantizer(
    #     "calib", edic_encoder, (img), device=device
    # )
    # quant_encoder = encoder_quantizer.quant_model

    # decoder_quantizer = torch_quantizer(
    #     "calib", edic_decoder, (feature, z), device=device
    # )
    # quant_decoder = encoder_quantizer.quant_model


if __name__ == "__main__":
    import time
    from torch.utils.data import dataloader
    from tfrecord.torch.dataset import TFRecordDataset

    description = {
        "image": "byte",
        "size": "int",
    }
    valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
    valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

    cnt = 0
    for record in valid_dataloader:
        cnt += 1
        if cnt == 1:
            img = record["image"].reshape(
                1,
                1,
                record["size"][0],
                record["size"][0],
            ).float()
            cv2.imwrite("img.jpg", img.squeeze().numpy())
            t0 = time.time()
            quantize(img, quant_mode="calib")
            quantize(img, quant_mode="test", deploy=True)
            t1 = time.time()
            print("quantize and dumpy xmodel, spend: {}s".format(t1 - t0))
            break
        else:
            continue
