import torch
import torch.onnx

import sys
sys.path.append("..")

from models_converted import *


class EDICConvert(EDICImageCompression):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super().__init__(out_channels_n=out_channels_n, out_channels_m=out_channels_m)

    def forward(self, x):
        feature = self.encoder(x)
        feature = torch.round(feature)
        z = self.encoder_prior(feature)
        z = torch.round(z)
        mu = self.decoder_prior_mu(z)
        feature = feature - mu
        delta = torch.floor(feature + 0.5) - feature
        feature = feature + delta
        feature = feature + mu
        recon_image = self.decoder(feature)
        return recon_image


# models init
model = EDICConvert().cuda()
model_params = torch.load("../model/pretrain_converted2.pth")
# for pytorch 1.4.0 and Vitis-AI 1.4.0
torch.save(model_params, "../model/pretrain_converted2_low.pth", _use_new_zipfile_serialization=False)
model.load_state_dict(model_params)
model.eval()

# modify the shape of input
dummy_input = torch.randn(1, 1, 64, 64).cuda()
input_names = ["image_input"]
output_names = ["image_output"]

torch.onnx.export(
    model, dummy_input, "pretrain.onnx", 
    verbose=True, 
    input_names=input_names, output_names=output_names, 
    opset_version=11
)
