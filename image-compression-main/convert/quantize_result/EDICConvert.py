# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class EDICConvert(torch.nn.Module):
    def __init__(self):
        super(EDICConvert, self).__init__()
        self.module_0 = py_nndct.nn.Input() #EDICConvert::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/AnalysisNet[encoder]/Conv2d[conv1]/input.2
        self.module_3 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/AnalysisNet[encoder]/Conv2d[conv2]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/AnalysisNet[encoder]/Conv2d[conv3]/input.6
        self.module_7 = py_nndct.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=False) #EDICConvert::EDICConvert/AnalysisNet[encoder]/Conv2d[conv4]/145
        self.module_8 = py_nndct.nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #EDICConvert::EDICConvert/SynthesisNet[decoder]/ConvTranspose2d[deconv1]/input.8
        self.module_9 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/SynthesisNet[decoder]/Conv2d[igdn1]/165
        self.module_10 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #EDICConvert::EDICConvert/SynthesisNet[decoder]/ConvTranspose2d[deconv2]/input.9
        self.module_11 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/SynthesisNet[decoder]/Conv2d[igdn2]/185
        self.module_12 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #EDICConvert::EDICConvert/SynthesisNet[decoder]/ConvTranspose2d[deconv3]/input
        self.module_13 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #EDICConvert::EDICConvert/SynthesisNet[decoder]/Conv2d[igdn3]/205
        self.module_14 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], output_padding=[1, 1], groups=1, bias=False, dilation=[1, 1]) #EDICConvert::EDICConvert/SynthesisNet[decoder]/ConvTranspose2d[deconv4]/216

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(self.output_module_1)
        self.output_module_5 = self.module_5(self.output_module_3)
        self.output_module_7 = self.module_7(self.output_module_5)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_14 = self.module_14(self.output_module_13)
        return self.output_module_14
