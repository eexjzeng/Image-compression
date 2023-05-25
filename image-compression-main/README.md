# image compression

image compression for `.nii.gz` / `.DICOM` (medical image)

requirement: torch-1.7.0-cu11.0-cudnn8

## Learned Compression

After downloading the data, please make sure there is a dictionary called "dataset" contained `.nii.gz` files.

```python
python make_dataset.py
```

Please check your `train.tfrecord` and `valid.tfrecord`, then

```shell
nohup sh scripts/train_oic.sh > /dev/null &
```

## demo

GDN's compression metrics:

```shell
eval psnr: 36.109871 eval ssim: 0.990302 eval bpp: 0.508365 valid_time: 0.120961s
eval psnr: 33.125748 eval ssim: 0.984152 eval bpp: 0.490635 valid_time: 0.110648s
eval psnr: 32.912016 eval ssim: 0.981523 eval bpp: 0.489126 valid_time: 0.110688s
eval psnr: 31.760520 eval ssim: 0.978599 eval bpp: 0.496626 valid_time: 0.111405s
eval psnr: 29.456129 eval ssim: 0.972837 eval bpp: 0.506207 valid_time: 0.111129s
```

Simple-GDN's compression metrics:

```shell
eval psnr: 35.135837 eval ssim: 0.989593 eval bpp: 0.539716 valid_time: 0.119185s
eval psnr: 33.102380 eval ssim: 0.984198 eval bpp: 0.522302 valid_time: 0.110642s
eval psnr: 32.611660 eval ssim: 0.981391 eval bpp: 0.520853 valid_time: 0.111069s
eval psnr: 31.173841 eval ssim: 0.977298 eval bpp: 0.528915 valid_time: 0.109509s
eval psnr: 28.908695 eval ssim: 0.970718 eval bpp: 0.539352 valid_time: 0.110749s
```

## quantize

use [vitis-ai](https://github.com/jkhu29/vitis-ai) to quantize the model

## Traditional Compression

### JPEG

support JPEG | JPEG2000 | WEBP | PNG

```shell
python -W ignore valid_jpeg.py
```

JPEG2000 compression metrics:

```shell
eval psnr: 40.119104 eval ssim: 0.995919 eval bpp: 1.183454 valid time: 0.009126s
eval psnr: 38.168457 eval ssim: 0.993579 eval bpp: 0.951849 valid time: 0.008166s
eval psnr: 35.744500 eval ssim: 0.988959 eval bpp: 0.712808 valid time: 0.007968s
eval psnr: 32.649787 eval ssim: 0.977836 eval bpp: 0.478709 valid time: 0.007724s
eval psnr: 28.855932 eval ssim: 0.943805 eval bpp: 0.283353 valid time: 0.008874s
```

### BPG

> in Linux

```shell
# get source code
wget https://bellard.org/bpg/libbpg-0.9.8.tar.gz
tar -zxvf libbpg-0.9.8.tar.gz
cd libbpg-0.9.8

# get independence & make
sudo apt install yasm
make
sudo make install

# check
bpgenc -h
```

BPG compression metrics:

```shell
```
