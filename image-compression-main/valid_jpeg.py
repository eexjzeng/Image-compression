import time
import os

import numpy as np
import cv2
import torch
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

import utils

"""
about how to use there params
see : https://docs.opencv.org/4.5.1/d8/d6a/group__imgcodecs__flags.html
find: 'cv::ImwriteFlags'
"""
file_name = {
    "JPEG": ".jpeg",
    "JPEG2000": "jp2",
    "PNG": ".png",
    "WEBP": ".webp"
}
compress = {
    "JPEG": cv2.IMWRITE_JPEG_QUALITY,                    # JPEG, 0-100
    "JPEG2000": cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,  # JPEG2000, 0-1000
    "PNG": cv2.IMWRITE_PNG_COMPRESSION,                  # PNG, 0-9
    "WEBP": cv2.IMWRITE_WEBP_QUALITY,                    # WEBP, 1-100
}

# dataset init, train file need .tfrecord
description = {
    "image": "byte",
    "size": "int",
}
valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

epoch_eval_pnsr = utils.AverageMeter()
epoch_eval_ssim = utils.AverageMeter()
epoch_eval_bpp = utils.AverageMeter()

t0 = time.time()
for cnt, record in enumerate(valid_dataloader):
    input = record["image"].reshape(
        record["size"][0],
        record["size"][0],
    )
    input_numpy = input.numpy()
    input_numpy.tofile("in.bin")

    img_encode = np.array(cv2.imencode(
        ".jp2", input_numpy,
        [compress["JPEG2000"], 35]
    )[1])
    img_encode.tofile("img.bin")

    img_bytes = img_encode.nbytes

    output = np.fromfile("img.bin", np.uint8)
    img_decode = cv2.imdecode(output, cv2.IMREAD_GRAYSCALE)

    torch_img = torch.from_numpy(img_decode).float()
    torch_out = input.float()
    torch_img = torch_img.unsqueeze(0).unsqueeze(0) / 255
    torch_out = torch_out.unsqueeze(0).unsqueeze(0) / 255

    epoch_eval_pnsr.update(utils.calc_psnr(torch_img, torch_out), 1)
    epoch_eval_ssim.update(utils.calc_msssim(torch_img, torch_out), 1)
    epoch_eval_bpp.update(img_bytes * 8 / input_numpy.nbytes, 1)

    if cnt == 100:
        break

t1 = time.time()

print('eval psnr: {:.6f} eval ssim: {:.6f} eval bpp: {:.6f} valid time: {:.6f}s'.format(
    epoch_eval_pnsr.avg, epoch_eval_ssim.avg, epoch_eval_bpp.avg, (t1-t0) / cnt
))
