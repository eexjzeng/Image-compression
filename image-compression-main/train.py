import os
import math
import random

import cv2
import torch
import torch.nn as nn
from torch import optim
from timm.models.layers import trunc_normal_

from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

from tqdm import tqdm

import trainers
import config
import utils
import criterions


opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')
torch.backends.cudnn.benchmark=True

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "image": "byte",
    "size": "int",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description, shuffle_queue_size=2048)
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
length = 209260   # NOTE: ans from make_dataset.py

valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1, drop_last=True)

# models init
model = trainers.OctaveEfficientDeepImageCompression(out_channels_n=192, out_channels_m=192, embed_dim=192).to(device)
if os.path.exists("pretrain_oic.pth"):
    params = torch.load("pretrain_oic.pth")
    model.load_state_dict(params)
else:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# NOTE: after training by only loss_mse
# for param in model.decoder.parameters():
#     param.requires_grad = False

# criterion init
criterion = torch.nn.MSELoss()

# optim and scheduler init
model_optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
# NOTE: after training by only loss_mse
# model_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

scale_table = [0.0]

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()
    # epoch_losses_feature = utils.AverageMeter()
    # epoch_losses_compress = utils.AverageMeter()
    epoch_psnr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()
    epoch_bpp_feature = utils.AverageMeter()
    # epoch_bpp_z = utils.AverageMeter()

    with tqdm(total=((length - length % opt.batch_size) * len(scale_table))) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for cnt, record in enumerate(train_dataloader):
            inputs = record["image"].reshape(
                opt.batch_size,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to("cuda") / 255

            for p in scale_table:
                model_optimizer.zero_grad()

                outputs_high, bpp_feature, bpp_z, loss_feature = model(inputs, p)
                loss_mse = criterion(inputs, outputs_high)

                if p == 0:
                    loss = loss_mse * 4096 + (bpp_feature + bpp_z)
                else:
                    loss = loss_mse * 255 ** 2 * p + (bpp_feature + bpp_z) + loss_feature / (2 * p)
                loss.backward()

                utils.clip_gradient(model_optimizer, 5)

                epoch_losses.update(loss_mse.item(), opt.batch_size)
                # epoch_losses_feature.update(loss_feature.item(), opt.batch_size)
                # epoch_losses_compress.update(loss_compress.item(), opt.batch_size)
                epoch_bpp_feature.update(bpp_feature.item(), opt.batch_size)
                # epoch_bpp_z.update(bpp_z.item(), opt.batch_size)
                epoch_ssim.update(utils.calc_ssim(inputs, outputs_high).item(), opt.batch_size)
                epoch_psnr.update(utils.calc_psnr(inputs, outputs_high), opt.batch_size)

                t.set_postfix(
                    loss_mse='{:.6f}'.format(epoch_losses.avg),
                    # loss_fea='{:.4f}'.format(epoch_losses_feature.avg),
                    bpp='{:.6f}'.format(epoch_bpp_feature.avg),
                    psnr='{:.4f}'.format(epoch_psnr.avg),
                    ssim='{:.4f}'.format(epoch_ssim.avg),
                )
                t.update(opt.batch_size)
            model_optimizer.step()
            torch.cuda.empty_cache()

    model_scheduler.step()

    model.eval()

    with torch.no_grad():
        epoch_eval_pnsr = utils.AverageMeter()
        epoch_eval_ssim = utils.AverageMeter()
        epoch_eval_bpp = utils.AverageMeter()

        for cnt, record in enumerate(valid_dataloader):
            input = record["image"].reshape(
                1,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to("cuda") / 255

            output_high, bpp_feature_val, bpp_z_val, _ = model(input, 0)
            epoch_eval_pnsr.update(utils.calc_psnr(output_high, input), 1)
            epoch_eval_ssim.update(utils.calc_ssim(output_high, input).item(), 1)
            epoch_eval_bpp.update(bpp_feature_val + bpp_z_val, 1)
            
            if cnt == 100:
                break

        print('eval psnr: {:.6f} eval ssim: {:.6f} eval bpp: {:.4f}'.format(epoch_eval_pnsr.avg, epoch_eval_ssim.avg, epoch_eval_bpp.avg))

    torch.save(model.state_dict(), "model/oic/epoch_{}.pth".format(epoch+1))
    # print()
    torch.cuda.empty_cache()
