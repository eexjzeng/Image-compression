import time
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

import trainers
import utils


models_path = "./model"
for name in os.listdir(models_path):
    model_name = os.path.join(models_path, name)
    # dataset init, train file need .tfrecord
    description = {
        "image": "byte",
        "size": "int",
    }
    valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
    valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

    # models init
    model = trainers.OctaveEfficientDeepImageCompression().to("cuda")
    model_params = torch.load(model_name)
    model.load_state_dict(model_params)

    model.eval()

    with torch.no_grad():
        for p1 in [0, 0.01, 0.02, 0.03, 0.04]:
            t0 = time.clock()
            epoch_eval_pnsr = utils.AverageMeter()
            epoch_eval_ssim = utils.AverageMeter()
            epoch_eval_bpp = utils.AverageMeter()
            cnt = 0
            for record in valid_dataloader:
                cnt += 1
                input = record["image"].reshape(
                    1,
                    1,
                    record["size"][0],
                    record["size"][0],
                ).float().to("cuda") / 255

                output_high, bpp_feature_val, bpp_z_val, _ = model(input, p1)
                epoch_eval_pnsr.update(utils.calc_psnr(output_high, input), 1)
                epoch_eval_ssim.update(utils.calc_msssim(output_high, input), 1)
                epoch_eval_bpp.update(bpp_feature_val + bpp_z_val, 1)

                # if cnt == 1:
                #     plt.imshow(output_high.cpu().squeeze().numpy(), cmap="gray")
                #     plt.show()
                #     break
            t1 = time.clock()
            print('model name: {} eval psnr: {:.6f} eval ssim: {:.6f} eval bpp: {:.6f} p: {:.2f} valid_time: {:.6f}s'.format(
        model_name, epoch_eval_pnsr.avg, epoch_eval_ssim.avg, epoch_eval_bpp.avg, p1, (t1-t0) / cnt
    ))
    print()
