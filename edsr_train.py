import os
import torch
import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from data import DIV2K
from model.edsr import edsr
from train import EdsrTrainer
from model import resolve_single
from utils import load_image, plot_sample
from gan_modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)
def get_image_path(dir_name):
    pattern_name = dir_name + '/**/*.[jbptJBPT][pnmiPNMI][gepfGEPF]'
    image_paths=[]
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    pattern_name = dir_name + '/**/*.[jtJT][piPI][efEF][gfGF]'
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    return image_paths
def get_image_name(path):
    i = 0
    name_flag = 0
    image_name = ""
    if(path[0]=='.'): i += 2
    while(i<len(path)):
        if(name_flag!=0):
            image_name += path[i]
        if(path[i]=='/'):
            name_flag = 1
        i += 1
    if('/' in image_name):
        return get_image_name(image_name)
    else:
        return image_name
def divide_image_name(name):
    i = 0
    before = ""
    after = ""
    divide_flag = 0
    while(i<len(name)):
        if(name[i]=='.'):
            divide_flag = 1
        if(divide_flag==0):
            before += name[i]
        else:
            after += name[i]
        i += 1
    return before, after

def create_lr_hr_pair(raw_img, scale=4.):
    lr_h, lr_w = raw_img.shape[0] // scale, raw_img.shape[1] // scale
    hr_h, hr_w = lr_h * scale, lr_w * scale
    hr_img = raw_img[:hr_h, :hr_w, :]
    lr_img = imresize_np(hr_img, 1 / scale)
    return lr_img, hr_img
def train(depth,scale,downgrade):
    weights_dir = f'weights/edsr-{depth}-x{scale}'
    weights_file = os.path.join(weights_dir, 'weights.h5')
    os.makedirs(weights_dir, exist_ok=True)

    div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)# 1-800 images
    div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)# 801-900 images

    train_batch_size = 16
    train_ds = div2k_train.dataset(batch_size=train_batch_size, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    trainer = EdsrTrainer(model=edsr(scale=scale, num_res_blocks=depth), 
                          checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}')

    steps_epoch = int(800/train_batch_size) # 50 steps/epoch
    # Train EDSR model for 300,000 steps and evaluate model
    # every 1000 steps on the first 10 images of the DIV2K
    # validation set.
    trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=6000*steps_epoch, 
                  evaluate_every=500*steps_epoch, 
                  save_best_only=True)

    # Restore from checkpoint with highest PSNR
    trainer.restore()

    # Evaluate model on full validation set
    psnrv = trainer.evaluate(valid_ds)
    print(f'PSNR = {psnrv.numpy():3f}')

    # Save weights
    trainer.model.save_weights(weights_file)

def test(depth,scale,inputs_path,outputs_path,is_validate,weight_file):
    weights_dir = f'weights/edsr-{depth}-x{scale}'
    weights_file = os.path.join(weights_dir, weight_file)

    model = edsr(scale=scale, num_res_blocks=depth)
    model.load_weights(weights_file)
    print("[*] weights loaded: ",weight_file)

    if(is_validate==0):
        print("[*] inferring, scale = ",scale)
        image_path_list = get_image_path(inputs_path)
        total_num = len(image_path_list)
        cnt = 0
        for img_path in image_path_list:
            t_start = time.time()
            cnt += 1
            img_name = get_image_name(img_path)
            print("[*] processing[%d/%d]:%s"%(cnt,total_num,img_name))
            lr_img = load_image(img_path)
            print("   [*] low res image shape = ",lr_img.shape)
            sr_img = resolve_single(model, lr_img)
            # sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_EDSR_4x" + img_name_after
            output_img_path = os.path.join(outputs_path, output_img_name)
            outputs_img = sr_img
            print("output_img_name = ",output_img_name)
            cv2.imwrite(output_img_path, outputs_img)
            t_end = time.time()
            print("   [*] done! Time = %.1fs"%(t_end - t_start))
    else:
        print("   image_name                   PSNR/SSIM        PSNR/SSIM (higher,better)")
        image_path_list = get_image_path(inputs_path)
        for img_path in image_path_list:
            img_name = get_image_name(img_path)
            raw_img = cv2.imread(img_path)
            # Generate low resolution image with original images
            lr_img, hr_img = create_lr_hr_pair(raw_img, 4) # scale=4
            sr_img = resolve_single(model, lr_img)
            bic_img = imresize_np(lr_img, 4).astype(np.uint8)
            str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
            sr_img = sr_img.numpy()
            b = rgb2ycbcr(sr_img)
            print(str_format.format(
                img_name + ' ' * max(0, 20 - len(img_name)),
                calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_ESRGAN_025x_4x" + img_name_after
            output_img_path = os.path.join(outputs_path, output_img_name)
            # outputs_img = np.concatenate((bic_img, sr_img, hr_img), 1)
            outputs_img = sr_img
            # cv2.imwrite(output_img_path, outputs_img) # write super resoltion images
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_ESRGAN_025x" + img_name_after
            output_lr_img_path = os.path.join(outputs_path, output_img_name)
            outputs_lr_img = lr_img
            # cv2.imwrite(output_lr_img_path, outputs_lr_img) # write low resoltion images

if __name__ == '__main__':
    is_validate = 1
    # Number of residual blocks
    depth = 16
    # Super-resolution factor
    scale = 4
    # Downgrade operator
    downgrade = 'bicubic'
    # train(depth,scale,downgrade)
    inputs_path = "./test_data/Set14"
    outputs_path = "./test_data/Set14_025x_4x"
    weight_file = "downloaded_weights"+".h5"
    os.system("mkdir "+inputs_path)
    os.system("mkdir "+outputs_path)
    test(depth,scale,inputs_path,outputs_path,is_validate,weight_file)



