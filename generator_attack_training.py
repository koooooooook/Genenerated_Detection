import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

import pytorch_ssim
import lpips

from models_progan.progressive_gan import ProgressiveGAN
from models_univdet import get_model

# yaml 파일로 뺄 것
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_with", type=str, help="Training image: class_id", default='real')
parser.add_argument("--train_id", type=str, help="Training image: class_id", default='cat_0005')
args = parser.parse_args()
train_with = args.train_with
train_id = args.train_id

batch_size = 10
shuffle_batch = False
traindata_size = 100
device = 'cuda'
criterion = 'MSE'
num_epochs = 500
learing_rate = 0.0002
betas = (0.5, 0.999)
data_root = './dataset'
detector_attack = False
pretrained_generator = 'scratch' # pretrained, scratch
# train_with = 'gen'
# train_id = 'cat_0005'
target_image_path = f'{train_with}_{train_id}.png'
# wandb
wandb_project = 'GenDet'
wandb_name = f'progan_{pretrained_generator}_{train_with}_{train_id}'
log_freq=10
# detector
det_arch='CLIP:ViT-L/14' # 'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152','Imagenet:vgg11','Imagenet:vgg19','Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t','Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32','CLIP:RN50','CLIP:RN101','CLIP:RN50x4','CLIP:RN50x16','CLIP:RN50x64','CLIP:ViT-B/32','CLIP:ViT-B/16','CLIP:ViT-L/14','CLIP:ViT-L/14@336px',
det_ckpt='experiments/detector/fc_weights.pth'

# LOSS 구현들, 밖으로 뺼 것
import torch.nn.functional as F
def mse_loss(input, target):
    return F.mse_loss(input, target)

def l1_loss(input, target):
    return F.l1_loss(input, target)

import torch
def psnr_loss(input, target):
    mse = torch.mean((input - target) ** 2)
    return 10 * torch.log10(1 / mse)

import pytorch_msssim
def ssim_loss(input, target):
    return 1 - pytorch_msssim.ssim(input, target)
def ssim_loss_for_print(input, target):
    return pytorch_msssim.ssim(input, target)

import lpips
# Initialize the LPIPS loss function
loss_fn = lpips.LPIPS(net='vgg').to('cuda')
def lpips_loss(input, target):
    return loss_fn(input, target)

# START MAIN
device = torch.device(device)

# 모델
# this model outputs 256 x 256 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                    'PGAN', model_name='celebAHQ-256',
                    pretrained=True, useGPU=True)
if pretrained_generator == 'pretrained':
    generator = model.getOriginalG().to(device)
elif pretrained_generator == 'scratch':
    generator = model.getNetG().to(device)

# 디텍터
if detector_attack:
    detector = get_model(det_arch)
    state_dict = torch.load(det_ckpt, map_location='cpu')
    detector.fc.load_state_dict(state_dict)
    print ("Detector loaded..")
    detector.eval()
    detector.cuda()

# 옵티마이저
optimizer = optim.Adam(generator.parameters(), lr=learing_rate, betas=betas)
# optimizer = model.getOptimizerG()

# 데이터로더
noises, _ = model.buildNoiseData(traindata_size)
# noises = torch.randn(100, 512)
dataloader = torch.utils.data.DataLoader(noises, batch_size=batch_size, shuffle=shuffle_batch)

# geneated image 타겟
img_path = os.path.join(data_root, train_with, target_image_path)
if target_image_path.endswith('.npy'):
    generated_image = np.load(img_path)
elif target_image_path.endswith('.png'):
    generated_image = np.array(Image.open(img_path))
    generated_image = generated_image.transpose(2, 0, 1)
generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
generated_image = (generated_image * 2) - 1
generated_images = []
for _ in range(batch_size):
    if len(generated_image.shape) == 4:
        generated_images.append(torch.Tensor(generated_image))
    elif len(generated_image.shape) == 3:
        generated_images.append(torch.Tensor(generated_image[np.newaxis, :]))
    elif len(generated_image.shape) == 2:
        generated_images.append(torch.Tensor(generated_image[np.newaxis, np.newaxis, :]))
    else:
        raise ValueError("Invalid image shape")
targets = torch.cat(generated_images, dim=0)
print("target shape : ", targets.shape)

# criterion
if criterion == 'MSE':
    criterion = mse_loss
elif criterion == 'L1':
    criterion = l1_loss
elif criterion == 'PSNR':
    criterion = psnr_loss
elif criterion == 'SSIM':
    criterion = ssim_loss
elif criterion == 'LPIPS':
    criterion = lpips_loss
else:
    raise ValueError("Invalid loss function")
if detector_attack:
    lambda_cri = 0.9
    lambda_det = 0.1
else:
    lambda_cri = 1
    lambda_det = 0
    
# wandb 초기화
wandb.init(project=wandb_project, name=wandb_name)
# wandb에 모델과 optimizer 등록
wandb.watch(generator, criterion, log='all', log_freq=log_freq)

generator.train()
for epoch in tqdm(range(num_epochs)):
    for inputs in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # forward pass
        outputs = generator(inputs)

        # loss for update
        loss_cri = 0
        loss_cri = criterion(outputs, targets)
        
        # Detector (input 224 224)
        loss_det = 0
        if detector_attack:
            crop_transform = transforms.CenterCrop((224, 224))
            outputs_cropped = crop_transform(outputs)
            det_output = detector(outputs_cropped)
            loss_det = torch.mean(det_output.sigmoid().flatten())
        
        # backward
        # MODIFICATION: loss_cri * lambda_cri + loss_det * lambda_det
        loss = loss_cri * lambda_cri + loss_det * lambda_det
        loss.backward()

        # update
        optimizer.step()
        
        # loss for analysis
        losses_for_wandb = {}        
        losses_for_wandb['MSE']    = mse_loss(outputs, targets).item()
        losses_for_wandb['L1']     = l1_loss(outputs, targets).item()
        losses_for_wandb['PSNR']   = psnr_loss(outputs, targets).item()
        losses_for_wandb['SSIM+']  = ssim_loss_for_print(outputs, targets).item()
        losses_for_wandb['LPIPS']  = lpips_loss(outputs, targets).mean().item()
        if detector_attack:
            losses_for_wandb['DET']    = loss_det.item()
        
    # wandb에 로그 기록
    wandb.log({"Epoch":epoch, "Loss": loss.item()})
    for loss_ in losses_for_wandb:
        wandb.log({"Epoch":epoch, loss_: losses_for_wandb[loss_]})
    if (epoch) % 10 == 0:
        wandb.log({"Generated Images": wandb.Image(outputs[0])})
    if (epoch) % 200 == 0:
        torch.save(generator.state_dict(), f"experiments/ProGan/progan_{target_image_path}_{epoch+1}.pt")