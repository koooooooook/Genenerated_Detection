import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from models_progan.progressive_gan import ProgressiveGAN

import pytorch_ssim
import lpips

# yaml 파일로 뺄 것
batch_size = 10
shuffle_batch = False
traindata_size = 100
device = 'cuda'
criterion = 'MSE'
num_epochs = 1000
learing_rate = 0.0002
betas = (0.5, 0.999)
data_root = './dataset'
pretrained_generator = 'pretrained' # pretrained, scratch
train_with = 'gen'
target_image_path = f'{train_with}_dalle_aahxgrlfqt.png'
# wandb
wandb_project = 'GenDet'
wandb_name = f'progan_{pretrained_generator}_{target_image_path}'
log_freq=10

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
if pretrained_generator == 'scratch':
    model = ProgressiveGAN()
    generator = model.getNetG().to(device)
elif pretrained_generator == 'pretrained':
    # this model outputs 256 x 256 pixel images
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-256',
                        pretrained=True, useGPU=True)
    generator = model.getNetG().to(device)

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
        loss = criterion(outputs, targets)
        
        # loss for analysis
        losses_for_wandb = {}
        losses_for_wandb['MSE']    = mse_loss(outputs, targets).item()
        losses_for_wandb['L1']     = l1_loss(outputs, targets).item()
        losses_for_wandb['PSNR']   = psnr_loss(outputs, targets).item()
        losses_for_wandb['SSIM']   = ssim_loss_for_print(outputs, targets).item()
        losses_for_wandb['LPIPS']  = lpips_loss(outputs, targets).mean().item()
                
        # backward
        loss.backward()

        # update
        optimizer.step()
        
    # wandb에 로그 기록
    wandb.log({"Loss": loss.item()})
    wandb.log(losses_for_wandb)
    if (epoch) % 10 == 0:
        wandb.log({"Generated Images": wandb.Image(outputs[0])})
    if (epoch) % 200 == 0:
        torch.save(generator.state_dict(), f"experiments/ProGan/progan_{target_image_path}_{epoch+1}.pt")