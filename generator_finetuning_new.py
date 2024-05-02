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

# yaml 파일로 뺄 것
batch_size = 10
shuffle_batch = False
traindata_size = 100
device = 'cuda'
criterion = 'MSE'
num_epochs = 500
learing_rate = 0.0002
betas = (0.5, 0.999)
data_root = './dataset/sy_test/'
lambda_cri = 1
lambda_det = 0
pretrained_generator = 'pretrained' # pretrained, scratch
train_with = 'gen'
# target_image_path = f'{train_with}_train_0000.png'
# wandb
wandb_project = 'source'
# wandb_name = f'progan_noloss_{pretrained_generator}_{target_image_path}'
# wandb_name = f'progan_detach_0109_UnivFD_{target_image_path}'
log_freq=10
# detector
det_arch='CLIP:ViT-L/14' # 'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152','Imagenet:vgg11','Imagenet:vgg19','Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t','Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32','CLIP:RN50','CLIP:RN101','CLIP:RN50x4','CLIP:RN50x16','CLIP:RN50x64','CLIP:ViT-B/32','CLIP:ViT-B/16','CLIP:ViT-L/14','CLIP:ViT-L/14@336px',
det_model='UnivFD' # 'UnivFD', 'CNNDetection' 
crop_images = None # 224?

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--source', type=str, default='./dataset/sy_test/', help='')
parser.add_argument('--ox', type=str, default='1_fake', help='')
parser.add_argument('--img_name', type=str, default='', help='')
opt = parser.parse_args()

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

generator = torch.nn.DataParallel(generator)

# # 디텍터
# if det_model == 'UnivFD':
#     from models_univdet import get_model
#     det_ckpt = 'experiments/detector/fc_weights.pth'
    
#     detector = get_model(det_arch)
#     state_dict = torch.load(det_ckpt, map_location='cpu')
#     detector.fc.load_state_dict(state_dict)
#     print ("Detector loaded..")
#     detector.eval()
#     detector.cuda()
#     detector = torch.nn.DataParallel(detector)
    
# elif det_model == 'CNNDetection':
#     from models_CNNDetection.demo_output import get_model
#     det_ckpt = 'experiments/detector/blur_jpg_prob0.5.pth'

#     detector = get_model()
#     state_dict = torch.load(det_ckpt, map_location='cpu')
#     detector.load_state_dict(state_dict['model'])
#     detector.cuda()
#     detector.eval()
#     detector = torch.nn.DataParallel(detector)

# 옵티마이저
optimizer = optim.Adam(generator.parameters(), lr=learing_rate, betas=betas)
# optimizer = model.getOptimizerG()

# 데이터로더
noises, _ = model.buildNoiseData(traindata_size)
# noises = torch.randn(100, 512)
dataloader = torch.utils.data.DataLoader(noises, batch_size=batch_size, shuffle=shuffle_batch)

# geneated image 타겟
root = './dataset/sy_test_resized/'
data_dir = os.path.join(root, opt.source, opt.ox, opt.img_name)

img_name = opt.source + '_' + opt.ox + '_' + opt.img_name
wandb_name = f'UnivFD_{img_name}'
target_image_path = data_dir

if target_image_path.endswith('.npy'):
    generated_image = np.load(data_dir)
elif target_image_path.endswith('.png') or target_image_path.endswith('.jpeg'):
    generated_image = np.array(Image.open(data_dir))
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
wandb.init(project=wandb_project, entity="junwon-ko", name=wandb_name)
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
        
        # # Detector (input 224 224)
        # loss_det = 0
        # if det_model == 'UnivFD':
        #     crop_transform = transforms.CenterCrop((224, 224))
        #     outputs_cropped = crop_transform(outputs)
        #     det_output = detector(outputs_cropped)
        # elif det_model == 'CNNDetection':
        #     trans_init=[]
        #     if(crop_images is not None):
        #         trans_init = [transforms.CenterCrop(crop_images),]
        #         print('Cropping to [%i]'%crop_images)
        #     else:
        #         # print('Not cropping')
        #         trans = transforms.Compose(trans_init + [
        #             # transforms.ToTensor(),
        #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         ])
            
        #     # img = trans(Image.open(img_path).convert('RGB'))
        #     img = trans(outputs) # 이거 빼야하나 고민
        #     # img = trans(targets)
            
        #     # img = img.unsqueeze(0)
        #     outputs = img.cuda()
        #     det_output = detector(outputs)
        #     # import pdb; pdb.set_trace()
        # loss_det = torch.mean(det_output.sigmoid().flatten())
        # # print(loss_det)
        # new_loss_det = loss_det.detach()
        # BCEcri = torch.nn.BCELoss()
        # target = torch.zeros_like(new_loss_det)
        # new_cri = BCEcri(new_loss_det, target)
        
        # backward
        # loss = loss_cri * lambda_cri + new_loss_det * lambda_det
        loss = loss_cri * lambda_cri
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
        # losses_for_wandb['DET']    = loss_det.item()
        
    # # wandb에 로그 기록
    # wandb.log({"Loss": loss.item()})
    # wandb.log(losses_for_wandb)
    # if (epoch) % 10 == 0:
    #     wandb.log({"Generated Images": wandb.Image(outputs[0])})
    # if (epoch) % 200 == 0:
    #     torch.save(generator.state_dict(), f"experiments/ProGan/progan_{img_name}_{epoch+1}.pt")
        
        
    # wandb에 로그 기록
    wandb.log({"Epoch":epoch, "Loss": loss.item()})
    for loss_ in losses_for_wandb:
        wandb.log({"Epoch":epoch, loss_: losses_for_wandb[loss_]})
    if (epoch) % 10 == 0:
        wandb.log({"Generated Images": wandb.Image(outputs[0])})
    if (epoch) % 200 == 0:
        torch.save(generator.state_dict(), f"experiments/ProGan/progan_{img_name}_{epoch+1}.pt")