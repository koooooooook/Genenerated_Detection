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
num_epochs = 1000
learing_rate = 0.0002
betas = (0.5, 0.999)
data_root = './dataset'
lambda_cri = 0.9
lambda_det = 0.1
pretrained_generator = 'pretrained' # pretrained, scratch
train_with = 'gen'
target_image_path = f'{train_with}_dalle_aahxgrlfqt.png' #gen_dalle_aahxgrlfqt {train_with}_face_0000.png
# wandb
wandb_project = 'GenDet'
# wandb_name = f'progan_noloss_{pretrained_generator}_{target_image_path}'
wandb_name = f'progan_detcri_0901_CNNDetection_{target_image_path}'
log_freq=10
# detector
det_arch='CLIP:ViT-L/14' # 'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152','Imagenet:vgg11','Imagenet:vgg19','Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t','Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32','CLIP:RN50','CLIP:RN101','CLIP:RN50x4','CLIP:RN50x16','CLIP:RN50x64','CLIP:ViT-B/32','CLIP:ViT-B/16','CLIP:ViT-L/14','CLIP:ViT-L/14@336px',
det_model='UnivFD' # 'UnivFD', 'CNNDetection' 
crop_images = None # 224?

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

# # 모델
# if pretrained_generator == 'scratch':
#     model = ProgressiveGAN()
#     generator = model.getNetG().to(device)
# elif pretrained_generator == 'pretrained':
#     # this model outputs 256 x 256 pixel images
#     model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                         'PGAN', model_name='celebAHQ-256',
#                         pretrained=True, useGPU=True)
#     generator = model.getNetG().to(device)

# generator = torch.nn.DataParallel(generator)

# 디텍터
if det_model == 'UnivFD':
    from models_univdet import get_model
    det_ckpt = 'experiments/detector/fc_weights.pth'
    
    detector = get_model(det_arch)
    state_dict = torch.load(det_ckpt, map_location='cpu')
    detector.fc.load_state_dict(state_dict)
    print ("Detector loaded..")
    detector.eval()
    detector.cuda()
    detector = torch.nn.DataParallel(detector)
    
elif det_model == 'CNNDetection':
    from models_CNNDetection.demo_output import get_model
    det_ckpt = 'experiments/detector/blur_jpg_prob0.5.pth'

    detector = get_model()
    state_dict = torch.load(det_ckpt, map_location='cpu')
    detector.load_state_dict(state_dict['model'])
    detector.cuda()
    detector.eval()
    detector = torch.nn.DataParallel(detector)

# # 옵티마이저
# optimizer = optim.Adam(generator.parameters(), lr=learing_rate, betas=betas)
# # optimizer = model.getOptimizerG()

# # 데이터로더
# noises, _ = model.buildNoiseData(traindata_size)
# # noises = torch.randn(100, 512)
# dataloader = torch.utils.data.DataLoader(noises, batch_size=batch_size, shuffle=shuffle_batch)

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
# wandb.init(project=wandb_project, name=wandb_name)
# # wandb에 모델과 optimizer 등록
# wandb.watch(generator, criterion, log='all', log_freq=log_freq)

from simba.simba import SimBA

target_image = Image.open(img_path).convert("RGB")
MEAN = {
    "clip":[0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "clip":[0.26862954, 0.26130258, 0.27577711]
}
transform_univfd = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] ),
        ])
target_image = transform_univfd(target_image)

attacker = SimBA(detector, target_image, targets.shape)
labels = torch.ones(1)
import time
start_time = time.time()
x = attacker.simba_single(target_image, labels) # [1, 3, 224, 224], [1.]
print(time.time() - start_time)

def tensor_to_pil(tensor):
    # Denormalize the image
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    tensor = tensor.clone()  # 복사본을 만들어 원본 데이터를 보존
    # tensor = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)  # 값의 범위를 0과 1 사이로 조정
    # PIL 이미지로 변환
    image = transforms.ToPILImage()(tensor)
    return image

# target_image를 PIL 이미지로 변환
pil_image = tensor_to_pil(x)

# 이미지 파일로 저장
pil_image.save('saved_image_dalle.png')