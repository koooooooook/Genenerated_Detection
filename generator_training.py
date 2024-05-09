import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import os
import argparse
import yaml
import wandb
from tqdm import tqdm

from models_progan.progressive_gan import ProgressiveGAN
from models_univdet import get_model
from loss import mse_loss, l1_loss, psnr_loss, ssim_loss, ssim_loss_for_print, lpips_loss
# import torch.nn.functional as F
# import pytorch_msssim
# import lpips

def load_config(config_path):
    """ YAML 파일을 로드하여 파이썬 딕셔너리로 변환합니다. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """ 커맨드 라인 인자를 파싱합니다. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="A file name", default='')
    args = parser.parse_args()
    return args

def main():
    # Debugging?
    wandb_debug = False
    
    # 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="A configuration file", default='')
    parser.add_argument("--training_list", type=str, help="A training dataset list", default='')
    args = parser.parse_args()

    # YAML 파일 로드
    config = load_config(args.config)
    print(config)
    
    ###########################################################################################################
    # exp kinds
    detector_attack = config['exp']['detector_attack']
    lambda_cri = config['exp']['lambda_cri']
    lambda_det = config['exp']['lambda_det']
    
    # wandb
    wandb_project = config['wandb']['wandb_project']
    log_freq = config['wandb']['log_freq']

    # generator
    gen_arch = config['generator']['gen_arch']
    pretrained_generator = config['generator']['pretrained_generator']
    
    # detector
    det_arch = config['detector']['det_arch']
    det_ckpt = config['detector']['det_ckpt']
    
    # hyperparameters
    criterion_name = config['hyperparameters']['criterion']
    num_epochs = config['hyperparameters']['num_epochs']
    batch_size = config['hyperparameters']['batch_size']

    # optimizer
    learning_rate = config['optimizer']['learning_rate']
    learning_rate_noise = config['optimizer']['learning_rate_noise']
    betas = config['optimizer']['betas']
    betas_noise = config['optimizer']['betas_noise']
    
    # scheduler
    T_max = config['scheduler']['T_max']
    T_max_noise = config['scheduler']['T_max_noise']
    eta_min = config['scheduler']['eta_min']
    eta_min_noise = config['scheduler']['eta_min_noise']
    ###########################################################################################################

    for target_image_path in open(args.training_list, 'r').readlines():
        target_image_path = target_image_path.strip()
        train_with  = target_image_path.split('/')[-2]
        train_class = target_image_path.split('/')[-1].split('_')[0]
        train_id    = ''.join(target_image_path.split('/')[-1].split('_')[1:])
        wandb_name  = f'{gen_arch}_{pretrained_generator}_{train_class}_{train_with}_{train_id}'
        
        device = torch.device('cuda')

        # 데이터로더
        # 노이즈 1개 로드
        noise = torch.randn(1, 512).to(device).requires_grad_()

        # 타겟 이미지 1장 로드
        generated_image = np.array(Image.open(target_image_path))
        generated_image = generated_image.transpose(2, 0, 1)
        generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image)) # 0 ~ 1
        generated_image = (generated_image * 2) - 1 # -1 ~ 1
        # for the only 1 target image
        target = torch.Tensor(generated_image).unsqueeze(0).to(device)
        print("target shape : ", target.shape)

        # 모델 설정
        if gen_arch == 'progan':
            # 이 모델은 output 256 x 256 pixel images
            model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                'PGAN', model_name='celebAHQ-256',
                                pretrained=True, useGPU=True)
            if pretrained_generator == 'pretrained':
                generator = model.getOriginalG().to(device)
            elif pretrained_generator == 'scratch':
                generator = model.getNetG().to(device)
        else:
            raise ValueError("generator not implemented")
            
        # 디텍터
        if detector_attack:
            detector = get_model(det_arch)
            state_dict = torch.load(det_ckpt, map_location='cpu')
            detector.fc.load_state_dict(state_dict)
            print ("Detector loaded..")
            detector.eval()
            detector.to(device)

        # 옵티마이저, 스케쥴러
        optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        optimizer_noise = optim.Adam([noise], lr=learning_rate, betas=betas)
        scheduler_noise = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_noise, T_max=T_max_noise, eta_min=eta_min_noise)

        # 로스
        if criterion_name == 'MSE':
            criterion = mse_loss
        elif criterion_name == 'L1':
            criterion = l1_loss
        elif criterion_name == 'PSNR':
            criterion = psnr_loss
        elif criterion_name == 'SSIM':
            criterion = ssim_loss
        elif criterion_name == 'LPIPS':
            criterion = lpips_loss
        else:
            raise ValueError("loss function not implemented")

        # wandb 초기화
        if not wandb_debug:
            wandb.init(project=wandb_project, name=wandb_name)
            # wandb에 모델과 optimizer 등록
            wandb.watch(generator, criterion, log='all', log_freq=log_freq)

        generator.train()
        for epoch in tqdm(range(num_epochs)):
            # forward pass
            output = generator(noise)

            # loss for update
            loss_cri = 0
            loss_cri = criterion(output, target)
            
            # Detector (input 224 224)
            loss_det = 0
            if detector_attack:
                crop_transform = transforms.CenterCrop((224, 224))
                output_cropped = crop_transform(output)
                det_output = detector(output_cropped)
                loss_det = torch.mean(det_output.sigmoid().flatten())
            
            # backward
            optimizer.zero_grad()
            optimizer_noise.zero_grad()
            
            loss = loss_cri * lambda_cri + loss_det * lambda_det
            loss.backward()

            # update
            optimizer.step()
            optimizer_noise.step()
            scheduler.step()
            scheduler_noise.step()
            
            # WANDB logging
            losses_for_wandb = {}        
            losses_for_wandb['MSE']    = mse_loss(output, target).item()
            losses_for_wandb['L1']     = l1_loss(output, target).item()
            losses_for_wandb['PSNR']   = psnr_loss(output, target).item()
            losses_for_wandb['SSIM+']  = ssim_loss_for_print(output, target).item()
            losses_for_wandb['LPIPS']  = lpips_loss(output, target).mean().item()
            if detector_attack:
                losses_for_wandb['DET']    = loss_det.item()
                
            # wandb에 로그 기록
            if not wandb_debug:
                wandb.log({"Epoch":epoch})
                wandb.log({"Epoch":epoch, "Loss": loss.item()})
                wandb.log({"Epoch":epoch, "lr": scheduler.get_last_lr()})
                wandb.log({"Epoch":epoch, "lr": scheduler_noise.get_last_lr()})
                for loss_ in losses_for_wandb:  
                    wandb.log({"Epoch":epoch, loss_: losses_for_wandb[loss_]})
                if (epoch) % 10 == 0:
                    wandb.log({"Generated Images": wandb.Image(output)})

        # Save train_with, train_id, and final loss into a txt file
        os.makedirs(f'experiments/{gen_arch}/pt', exist_ok=True)
        torch.save(generator.state_dict(), f"experiments/{gen_arch}/pt/{train_with}_{train_class}_{train_id}_{epoch+1}.pt")
        with open(f'experiments/{gen_arch}/final_losses.txt', 'a') as f:
            # f.write(f"train with, class, id, final loss\n")
            f.write(f"{train_with} {train_class} {train_id} {loss.item()}\n")
            
        if not wandb_debug:
            wandb.finish()
if __name__ == "__main__":
    main()