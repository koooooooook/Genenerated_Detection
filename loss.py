import torch
import torch.nn.functional as F
import pytorch_msssim
# import pytorch_ssim
import lpips

def mse_loss(input, target):
    return F.mse_loss(input, target)

def l1_loss(input, target):
    return F.l1_loss(input, target)

def psnr_loss(input, target):
    mse = torch.mean((input - target) ** 2)
    return 10 * torch.log10(1 / mse)

def ssim_loss(input, target):
    return 1 - pytorch_msssim.ssim(input, target)

def ssim_loss_for_print(input, target):
    return pytorch_msssim.ssim(input, target)

# Initialize the LPIPS loss function
loss_fn = lpips.LPIPS(net='vgg').to('cuda')
def lpips_loss(input, target):
    return loss_fn(input, target)