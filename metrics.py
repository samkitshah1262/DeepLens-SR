from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from torch.nn import functional as F

class Metrics:
    @staticmethod
    def mse(pred, target):
        return F.mse_loss(pred, target)
    
    @staticmethod
    def residual_map(sr, hr):
        return (sr - hr).abs()

def calculate_psnr(img1, img2):
    """Calculate PSNR between two image tensors"""
    img1 = img1.detach().cpu().numpy().transpose(0,2,3,1)
    img2 = img2.detach().cpu().numpy().transpose(0,2,3,1)
    return np.mean([psnr(im1, im2, data_range=1.0) 
                   for im1, im2 in zip(img1, img2)])

def calculate_ssim(img1, img2):
    """Calculate SSIM between two grayscale image tensors"""
    # Convert tensors to numpy arrays and remove channel dimension
    img1 = img1.detach().cpu().numpy().squeeze(1)  # [B, 1, H, W] → [B, H, W]
    img2 = img2.detach().cpu().numpy().squeeze(1)
    
    ssim_values = []
    for im1, im2 in zip(img1, img2):
        # Ensure minimum size for SSIM computation
        min_side = min(im1.shape)
        win_size = 7 if min_side >= 7 else min_side - (min_side % 2 == 0)
        win_size = max(3, win_size)  # Ensure window size ≥ 3
        
        ssim_val = ssim(
            im1, im2,
            data_range=1.0,
            win_size=win_size,
            channel_axis=None  # Explicitly specify grayscale
        )
        ssim_values.append(ssim_val)
        
    return np.mean(ssim_values)
