from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def calculate_psnr(img1, img2):
    """Calculate PSNR between two image tensors"""
    img1 = img1.detach().cpu().numpy().transpose(0,2,3,1)
    img2 = img2.detach().cpu().numpy().transpose(0,2,3,1)
    return np.mean([psnr(im1, im2, data_range=1.0) 
                   for im1, im2 in zip(img1, img2)])

def calculate_ssim(img1, img2):
    """Calculate SSIM between two image tensors"""
    img1 = img1.detach().cpu().numpy().transpose(0,2,3,1)
    img2 = img2.detach().cpu().numpy().transpose(0,2,3,1)
    return np.mean([ssim(im1, im2, multichannel=True, data_range=1.0)
                   for im1, im2 in zip(img1, img2)])