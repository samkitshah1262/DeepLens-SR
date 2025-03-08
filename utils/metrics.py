from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from torch.nn import functional as F
import torch

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

def calculate_ssim(img1, img2, data_range=1.0, eps=1e-8):
    """Numerically stable SSIM calculation for grayscale images"""
    # Input validation and clamping
    img1 = torch.clamp(img1, -data_range, data_range).detach()
    img2 = torch.clamp(img2, -data_range, data_range).detach()
    
    # Convert to numpy with double precision
    img1_np = img1.cpu().numpy().squeeze(1).astype(np.float64)
    img2_np = img2.cpu().numpy().squeeze(1).astype(np.float64)
    
    ssim_values = []
    
    for i in range(img1_np.shape[0]):
        im1 = img1_np[i]
        im2 = img2_np[i]
        
        try:
            # Dynamic window size selection with safety checks
            min_dim = min(im1.shape)
            win_size = min(7, min_dim - 1 if min_dim % 2 == 0 else min_dim)
            win_size = max(3, win_size)
            
            # Calculate SSIM with stability parameters
            ssim_val = ssim(
                im1, im2,
                data_range=data_range,
                win_size=win_size,
                channel_axis=None,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False
            )
            
            # Handle potential NaN/Inf
            if np.isnan(ssim_val) or np.isinf(ssim_val):
                raise ValueError("Invalid SSIM value")
                
        except Exception as e:
            print(f"SSIM calculation failed for image {i}: {str(e)}")
            ssim_val = -1  # Sentinel value for failures
            
        ssim_values.append(ssim_val)
    
    # Filter out failed calculations
    valid_ssim = [v for v in ssim_values if v >= 0]
    
    return np.mean(valid_ssim) if valid_ssim else 0.0