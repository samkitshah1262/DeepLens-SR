import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 14]): 
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features 
        self.layers = layers
        self.vgg_layers = nn.ModuleList([vgg[i] for i in layers])
        for param in self.parameters():
            param.requires_grad = False 
        self.vgg = vgg.eval()
        

    def forward(self, sr, hr):
        sr = sr.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
        hr = hr.repeat(1, 3, 1, 1)
        
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return F.l1_loss(sr_features, hr_features)
    

class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01, device=device):
        super().__init__()
        self.alpha = alpha  # Weight for mass conservation
        self.beta = beta    # Weight for lensing equation
        self.device = device
        
        # Sobel filters for gradient calculation
        self.sobel_x = torch.tensor([[[[1, 0, -1], 
                                      [2, 0, -2], 
                                      [1, 0, -1]]]], dtype=torch.float32, device=device)
        self.sobel_y = torch.tensor([[[[1, 2, 1], 
                                      [0, 0, 0], 
                                      [-1, -2, -1]]]], dtype=torch.float32, device=device)

    def gradient(self, img):
        """Calculate image gradients using Sobel operators"""
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        return grad_x, grad_y

    def laplacian(self, img):
        """Calculate image Laplacian"""
        kernel = torch.tensor([[[[0, 1, 0], 
                               [1, -4, 1], 
                               [0, 1, 0]]]], dtype=torch.float32, device=self.device)
        return F.conv2d(img, kernel, padding=1)

    def mass_conservation_loss(self, sr, hr):
        """
        Enforce conservation of total flux/mass between 
        LR upscaled and SR reconstruction
        """
        lr_upscaled = F.interpolate(sr, scale_factor=0.5, mode='bicubic')
        return F.mse_loss(lr_upscaled.mean(dim=(2,3)), sr.mean(dim=(2,3)))

    def lensing_equation_loss(self, sr):
        """
        Enforce weak lensing approximation:
        ∇²ψ = 2κ where ψ is lensing potential, κ is convergence
        Approximated using image gradients and Laplacian
        """
        grad_x, grad_y = self.gradient(sr)
        lap = self.laplacian(sr)
        
        # Simulated convergence (κ) from image intensity
        kappa = sr.mean(dim=1, keepdim=True)  # Simplified assumption
        
        # Lensing equation residual
        residual = lap - 2*kappa
        return torch.mean(residual**2)

    def forward(self, sr, hr):
        base_loss = F.l1_loss(sr, hr)
        mass_loss = self.mass_conservation_loss(sr, hr)
        lens_loss = self.lensing_equation_loss(sr)
        
        return base_loss + self.alpha*mass_loss + self.beta*lens_loss

class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        h_variation = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_variation = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (h_variation + w_variation) / (batch_size * channels * height * width)

class HybridLoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.physics = PhysicsConstrainedLoss(device=device)
        self.vgg = VGGPerceptualLoss().to(device)
        self.tv = TVLoss()
    def forward(self, sr, hr):
        return (0.7*self.physics(sr, hr) + 
                0.2*self.vgg(sr, hr) + 
                0.1*self.tv(sr))