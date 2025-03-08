import torch
from torch import nn
from torch.nn import functional as F
from task3A.model import EquiformerBlock,GroupConv

class DyMoE_SR(nn.Module):
    """Equivariant Dynamic Mixture of Experts for Super-Resolution"""
    def __init__(self, in_channels=1, out_channels=1, dim=64, 
                 num_blocks=8, num_heads=4, groups=4, num_experts=4):
        super().__init__()
        self.groups = groups
        self.num_experts = num_experts

        # Initial embedding layer
        self.embed = nn.Conv2d(in_channels, dim, 3, padding=1)
        
        # Expert blocks with shared base architecture
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                EquiformerBlock(dim, num_heads, groups)
                for _ in range(num_blocks)
            ]) for _ in range(num_experts)
        ])
        
        # Equivariant gating network
        self.gate = nn.Sequential(
            GroupConv(dim, num_experts * groups, 1, groups=groups),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_experts * groups, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Equivariant upsampling
        self.upsampler = nn.Sequential(
            GroupConv(dim, dim * 4, 3, groups=groups),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, out_channels, 3,  padding=1)
        )

    def forward(self, x):
        x_low = x
        x = self.embed(x)
        if torch.isnan(x).any():
            raise ValueError("NaN in model input")
        # Compute gating weights
        gate_weights = self.gate(x)  # [B, num_experts]
        x = torch.clamp(x, -1, 1)  # Assuming data range [-1, 1]
        x = torch.nan_to_num(x)
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # Combine expert outputs
        combined = torch.stack(expert_outputs, dim=1)  # [B, E, C, H, W]
        weights = gate_weights.view(-1, self.num_experts, 1, 1, 1)
        x = torch.sum(combined * weights, dim=1)
        
        # Final upsampling
        x = self.upsampler(x)
        return x + F.interpolate(x_low, scale_factor=2, mode='bilinear')
