import torch.nn as nn
import torch.nn.functional as F

class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=4):
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding=kernel_size//2, groups=groups)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.groups, c//self.groups, h, w)
        x = self.conv(x.reshape(b, c, h, w))
        return x.view(b, -1, h, w)

class EquivariantAttention(nn.Module):
    def __init__(self, dim, num_heads=4, groups=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.groups = groups
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = GroupConv(dim, dim*3, 1, groups=groups)
        self.proj = GroupConv(dim, dim, 1, groups=groups)
        
        self.norm = nn.GroupNorm(groups, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).chunk(3, dim=1)
        
        q, k, v = map(lambda t: t.view(B, self.num_heads, C // self.num_heads, H, W), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        return x + x

class EquivariantFFN(nn.Module):
    def __init__(self, dim, expansion=4, groups=4):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            GroupConv(dim, hidden_dim, 1, groups=groups),
            nn.GELU(),
            GroupConv(hidden_dim, dim, 1, groups=groups)
        )
        self.norm = nn.GroupNorm(groups, dim)
        
    def forward(self, x):
        return self.net(self.norm(x)) + x

class EquiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, groups=4, mlp_expansion=4):
        super().__init__()
        self.attn = EquivariantAttention(dim, num_heads, groups)
        self.ffn = EquivariantFFN(dim, mlp_expansion, groups)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.norm2 = nn.GroupNorm(groups, dim)
        
    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

class Equiformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=64, 
                 num_blocks=8, num_heads=4, groups=4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, 3, padding=1)
        
        self.blocks = nn.Sequential(*[
            EquiformerBlock(dim, num_heads, groups)
            for _ in range(num_blocks)
        ])

        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, padding=1),  # 2² = 4 channels for PixelShuffle
            nn.PixelShuffle(2),  # 2× upscale
            nn.Conv2d(dim, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        x_low = x
        x = self.embed(x)
        x = self.blocks(x)
        x = self.upsampler(x)
        return x + F.interpolate(x_low, scale_factor=2, mode='bilinear')  
        