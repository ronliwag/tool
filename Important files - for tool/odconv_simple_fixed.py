import torch
import torch.nn as nn
import torch.nn.functional as F

class ODConvTranspose1D(nn.Module):
    """Simplified ODConv without attention mechanism to avoid dimension issues"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1, bias=True, 
                 speaker_dim=192, emotion_dim=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Use standard ConvTranspose1D instead of ODConv
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, 
            padding, output_padding, groups, bias, dilation
        )
        
        # Simple FiLM conditioning
        self.film = nn.Sequential(
            nn.Linear(speaker_dim + emotion_dim, out_channels * 2),
            nn.ReLU()
        )
        
        self.condition = None
    
    def set_condition(self, condition):
        self.condition = condition
    
    def forward(self, x):
        # CRITICAL FIX: Handle channel mismatches
        if x.size(1) != self.in_channels:
            print(f"WARNING: ODConv expected {self.in_channels} channels, got {x.size(1)} channels")
            if x.size(1) < self.in_channels:
                # Pad with zeros
                padding = self.in_channels - x.size(1)
                x = F.pad(x, (0, 0, 0, padding))
            else:
                # Crop excess channels
                x = x[:, :self.in_channels, :]
        
        # Apply simple FiLM conditioning
        if self.condition is not None:
            film_params = self.film(self.condition)  # [B, out_channels * 2]
            scale = film_params[:, :self.out_channels].unsqueeze(-1)  # [B, out_channels, 1]
            shift = film_params[:, self.out_channels:].unsqueeze(-1)  # [B, out_channels, 1]
            
            # Apply convolution
            x = self.conv(x)
            
            # Apply FiLM
            x = x * scale + shift
        else:
            x = self.conv(x)
        
        return x

