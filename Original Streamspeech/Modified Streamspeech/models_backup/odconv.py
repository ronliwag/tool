"""
Omni-Dimensional Dynamic Convolution (ODConv) Implementation
Replaces static ConvTranspose1D layers in HiFi-GAN generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ODConv(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution (ODConv)
    Dynamically adjusts convolutional kernel behavior along all four dimensions:
    - kernel (spatial/temporal)
    - spatial/time-lag
    - in-channel
    - out-channel
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, reduction=0.25):
        super(ODConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Calculate reduction factor
        self.reduction = max(int(in_channels * reduction), 4)
        
        # Standard convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        
        # Dynamic attention mechanisms for each dimension
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, kernel_size, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.output_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, out_channels, 1),
            nn.Sigmoid()
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        batch_size, in_channels, length = x.size()
        
        # Generate dynamic attention weights
        kernel_att = self.kernel_attention(x)  # [B, kernel_size, 1]
        spatial_att = self.spatial_attention(x)  # [B, 1, 1]
        channel_att = self.channel_attention(x)  # [B, in_channels, 1]
        output_att = self.output_attention(x)  # [B, out_channels, 1]
        
        # Apply dynamic modulation to weights
        # Reshape for broadcasting
        weight = self.weight.unsqueeze(0)  # [1, out_channels, in_channels, kernel_size]
        
        # CRITICAL FIX: Proper tensor broadcasting to avoid dimension mismatch
        # kernel_att: [B, kernel_size, 1] -> [B, 1, 1, kernel_size]
        kernel_att_expanded = kernel_att.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 1, kernel_size]
        weight = weight * kernel_att_expanded
        
        # spatial_att: [B, 1, 1] -> [B, 1, 1, 1]
        spatial_att_expanded = spatial_att.unsqueeze(-1)  # [B, 1, 1, 1]
        weight = weight * spatial_att_expanded
        
        # channel_att: [B, in_channels, 1] -> [B, 1, in_channels, 1]
        channel_att_expanded = channel_att.permute(0, 2, 1).unsqueeze(1)  # [B, 1, in_channels, 1]
        weight = weight * channel_att_expanded
        
        # output_att: [B, out_channels, 1] -> [B, out_channels, 1, 1]
        output_att_expanded = output_att.unsqueeze(-1)  # [B, out_channels, 1, 1]
        weight = weight * output_att_expanded
        
        # Perform convolution with dynamic weights
        output = F.conv_transpose1d(
            x, weight.view(-1, in_channels // self.groups, self.kernel_size),
            bias=self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        return output

class ODConvTranspose1D(ODConv):
    """
    ODConv wrapper for ConvTranspose1D replacement
    Maintains same interface as ConvTranspose1D
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=True, dilation=1):
        super(ODConvTranspose1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias
        )
        self.output_padding = output_padding
        
        # Alternative: Nearest upsample + conv (less artifacts) - DISABLED to fix buzzing
        self.use_nearest_upsample = False  # DISABLED: Use proper ODConv to fix buzzing artifacts
        self.nearest_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        if self.use_nearest_upsample:
            # CRITICAL FIX: Use proper upsampling for ConvTranspose1D replacement
            # Calculate target size based on stride
            target_size = x.size(2) * self.stride
            x_upsampled = F.interpolate(x, size=target_size, mode='linear', align_corners=False)
            output = self.nearest_conv(x_upsampled)
        else:
            # Use ODConv (original approach) with buzzing reduction
            output = super().forward(x)
            
            # BUZZING FIX: Apply gentle smoothing to reduce artifacts
            if hasattr(self, 'smoothing_enabled') and self.smoothing_enabled:
                from scipy.ndimage import gaussian_filter1d
                import numpy as np
                # Convert to numpy, apply smoothing, convert back
                output_np = output.detach().cpu().numpy()
                smoothed_np = gaussian_filter1d(output_np, sigma=0.3, axis=2)
                output = torch.from_numpy(smoothed_np).to(output.device)
        
        # Apply output padding if specified
        if self.output_padding > 0:
            output = F.pad(output, (0, self.output_padding))
        
        return output
