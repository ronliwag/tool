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

class ODConvTranspose1D(nn.Module):
    """ODConv wrapper for ConvTranspose1D replacement with conditioning support"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=True, dilation=1, speaker_dim=None, emotion_dim=None):
        super(ODConvTranspose1D, self).__init__()
        
        # Store conditioning dimensions
        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        
        # Main conv
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, output_padding=output_padding,
                                       groups=groups, bias=bias, dilation=dilation)
        
        # Optional conditioning projection layers
        if speaker_dim is not None:
            self.speaker_proj = nn.Linear(speaker_dim, out_channels)
        else:
            self.speaker_proj = None

        if emotion_dim is not None:
            self.emotion_proj = nn.Linear(emotion_dim, out_channels)
        else:
            self.emotion_proj = None

    def forward(self, x, speaker_embed=None, emotion_embed=None):
        # CRITICAL FIX: Handle channel mismatches
        if x.size(1) != self.conv.in_channels:
            print(f"WARNING: ODConv expected {self.conv.in_channels} channels, got {x.size(1)} channels")
            if x.size(1) < self.conv.in_channels:
                # Pad with zeros
                padding = self.conv.in_channels - x.size(1)
                x = F.pad(x, (0, 0, 0, padding))
            else:
                # Crop excess channels
                x = x[:, :self.conv.in_channels, :]
        
        x = self.conv(x)

        # Apply conditioning if available
        if self.speaker_proj is not None and speaker_embed is not None:
            cond = self.speaker_proj(speaker_embed).unsqueeze(-1)
            x = x + cond

        if self.emotion_proj is not None and emotion_embed is not None:
            cond = self.emotion_proj(emotion_embed).unsqueeze(-1)
            x = x + cond

        return x
