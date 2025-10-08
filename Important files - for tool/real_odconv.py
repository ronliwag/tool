"""
Real ODConv Implementation for Thesis
Omni-Dimensional Dynamic Convolution (ODConv) as specified in thesis
Replaces static ConvTranspose1D layers in HiFi-GAN generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ODConvTranspose1D(nn.Module):
    """
    Real ODConv Implementation for HiFi-GAN
    Based on Li et al. (2022) - Omni-Dimensional Dynamic Convolution
    Dynamically modulates convolution weights along four dimensions:
    1. Kernel number (spatial/temporal)
    2. Spatial position (time-lag)
    3. Input channels
    4. Output channels
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1, bias=True, reduction=0.25):
        super(ODConvTranspose1D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Calculate reduction factor for attention mechanisms
        self.reduction = max(int(in_channels * reduction), 4)
        
        # Base ConvTranspose1D weights
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size)
        )
        
        # Initialize weights properly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Dynamic attention mechanisms for each dimension
        # 1. Kernel attention - modulates kernel weights across spatial/temporal positions
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, kernel_size, 1),
            nn.Sigmoid()
        )
        
        # 2. Spatial attention - modulates across time-lag positions
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # 3. Input channel attention - modulates input channel importance
        self.input_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 4. Output channel attention - modulates output channel importance
        self.output_channel_attention = nn.Sequential(
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
        """
        Forward pass with omni-dimensional dynamic convolution
        """
        batch_size, in_channels, length = x.size()
        
        # Validate input channels
        if in_channels != self.in_channels:
            if in_channels < self.in_channels:
                # Pad with zeros
                padding = self.in_channels - in_channels
                x = F.pad(x, (0, 0, 0, padding))
            else:
                # Crop excess channels
                x = x[:, :self.in_channels, :]
        
        # Generate dynamic attention weights for all four dimensions
        # 1. Kernel attention: [B, kernel_size, 1]
        kernel_att = self.kernel_attention(x)
        
        # 2. Spatial attention: [B, 1, 1]
        spatial_att = self.spatial_attention(x)
        
        # 3. Input channel attention: [B, in_channels, 1]
        input_ch_att = self.input_channel_attention(x)
        
        # 4. Output channel attention: [B, out_channels, 1]
        output_ch_att = self.output_channel_attention(x)
        
        # Apply dynamic modulation to convolution weights
        # Start with base weight: [in_channels, out_channels//groups, kernel_size]
        dynamic_weight = self.weight.clone()
        
        # Reshape for broadcasting operations
        dynamic_weight = dynamic_weight.unsqueeze(0)  # [1, in_channels, out_channels//groups, kernel_size]
        
        # Apply kernel attention modulation
        # kernel_att: [B, kernel_size, 1] -> [B, 1, 1, kernel_size]
        kernel_att_expanded = kernel_att.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 1, kernel_size]
        dynamic_weight = dynamic_weight * kernel_att_expanded
        
        # Apply spatial attention modulation
        # spatial_att: [B, 1, 1] -> [B, 1, 1, 1]
        spatial_att_expanded = spatial_att.unsqueeze(-1)  # [B, 1, 1, 1]
        dynamic_weight = dynamic_weight * spatial_att_expanded
        
        # Apply input channel attention modulation
        # input_ch_att: [B, in_channels, 1] -> [B, in_channels, 1, 1]
        input_ch_att_expanded = input_ch_att.unsqueeze(-1)  # [B, in_channels, 1, 1]
        dynamic_weight = dynamic_weight * input_ch_att_expanded
        
        # Apply output channel attention modulation
        # output_ch_att: [B, out_channels, 1] -> [B, 1, out_channels//groups, 1]
        output_ch_att_reshaped = output_ch_att.view(batch_size, self.groups, -1).unsqueeze(-1)  # [B, groups, out_channels//groups, 1]
        dynamic_weight = dynamic_weight * output_ch_att_reshaped
        
        # Reshape for convolution operation
        dynamic_weight = dynamic_weight.view(
            batch_size * self.in_channels, 
            self.out_channels // self.groups, 
            self.kernel_size
        )
        
        # Apply dynamic convolution
        # Reshape input for grouped convolution
        x_reshaped = x.view(1, batch_size * self.in_channels, length)
        
        # Apply ConvTranspose1D with dynamic weights
        output = F.conv_transpose1d(
            x_reshaped, 
            dynamic_weight, 
            bias=None,  # Handle bias separately
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=batch_size * self.groups,
            dilation=self.dilation
        )
        
        # Reshape output back to [B, out_channels, length]
        output = output.view(batch_size, self.out_channels, -1)
        
        # Apply bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        
        return output

class ODConv1D(nn.Module):
    """
    ODConv for regular 1D convolution (for MRF blocks)
    Same omni-dimensional dynamic convolution but for standard convolution
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, reduction=0.25):
        super(ODConv1D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Calculate reduction factor
        self.reduction = max(int(in_channels * reduction), 4)
        
        # Base Conv1D weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size)
        )
        
        # Initialize weights properly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Dynamic attention mechanisms (same as ODConvTranspose1D)
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
        
        self.input_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.output_channel_attention = nn.Sequential(
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
        """
        Forward pass with omni-dimensional dynamic convolution
        """
        batch_size, in_channels, length = x.size()
        
        # Validate input channels
        if in_channels != self.in_channels:
            if in_channels < self.in_channels:
                # Pad with zeros
                padding = self.in_channels - in_channels
                x = F.pad(x, (0, 0, 0, padding))
            else:
                # Crop excess channels
                x = x[:, :self.in_channels, :]
        
        # Generate dynamic attention weights
        kernel_att = self.kernel_attention(x)  # [B, kernel_size, 1]
        spatial_att = self.spatial_attention(x)  # [B, 1, 1]
        input_ch_att = self.input_channel_attention(x)  # [B, in_channels, 1]
        output_ch_att = self.output_channel_attention(x)  # [B, out_channels, 1]
        
        # Apply dynamic modulation to weights
        dynamic_weight = self.weight.unsqueeze(0)  # [1, out_channels, in_channels//groups, kernel_size]
        
        # Apply kernel attention
        kernel_att_expanded = kernel_att.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 1, kernel_size]
        dynamic_weight = dynamic_weight * kernel_att_expanded
        
        # Apply spatial attention
        spatial_att_expanded = spatial_att.unsqueeze(-1)  # [B, 1, 1, 1]
        dynamic_weight = dynamic_weight * spatial_att_expanded
        
        # Apply input channel attention
        input_ch_att_expanded = input_ch_att.unsqueeze(-1)  # [B, in_channels, 1, 1]
        dynamic_weight = dynamic_weight * input_ch_att_expanded
        
        # Apply output channel attention
        output_ch_att_expanded = output_ch_att.unsqueeze(-1)  # [B, out_channels, 1, 1]
        dynamic_weight = dynamic_weight * output_ch_att_expanded
        
        # Reshape for convolution
        dynamic_weight = dynamic_weight.view(
            batch_size * self.out_channels, 
            self.in_channels // self.groups, 
            self.kernel_size
        )
        
        # Apply dynamic convolution
        x_reshaped = x.view(1, batch_size * self.in_channels, length)
        
        output = F.conv1d(
            x_reshaped, 
            dynamic_weight, 
            bias=None,
            stride=self.stride, 
            padding=self.padding, 
            groups=batch_size * self.groups,
            dilation=self.dilation
        )
        
        # Reshape output
        output = output.view(batch_size, self.out_channels, -1)
        
        # Apply bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        
        return output

