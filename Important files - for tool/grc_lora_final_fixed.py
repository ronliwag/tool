import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """Simplified LoRA Linear layer"""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = max(1, min(rank, in_features))  # Ensure rank is valid
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.randn(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(x, self.lora_B @ self.lora_A)

class GroupedResidualConvolution(nn.Module):
    """FIXED Grouped Residual Convolution with proper tensor size handling"""
    def __init__(self, channels, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = max(1, min(groups, channels))  # Ensure groups is valid

        # Calculate padding to maintain same size
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                              dilation=dilation, padding=padding, groups=self.groups)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                              dilation=dilation, padding=padding, groups=self.groups)
        self.norm2 = nn.InstanceNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        # CRITICAL FIX: Ensure residual and output have same size
        if residual.size() != out.size():
            # If sizes don't match, crop or pad residual to match output
            if residual.size(2) > out.size(2):
                # Crop residual if it's longer
                residual = residual[:, :, :out.size(2)]
            else:
                # Pad residual if it's shorter
                padding = out.size(2) - residual.size(2)
                residual = F.pad(residual, (0, padding))
        
        # Residual connection
        out = out + residual
        
        return out

class MultiReceptiveFieldFusion(nn.Module):
    """FIXED Multi-Receptive Field Fusion with proper channel handling"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Use different kernel sizes and dilations for multi-receptive field
        self.convs = nn.ModuleList([
            GroupedResidualConvolution(channels, kernel_size=3, dilation=1),
            GroupedResidualConvolution(channels, kernel_size=5, dilation=2),
            GroupedResidualConvolution(channels, kernel_size=7, dilation=4)
        ])

    def forward(self, x):
        # Verify input channels match expected channels
        if x.size(1) != self.channels:
            print(f"WARNING: Expected {self.channels} channels, got {x.size(1)} channels")
            # If channels don't match, we need to handle this
            if x.size(1) < self.channels:
                # Pad with zeros
                padding = self.channels - x.size(1)
                x = F.pad(x, (0, 0, 0, padding))
            else:
                # Crop excess channels
                x = x[:, :self.channels, :]
        
        # Apply all convolutions and sum them
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        
        # Sum all outputs
        return torch.sum(torch.stack(outputs), dim=0)

