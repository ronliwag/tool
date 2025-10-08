"""
Grouped Residual Convolution (GRC) with Low-Rank Adaptation (LoRA)
Replaces original Residual Blocks in Multi-Receptive Field Fusion (MRF) module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for efficient fine-tuning
    """
    
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        # CRITICAL FIX: Ensure rank doesn't exceed input features and is at least 1
        self.rank = max(1, min(rank, in_features))  # FIXED: rank must be at least 1
        self.alpha = alpha
        
        # Low-rank matrices - ensure minimum dimensions
        if self.rank > 0 and in_features > 0:
            self.lora_A = nn.Parameter(torch.randn(self.rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        else:
            # Fallback: identity mapping
            self.lora_A = nn.Parameter(torch.zeros(1, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, 1))
            self.rank = 1
        
        # Scaling factor
        self.scaling = alpha / self.rank
    
    def forward(self, x):
        # CRITICAL FIX: Handle different input shapes properly
        if x.dim() == 3:  # [B, T, C] format
            batch_size, time_steps, channels = x.size()
            x_flat = x.view(-1, channels)  # [B*T, C]
        else:  # [B, C] format
            x_flat = x
        
        # LoRA adaptation: x @ (A @ B).T
        lora_result = F.linear(x_flat, self.lora_A.T) @ self.lora_B.T
        lora_result = lora_result * self.scaling
        
        # Reshape back to original format
        if x.dim() == 3:
            lora_result = lora_result.view(batch_size, time_steps, channels)
        
        return lora_result

class GroupedResidualConvolution(nn.Module):
    """
    Grouped Residual Convolution (GRC) with LoRA adaptation
    Replaces standard Residual Blocks in MRF module
    """
    
    def __init__(self, channels, kernel_size=3, dilation=1, groups=4, lora_rank=4):
        super(GroupedResidualConvolution, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # CRITICAL FIX: Ensure groups doesn't exceed channels and is at least 1
        self.groups = max(1, min(groups, channels))  # FIXED: groups must be at least 1 and cannot exceed channels
        
        # Grouped convolutions
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=(kernel_size-1)//2, dilation=dilation, groups=self.groups)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=(kernel_size-1)//2, dilation=dilation, groups=self.groups)
        
        # Batch normalization
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
        
        # LoRA adaptation layers
        self.lora_conv1 = LoRALinear(channels, channels, rank=lora_rank)
        self.lora_conv2 = LoRALinear(channels, channels, rank=lora_rank)
        
        # Channel attention for grouped processing
        # CRITICAL FIX: Ensure minimum channel dimensions
        bottleneck_channels = max(1, channels // 4)  # FIXED: Ensure at least 1 channel
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = x
        
        # First grouped convolution with LoRA
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Apply LoRA adaptation
        batch_size, channels, time_steps = out.size()
        # CRITICAL FIX: Proper tensor reshaping for LoRA
        out_flat = out.transpose(1, 2)  # [B, T, C]
        lora_out = self.lora_conv1(out_flat)  # [B, T, C]
        lora_out = lora_out.transpose(1, 2)  # [B, C, T]
        out = out + lora_out
        
        # Second grouped convolution with LoRA
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply LoRA adaptation
        # CRITICAL FIX: Proper tensor reshaping for LoRA
        out_flat = out.transpose(1, 2)  # [B, T, C]
        lora_out = self.lora_conv2(out_flat)  # [B, T, C]
        lora_out = lora_out.transpose(1, 2)  # [B, C, T]
        out = out + lora_out
        
        # Channel attention
        channel_att = self.channel_attention(out)
        out = out * channel_att
        
        # Temporal attention
        temporal_att = self.temporal_attention(out)
        out = out * temporal_att
        
        # CRITICAL FIX: Ensure residual connection has same size
        if residual.size() != out.size():
            # Pad or crop residual to match output size
            if residual.size(2) > out.size(2):
                residual = residual[:, :, :out.size(2)]
            else:
                padding = out.size(2) - residual.size(2)
                residual = F.pad(residual, (0, padding))
        
        # Residual connection
        out = out + residual
        
        # BUZZING FIX: Apply gentle normalization to prevent artifacts
        out = torch.clamp(out, -1.0, 1.0)
        
        return out

class MultiReceptiveFieldFusion(nn.Module):
    """
    Multi-Receptive Field Fusion (MRF) module with GRC+LoRA
    Replaces original MRF with grouped residual convolutions
    """
    
    def __init__(self, channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], groups=4, lora_rank=4):
        super(MultiReceptiveFieldFusion, self).__init__()
        
        self.channels = channels
        self.num_fields = len(kernel_sizes)
        
        # Multiple GRC blocks with different receptive fields
        self.grc_blocks = nn.ModuleList([
            GroupedResidualConvolution(channels, kernel_size, dilation, groups, lora_rank)
            for kernel_size, dilation in zip(kernel_sizes, dilations)
        ])
        
        # Fusion layer
        self.fusion = nn.Conv1d(channels * self.num_fields, channels, 1)
        
        # Final normalization and activation
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Process through multiple GRC blocks
        grc_outputs = []
        for grc_block in self.grc_blocks:
            grc_outputs.append(grc_block(x))
        
        # CRITICAL FIX: Ensure all outputs have the same size before concatenation
        target_size = grc_outputs[0].size(2)  # Use first output as reference
        aligned_outputs = []
        for output in grc_outputs:
            if output.size(2) != target_size:
                # Pad or crop to match target size
                if output.size(2) > target_size:
                    output = output[:, :, :target_size]
                else:
                    padding = target_size - output.size(2)
                    output = F.pad(output, (0, padding))
            aligned_outputs.append(output)
        
        # Concatenate outputs from different receptive fields
        fused = torch.cat(aligned_outputs, dim=1)  # [B, channels * num_fields, T]
        
        # Fuse the multi-scale features
        output = self.fusion(fused)  # [B, channels, T]
        output = self.norm(output)
        output = self.activation(output)
        
        # BUZZING FIX: Apply gentle normalization to prevent artifacts
        output = torch.clamp(output, -1.0, 1.0)
        
        return output
