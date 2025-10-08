"""
Complete Modified HiFi-GAN Generator with Real ODConv Implementation
Implements the complete thesis modifications for expressive voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .real_odconv import ODConvTranspose1D, ODConv1D

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Applies speaker and emotion conditioning to feature maps
    """
    
    def __init__(self, feature_dim, speaker_dim=192, emotion_dim=256):
        super(FiLMLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        
        # FiLM parameter generation
        self.film_generator = nn.Sequential(
            nn.Linear(speaker_dim + emotion_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
    def forward(self, x, speaker_embed, emotion_embed):
        """
        Apply FiLM conditioning
        x: feature map [B, C, T]
        speaker_embed: [B, speaker_dim]
        emotion_embed: [B, emotion_dim]
        """
        # Combine embeddings
        condition = torch.cat([speaker_embed, emotion_embed], dim=-1)  # [B, speaker_dim + emotion_dim]
        
        # Generate FiLM parameters
        film_params = self.film_generator(condition)  # [B, feature_dim * 2]
        
        # Split into scale and shift
        scale = film_params[:, :self.feature_dim].unsqueeze(-1)  # [B, feature_dim, 1]
        shift = film_params[:, self.feature_dim:].unsqueeze(-1)  # [B, feature_dim, 1]
        
        # Apply FiLM: y = scale * x + shift
        return x * scale + shift

class GRCWithLoRA(nn.Module):
    """
    Grouped Residual Convolution with Low-Rank Adaptation (LoRA)
    Replaces original Residual Blocks in MRF module
    """
    
    def __init__(self, channels, kernel_size=3, dilation=1, groups=4, lora_rank=4):
        super(GRCWithLoRA, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.lora_rank = lora_rank
        
        # Base convolution
        self.base_conv = ODConv1D(
            channels, channels, kernel_size, 
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation, groups=groups
        )
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.randn(channels, lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, channels) * 0.01)
        self.lora_alpha = 1.0 / lora_rank
        
        # Grouped convolution for residual connection
        self.grouped_conv = nn.Conv1d(
            channels, channels, 1, groups=groups
        )
        
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        """
        Forward pass with GRC and LoRA
        """
        # Apply base ODConv
        out = self.base_conv(x)
        
        # Apply LoRA adaptation
        # Reshape for LoRA: [B, C, T] -> [B*T, C]
        B, C, T = out.shape
        out_reshaped = out.permute(0, 2, 1).contiguous().view(-1, C)  # [B*T, C]
        
        # Apply LoRA: x + (x @ A @ B) * alpha
        lora_out = out_reshaped @ self.lora_A @ self.lora_B * self.lora_alpha
        out_reshaped = out_reshaped + lora_out
        
        # Reshape back: [B*T, C] -> [B, C, T]
        out = out_reshaped.view(B, T, C).permute(0, 2, 1).contiguous()
        
        # Apply grouped convolution for residual connection
        residual = self.grouped_conv(x)
        
        # Combine with residual connection
        out = out + residual
        
        return self.activation(out)

class MultiReceptiveFieldFusion(nn.Module):
    """
    Modified MRF module with GRC+LoRA instead of original ResBlocks
    """
    
    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], groups=4, lora_rank=4):
        super(MultiReceptiveFieldFusion, self).__init__()
        
        self.grc_blocks = nn.ModuleList()
        
        for i, (kernel_size, dilation_group) in enumerate(zip(kernel_sizes, dilations)):
            for dilation in dilation_group:
                self.grc_blocks.append(
                    GRCWithLoRA(channels, kernel_size, dilation, groups, lora_rank)
                )
        
        # Final fusion
        self.fusion_conv = nn.Conv1d(channels * len(self.grc_blocks), channels, 1)
        
    def forward(self, x):
        """
        Apply GRC blocks and fuse outputs
        """
        outputs = []
        for grc_block in self.grc_blocks:
            outputs.append(grc_block(x))
        
        # Concatenate all outputs
        concat_output = torch.cat(outputs, dim=1)  # [B, C*num_blocks, T]
        
        # Fusion
        return self.fusion_conv(concat_output)

class ModifiedHiFiGANGenerator(nn.Module):
    """
    Complete Modified HiFi-GAN Generator with ODConv, GRC+LoRA, and FiLM
    Implements all thesis modifications
    """
    
    def __init__(self, 
                 mel_channels=80,
                 audio_channels=1,
                 upsample_rates=[8, 8, 2, 2],
                 upsample_kernel_sizes=[16, 16, 4, 4],
                 upsample_initial_channel=512,
                 resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 speaker_embed_dim=192,
                 emotion_embed_dim=256,
                 lora_rank=4):
        super(ModifiedHiFiGANGenerator, self).__init__()
        
        self.mel_channels = mel_channels
        self.audio_channels = audio_channels
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(mel_channels, upsample_initial_channel, 7, padding=3)
        
        # Upsampling layers with ODConv
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        current_channels = upsample_initial_channel
        
        for i, (upsample_rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate output channels
            out_channels = current_channels // 2 if i < len(upsample_rates) - 1 else current_channels
            
            # ODConv upsampling layer
            self.ups.append(
                ODConvTranspose1D(
                    current_channels, out_channels, kernel_size,
                    stride=upsample_rate, padding=(kernel_size - upsample_rate) // 2,
                    output_padding=upsample_rate - 1
                )
            )
            
            # Modified MRF with GRC+LoRA
            self.mrfs.append(
                MultiReceptiveFieldFusion(
                    out_channels, resblock_kernel_sizes, resblock_dilations,
                    groups=4, lora_rank=lora_rank
                )
            )
            
            current_channels = out_channels
        
        # FiLM layers for conditioning
        self.film_layers = nn.ModuleList()
        for i in range(len(upsample_rates)):
            out_channels = upsample_initial_channel // (2 ** (i + 1)) if i < len(upsample_rates) - 1 else upsample_initial_channel // (2 ** len(upsample_rates))
            self.film_layers.append(
                FiLMLayer(out_channels, speaker_embed_dim, emotion_embed_dim)
            )
        
        # Final convolution
        self.conv_post = nn.Conv1d(current_channels, audio_channels, 7, padding=3)
        
    def forward(self, mel, speaker_embed, emotion_embed):
        """
        Forward pass with all modifications
        mel: mel-spectrogram [B, mel_channels, T]
        speaker_embed: speaker embedding [B, speaker_embed_dim]
        emotion_embed: emotion embedding [B, emotion_embed_dim]
        """
        # Initial convolution
        x = self.conv_pre(mel)
        
        # Upsampling with ODConv and FiLM conditioning
        for i, (up, mrf, film) in enumerate(zip(self.ups, self.mrfs, self.film_layers)):
            # Apply ODConv upsampling
            x = up(x)
            
            # Apply FiLM conditioning
            x = film(x, speaker_embed, emotion_embed)
            
            # Apply modified MRF with GRC+LoRA
            x = mrf(x)
            
            # Apply activation
            x = F.leaky_relu(x, 0.1)
        
        # Final convolution
        x = self.conv_post(x)
        
        # Apply tanh activation for final output
        x = torch.tanh(x)
        
        return x

class ModifiedHiFiGANVocoder:
    """
    Complete Modified HiFi-GAN Vocoder wrapper
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'mel_channels': 80,
                'audio_channels': 1,
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'upsample_initial_channel': 512,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilations': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                'speaker_embed_dim': 192,
                'emotion_embed_dim': 256,
                'lora_rank': 4
            }
        
        self.config = config
        self.generator = ModifiedHiFiGANGenerator(**config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load trained model checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.generator.load_state_dict(checkpoint)
            
            self.generator.eval()
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def generate(self, mel, speaker_embed=None, emotion_embed=None):
        """
        Generate audio from mel-spectrogram with conditioning
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Create default embeddings if not provided
            if speaker_embed is None:
                speaker_embed = torch.randn(mel.size(0), 192).to(self.device)
            if emotion_embed is None:
                emotion_embed = torch.randn(mel.size(0), 256).to(self.device)
            
            # Generate audio
            audio = self.generator(mel, speaker_embed, emotion_embed)
            
        return audio

