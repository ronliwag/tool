"""
Modified HiFi-GAN Generator with ODConv, GRC+LoRA, and FiLM Conditioning
Implements the complete thesis modifications for expressive voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .odconv import ODConvTranspose1D
from .film_conditioning import FiLMLayer, SpeakerEmotionExtractor
from .grc_lora import MultiReceptiveFieldFusion

class ModifiedHiFiGANGenerator(nn.Module):
    """
    Modified HiFi-GAN Generator with thesis modifications:
    1. ODConv replaces static ConvTranspose1D layers
    2. FiLM conditioning after every ODConv layer
    3. GRC+LoRA replaces original Residual Blocks in MRF
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
        
        # Speaker and emotion embedding extractor
        self.embed_extractor = SpeakerEmotionExtractor(
            input_dim=mel_channels,
            speaker_embed_dim=speaker_embed_dim,
            emotion_embed_dim=emotion_embed_dim
        )
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(mel_channels, upsample_initial_channel, 7, padding=3)
        
        # Upsampling layers with ODConv
        self.ups = nn.ModuleList()
        self.fiLMs = nn.ModuleList()
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate input and output channels
            in_channels = upsample_initial_channel // (2 ** i)
            if i == len(upsample_rates) - 1:
                out_channels = audio_channels
            else:
                out_channels = upsample_initial_channel // (2 ** (i + 1))
            
            # Ensure minimum channel dimensions
            in_channels = max(in_channels, 1)
            out_channels = max(out_channels, 1)
            
            # ODConv upsampling layer
            self.ups.append(ODConvTranspose1D(
                in_channels,
                out_channels,
                k, u, (k - u) // 2
            ))
            
            # FiLM conditioning after each upsampling
            self.fiLMs.append(FiLMLayer(
                feature_dim=out_channels,
                speaker_embed_dim=speaker_embed_dim,
                emotion_embed_dim=emotion_embed_dim
            ))
        
        # Multi-Receptive Field Fusion with GRC+LoRA
        # CRITICAL FIX: Use upsample_initial_channel because after upsampling, we have upsample_initial_channel channels
        self.mrf = MultiReceptiveFieldFusion(
            channels=upsample_initial_channel,  # FIXED: Use upsample_initial_channel (512) after upsampling
            kernel_sizes=resblock_kernel_sizes,
            dilations=resblock_dilations[0],
            groups=4,  # FIXED: Use groups=4 for multi-channel processing
            lora_rank=lora_rank
        )
        
        # Post-processing
        self.conv_post = nn.Conv1d(upsample_initial_channel, audio_channels, 7, padding=3)  # FIXED: Use upsample_initial_channel input, audio_channels output
        
        # Voice cloning enhancement
        self.voice_cloning_enhancer = nn.Sequential(
            nn.Conv1d(audio_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, audio_channels, 3, padding=1)
        )
    
    def forward(self, mel, speaker_embed=None, emotion_embed=None):
        """
        Forward pass with voice cloning conditioning
        
        Args:
            mel: Mel-spectrogram input [B, mel_channels, T]
            speaker_embed: Optional speaker embedding [B, speaker_embed_dim]
            emotion_embed: Optional emotion embedding [B, emotion_embed_dim]
        """
        # Extract embeddings if not provided
        if speaker_embed is None or emotion_embed is None:
            extracted_speaker, extracted_emotion = self.embed_extractor(mel)
            if speaker_embed is None:
                speaker_embed = extracted_speaker
            if emotion_embed is None:
                emotion_embed = extracted_emotion
        
        # Initial convolution
        x = self.conv_pre(mel)
        
        # Upsampling with ODConv and FiLM conditioning
        for up, fiLM in zip(self.ups, self.fiLMs):
            x = up(x)
            x = fiLM(x, speaker_embed, emotion_embed)
            x = F.leaky_relu(x, 0.1)
        
        # Multi-Receptive Field Fusion with GRC+LoRA
        x = self.mrf(x)
        
        # Post-processing
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # Voice cloning enhancement
        x = self.voice_cloning_enhancer(x)
        
        return x
    
    def voice_cloning_forward(self, mel, source_speaker_embed, target_speaker_embed, emotion_embed):
        """
        Special forward pass for voice cloning
        Transfers voice characteristics while preserving content
        """
        # Use source speaker for content, target speaker for voice characteristics
        x = self.conv_pre(mel)
        
        # Upsampling with mixed conditioning
        for i, (up, fiLM) in enumerate(zip(self.ups, self.fiLMs)):
            x = up(x)
            
            # Mix source and target speaker embeddings
            mixed_speaker_embed = 0.7 * target_speaker_embed + 0.3 * source_speaker_embed
            
            x = fiLM(x, mixed_speaker_embed, emotion_embed)
            x = F.leaky_relu(x, 0.1)
        
        # MRF processing
        x = self.mrf(x)
        
        # Post-processing
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # Enhanced voice cloning
        x = self.voice_cloning_enhancer(x)
        
        return x

class ModifiedStreamSpeechVocoder(nn.Module):
    """
    Modified StreamSpeech Vocoder wrapper
    Integrates the modified HiFi-GAN generator
    """
    
    def __init__(self, config):
        super(ModifiedStreamSpeechVocoder, self).__init__()
        
        # Initialize modified generator
        self.generator = ModifiedHiFiGANGenerator(
            mel_channels=config.get('mel_channels', 80),
            audio_channels=config.get('audio_channels', 1),
            upsample_rates=config.get('upsample_rates', [8, 8, 2, 2]),
            upsample_kernel_sizes=config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
            upsample_initial_channel=config.get('upsample_initial_channel', 512),
            speaker_embed_dim=config.get('speaker_embed_dim', 192),
            emotion_embed_dim=config.get('emotion_embed_dim', 256),
            lora_rank=config.get('lora_rank', 4)
        )
        
        # Voice cloning mode flag
        self.voice_cloning_mode = config.get('voice_cloning_mode', True)
        
    def forward(self, mel, speaker_embed=None, emotion_embed=None, 
                source_speaker_embed=None, target_speaker_embed=None):
        """
        Forward pass with voice cloning capabilities
        """
        if self.voice_cloning_mode and source_speaker_embed is not None and target_speaker_embed is not None:
            # Voice cloning mode
            return self.generator.voice_cloning_forward(
                mel, source_speaker_embed, target_speaker_embed, emotion_embed
            )
        else:
            # Standard mode
            return self.generator(mel, speaker_embed, emotion_embed)
    
    def enable_voice_cloning(self):
        """Enable voice cloning mode"""
        self.voice_cloning_mode = True
    
    def disable_voice_cloning(self):
        """Disable voice cloning mode"""
        self.voice_cloning_mode = False
