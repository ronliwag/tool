"""
Feature-wise Linear Modulation (FiLM) for Speaker and Emotion Conditioning
Applied after every ODConv layer for expressive voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer
    Applies affine transformations using speaker and emotion embeddings
    """
    
    def __init__(self, feature_dim, speaker_embed_dim=192, emotion_embed_dim=256):
        super(FiLMLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.emotion_embed_dim = emotion_embed_dim
        
        # Speaker conditioning networks
        self.speaker_gamma_net = nn.Sequential(
            nn.Linear(speaker_embed_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.speaker_beta_net = nn.Sequential(
            nn.Linear(speaker_embed_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Emotion conditioning networks
        self.emotion_gamma_net = nn.Sequential(
            nn.Linear(emotion_embed_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.emotion_beta_net = nn.Sequential(
            nn.Linear(emotion_embed_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Fusion network for combined conditioning
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, x, speaker_embed=None, emotion_embed=None):
        """
        Apply FiLM conditioning to input features
        
        Args:
            x: Input features [B, C, T]
            speaker_embed: Speaker embedding [B, speaker_embed_dim]
            emotion_embed: Emotion embedding [B, emotion_embed_dim]
        """
        batch_size, channels, time_steps = x.size()
        
        # Initialize with identity if no embeddings provided
        if speaker_embed is None:
            speaker_gamma = torch.ones(batch_size, channels, device=x.device)
            speaker_beta = torch.zeros(batch_size, channels, device=x.device)
        else:
            speaker_gamma = self.speaker_gamma_net(speaker_embed)  # [B, C]
            speaker_beta = self.speaker_beta_net(speaker_embed)    # [B, C]
        
        if emotion_embed is None:
            emotion_gamma = torch.ones(batch_size, channels, device=x.device)
            emotion_beta = torch.zeros(batch_size, channels, device=x.device)
        else:
            emotion_gamma = self.emotion_gamma_net(emotion_embed)  # [B, C]
            emotion_beta = self.emotion_beta_net(emotion_embed)    # [B, C]
        
        # Apply speaker conditioning
        x_speaker = x * speaker_gamma.unsqueeze(-1) + speaker_beta.unsqueeze(-1)
        
        # Apply emotion conditioning
        x_emotion = x * emotion_gamma.unsqueeze(-1) + emotion_beta.unsqueeze(-1)
        
        # Fuse both conditioning effects
        x_combined = torch.cat([x_speaker, x_emotion], dim=1)  # [B, 2C, T]
        
        # Apply fusion network with proper dimension handling
        # Global average pooling to reduce dimensions before fusion
        x_pooled = torch.mean(x_combined, dim=-1)  # [B, 2C]
        x_fused = self.fusion_net(x_pooled)  # [B, C]
        x_fused = x_fused.view(batch_size, channels, 1)  # [B, C, 1]
        
        # Final conditioning
        output = x * x_fused
        
        return output

class SpeakerEmotionExtractor(nn.Module):
    """
    Extracts speaker and emotion embeddings from audio features
    """
    
    def __init__(self, input_dim=80, speaker_embed_dim=192, emotion_embed_dim=256):
        super(SpeakerEmotionExtractor, self).__init__()
        
        self.speaker_embed_dim = speaker_embed_dim
        self.emotion_embed_dim = emotion_embed_dim
        
        # Speaker embedding network (ECAPA-TDNN inspired)
        self.speaker_net = nn.Sequential(
            nn.Conv1d(input_dim, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, speaker_embed_dim)
        )
        
        # Emotion embedding network (Emotion2Vec inspired)
        self.emotion_net = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, emotion_embed_dim)
        )
    
    def forward(self, mel_features):
        """
        Extract speaker and emotion embeddings from mel-spectrogram features
        
        Args:
            mel_features: Mel-spectrogram features [B, mel_bins, T]
        """
        # Extract speaker embedding
        speaker_embed = self.speaker_net(mel_features)  # [B, speaker_embed_dim]
        
        # Extract emotion embedding
        emotion_embed = self.emotion_net(mel_features)  # [B, emotion_embed_dim]
        
        return speaker_embed, emotion_embed
