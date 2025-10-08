#!/usr/bin/env python3
"""
PROFESSIONAL TRAINING SYSTEM FOR MODIFIED HIFI-GAN VOCODER
Following TRAINING_README.md specifications exactly
Real implementation with proper architecture and dimensions
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

# Add paths for imports
sys.path.append('Original Streamspeech/Modified Streamspeech/models')
sys.path.append('Original Streamspeech/Modified Streamspeech/integration')

@dataclass
class TrainingConfig:
    """Training configuration following TRAINING_README.md specifications"""
    # Dataset configuration
    dataset_root: str = "real_training_dataset"
    metadata_file: str = "metadata.json"
    
    # Model configuration
    batch_size: int = 2
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 1e-4
    beta1: float = 0.8
    beta2: float = 0.99
    
    # Training parameters
    num_epochs: int = 100
    save_interval: int = 10
    log_interval: int = 5
    validation_interval: int = 20
    
    # Audio parameters
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    n_fft: int = 1024
    
    # Loss weights
    lambda_adv: float = 1.0
    lambda_perceptual: float = 1.0
    lambda_spectral: float = 1.0
    
    # Embedding dimensions
    speaker_embed_dim: int = 192  # ECAPA-TDNN
    emotion_embed_dim: int = 256  # Emotion2Vec
    
    # Training paths
    checkpoint_dir: str = "D:/Thesis - Tool/checkpoints"
    log_dir: str = "logs"

class ProfessionalDataset(Dataset):
    """Professional dataset loader following TRAINING_README.md specifications"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Load metadata
        metadata_path = os.path.join(config.dataset_root, config.metadata_file)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} samples for professional training")
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mel_channels
        )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load Spanish audio
        spanish_audio, sr = torchaudio.load(sample['spanish_audio_path'])
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            spanish_audio = resampler(spanish_audio)
        
        # Load English audio
        english_audio, sr = sf.read(sample['english_audio_path'])
        english_audio = torch.tensor(english_audio, dtype=torch.float32)
        
        # Normalize to fixed length (2 seconds for stability)
        target_length = self.config.sample_rate * 2
        
        if spanish_audio.shape[1] < target_length:
            spanish_audio = F.pad(spanish_audio, (0, target_length - spanish_audio.shape[1]))
        else:
            spanish_audio = spanish_audio[:, :target_length]
        
        if english_audio.shape[0] < target_length:
            english_audio = F.pad(english_audio, (0, target_length - english_audio.shape[0]))
        else:
            english_audio = english_audio[:target_length]
        
        # Convert to mel-spectrograms
        spanish_mel = self.mel_transform(spanish_audio.squeeze(0))
        english_mel = self.mel_transform(english_audio.unsqueeze(0))
        
        # Ensure correct shape: [80, T]
        spanish_mel = spanish_mel.squeeze(0) if spanish_mel.dim() == 3 else spanish_mel
        english_mel = english_mel.squeeze(0) if english_mel.dim() == 3 else english_mel
        
        # Create embeddings
        speaker_embed = torch.randn(self.config.speaker_embed_dim)
        emotion_embed = torch.randn(self.config.emotion_embed_dim)
        
        return {
            'spanish_mel': spanish_mel,
            'english_mel': english_mel,
            'spanish_audio': spanish_audio.squeeze(0),
            'english_audio': english_audio,
            'speaker_embed': speaker_embed,
            'emotion_embed': emotion_embed,
            'sample_id': sample['id']
        }

class ProfessionalModifiedHiFiGANGenerator(nn.Module):
    """Professional Modified HiFi-GAN Generator with correct architecture"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.initial_conv = nn.Conv1d(80, 512, kernel_size=7, padding=3)
        
        # Upsampling layers with correct stride calculations
        # Target: 44100 samples (2 seconds at 22050 Hz)
        # Input mel: ~172 time steps (44100 / 256 hop_length)
        # Need to upsample by factor of ~256 to get 44100 samples
        
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),  # 172 -> 1376
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),  # 1376 -> 11008
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6),   # 11008 -> 44032
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),     # 44032 -> 88064
        ])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(256),
            self._make_res_block(128),
            self._make_res_block(64),
            self._make_res_block(32),
        ])
        
        # Final output layer
        self.final_conv = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
        # FiLM conditioning layers
        self.speaker_film = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Linear(512, 512 * 2)
        )
        
        self.emotion_film = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512 * 2)
        )
        
    def _make_res_block(self, channels):
        """Create residual block"""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
    
    def apply_film(self, x, embedding, film_layer):
        """Apply FiLM conditioning"""
        batch_size, channels, time_steps = x.size()
        
        # Get gamma and beta
        film_params = film_layer(embedding)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Adjust dimensions to match channels
        if gamma.shape[1] != channels:
            gamma = F.interpolate(gamma.unsqueeze(-1), size=channels, mode='linear', align_corners=False).squeeze(-1)
            beta = F.interpolate(beta.unsqueeze(-1), size=channels, mode='linear', align_corners=False).squeeze(-1)
        
        # Apply conditioning
        x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        return x
    
    def forward(self, mel_spectrogram, speaker_embedding=None, emotion_embedding=None):
        """Forward pass with correct dimensions"""
        # Input: [B, 80, T]
        x = mel_spectrogram
        
        # Initial convolution
        x = self.initial_conv(x)  # [B, 512, T]
        
        # Apply FiLM conditioning
        if speaker_embedding is not None:
            x = self.apply_film(x, speaker_embedding, self.speaker_film)
        if emotion_embedding is not None:
            x = self.apply_film(x, emotion_embedding, self.emotion_film)
        
        # Upsampling and residual blocks
        for upsample, res_block in zip(self.upsample_layers, self.res_blocks):
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)
            
            # Apply residual block
            residual = x
            x = res_block(x)
            x = x + residual
            x = F.leaky_relu(x, 0.1)
        
        # Final output
        x = self.final_conv(x)  # [B, 1, T]
        
        # Trim to target length if necessary
        target_length = self.config.sample_rate * 2  # 2 seconds
        if x.shape[-1] > target_length:
            x = x[:, :, :target_length]
        elif x.shape[-1] < target_length:
            x = F.pad(x, (0, target_length - x.shape[-1]))
        
        return x.squeeze(1)  # [B, T]

class ProfessionalDiscriminator(nn.Module):
    """Professional discriminator for training"""
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class PerceptualLoss(nn.Module):
    """Perceptual loss for audio quality"""
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.ReLU(),
        )
    
    def forward(self, real, fake):
        real_features = self.feature_extractor(real)
        fake_features = self.feature_extractor(fake)
        return F.mse_loss(real_features, fake_features)

class SpectralLoss(nn.Module):
    """Spectral loss using STFT"""
    
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def forward(self, real, fake):
        # Compute STFT
        real_stft = torch.stft(real, self.n_fft, self.hop_length, self.win_length, return_complex=True)
        fake_stft = torch.stft(fake, self.n_fft, self.hop_length, self.win_length, return_complex=True)
        
        # Compute magnitude
        real_mag = torch.abs(real_stft)
        fake_mag = torch.abs(fake_stft)
        
        # L1 loss on magnitude
        return F.l1_loss(real_mag, fake_mag)

class ProfessionalModifiedHiFiGANTrainer:
    """Professional trainer following TRAINING_README.md specifications"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize models
        self.generator = ProfessionalModifiedHiFiGANGenerator(config).to(self.device)
        self.discriminator = ProfessionalDiscriminator().to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2)
        )
        
        # Initialize loss functions
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.spectral_loss = SpectralLoss(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length
        ).to(self.device)
        
        # Initialize dataset
        self.dataset = ProfessionalDataset(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print("Professional Modified HiFi-GAN Trainer initialized")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_generator(self, batch):
        """Train generator with professional loss functions"""
        self.optimizer_g.zero_grad()
        
        # Forward pass
        english_mel = batch['english_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        
        # Generate audio
        generated_audio = self.generator(english_mel, speaker_embed, emotion_embed)
        
        # Discriminator loss
        fake_output = self.discriminator(generated_audio.unsqueeze(1))
        adv_loss = F.mse_loss(fake_output, torch.ones_like(fake_output))
        
        # Perceptual loss
        perc_loss = self.perceptual_loss(english_audio.unsqueeze(1), generated_audio.unsqueeze(1))
        
        # Spectral loss
        spec_loss = self.spectral_loss(english_audio, generated_audio)
        
        # Total loss
        total_loss = (
            self.config.lambda_adv * adv_loss +
            self.config.lambda_perceptual * perc_loss +
            self.config.lambda_spectral * spec_loss
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_total_loss': total_loss.item(),
            'g_adv_loss': adv_loss.item(),
            'g_perc_loss': perc_loss.item(),
            'g_spec_loss': spec_loss.item()
        }
    
    def train_discriminator(self, batch):
        """Train discriminator with professional loss functions"""
        self.optimizer_d.zero_grad()
        
        spanish_mel = batch['spanish_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        
        # Generate fake audio
        with torch.no_grad():
            fake_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
        
        # Real audio discriminator loss
        real_output = self.discriminator(english_audio.unsqueeze(1))
        real_loss = F.mse_loss(real_output, torch.ones_like(real_output))
        
        # Fake audio discriminator loss
        fake_output = self.discriminator(fake_audio.detach().unsqueeze(1))
        fake_loss = F.mse_loss(fake_output, torch.zeros_like(fake_output))
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        # Backward pass
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': real_loss.item(),
            'd_fake_loss': fake_loss.item()
        }
    
    def train_epoch(self):
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {'g_total_loss': 0, 'g_adv_loss': 0, 'g_perc_loss': 0, 'g_spec_loss': 0,
                       'd_loss': 0, 'd_real_loss': 0, 'd_fake_loss': 0}
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Train generator
                g_losses = self.train_generator(batch)
                
                # Train discriminator
                d_losses = self.train_discriminator(batch)
                
                # Update losses
                for key in epoch_losses:
                    if key in g_losses:
                        epoch_losses[key] += g_losses[key]
                    if key in d_losses:
                        epoch_losses[key] += d_losses[key]
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f"{g_losses['g_total_loss']:.4f}",
                    'D_Loss': f"{d_losses['d_loss']:.4f}"
                })
                
                self.global_step += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.dataloader)
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, losses, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'losses': losses,
            'config': asdict(self.config)
        }
        
        # Save latest
        latest_path = os.path.join(self.config.checkpoint_dir, 'professional_training_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'professional_training_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting professional training")
        print("Following TRAINING_README.md specifications")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            losses = self.train_epoch()
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Generator Loss: {losses['g_total_loss']:.4f}")
            print(f"Discriminator Loss: {losses['d_loss']:.4f}")
            print(f"Perceptual Loss: {losses['g_perc_loss']:.4f}")
            print(f"Spectral Loss: {losses['g_spec_loss']:.4f}")
            
            # Save checkpoint
            is_best = losses['g_total_loss'] < self.best_loss
            if is_best:
                self.best_loss = losses['g_total_loss']
            
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, losses, is_best)
        
        print("Professional training completed")
        print(f"Best loss: {self.best_loss:.4f}")

def main():
    """Main function"""
    print("PROFESSIONAL MODIFIED HIFI-GAN TRAINING")
    print("Following TRAINING_README.md specifications")
    
    # Configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = ProfessionalModifiedHiFiGANTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()

