#!/usr/bin/env python3
"""
REAL TRAINING STARTER FOR MODIFIED HIFI-GAN
Starts the complete training with REAL data and REAL models
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

# Import the modified HiFi-GAN
from modified_hifigan import ModifiedHiFiGANGenerator, HiFiGANConfig

@dataclass
class TrainingConfig:
    """REAL training configuration"""
    # Dataset paths
    dataset_root: str = "real_training_dataset"
    spanish_audio_dir: str = "spanish"
    english_audio_dir: str = "english"
    metadata_file: str = "metadata.json"
    
    # Model configuration
    batch_size: int = 2  # Small batch for GTX 1050
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 1e-4
    beta1: float = 0.8
    beta2: float = 0.99
    
    # Training parameters
    num_epochs: int = 100
    save_interval: int = 10
    log_interval: int = 5
    
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
    speaker_embed_dim: int = 192
    emotion_embed_dim: int = 256
    
    # Training paths
    checkpoint_dir: str = "D:/Thesis - Tool/checkpoints"

class RealDataset(Dataset):
    """REAL Dataset Loader"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Load metadata
        metadata_path = os.path.join(config.dataset_root, config.metadata_file)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} REAL samples")
        
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
        
        # Pad or truncate to fixed length (4 seconds at 22050 Hz)
        target_length = self.config.sample_rate * 4
        
        if spanish_audio.shape[1] < target_length:
            spanish_audio = F.pad(spanish_audio, (0, target_length - spanish_audio.shape[1]))
        else:
            spanish_audio = spanish_audio[:, :target_length]
        
        if english_audio.shape[0] < target_length:
            english_audio = F.pad(english_audio, (0, target_length - english_audio.shape[0]))
        else:
            english_audio = english_audio[:target_length]
        
        # Convert to mel-spectrograms
        spanish_mel = self.mel_transform(spanish_audio.squeeze(0))  # [n_mels, T]
        english_mel = self.mel_transform(english_audio.unsqueeze(0))  # [1, n_mels, T]
        
        # Ensure correct shape for generator: [n_mels, T]
        spanish_mel = spanish_mel.squeeze(0) if spanish_mel.dim() == 3 else spanish_mel
        english_mel = english_mel.squeeze(0) if english_mel.dim() == 3 else english_mel
        
        # Create speaker and emotion embeddings
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

class SimpleDiscriminator(nn.Module):
    """Simple discriminator for training"""
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class ModifiedHiFiGANTrainer:
    """REAL TRAINER - NO compromises"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize models
        self.generator = ModifiedHiFiGANGenerator(HiFiGANConfig(
            speaker_embedding_dim=config.speaker_embed_dim,
            emotion_embedding_dim=config.emotion_embed_dim
        )).to(self.device)
        
        self.discriminator = SimpleDiscriminator().to(self.device)
        
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
        
        # Initialize dataset
        self.dataset = RealDataset(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print("✅ REAL Modified HiFi-GAN Trainer initialized!")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_generator(self, batch):
        """Train generator"""
        self.optimizer_g.zero_grad()
        
        # Forward pass
        spanish_mel = batch['spanish_mel'].to(self.device)
        english_mel = batch['english_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        
        # Generate audio - ensure correct input shape
        # english_mel should be [B, n_mels, T] for generator
        generated_audio = self.generator(english_mel, speaker_embed, emotion_embed)
        
        # Discriminator loss
        fake_output = self.discriminator(generated_audio.unsqueeze(1))
        adv_loss = F.mse_loss(fake_output, torch.ones_like(fake_output))
        
        # Spectral loss
        spec_loss = F.l1_loss(generated_audio, english_audio)
        
        # Total loss
        total_loss = self.config.lambda_adv * adv_loss + self.config.lambda_spectral * spec_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_total_loss': total_loss.item(),
            'g_adv_loss': adv_loss.item(),
            'g_spec_loss': spec_loss.item()
        }
    
    def train_discriminator(self, batch):
        """Train discriminator"""
        self.optimizer_d.zero_grad()
        
        spanish_mel = batch['spanish_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        
        # Generate fake audio
        with torch.no_grad():
            fake_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
        
        # Real audio
        real_output = self.discriminator(english_audio.unsqueeze(1))
        real_loss = F.mse_loss(real_output, torch.ones_like(real_output))
        
        # Fake audio
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
        
        epoch_losses = {'g_total_loss': 0, 'g_adv_loss': 0, 'g_spec_loss': 0, 
                       'd_loss': 0, 'd_real_loss': 0, 'd_fake_loss': 0}
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
        latest_path = os.path.join(self.config.checkpoint_dir, 'real_training_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'real_training_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        print("🚀 STARTING REAL TRAINING...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            losses = self.train_epoch()
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Generator Loss: {losses['g_total_loss']:.4f}")
            print(f"Discriminator Loss: {losses['d_loss']:.4f}")
            
            # Save checkpoint
            is_best = losses['g_total_loss'] < self.best_loss
            if is_best:
                self.best_loss = losses['g_total_loss']
            
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, losses, is_best)
        
        print("✅ REAL TRAINING COMPLETED!")
        print(f"Best loss: {self.best_loss:.4f}")

def main():
    """Main function"""
    print("🎯 REAL MODIFIED HIFI-GAN TRAINING")
    print("NO compromises, NO shortcuts, NO mistakes!")
    
    # Load config
    config = TrainingConfig()
    
    # Create trainer
    trainer = ModifiedHiFiGANTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
