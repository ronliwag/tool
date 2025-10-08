#!/usr/bin/env python3
"""
PROFESSIONAL TRAINING WITH REAL CVSS DATASET
Real implementation using your CVSS-T dataset with real embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import json
import logging
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys

# Import the professional training system
sys.path.append('.')
from professional_training_system import (
    ProfessionalModifiedHiFiGANGenerator, 
    TrainingConfig,
    ProfessionalDiscriminator,
    PerceptualLoss,
    SpectralLoss
)

# Import embedding extractors
from ecapa_tdnn_integration import ECAPATDNNSpeakerExtractor
from emotion2vec_integration import Emotion2VecExtractor

class RealCVSSDataset(Dataset):
    """Real CVSS dataset with actual embeddings"""
    
    def __init__(self, config: TrainingConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Load metadata
        metadata_path = "professional_cvss_dataset/metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Filter for split
        self.samples = [item for item in self.metadata if item['split'] == split]
        
        print(f"Loaded {len(self.samples)} {split} samples from real CVSS dataset")
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mel_channels
        )
        
        # Initialize embedding extractors
        self.speaker_extractor = ECAPATDNNSpeakerExtractor()
        self.emotion_extractor = Emotion2VecExtractor()
        
        # Load pre-computed embeddings if available
        self.speaker_embeddings = self.load_embeddings("ecapa_speaker_embeddings.npy")
        self.emotion_embeddings = self.load_embeddings("emotion2vec_embeddings.npy")
        
    def load_embeddings(self, embeddings_path: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed embeddings"""
        if os.path.exists(embeddings_path):
            embeddings_np = np.load(embeddings_path, allow_pickle=True).item()
            return {path: torch.from_numpy(emb) for path, emb in embeddings_np.items()}
        return {}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio files
        spanish_audio, sr = torchaudio.load(sample['spanish_audio_path'])
        english_audio, _ = torchaudio.load(sample['english_audio_path'])
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            spanish_audio = resampler(spanish_audio)
            english_audio = resampler(english_audio)
        
        # Pad or truncate to fixed length (2 seconds for stability)
        target_length = self.config.sample_rate * 2
        
        if spanish_audio.shape[1] < target_length:
            spanish_audio = F.pad(spanish_audio, (0, target_length - spanish_audio.shape[1]))
        else:
            spanish_audio = spanish_audio[:, :target_length]
        
        if english_audio.shape[1] < target_length:
            english_audio = F.pad(english_audio, (0, target_length - english_audio.shape[1]))
        else:
            english_audio = english_audio[:, :target_length]
        
        # Convert to mel-spectrograms
        spanish_mel = self.mel_transform(spanish_audio.squeeze(0))
        english_mel = self.mel_transform(english_audio.squeeze(0))
        
        # Ensure correct shape for generator: [n_mels, T]
        spanish_mel = spanish_mel.squeeze(0) if spanish_mel.dim() == 3 else spanish_mel
        english_mel = english_mel.squeeze(0) if english_mel.dim() == 3 else english_mel
        
        # Get embeddings
        speaker_embed = self.get_speaker_embedding(sample['spanish_audio_path'])
        emotion_embed = self.get_emotion_embedding(sample['spanish_audio_path'])
        
        return {
            'spanish_mel': spanish_mel,
            'english_mel': english_mel,
            'spanish_audio': spanish_audio.squeeze(0),
            'english_audio': english_audio.squeeze(0),
            'speaker_embed': speaker_embed,
            'emotion_embed': emotion_embed,
            'sample_id': sample['id']
        }
    
    def get_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Get speaker embedding"""
        if audio_path in self.speaker_embeddings:
            return self.speaker_embeddings[audio_path]
        else:
            # Extract on-the-fly
            return self.speaker_extractor.get_speaker_embedding(audio_path)
    
    def get_emotion_embedding(self, audio_path: str) -> torch.Tensor:
        """Get emotion embedding"""
        if audio_path in self.emotion_embeddings:
            return self.emotion_embeddings[audio_path]
        else:
            # Extract on-the-fly
            return self.emotion_extractor.get_emotion_embedding(audio_path)

class ProfessionalCVSSTrainer:
    """Professional trainer with real CVSS dataset"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs("professional_cvss_checkpoints", exist_ok=True)
        os.makedirs("professional_cvss_logs", exist_ok=True)
        
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
        
        # Initialize datasets
        self.train_dataset = RealCVSSDataset(config, split="train")
        self.val_dataset = RealCVSSDataset(config, split="validation")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print("Professional CVSS Trainer initialized successfully!")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_generator(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train generator with real loss functions"""
        self.optimizer_g.zero_grad()
        
        # Forward pass
        spanish_mel = batch['spanish_mel'].to(self.device)
        english_mel = batch['english_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        
        # Generate audio
        generated_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
        
        # Discriminator loss
        fake_output = self.discriminator(generated_audio.unsqueeze(1))
        adv_loss = F.mse_loss(fake_output, torch.ones_like(fake_output))
        
        # Perceptual loss - ensure same length
        min_length = min(english_audio.shape[0], generated_audio.shape[0])
        english_audio_trimmed = english_audio[:min_length].unsqueeze(1)
        generated_audio_trimmed = generated_audio[:min_length].unsqueeze(1)
        perc_loss = self.perceptual_loss(english_audio_trimmed, generated_audio_trimmed)
        
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
    
    def train_discriminator(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train discriminator with real loss functions"""
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'g_total_loss': 0.0,
            'g_adv_loss': 0.0,
            'g_perc_loss': 0.0,
            'g_spec_loss': 0.0,
            'd_loss': 0.0,
            'd_real_loss': 0.0,
            'd_fake_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Train generator
            g_losses = self.train_generator(batch)
            
            # Train discriminator
            d_losses = self.train_discriminator(batch)
            
            # Update epoch losses
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
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'g_total_loss': 0.0,
            'g_adv_loss': 0.0,
            'g_perc_loss': 0.0,
            'g_spec_loss': 0.0,
            'd_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Forward pass
                spanish_mel = batch['spanish_mel'].to(self.device)
                speaker_embed = batch['speaker_embed'].to(self.device)
                emotion_embed = batch['emotion_embed'].to(self.device)
                english_audio = batch['english_audio'].to(self.device)
                
                # Generate audio
                generated_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
                
                # Compute losses
                fake_output = self.discriminator(generated_audio.unsqueeze(1))
                adv_loss = F.mse_loss(fake_output, torch.ones_like(fake_output))
                
                # Perceptual loss - ensure same length
                min_length = min(english_audio.shape[0], generated_audio.shape[0])
                english_audio_trimmed = english_audio[:min_length].unsqueeze(1)
                generated_audio_trimmed = generated_audio[:min_length].unsqueeze(1)
                perc_loss = self.perceptual_loss(english_audio_trimmed, generated_audio_trimmed)
                spec_loss = self.spectral_loss(english_audio, generated_audio)
                
                total_loss = (
                    self.config.lambda_adv * adv_loss +
                    self.config.lambda_perceptual * perc_loss +
                    self.config.lambda_spectral * spec_loss
                )
                
                val_losses['g_total_loss'] += total_loss.item()
                val_losses['g_adv_loss'] += adv_loss.item()
                val_losses['g_perc_loss'] += perc_loss.item()
                val_losses['g_spec_loss'] += spec_loss.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, losses: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'losses': losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = "professional_cvss_checkpoints/professional_cvss_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = "professional_cvss_checkpoints/professional_cvss_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting professional training with real CVSS dataset")
        print("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Generator Loss: {train_losses['g_total_loss']:.6f}")
            print(f"Discriminator Loss: {train_losses['d_loss']:.6f}")
            print(f"Perceptual Loss: {train_losses['g_perc_loss']:.6f}")
            print(f"Spectral Loss: {train_losses['g_spec_loss']:.6f}")
            
            # Save checkpoint
            is_best = train_losses['g_total_loss'] < self.best_loss
            if is_best:
                self.best_loss = train_losses['g_total_loss']
            
            # Save after every epoch
            self.save_checkpoint(epoch, train_losses, is_best)
            print(f"Checkpoint saved for epoch {epoch}")
        
        print("\nProfessional training completed")
        print(f"Best loss: {self.best_loss:.6f}")

def main():
    """Main function"""
    print("PROFESSIONAL TRAINING WITH REAL CVSS DATASET")
    print("Real implementation using your CVSS-T dataset")
    print("=" * 80)
    
    # Configuration
    config = TrainingConfig()
    config.batch_size = 2  # Reduced for Windows stability
    config.num_epochs = 10  # Reduced for faster completion
    config.save_interval = 1  # Save after every epoch
    config.log_interval = 10
    config.validation_interval = 25
    
    # Initialize trainer
    trainer = ProfessionalCVSSTrainer(config)
    
    # Start training
    trainer.train()
    
    print("\nPROFESSIONAL CVSS TRAINING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()

