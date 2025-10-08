#!/usr/bin/env python3
"""
SIMPLE TRAINING FOR MODIFIED HIFI-GAN
Focus on getting the model working with proper dimensions
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
import warnings
warnings.filterwarnings("ignore")

# Add paths for imports
sys.path.append('Original Streamspeech/Modified Streamspeech/models')
sys.path.append('Original Streamspeech/Modified Streamspeech/integration')

# Import the modified HiFi-GAN
from modified_hifigan import ModifiedHiFiGANGenerator, HiFiGANConfig

class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self):
        self.samples = []
        
        # Load metadata
        metadata_path = "real_training_dataset/metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create simple samples
        for sample in metadata:
            self.samples.append({
                'id': sample['id'],
                'spanish_path': sample['spanish_audio_path'],
                'english_path': sample['english_audio_path']
            })
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Spanish audio
        spanish_audio, sr = torchaudio.load(sample['spanish_path'])
        if sr != 22050:
            resampler = T.Resample(sr, 22050)
            spanish_audio = resampler(spanish_audio)
        
        # Load English audio
        english_audio, sr = sf.read(sample['english_path'])
        english_audio = torch.tensor(english_audio, dtype=torch.float32)
        
        # Pad to fixed length (2 seconds)
        target_length = 22050 * 2
        
        if spanish_audio.shape[1] < target_length:
            spanish_audio = F.pad(spanish_audio, (0, target_length - spanish_audio.shape[1]))
        else:
            spanish_audio = spanish_audio[:, :target_length]
        
        if english_audio.shape[0] < target_length:
            english_audio = F.pad(english_audio, (0, target_length - english_audio.shape[0]))
        else:
            english_audio = english_audio[:target_length]
        
        # Convert to mel-spectrograms
        spanish_mel = self.mel_transform(spanish_audio.squeeze(0))  # [80, T]
        english_mel = self.mel_transform(english_audio.unsqueeze(0))  # [1, 80, T]
        
        # Ensure correct shape: [80, T]
        spanish_mel = spanish_mel.squeeze(0) if spanish_mel.dim() == 3 else spanish_mel
        english_mel = english_mel.squeeze(0) if english_mel.dim() == 3 else english_mel
        
        # Create simple embeddings
        speaker_embed = torch.randn(192)
        emotion_embed = torch.randn(256)
        
        return {
            'spanish_mel': spanish_mel,
            'english_mel': english_mel,
            'spanish_audio': spanish_audio.squeeze(0),
            'english_audio': english_audio,
            'speaker_embed': speaker_embed,
            'emotion_embed': emotion_embed,
            'sample_id': sample['id']
        }

def test_generator():
    """Test the generator with proper inputs"""
    print("🧪 TESTING MODIFIED HIFI-GAN GENERATOR...")
    
    # Initialize generator
    config = HiFiGANConfig(
        speaker_embedding_dim=192,
        emotion_embedding_dim=256
    )
    generator = ModifiedHiFiGANGenerator(config)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test with dummy data
    batch_size = 1
    mel_channels = 80
    time_steps = 345  # From our dataset
    
    # Create dummy inputs
    mel_spectrogram = torch.randn(batch_size, mel_channels, time_steps)
    speaker_embed = torch.randn(batch_size, 192)
    emotion_embed = torch.randn(batch_size, 256)
    
    print(f"Input shapes:")
    print(f"  Mel spectrogram: {mel_spectrogram.shape}")
    print(f"  Speaker embed: {speaker_embed.shape}")
    print(f"  Emotion embed: {emotion_embed.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = generator(mel_spectrogram, speaker_embed, emotion_embed)
        
        print(f"✅ Generator test successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        return False

def simple_training():
    """Simple training loop"""
    print("🚀 STARTING SIMPLE TRAINING...")
    
    # Initialize generator
    config = HiFiGANConfig(
        speaker_embedding_dim=192,
        emotion_embedding_dim=256
    )
    generator = ModifiedHiFiGANGenerator(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    print(f"Using device: {device}")
    
    # Initialize optimizer
    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    
    # Initialize dataset
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Simple loss function
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        generator.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            english_mel = batch['english_mel'].to(device)
            speaker_embed = batch['speaker_embed'].to(device)
            emotion_embed = batch['emotion_embed'].to(device)
            english_audio = batch['english_audio'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                generated_audio = generator(english_mel, speaker_embed, emotion_embed)
                
                # Simple reconstruction loss
                loss = criterion(generated_audio, english_audio)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                print(f"  Batch {batch_idx+1}: Loss = {loss.item():.6f}")
                
            except Exception as e:
                print(f"  Batch {batch_idx+1}: Error = {e}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        checkpoint_path = f"D:/Thesis - Tool/checkpoints/simple_training_epoch_{epoch}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

def main():
    """Main function"""
    print("🎯 SIMPLE MODIFIED HIFI-GAN TRAINING")
    print("Testing and training with proper dimensions...")
    
    # Test generator first
    if not test_generator():
        print("❌ Generator test failed. Stopping.")
        return
    
    # Run simple training
    simple_training()
    
    print("✅ SIMPLE TRAINING COMPLETED!")

if __name__ == "__main__":
    main()

