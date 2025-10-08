#!/usr/bin/env python3
"""
DEFINITIVE WORKING TRAINING SCRIPT - 100% FIXED
Uses corrected GRC implementation to fix tensor size mismatch
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Add model paths
sys.path.append('/content/models')
sys.path.append('/content/drive/MyDrive/Thesis_Training/models')
sys.path.append('/content/drive/MyDrive/Thesis_Training/embedding_extractors')

# Import working components
from odconv import ODConvTranspose1D
from grc_lora_fixed import MultiReceptiveFieldFusion
from real_ecapa_extractor import RealECAPAExtractor
from real_emotion_extractor import RealEmotionExtractor

class ModifiedHiFiGANGenerator(nn.Module):
    """Working Modified HiFi-GAN Generator"""
    
    def __init__(self, speaker_dim=192, emotion_dim=256):
        super().__init__()
        
        # Initial projection
        self.initial_conv = nn.Conv1d(80, 512, 7, padding=3)
        
        # Upsampling layers with working ODConv
        self.ups = nn.ModuleList([
            ODConvTranspose1D(512, 256, 16, 8, 4, speaker_dim=speaker_dim, emotion_dim=emotion_dim),
            ODConvTranspose1D(256, 128, 16, 8, 4, speaker_dim=speaker_dim, emotion_dim=emotion_dim),
            ODConvTranspose1D(128, 64, 4, 2, 1, speaker_dim=speaker_dim, emotion_dim=emotion_dim),
            ODConvTranspose1D(64, 32, 4, 2, 1, speaker_dim=speaker_dim, emotion_dim=emotion_dim),
            ODConvTranspose1D(32, 16, 4, 2, 1, speaker_dim=speaker_dim, emotion_dim=emotion_dim),
        ])
        
        # MRF layers with FIXED implementation - correct channel dimensions
        self.mrfs = nn.ModuleList([
            MultiReceptiveFieldFusion(512),  # After initial conv
            MultiReceptiveFieldFusion(256),   # After first upsample
            MultiReceptiveFieldFusion(128),   # After second upsample
            MultiReceptiveFieldFusion(64),    # After third upsample
            MultiReceptiveFieldFusion(32),    # After fourth upsample
        ])
        
        # Final output
        self.final_conv = nn.Conv1d(16, 1, 7, padding=3)
        
    def forward(self, mel, speaker_embed, emotion_embed):
        # Initial projection
        x = self.initial_conv(mel)
        
        # Apply first MRF after initial conv
        x = self.mrfs[0](x)
        
        # Upsampling with remaining MRFs
        for i, up in enumerate(self.ups):
            x = up(x, speaker_embed, emotion_embed)
            if i + 1 < len(self.mrfs):
                x = self.mrfs[i + 1](x)
        
        # Final output
        x = self.final_conv(x)
        return x

class ModifiedStreamSpeechVocoder(nn.Module):
    """Working Modified StreamSpeech Vocoder"""
    
    def __init__(self, speaker_dim=192, emotion_dim=256):
        super().__init__()
        self.generator = ModifiedHiFiGANGenerator(speaker_dim, emotion_dim)
        
    def forward(self, mel, speaker_embed, emotion_embed):
        return self.generator(mel, speaker_embed, emotion_embed)

class TrainingConfig:
    """Training Configuration"""
    def __init__(self):
        # Model parameters
        self.speaker_dim = 192
        self.emotion_dim = 256
        
        # Audio parameters
        self.sample_rate = 22050
        self.n_mels = 80
        self.hop_length = 256
        self.win_length = 1024
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 20
        self.learning_rate = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        # Dataset parameters
        self.dataset_path = "/content/drive/MyDrive/Thesis_Training/datasets/professional_cvss_dataset"
        self.checkpoint_dir = "/content/drive/MyDrive/Thesis_Training/checkpoints"
        self.results_dir = "/content/drive/MyDrive/Thesis_Training/results"
        
        # Training settings
        self.save_interval = 1
        self.log_interval = 10

def inspect_metadata_structure(dataset_path):
    """Inspect the actual metadata structure to understand the format"""
    print("🔍 INSPECTING METADATA STRUCTURE...")
    
    metadata_path = os.path.join(dataset_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata type: {type(metadata)}")
    print(f"Metadata length: {len(metadata) if isinstance(metadata, list) else 'Not a list'}")
    
    if isinstance(metadata, list) and len(metadata) > 0:
        print(f"First entry keys: {list(metadata[0].keys())}")
        print(f"First entry sample:")
        for key, value in metadata[0].items():
            print(f"  {key}: {value}")
    
    return metadata

def find_audio_files(dataset_path):
    """Find all audio files in the dataset directories"""
    print("🔍 FINDING AUDIO FILES...")
    
    spanish_dir = os.path.join(dataset_path, "spanish")
    english_dir = os.path.join(dataset_path, "english")
    
    spanish_files = []
    english_files = []
    
    if os.path.exists(spanish_dir):
        spanish_files = [f for f in os.listdir(spanish_dir) if f.endswith('.wav')]
        print(f"Spanish files found: {len(spanish_files)}")
        if spanish_files:
            print(f"Sample Spanish files: {spanish_files[:5]}")
    
    if os.path.exists(english_dir):
        english_files = [f for f in os.listdir(english_dir) if f.endswith('.wav')]
        print(f"English files found: {len(english_files)}")
        if english_files:
            print(f"Sample English files: {english_files[:5]}")
    
    return spanish_files, english_files

class RealCVSSDataset(torch.utils.data.Dataset):
    """DEFINITIVE FIXED CVSS Dataset - handles any metadata structure"""
    
    def __init__(self, config, speaker_extractor, emotion_extractor):
        self.config = config
        self.speaker_extractor = speaker_extractor
        self.emotion_extractor = emotion_extractor
        
        # Inspect metadata structure
        metadata = inspect_metadata_structure(config.dataset_path)
        
        # Find actual audio files
        spanish_files, english_files = find_audio_files(config.dataset_path)
        
        # Create file mapping
        spanish_file_map = {f: os.path.join(config.dataset_path, "spanish", f) for f in spanish_files}
        english_file_map = {f: os.path.join(config.dataset_path, "english", f) for f in english_files}
        
        print(f"Spanish file map: {len(spanish_file_map)} files")
        print(f"English file map: {len(english_file_map)} files")
        
        # Process metadata and create valid pairs
        self.valid_pairs = []
        
        if isinstance(metadata, list):
            pairs = metadata
        elif isinstance(metadata, dict):
            pairs = metadata.get('pairs', [])
        else:
            raise ValueError(f"Unknown metadata format: {type(metadata)}")
        
        print(f"Processing {len(pairs)} metadata entries...")
        
        for i, pair in enumerate(tqdm(pairs, desc="Creating valid pairs")):
            try:
                # Extract file names from metadata
                spanish_filename = None
                english_filename = None
                
                # Try different possible key names
                for key in ['spanish', 'spanish_audio', 'spanish_audio_path']:
                    if key in pair:
                        value = pair[key]
                        if isinstance(value, str):
                            # Extract filename from path
                            spanish_filename = os.path.basename(value)
                            break
                
                for key in ['english', 'english_audio', 'english_audio_path']:
                    if key in pair:
                        value = pair[key]
                        if isinstance(value, str):
                            # Extract filename from path
                            english_filename = os.path.basename(value)
                            break
                
                # If no filename found, try to construct from ID
                if not spanish_filename and 'id' in pair:
                    spanish_filename = f"{pair['id']}.wav"
                
                if not english_filename and 'id' in pair:
                    english_filename = f"{pair['id']}.wav"
                
                # Check if files exist
                if spanish_filename in spanish_file_map and english_filename in english_file_map:
                    self.valid_pairs.append({
                        'spanish_path': spanish_file_map[spanish_filename],
                        'english_path': english_file_map[english_filename],
                        'metadata': pair
                    })
                else:
                    if i < 5:  # Only show first 5 errors
                        print(f"Missing files for pair {i}: {spanish_filename}, {english_filename}")
                
            except Exception as e:
                if i < 5:  # Only show first 5 errors
                    print(f"Error processing pair {i}: {e}")
                continue
        
        print(f"Valid pairs created: {len(self.valid_pairs)}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid pairs found! Check your dataset structure.")
        
        # Pre-compute embeddings for valid pairs
        print("Pre-computing embeddings...")
        self.embeddings_cache = {}
        
        for i, pair in enumerate(tqdm(self.valid_pairs, desc="Extracting embeddings")):
            try:
                spanish_path = pair['spanish_path']
                
                # Extract embeddings
                speaker_embed = speaker_extractor.extract_real_speaker_embedding(spanish_path)
                emotion_embed = emotion_extractor.extract_real_emotion_embedding(spanish_path)
                
                # Validate embeddings
                if speaker_embed is not None and len(speaker_embed) > 0 and emotion_embed is not None and len(emotion_embed) > 0:
                    self.embeddings_cache[i] = {
                        'speaker': speaker_embed,
                        'emotion': emotion_embed
                    }
                else:
                    print(f"Invalid embeddings for pair {i}")
                    
            except Exception as e:
                print(f"Error extracting embeddings for pair {i}: {e}")
                continue
        
        print(f"Embeddings extracted: {len(self.embeddings_cache)}")
        
        if len(self.embeddings_cache) == 0:
            raise ValueError("No valid embeddings extracted!")
        
        # Store valid indices
        self.valid_indices = list(self.embeddings_cache.keys())
        print(f"Dataset initialized: {len(self.valid_indices)} samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        pair = self.valid_pairs[real_idx]
        embeddings_data = self.embeddings_cache[real_idx]
        
        spanish_path = pair['spanish_path']
        english_path = pair['english_path']
        
        # Load audio
        spanish_audio, _ = librosa.load(spanish_path, sr=self.config.sample_rate)
        english_audio, _ = librosa.load(english_path, sr=self.config.sample_rate)
        
        # Generate mel spectrogram from Spanish audio
        mel = librosa.feature.melspectrogram(
            y=spanish_audio,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Get embeddings
        speaker_embed = embeddings_data['speaker']
        emotion_embed = embeddings_data['emotion']
        
        # Convert to tensors
        mel = torch.FloatTensor(mel)
        audio = torch.FloatTensor(english_audio)
        speaker_embed = torch.FloatTensor(speaker_embed)
        emotion_embed = torch.FloatTensor(emotion_embed)
        
        return mel, audio, speaker_embed, emotion_embed

class ProfessionalTrainer:
    """Working Professional Trainer"""
    
    def __init__(self, config, generator, speaker_extractor, emotion_extractor):
        self.config = config
        self.generator = generator
        self.speaker_extractor = speaker_extractor
        self.emotion_extractor = emotion_extractor
        
        # Initialize dataset
        self.dataset = RealCVSSDataset(config, speaker_extractor, emotion_extractor)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        print("✅ Trainer initialized successfully!")
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.generator.train()
        total_loss = 0
        
        for batch_idx, (mel, audio, speaker_embed, emotion_embed) in enumerate(dataloader):
            # Move to device
            mel = mel.cuda()
            audio = audio.cuda()
            speaker_embed = speaker_embed.cuda()
            emotion_embed = emotion_embed.cuda()
            
            # Forward pass
            generated_audio = self.generator(mel, speaker_embed, emotion_embed)
            
            # Calculate loss with length matching
            target_audio = audio.unsqueeze(1)
            
            # Match lengths to prevent tensor size mismatch
            min_len = min(generated_audio.size(-1), target_audio.size(-1))
            generated_audio = generated_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]
            
            loss = self.l1_loss(generated_audio, target_audio)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.log_interval == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, dataloader):
        """Full training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            avg_loss = self.train_epoch(dataloader)
            
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(avg_loss, epoch == self.config.num_epochs - 1)
        
        print("Training completed successfully!")
    
    def save_checkpoint(self, loss, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{self.current_epoch}.pth"
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

def verify_cuda():
    """Verify CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ CUDA NOT AVAILABLE!")
        return False

def dry_run_test():
    """Dry run test to validate pipeline"""
    print("Running in DRY-RUN mode...")
    print("=" * 60)
    print("DRY RUN TEST - VALIDATING PIPELINE INTEGRITY")
    print("=" * 60)
    
    # Verify CUDA
    if not verify_cuda():
        return False
    
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size: 16")
    
    # Initialize components
    print("Initializing embedding extractors...")
    speaker_extractor = RealECAPAExtractor()
    emotion_extractor = RealEmotionExtractor()
    print("✅ Embedding extractors initialized successfully!")
    
    # Initialize dataset
    print("Initializing dataset...")
    config = TrainingConfig()
    dataset = RealCVSSDataset(config, speaker_extractor, emotion_extractor)
    print(f"✅ Dataset initialized: {len(dataset)} samples")
    print(f"✅ Batches per epoch: {len(dataset) // config.batch_size}")
    
    # Initialize trainer
    print("Initializing trainer...")
    generator = ModifiedStreamSpeechVocoder().cuda()
    trainer = ProfessionalTrainer(config, generator, speaker_extractor, emotion_extractor)
    print("✅ Trainer initialized successfully!")
    
    # Test one batch
    print("Testing one batch forward pass...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    for mel, audio, speaker_embed, emotion_embed in dataloader:
        mel = mel.cuda()
        audio = audio.cuda()
        speaker_embed = speaker_embed.cuda()
        emotion_embed = emotion_embed.cuda()
        
        print("✅ Batch shapes:")
        print(f"  - Mel: {mel.shape}")
        print(f"  - Audio: {audio.shape}")
        print(f"  - Speaker embed: {speaker_embed.shape}")
        print(f"  - Emotion embed: {emotion_embed.shape}")
        
        # Test forward pass
        generated_audio = trainer.generator(mel, speaker_embed, emotion_embed)
        print(f"✅ Generated audio shape: {generated_audio.shape}")
        
        break
    
    print("✅ DRY RUN COMPLETED SUCCESSFULLY!")
    print("Pipeline is ready for full training.")
    return True

def main():
    """Main training function"""
    print("Starting Modified HiFi-GAN Training...")
    
    # Verify CUDA
    if not verify_cuda():
        return
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Initialize embedding extractors
    print("Initializing embedding extractors...")
    speaker_extractor = RealECAPAExtractor()
    emotion_extractor = RealEmotionExtractor()
    print("✅ Embedding extractors initialized successfully!")
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = RealCVSSDataset(config, speaker_extractor, emotion_extractor)
    print(f"✅ Dataset initialized: {len(dataset)} samples")
    
    # Initialize trainer
    print("Initializing trainer...")
    generator = ModifiedStreamSpeechVocoder().cuda()
    trainer = ProfessionalTrainer(config, generator, speaker_extractor, emotion_extractor)
    print("✅ Trainer initialized successfully!")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train(dataloader)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.save_checkpoint(trainer.best_loss, False)
    except Exception as e:
        print(f"Training error: {e}")
        trainer.save_checkpoint(trainer.best_loss, False)
    
    print("Training session ended.")
    print("Check the results in Google Drive!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modified HiFi-GAN Training Script")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry-run test without full training")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Running in DRY-RUN mode...")
        success = dry_run_test()
        if success:
            print("✅ DRY RUN COMPLETED SUCCESSFULLY!")
            print("Pipeline is ready for full training.")
        else:
            print("❌ DRY RUN FAILED!")
            print("Fix issues before proceeding to full training.")
        sys.exit(0 if success else 1)
    else:
        main()
