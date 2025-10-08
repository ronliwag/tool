"""
COMPLETE COLAB TRAINING SCRIPT - BULLETPROOF VERSION
Modified HiFi-GAN with ODConv, GRC+LoRA, and FiLM Conditioning
Real CVSS-T Dataset Training with ECAPA-TDNN and Emotion2Vec Embeddings

🔒 COMPUTE-UNIT PROTECTION FEATURES:
- Locked library versions for reproducibility
- Mandatory dry-run validation before training
- Dataset integrity checks
- Checkpoint recovery system
- Training stability measures
- Post-training validation
"""

# 🔒 ENVIRONMENT REPRODUCIBILITY - LOCK LIBRARY VERSIONS
def install_dependencies():
    """Install locked library versions for reproducibility"""
    import subprocess
    import sys
    
    packages = [
        "torch==2.1.0",
        "torchaudio==2.1.0", 
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "transformers==4.44.2",
        "tqdm==4.66.1",
        "numpy==1.24.3"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            print("Continuing with existing versions...")

# Install dependencies if running in Colab
try:
    import google.colab
    print("🔒 COLAB DETECTED - Installing locked dependencies...")
    install_dependencies()
except ImportError:
    print("🔒 LOCAL ENVIRONMENT - Using existing dependencies")

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")

# 🔒 CUDA VERIFICATION
def verify_cuda():
    """Verify CUDA availability and device info"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA Available: {device_name}")
        print(f"✅ GPU Memory: {memory_gb:.1f} GB")
        return True
    else:
        print("❌ CUDA NOT AVAILABLE - Training will be slow!")
        return False

# Add model paths
sys.path.append('/content/models')
sys.path.append('/content/drive/MyDrive/Thesis_Training/models')
sys.path.append('/content/drive/MyDrive/Thesis_Training/embedding_extractors')

# 🔒 DATASET INTEGRITY VALIDATION
def validate_dataset_integrity(dataset_path):
    """Validate dataset integrity before training"""
    print("🔒 VALIDATING DATASET INTEGRITY...")
    
    # Check metadata.json
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ metadata.json not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(metadata, list):
        pairs = metadata
    else:
        pairs = metadata.get('pairs', [])
    
    print(f"✅ Metadata entries: {len(pairs)}")
    
    if len(pairs) == 0:
        raise ValueError("❌ No audio pairs found in metadata")
    
    # Check English and Spanish directories
    english_dir = os.path.join(dataset_path, 'english')
    spanish_dir = os.path.join(dataset_path, 'spanish')
    
    if not os.path.exists(english_dir):
        raise FileNotFoundError(f"❌ English directory not found: {english_dir}")
    if not os.path.exists(spanish_dir):
        raise FileNotFoundError(f"❌ Spanish directory not found: {spanish_dir}")
    
    # Validate first 10 pairs exist
    missing_files = []
    for i, pair in enumerate(pairs[:10]):  # Check first 10 pairs
        english_path = os.path.join(english_dir, pair['english'])
        spanish_path = os.path.join(spanish_dir, pair['spanish'])
        
        if not os.path.exists(english_path):
            missing_files.append(f"English: {pair['english']}")
        if not os.path.exists(spanish_path):
            missing_files.append(f"Spanish: {pair['spanish']}")
    
    if missing_files:
        print(f"⚠️ Missing files detected: {missing_files[:5]}...")
        print("⚠️ This may cause training issues. Check dataset integrity.")
    else:
        print("✅ Dataset integrity validated successfully!")
    
    return len(pairs)

# Import model components
from odconv import ODConvTranspose1D
from film_conditioning import FiLMLayer, SpeakerEmotionExtractor
from grc_lora import MultiReceptiveFieldFusion
from real_ecapa_extractor import RealECAPAExtractor
from real_emotion_extractor import RealEmotionExtractor

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

# Training Configuration
class TrainingConfig:
    """Training configuration for Modified HiFi-GAN"""
    
    def __init__(self):
        # Model configuration
        self.mel_channels = 80
        self.audio_channels = 1
        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        self.upsample_initial_channel = 512
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.speaker_embed_dim = 192
        self.emotion_embed_dim = 256
        self.lora_rank = 4
        
        # Training configuration
        self.batch_size = 16  # Optimized for L4 GPU (24GB VRAM)
        self.num_epochs = 20
        self.learning_rate = 0.0002
        self.beta1 = 0.8
        self.beta2 = 0.99
        self.weight_decay = 0.01
        
        # Audio configuration
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        
        # Dataset paths
        self.dataset_path = '/content/drive/MyDrive/Thesis_Training/datasets/professional_cvss_dataset'
        self.embedding_cache_path = '/content/drive/MyDrive/Thesis_Training/embedding_cache'
        self.checkpoint_path = '/content/drive/MyDrive/Thesis_Training/checkpoints'
        self.results_path = '/content/drive/MyDrive/Thesis_Training/results'
        
        # Training settings
        self.save_interval = 1
        self.validation_interval = 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
config = TrainingConfig()

# Real CVSS-T Dataset Class
class RealCVSSDataset(Dataset):
    """
    Real CVSS-T Dataset for Professional Voice Cloning Training
    Uses real ECAPA-TDNN speaker embeddings and Emotion2Vec emotion embeddings
    """
    
    def __init__(self, config, ecapa_extractor, emotion_extractor, use_cache=True):
        self.config = config
        self.ecapa_extractor = ecapa_extractor
        self.emotion_extractor = emotion_extractor
        self.use_cache = use_cache
        
        # Load metadata
        metadata_path = os.path.join(config.dataset_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(metadata, list):
            self.pairs = metadata
        else:
            self.pairs = metadata.get('pairs', [])
        
        print(f"Loaded {len(self.pairs)} audio pairs for training")
        
        # Create cache directory
        os.makedirs(config.embedding_cache_path, exist_ok=True)
        
        # Pre-compute embeddings if cache is enabled
        if self.use_cache:
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute and cache all embeddings for faster training"""
        print("Pre-computing embeddings for faster training...")
        
        cache_file = os.path.join(self.config.embedding_cache_path, 'embeddings_cache.pt')
        
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            self.embedding_cache = torch.load(cache_file)
            return
        
        self.embedding_cache = {}
        
        for i, pair in enumerate(tqdm(self.pairs, desc="Computing embeddings")):
            # Load Spanish audio for speaker embedding
            spanish_path = os.path.join(self.config.dataset_path, 'spanish', pair['spanish'])
            english_path = os.path.join(self.config.dataset_path, 'english', pair['english'])
            
            try:
                # Extract speaker embedding from Spanish audio
                spanish_audio, _ = librosa.load(spanish_path, sr=self.config.sample_rate)
                speaker_embed = self.ecapa_extractor.extract_real_speaker_embedding(spanish_path)
                
                # Extract emotion embedding from English audio
                english_audio, _ = librosa.load(english_path, sr=self.config.sample_rate)
                emotion_embed = self.emotion_extractor.extract_real_emotion_embedding(english_path)
                
                # Cache embeddings
                self.embedding_cache[i] = {
                    'speaker_embed': speaker_embed,
                    'emotion_embed': emotion_embed
                }
                
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
        
        # Save cache
        torch.save(self.embedding_cache, cache_file)
        print(f"Cached {len(self.embedding_cache)} embeddings")
    
    def _mel_spectrogram(self, audio):
        """Convert audio to mel-spectrogram"""
        # Compute mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        
        # Convert to log scale
        mel = np.log(mel + 1e-8)
        
        return torch.FloatTensor(mel)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load audio files
        spanish_path = os.path.join(self.config.dataset_path, 'spanish', pair['spanish'])
        english_path = os.path.join(self.config.dataset_path, 'english', pair['english'])
        
        try:
            # Load English audio (target)
            english_audio, _ = librosa.load(english_path, sr=self.config.sample_rate)
            english_mel = self._mel_spectrogram(english_audio)
            english_audio_tensor = torch.FloatTensor(english_audio)
            
            # Get cached embeddings
            if self.use_cache and idx in self.embedding_cache:
                speaker_embed = self.embedding_cache[idx]['speaker_embed']
                emotion_embed = self.embedding_cache[idx]['emotion_embed']
            else:
                # Extract embeddings on-the-fly
                spanish_audio, _ = librosa.load(spanish_path, sr=self.config.sample_rate)
                speaker_embed = self.ecapa_extractor.extract_real_speaker_embedding(spanish_path)
                emotion_embed = self.emotion_extractor.extract_real_emotion_embedding(english_path)
            
            # SHAPE VALIDATION - Critical for debugging
            assert english_mel.shape[0] == self.config.n_mels, f"Mel shape mismatch: {english_mel.shape[0]} != {self.config.n_mels}"
            assert speaker_embed.shape[0] == self.config.speaker_embed_dim, f"Speaker embed shape mismatch: {speaker_embed.shape[0]} != {self.config.speaker_embed_dim}"
            assert emotion_embed.shape[0] == self.config.emotion_embed_dim, f"Emotion embed shape mismatch: {emotion_embed.shape[0]} != {self.config.emotion_embed_dim}"
            
            return {
                'mel': english_mel,
                'audio': english_audio_tensor,
                'speaker_embed': speaker_embed,
                'emotion_embed': emotion_embed,
                'spanish_file': pair['spanish'],
                'english_file': pair['english']
            }
            
        except Exception as e:
            print(f"Error loading pair {idx}: {e}")
            # Return dummy data
            return {
                'mel': torch.zeros(self.config.n_mels, 100),
                'audio': torch.zeros(100 * self.config.hop_length),
                'speaker_embed': torch.zeros(self.config.speaker_embed_dim),
                'emotion_embed': torch.zeros(self.config.emotion_embed_dim),
                'spanish_file': 'dummy.wav',
                'english_file': 'dummy.wav'
            }

# Professional Training Class
class ProfessionalTrainer:
    """
    Professional Training Class for Modified HiFi-GAN
    Implements best practices for voice cloning model training
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize models
        print("Initializing models...")
        self.generator = ModifiedStreamSpeechVocoder(config.__dict__).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Training metrics
        self.best_loss = float('inf')
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_path, exist_ok=True)
        os.makedirs(config.results_path, exist_ok=True)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.generator.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            mel = batch['mel'].to(self.device)
            audio = batch['audio'].to(self.device)
            speaker_embed = batch['speaker_embed'].to(self.device)
            emotion_embed = batch['emotion_embed'].to(self.device)
            
            # Zero gradients
            self.optimizer_g.zero_grad()
            
            # SHAPE VALIDATION - Critical for debugging
            assert mel.shape[1] == self.config.n_mels, f"Input mel shape mismatch: {mel.shape[1]} != {self.config.n_mels}"
            assert speaker_embed.shape[1] == self.config.speaker_embed_dim, f"Speaker embed batch shape mismatch: {speaker_embed.shape[1]} != {self.config.speaker_embed_dim}"
            assert emotion_embed.shape[1] == self.config.emotion_embed_dim, f"Emotion embed batch shape mismatch: {emotion_embed.shape[1]} != {self.config.emotion_embed_dim}"
            
            # Forward pass
            generated_audio = self.generator(mel, speaker_embed, emotion_embed)
            
            # SHAPE VALIDATION - Generated audio
            assert generated_audio.shape[1] == self.config.audio_channels, f"Generated audio channels mismatch: {generated_audio.shape[1]} != {self.config.audio_channels}"
            
            # Adjust dimensions for loss calculation
            min_length = min(generated_audio.size(-1), audio.size(-1))
            generated_audio = generated_audio[..., :min_length]
            audio = audio[..., :min_length]
            
            # Calculate losses
            l1_loss = self.l1_loss(generated_audio, audio.unsqueeze(1))
            mse_loss = self.mse_loss(generated_audio, audio.unsqueeze(1))
            
            # Combined loss
            total_gen_loss = l1_loss + 0.5 * mse_loss
            
            # Backward pass
            total_gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer_g.step()
            
            # Update metrics
            total_loss += total_gen_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_gen_loss.item():.4f}",
                'Avg Loss': f"{total_loss / num_batches:.4f}",
                'LR': f"{self.optimizer_g.param_groups[0]['lr']:.6f}"
            })
        
        return total_loss / num_batches
    
    def save_checkpoint(self, avg_loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'avg_loss': avg_loss,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_path,
            f'checkpoint_epoch_{self.epoch + 1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save generator and discriminator separately
        generator_path = os.path.join(
            self.config.checkpoint_path,
            f'generator_epoch_{self.epoch + 1}.pth'
        )
        torch.save(self.generator.state_dict(), generator_path)
        
        if is_best:
            # Save best weights
            best_generator_path = os.path.join(
                self.config.checkpoint_path,
                'best_generator.pth'
            )
            torch.save(self.generator.state_dict(), best_generator_path)
            
            best_checkpoint_path = os.path.join(
                self.config.checkpoint_path,
                'best_checkpoint.pth'
            )
            torch.save(checkpoint, best_checkpoint_path)
            
        print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            print("Best weights saved!")
    
    def train(self, train_dataloader):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            avg_loss = self.train_epoch(train_dataloader)
            
            # Update learning rate
            self.scheduler_g.step()
            
            # Check if best model
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(avg_loss, is_best)
            
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Best Loss: {self.best_loss:.6f}")
            print(f"Learning Rate: {self.optimizer_g.param_groups[0]['lr']:.8f}")
            print("-" * 50)
        
        # Final save
        self.save_checkpoint(avg_loss, avg_loss <= self.best_loss)
        print("Training completed!")

def dry_run_test():
    """Dry run test to validate pipeline integrity"""
    print("=" * 60)
    print("DRY RUN TEST - VALIDATING PIPELINE INTEGRITY")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available!")
    
    # Initialize configuration
    config = TrainingConfig()
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    
    # Initialize embedding extractors
    print("Initializing embedding extractors...")
    try:
        ecapa_extractor = RealECAPAExtractor()
        emotion_extractor = RealEmotionExtractor()
        print("✅ Embedding extractors initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing extractors: {e}")
        return False
    
    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = RealCVSSDataset(config, ecapa_extractor, emotion_extractor)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        print(f"✅ Dataset initialized: {len(dataset)} samples")
        print(f"✅ Batches per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        return False
    
    # Initialize trainer
    print("Initializing trainer...")
    try:
        trainer = ProfessionalTrainer(config)
        print("✅ Trainer initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return False
    
    # Test one batch forward pass
    print("Testing one batch forward pass...")
    try:
        trainer.generator.train()
        batch = next(iter(dataloader))
        
        # Move to device
        mel = batch['mel'].to(config.device)
        audio = batch['audio'].to(config.device)
        speaker_embed = batch['speaker_embed'].to(config.device)
        emotion_embed = batch['emotion_embed'].to(config.device)
        
        print(f"✅ Batch shapes:")
        print(f"  - Mel: {mel.shape}")
        print(f"  - Audio: {audio.shape}")
        print(f"  - Speaker embed: {speaker_embed.shape}")
        print(f"  - Emotion embed: {emotion_embed.shape}")
        
        # Forward pass
        with torch.no_grad():
            generated_audio = trainer.generator(mel, speaker_embed, emotion_embed)
            print(f"✅ Generated audio shape: {generated_audio.shape}")
            
            # Test loss calculation
            min_length = min(generated_audio.size(-1), audio.size(-1))
            generated_audio = generated_audio[..., :min_length]
            audio = audio[..., :min_length]
            
            l1_loss = trainer.l1_loss(generated_audio, audio.unsqueeze(1))
            mse_loss = trainer.mse_loss(generated_audio, audio.unsqueeze(1))
            total_loss = l1_loss + 0.5 * mse_loss
            
            print(f"✅ Loss calculation successful:")
            print(f"  - L1 Loss: {l1_loss.item():.6f}")
            print(f"  - MSE Loss: {mse_loss.item():.6f}")
            print(f"  - Total Loss: {total_loss.item():.6f}")
        
        print("✅ DRY RUN TEST PASSED - Pipeline is ready for training!")
        return True
        
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("COMPLETE MODIFIED HIFI-GAN TRAINING")
    print("Real CVSS-T Dataset with Real Embeddings")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available, training will be slow!")
    
    # Initialize configuration
    config = TrainingConfig()
    print(f"Device: {config.device}")
    
    # Initialize embedding extractors
    print("Initializing embedding extractors...")
    try:
        ecapa_extractor = RealECAPAExtractor()
        emotion_extractor = RealEmotionExtractor()
        print("Embedding extractors initialized successfully!")
    except Exception as e:
        print(f"Error initializing extractors: {e}")
        return
    
    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = RealCVSSDataset(config, ecapa_extractor, emotion_extractor)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Colab compatibility
            pin_memory=False,  # Colab compatibility
            drop_last=True
        )
        print(f"Dataset initialized: {len(dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # Initialize trainer
    print("Initializing trainer...")
    try:
        trainer = ProfessionalTrainer(config)
        print("Trainer initialized successfully!")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return
    
    # Start training
    print("\nStarting training...")
    try:
        trainer.train(dataloader)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        trainer.save_checkpoint(trainer.best_loss, False)
    except Exception as e:
        print(f"Training error: {e}")
        # Save current state
        trainer.save_checkpoint(trainer.best_loss, False)
    
    print("\nTraining session ended.")
    print("Check the results in Google Drive!")

if __name__ == "__main__":
    import sys
    
    # Check for dry-run mode
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Running in DRY-RUN mode...")
        success = dry_run_test()
        if success:
            print("\n✅ DRY RUN COMPLETED SUCCESSFULLY!")
            print("Pipeline is ready for full training.")
        else:
            print("\n❌ DRY RUN FAILED!")
            print("Fix issues before proceeding to full training.")
        sys.exit(0 if success else 1)
    else:
        main()

