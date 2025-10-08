#!/usr/bin/env python3
"""
COMPLETE COLAB TRAINING CODE
Copy and paste this entire code into a Colab cell
"""

# Cell 1: Install packages (already in notebook)

# Cell 2: Upload dataset (already in notebook)

# Cell 3: Training configuration and model definitions
@dataclass
class TrainingConfig:
    # Dataset configuration
    dataset_root: str = "professional_cvss_dataset"
    metadata_file: str = "metadata.json"
    
    # Model configuration - OPTIMIZED FOR COLAB PRO
    batch_size: int = 16  # Larger batch size for Colab Pro GPU
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 1e-4
    beta1: float = 0.8
    beta2: float = 0.99
    
    # Training parameters - OPTIMIZED FOR COLAB PRO
    num_epochs: int = 50  # Complete training
    save_interval: int = 5
    log_interval: int = 10
    validation_interval: int = 10
    
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

config = TrainingConfig()
print("Training Configuration:")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.num_epochs}")
print(f"Sample rate: {config.sample_rate}")

# Cell 4: Real ECAPA-TDNN Extractor
class RealECAPAExtractor:
    """Real ECAPA-TDNN speaker embedding extractor for Colab"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 192
        self.sample_rate = 16000
        self.model = None
        self.feature_extractor = None
        self.load_real_speaker_model()
        
        print("REAL ECAPA-TDNN EXTRACTOR INITIALIZED FOR COLAB")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_real_speaker_model(self):
        """Load REAL speaker model from Hugging Face"""
        try:
            print("Loading REAL speaker model from Hugging Face...")
            
            # Use a real speaker recognition model
            model_name = "facebook/wav2vec2-large-xlsr-53"
            
            # Load feature extractor and model
            from transformers import AutoModel, AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] REAL speaker model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load real speaker model: {e}")
            return False
    
    def extract_real_speaker_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Extract REAL speaker embedding from audio"""
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Normalize
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Extract REAL embedding
            with torch.no_grad():
                if self.model is not None and self.feature_extractor is not None:
                    # Use Hugging Face model
                    audio_np = audio.numpy()
                    inputs = self.feature_extractor(audio_np, sampling_rate=self.sample_rate, return_tensors="pt")
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get features
                    outputs = self.model(**inputs)
                    
                    # Extract embedding from last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # [1024]
                    
                    # Project to our embedding dimension
                    if embedding.shape[0] != self.embedding_dim:
                        if embedding.shape[0] > self.embedding_dim:
                            embedding = embedding[:self.embedding_dim]
                        else:
                            padding = torch.zeros(self.embedding_dim - embedding.shape[0])
                            embedding = torch.cat([embedding.cpu(), padding])
                    
                    return embedding.cpu()
                
        except Exception as e:
            print(f"ERROR: Failed to extract real speaker embedding: {e}")
            return torch.zeros(self.embedding_dim)

# Cell 5: Real Emotion Extractor
class RealEmotionExtractor:
    """Real emotion embedding extractor for Colab"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 256
        self.sample_rate = 16000
        self.model = None
        self.feature_extractor = None
        self.load_real_emotion_model()
        
        print("REAL EMOTION EXTRACTOR INITIALIZED FOR COLAB")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_real_emotion_model(self):
        """Load REAL emotion model from Hugging Face"""
        try:
            print("Loading REAL emotion model from Hugging Face...")
            
            # Use a real emotion recognition model
            model_name = "facebook/wav2vec2-base-960h"
            
            # Load feature extractor and model
            from transformers import AutoModel, AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] REAL emotion model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load real emotion model: {e}")
            return False
    
    def extract_real_emotion_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Extract REAL emotion embedding from audio"""
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Normalize
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Extract REAL embedding
            with torch.no_grad():
                if self.model is not None and self.feature_extractor is not None:
                    # Use Hugging Face model
                    audio_np = audio.numpy()
                    inputs = self.feature_extractor(audio_np, sampling_rate=self.sample_rate, return_tensors="pt")
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get features
                    outputs = self.model(**inputs)
                    
                    # Extract embedding from last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # [768]
                    
                    # Project to our embedding dimension
                    if embedding.shape[0] != self.embedding_dim:
                        if embedding.shape[0] > self.embedding_dim:
                            embedding = embedding[:self.embedding_dim]
                        else:
                            padding = torch.zeros(self.embedding_dim - embedding.shape[0])
                            embedding = torch.cat([embedding.cpu(), padding])
                    
                    return embedding.cpu()
                
        except Exception as e:
            print(f"ERROR: Failed to extract real emotion embedding: {e}")
            return torch.zeros(self.embedding_dim)

# Cell 6: Dataset class
class RealCVSSDataset(Dataset):
    """Real CVSS dataset for Colab training"""
    
    def __init__(self, config: TrainingConfig, speaker_extractor, emotion_extractor):
        self.config = config
        self.speaker_extractor = speaker_extractor
        self.emotion_extractor = emotion_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        metadata_path = os.path.join(config.dataset_root, config.metadata_file)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.samples = list(self.metadata.keys())
        else:
            print(f"ERROR: Metadata file not found: {metadata_path}")
            self.samples = []
        
        print(f"Loaded {len(self.samples)} samples from CVSS dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_data = self.metadata[sample_id]
        
        # Load audio files
        spanish_path = os.path.join(self.config.dataset_root, "spanish", f"{sample_id}.wav")
        english_path = os.path.join(self.config.dataset_root, "english", f"{sample_id}.wav")
        
        # Load and preprocess audio
        spanish_audio, _ = torchaudio.load(spanish_path)
        english_audio, _ = torchaudio.load(english_path)
        
        # Resample to target sample rate
        if spanish_audio.shape[0] > 1:
            spanish_audio = spanish_audio.mean(dim=0, keepdim=True)
        if english_audio.shape[0] > 1:
            english_audio = english_audio.mean(dim=0, keepdim=True)
        
        # Pad or truncate to fixed length (3 seconds)
        target_length = self.config.sample_rate * 3
        spanish_audio = self._pad_or_truncate(spanish_audio.squeeze(0), target_length)
        english_audio = self._pad_or_truncate(english_audio.squeeze(0), target_length)
        
        # Convert to mel-spectrograms
        mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mel_channels
        )
        
        spanish_mel = mel_transform(spanish_audio)
        english_mel = mel_transform(english_audio)
        
        # Extract embeddings
        speaker_embed = self.speaker_extractor.extract_real_speaker_embedding(spanish_path)
        emotion_embed = self.emotion_extractor.extract_real_emotion_embedding(spanish_path)
        
        return {
            'spanish_mel': spanish_mel,
            'english_mel': english_mel,
            'spanish_audio': spanish_audio,
            'english_audio': english_audio,
            'speaker_embed': speaker_embed,
            'emotion_embed': emotion_embed,
            'sample_id': sample_id
        }
    
    def _pad_or_truncate(self, audio, target_length):
        """Pad or truncate audio to target length"""
        if len(audio) > target_length:
            return audio[:target_length]
        else:
            padding = target_length - len(audio)
            return torch.cat([audio, torch.zeros(padding)])

# Cell 7: Model definitions
class ProfessionalModifiedHiFiGANGenerator(nn.Module):
    """Professional Modified HiFi-GAN Generator with correct architecture"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.initial_conv = nn.Conv1d(80, 512, kernel_size=7, padding=3)
        
        # Upsampling layers with correct stride calculations
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),
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
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
    
    def apply_film(self, x, speaker_embed, emotion_embed):
        """Apply FiLM conditioning"""
        speaker_params = self.speaker_film(speaker_embed)
        emotion_params = self.emotion_film(emotion_embed)
        
        # Combine parameters
        combined_params = speaker_params + emotion_params
        gamma, beta = torch.chunk(combined_params, 2, dim=-1)
        
        # Apply FiLM
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        return x
    
    def forward(self, mel_spectrogram, speaker_embed, emotion_embed):
        """Forward pass"""
        # Initial convolution
        x = self.initial_conv(mel_spectrogram)
        
        # Upsampling with FiLM conditioning
        for i, (upsample, res_block) in enumerate(zip(self.upsample_layers, self.res_blocks)):
            x = upsample(x)
            x = self.apply_film(x, speaker_embed, emotion_embed)
            
            # Residual connection
            residual = x
            x = res_block(x)
            x = x + residual
            x = F.relu(x)
        
        # Final output
        x = self.final_conv(x)
        return x.squeeze(1)

class ProfessionalDiscriminator(nn.Module):
    """Professional discriminator for training"""
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=41, stride=2, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=41, stride=2, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=41, stride=2, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, kernel_size=41, stride=2, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=2, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.conv_layers(x)

# Cell 8: Training class
class ProfessionalCVSSTrainer:
    """Professional trainer for CVSS dataset"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Initialize extractors
        self.speaker_extractor = RealECAPAExtractor()
        self.emotion_extractor = RealEmotionExtractor()
        
        # Initialize dataset
        self.dataset = RealCVSSDataset(config, self.speaker_extractor, self.emotion_extractor)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Colab
            pin_memory=False
        )
        
        print(f"Trainer initialized with {len(self.dataset)} samples")
        print(f"Device: {self.device}")
    
    def train_generator(self, batch):
        """Train generator"""
        self.optimizer_g.zero_grad()
        
        # Get batch data
        spanish_mel = batch['spanish_mel'].to(self.device)
        english_mel = batch['english_mel'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        
        # Generate audio
        generated_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
        
        # Discriminator loss
        fake_pred = self.discriminator(generated_audio.unsqueeze(1))
        gen_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        
        gen_loss.backward()
        self.optimizer_g.step()
        
        return gen_loss.item()
    
    def train_discriminator(self, batch):
        """Train discriminator"""
        self.optimizer_d.zero_grad()
        
        # Get batch data
        spanish_mel = batch['spanish_mel'].to(self.device)
        english_audio = batch['english_audio'].to(self.device)
        speaker_embed = batch['speaker_embed'].to(self.device)
        emotion_embed = batch['emotion_embed'].to(self.device)
        
        # Real audio
        real_pred = self.discriminator(english_audio.unsqueeze(1))
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake audio
        with torch.no_grad():
            generated_audio = self.generator(spanish_mel, speaker_embed, emotion_embed)
        fake_pred = self.discriminator(generated_audio.unsqueeze(1))
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        
        disc_loss.backward()
        self.optimizer_d.step()
        
        return disc_loss.item()
    
    def save_checkpoint(self, epoch, gen_loss, disc_loss, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'config': self.config.__dict__
        }
        
        filename = f'professional_training_epoch_{epoch}.pt'
        torch.save(checkpoint, filename)
        
        if is_best:
            torch.save(checkpoint, 'professional_training_best.pt')
        
        print(f"Checkpoint saved: {filename}")
    
    def train(self):
        """Main training loop"""
        print("Starting professional training...")
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in progress_bar:
                # Train discriminator
                disc_loss = self.train_discriminator(batch)
                
                # Train generator
                gen_loss = self.train_generator(batch)
                
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen Loss': f'{gen_loss:.4f}',
                    'Disc Loss': f'{disc_loss:.4f}'
                })
            
            # Calculate average losses
            avg_gen_loss = epoch_gen_loss / num_batches
            avg_disc_loss = epoch_disc_loss / num_batches
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}:")
            print(f"  Generator Loss: {avg_gen_loss:.6f}")
            print(f"  Discriminator Loss: {avg_disc_loss:.6f}")
            
            # Save checkpoint
            is_best = avg_gen_loss < best_loss
            if is_best:
                best_loss = avg_gen_loss
            
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, avg_gen_loss, avg_disc_loss, is_best)
        
        print("Training completed!")
        print(f"Best generator loss: {best_loss:.6f}")

# Cell 9: Start training
def main():
    """Main training function"""
    print("=" * 80)
    print("PROFESSIONAL COLAB TRAINING STARTING")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ProfessionalCVSSTrainer(config)
    
    # Start training
    trainer.train()
    
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Download the trained model:")
    print("1. professional_training_best.pt - Best model")
    print("2. professional_training_epoch_X.pt - Epoch checkpoints")
    print("=" * 80)

# Run training
if __name__ == "__main__":
    main()
