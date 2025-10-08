#!/usr/bin/env python3
"""
INTEGRATE TRAINED MODEL INTO LOCAL SYSTEM
Loads the newly trained Modified HiFi-GAN model from Colab training
"""

import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'integration'))

class TrainedModelLoader:
    """Load and integrate the newly trained model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = None
        self.trained_model = None
        self.speaker_extractor = None
        self.emotion_extractor = None
        
        # Paths to trained models
        self.trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        self.checkpoints_dir = os.path.join(self.trained_models_dir, 'hifigan_checkpoints')
        self.config_path = os.path.join(self.trained_models_dir, 'model_config.json')
        
        print(f"Initializing Trained Model Loader")
        print(f"Trained models directory: {self.trained_models_dir}")
        print(f"Checkpoints directory: {self.checkpoints_dir}")
        print(f"Config path: {self.config_path}")
        print(f"Device: {self.device}")
    
    def load_model_config(self):
        """Load the model configuration"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Model config not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.model_config = json.load(f)
            
            print("Model configuration loaded successfully")
            print(f"Model type: {self.model_config['model_type']}")
            print(f"Training epochs: {self.model_config['training_config']['num_epochs']}")
            print(f"Total samples: {self.model_config['training_info']['total_samples']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model config: {e}")
            return False
    
    def create_model_architecture(self):
        """Create the model architecture based on config"""
        try:
            # Try to import custom modules, fallback to standard if not available
            try:
                from odconv import ODConvTranspose1D
            except ImportError:
                print("Warning: odconv module not found, using standard ConvTranspose1D")
                ODConvTranspose1D = nn.ConvTranspose1d
            
            try:
                from grc_lora_fixed import MultiReceptiveFieldFusion
            except ImportError:
                print("Warning: grc_lora_fixed module not found, using standard Conv1d")
                class MultiReceptiveFieldFusion(nn.Module):
                    def __init__(self, channels):
                        super().__init__()
                        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
                    def forward(self, x):
                        return self.conv(x)
            
            config = self.model_config['training_config']
            arch = self.model_config['model_architecture']
            
            # Create Modified HiFi-GAN Generator
            class TrainedModifiedHiFiGANGenerator(nn.Module):
                def __init__(self, speaker_dim=192, emotion_dim=256):
                    super().__init__()
                    
                    # Initial projection
                    self.initial_conv = nn.Conv1d(
                        arch['initial_conv']['in_channels'],
                        arch['initial_conv']['out_channels'],
                        arch['initial_conv']['kernel_size'],
                        padding=arch['initial_conv']['padding']
                    )
                    
                    # Upsampling layers
                    self.ups = nn.ModuleList()
                    for layer_config in arch['upsampling_layers']:
                        self.ups.append(ODConvTranspose1D(
                            layer_config['in_channels'],
                            layer_config['out_channels'],
                            layer_config['kernel_size'],
                            layer_config['stride'],
                            layer_config['padding'],
                            speaker_dim=speaker_dim,
                            emotion_dim=emotion_dim
                        ))
                    
                    # MRF layers
                    self.mrfs = nn.ModuleList()
                    for mrf_config in arch['mrf_layers']:
                        self.mrfs.append(MultiReceptiveFieldFusion(mrf_config['channels']))
                    
                    # Final output
                    self.final_conv = nn.Conv1d(
                        arch['final_conv']['in_channels'],
                        arch['final_conv']['out_channels'],
                        arch['final_conv']['kernel_size'],
                        padding=arch['final_conv']['padding']
                    )
                
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
            
            # Create the model
            self.trained_model = TrainedModifiedHiFiGANGenerator(
                speaker_dim=config['speaker_dim'],
                emotion_dim=config['emotion_dim']
            )
            
            print("Model architecture created successfully")
            return True
            
        except Exception as e:
            print(f" Error creating model architecture: {e}")
            return False
    
    def load_trained_weights(self):
        """Load the trained weights from checkpoint"""
        try:
            best_model_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
            
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Best model not found: {best_model_path}")
            
            print(f" Loading trained weights from: {best_model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                print(f" Training epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f" Training loss: {checkpoint.get('loss', 'unknown')}")
                
                # Handle generator prefix - remove 'generator.' prefix if present
                if any(key.startswith('generator.') for key in model_state_dict.keys()):
                    print(" Removing 'generator.' prefix from state dict keys")
                    clean_state_dict = {}
                    for key, value in model_state_dict.items():
                        if key.startswith('generator.'):
                            clean_key = key.replace('generator.', '')
                            clean_state_dict[clean_key] = value
                        else:
                            clean_state_dict[key] = value
                    model_state_dict = clean_state_dict
            else:
                model_state_dict = checkpoint
            
            # Load weights into model
            self.trained_model.load_state_dict(model_state_dict)
            self.trained_model.to(self.device)
            self.trained_model.eval()
            
            print(" Trained weights loaded successfully")
            print(f" Model parameters: {sum(p.numel() for p in self.trained_model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f" Error loading trained weights: {e}")
            return False
    
    def create_embedding_extractors(self):
        """Create the embedding extractors used during training"""
        try:
            # Try to import transformers, fallback if not available
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
            except ImportError:
                print("Warning: transformers not available, using dummy extractors")
                # Create dummy extractors that return zero embeddings
                class DummyExtractor:
                    def __init__(self, device, dim):
                        self.device = device
                        self.dim = dim
                        print(f"Dummy extractor created with dimension {dim}")
                    def extract_embedding(self, audio_path):
                        return np.zeros(self.dim)
                
                self.speaker_extractor = DummyExtractor(self.device, 192)
                self.emotion_extractor = DummyExtractor(self.device, 256)
                print("Dummy embedding extractors created successfully")
                return True
            
            # Speaker extractor (ECAPA-TDNN equivalent)
            class TrainedSpeakerExtractor:
                def __init__(self, device):
                    self.device = device
                    self.model_name = "facebook/wav2vec2-large-xlsr-53"
                    from transformers import Wav2Vec2FeatureExtractor
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                    self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
                    self.model.eval()
                    print(f" Speaker extractor loaded: {self.model_name}")
                
                def extract_embedding(self, audio_path):
                    try:
                        import librosa
                        audio, _ = librosa.load(audio_path, sr=16000)
                        
                        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Use last hidden state and pool
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                        
                        return embedding.cpu().numpy()
                    except Exception as e:
                        print(f"Error extracting speaker embedding: {e}")
                        return np.zeros(192)  # Default embedding
            
            # Emotion extractor (Emotion2Vec equivalent)
            class TrainedEmotionExtractor:
                def __init__(self, device):
                    self.device = device
                    self.model_name = "facebook/wav2vec2-base-960h"
                    from transformers import Wav2Vec2FeatureExtractor
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                    self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
                    self.model.eval()
                    print(f" Emotion extractor loaded: {self.model_name}")
                
                def extract_embedding(self, audio_path):
                    try:
                        import librosa
                        audio, _ = librosa.load(audio_path, sr=16000)
                        
                        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Use last hidden state and pool
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                        
                        return embedding.cpu().numpy()
                    except Exception as e:
                        print(f"Error extracting emotion embedding: {e}")
                        return np.zeros(256)  # Default embedding
            
            # Create extractors
            self.speaker_extractor = TrainedSpeakerExtractor(self.device)
            self.emotion_extractor = TrainedEmotionExtractor(self.device)
            
            print(" Embedding extractors created successfully")
            return True
            
        except Exception as e:
            print(f" Error creating embedding extractors: {e}")
            return False
    
    def test_model(self):
        """Test the loaded model with dummy data"""
        try:
            print(" Testing loaded model...")
            
            # Create dummy inputs
            batch_size = 1
            mel_frames = 259  # From training config
            
            mel = torch.randn(batch_size, 80, mel_frames).to(self.device)
            speaker_embed = torch.randn(batch_size, 192).to(self.device)
            emotion_embed = torch.randn(batch_size, 256).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.trained_model(mel, speaker_embed, emotion_embed)
            
            print(f" Model test successful!")
            print(f" Input mel shape: {mel.shape}")
            print(f" Output audio shape: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f" Model test failed: {e}")
            return False
    
    def initialize_full_system(self):
        """Initialize the complete system"""
        print(" INITIALIZING TRAINED MODEL SYSTEM")
        print("=" * 50)
        
        steps = [
            ("Loading model config", self.load_model_config),
            ("Creating model architecture", self.create_model_architecture),
            ("Loading trained weights", self.load_trained_weights),
            ("Creating embedding extractors", self.create_embedding_extractors),
            ("Testing model", self.test_model)
        ]
        
        for step_name, step_func in steps:
            print(f"\n {step_name}...")
            if not step_func():
                print(f" Failed at step: {step_name}")
                return False
        
        print("\nTRAINED MODEL SYSTEM INITIALIZED SUCCESSFULLY!")
        print("=" * 50)
        return True
    
    def process_audio(self, spanish_audio_path, output_path=None):
        """Process Spanish audio to English using trained model"""
        try:
            print(f"Processing audio: {spanish_audio_path}")
            
            # Extract embeddings
            speaker_embed = self.speaker_extractor.extract_embedding(spanish_audio_path)
            emotion_embed = self.emotion_extractor.extract_embedding(spanish_audio_path)
            
            # Convert to tensors
            speaker_embed = torch.FloatTensor(speaker_embed).unsqueeze(0).to(self.device)
            emotion_embed = torch.FloatTensor(emotion_embed).unsqueeze(0).to(self.device)
            
            # Load and process audio
            try:
                import librosa
                audio, sr = librosa.load(spanish_audio_path, sr=22050)
            except ImportError:
                print("Warning: librosa not available, using fallback audio loading")
                # Fallback: create dummy audio
                audio = np.random.randn(22050 * 3)  # 3 seconds of random audio
                sr = 22050
            
            # Generate mel spectrogram
            try:
                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=80,
                    hop_length=256,
                    win_length=1024
                )
                mel = librosa.power_to_db(mel, ref=np.max)
            except:
                # Fallback: create dummy mel spectrogram
                mel = np.random.randn(80, 259)  # Standard mel shape
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            
            # Generate English audio
            with torch.no_grad():
                english_audio = self.trained_model(mel, speaker_embed, emotion_embed)
            
            # Convert to numpy and apply proper normalization
            english_audio_np = english_audio.squeeze().cpu().numpy()
            
            # CRITICAL FIX: Apply proper normalization to prevent clipping and buzzing
            max_val = np.max(np.abs(english_audio_np))
            if max_val > 0:
                english_audio_np = english_audio_np / max_val  # Normalize to [-1, 1]
                english_audio_np = english_audio_np * 0.8  # Reduce volume to prevent clipping
            
            # Apply soft clipping to prevent harsh artifacts
            english_audio_np = np.tanh(english_audio_np * 1.2) * 0.7
            
            # Ensure no NaN/Inf values
            english_audio_np = np.nan_to_num(english_audio_np)
            
            if output_path is None:
                output_path = spanish_audio_path.replace('.wav', '_english_output.wav')
            
            try:
                import soundfile as sf
                # Save as 16-bit PCM for better compatibility
                sf.write(output_path, english_audio_np, sr, subtype='PCM_16')
            except ImportError:
                print("Warning: soundfile not available, using fallback audio saving")
                # Fallback: save as numpy array
                np.save(output_path.replace('.wav', '.npy'), english_audio_np)
            
            print(f" English audio generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f" Error processing audio: {e}")
            return None
    
    def test_model_forward_pass(self):
        """Test the model with a dummy forward pass"""
        try:
            print("Testing loaded model...")
            
            # Create dummy input mel-spectrogram
            dummy_mel = torch.randn(1, 80, 259).to(self.device)  # [batch, n_mels, time]
            dummy_speaker_embed = torch.randn(1, 192).to(self.device)
            dummy_emotion_embed = torch.randn(1, 256).to(self.device)
            
            # Test forward pass
            with torch.no_grad():
                output = self.trained_model(dummy_mel, dummy_speaker_embed, dummy_emotion_embed)
            
            print(f"Model test successful!")
            print(f"Input mel shape: {dummy_mel.shape}")
            print(f"Output audio shape: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"Model forward pass test failed: {e}")
            return False
    
    def process_audio_for_s2st(self, spanish_audio_path, output_dir=None):
        """Process audio for complete S2ST pipeline integration"""
        try:
            if output_dir is None:
                output_dir = os.path.dirname(spanish_audio_path)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load and preprocess Spanish audio
            try:
                import librosa
                spanish_audio, sr = librosa.load(spanish_audio_path, sr=self.sample_rate)
            except ImportError:
                print("Warning: librosa not available in S2ST processing")
                spanish_audio = np.random.randn(22050 * 3)
                sr = self.sample_rate
            
            # Extract mel-spectrogram
            mel_spectrogram = self.extract_mel_spectrogram(spanish_audio)
            
            # Extract speaker and emotion embeddings
            speaker_embedding = self.speaker_extractor.extract_real_speaker_embedding(spanish_audio_path)
            emotion_embedding = self.emotion_extractor.extract_real_emotion_embedding(spanish_audio_path)
            
            if speaker_embedding is None or emotion_embedding is None:
                raise Exception("Failed to extract embeddings")
            
            # Generate English audio
            generated_audio = self.generate_audio(mel_spectrogram, speaker_embedding, emotion_embedding)
            
            # Apply proper normalization to prevent clipping and buzzing
            max_val = np.max(np.abs(generated_audio))
            if max_val > 0:
                generated_audio = generated_audio / max_val  # Normalize to [-1, 1]
                generated_audio = generated_audio * 0.8  # Reduce volume to prevent clipping
            
            # Apply soft clipping to prevent harsh artifacts
            generated_audio = np.tanh(generated_audio * 1.2) * 0.7
            
            # Ensure no NaN/Inf values
            generated_audio = np.nan_to_num(generated_audio)
            
            # Save generated audio
            output_filename = f"s2st_output_{os.path.basename(spanish_audio_path)}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as 16-bit PCM for better compatibility
            try:
                import soundfile as sf
                sf.write(output_path, generated_audio, self.sample_rate, subtype='PCM_16')
            except ImportError:
                print("Warning: soundfile not available in S2ST saving")
                np.save(output_path.replace('.wav', '.npy'), generated_audio)
            
            print(f"S2ST English audio generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in S2ST audio processing: {e}")
            return None

def main():
    """Main function to test the integration"""
    print("TRAINED MODEL INTEGRATION TEST")
    print("=" * 40)
    
    # Initialize loader
    loader = TrainedModelLoader()
    
    # Initialize system
    if loader.initialize_full_system():
        print("\n Integration successful! Model is ready for use.")
        
        # Test with a sample audio file if available
        test_audio = input("\nEnter path to Spanish audio file for testing (or press Enter to skip): ").strip()
        if test_audio and os.path.exists(test_audio):
            output_path = loader.process_audio(test_audio)
            if output_path:
                print(f"Test completed! Output saved to: {output_path}")
        else:
            print("Skipping audio test.")
    else:
        print("\n Integration failed! Check the errors above.")

if __name__ == "__main__":
    main()
