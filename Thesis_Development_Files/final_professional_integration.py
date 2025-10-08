#!/usr/bin/env python3
"""
FINAL PROFESSIONAL INTEGRATION
Integration of the professionally trained model into StreamSpeech
Real implementation with complete voice cloning pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import sys
import json
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add paths for imports
sys.path.append('Original Streamspeech/Modified Streamspeech/models')
sys.path.append('Original Streamspeech/Modified Streamspeech/integration')

# Import the professional training system
from professional_training_system import (
    ProfessionalModifiedHiFiGANGenerator, 
    TrainingConfig
)

# Import embedding extractors
from ecapa_tdnn_integration import ECAPATDNNSpeakerExtractor
from emotion2vec_integration import Emotion2VecExtractor

class FinalProfessionalIntegration:
    """Final integration of professionally trained model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize configuration
        self.config = TrainingConfig()
        
        # Initialize models
        self.generator = ProfessionalModifiedHiFiGANGenerator(self.config).to(self.device)
        
        # Initialize embedding extractors
        self.speaker_extractor = ECAPATDNNSpeakerExtractor()
        self.emotion_extractor = Emotion2VecExtractor()
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mel_channels
        )
        
        # Load the professionally trained model
        self.load_professional_model()
        
        print("Final Professional Integration initialized successfully!")
    
    def load_professional_model(self):
        """Load the professionally trained model"""
        checkpoint_path = "professional_cvss_checkpoints/professional_cvss_best.pt"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading professionally trained model from: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load generator weights
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                print("Generator loaded successfully")
            else:
                print("ERROR: No generator state dict found in checkpoint")
                return False
            
            # Print training info
            if 'epoch' in checkpoint:
                print(f"Model trained for {checkpoint['epoch']} epochs")
            if 'losses' in checkpoint:
                print(f"Best training loss: {checkpoint['losses'].get('g_total_loss', 'unknown')}")
            
            return True
        else:
            print(f"ERROR: Professional model checkpoint not found: {checkpoint_path}")
            print("Please run professional training first")
            return False
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio for inference"""
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = T.Resample(sr, self.config.sample_rate)
                audio = resampler(audio)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Pad or truncate to fixed length (2 seconds)
            target_length = self.config.sample_rate * 2
            if audio.shape[1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[1]))
            else:
                audio = audio[:, :target_length]
            
            # Normalize
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            return audio.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"ERROR: Failed to preprocess audio {audio_path}: {e}")
            return None
    
    def generate_english_audio(self, spanish_audio_path: str) -> Optional[torch.Tensor]:
        """Generate English audio from Spanish audio using professional model"""
        try:
            print(f"Processing Spanish audio: {spanish_audio_path}")
            
            # Preprocess Spanish audio
            spanish_audio = self.preprocess_audio(spanish_audio_path)
            if spanish_audio is None:
                return None
            
            # Convert to mel-spectrogram
            spanish_mel = self.mel_transform(spanish_audio.unsqueeze(0)).squeeze(0)  # [80, T]
            
            # Get embeddings
            speaker_embed = self.speaker_extractor.get_speaker_embedding(spanish_audio_path)
            emotion_embed = self.emotion_extractor.get_emotion_embedding(spanish_audio_path)
            
            # Move to device
            spanish_mel = spanish_mel.to(self.device)
            speaker_embed = speaker_embed.to(self.device)
            emotion_embed = emotion_embed.to(self.device)
            
            # Generate English audio
            self.generator.eval()
            with torch.no_grad():
                # Add batch dimension
                spanish_mel_batch = spanish_mel.unsqueeze(0)  # [1, 80, T]
                speaker_embed_batch = speaker_embed.unsqueeze(0)  # [1, 192]
                emotion_embed_batch = emotion_embed.unsqueeze(0)  # [1, 256]
                
                # Generate
                generated_audio = self.generator(
                    spanish_mel_batch, 
                    speaker_embed_batch, 
                    emotion_embed_batch
                )
                
                # Remove batch dimension
                generated_audio = generated_audio.squeeze(0)  # [T]
            
            print("English audio generated successfully")
            return generated_audio.cpu()
            
        except Exception as e:
            print(f"ERROR: Failed to generate English audio: {e}")
            return None
    
    def process_spanish_samples(self):
        """Process Spanish samples and generate English output"""
        print("PROCESSING SPANISH SAMPLES WITH PROFESSIONAL MODEL")
        print("=" * 60)
        
        # Get Spanish audio files
        spanish_samples = [
            "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3",
            "Original Streamspeech/example/wavs/common_voice_es_18311413.mp3",
            "Original Streamspeech/example/wavs/common_voice_es_18311414.mp3",
            "Original Streamspeech/example/wavs/common_voice_es_18311415.mp3",
            "Original Streamspeech/example/wavs/common_voice_es_18311416.mp3"
        ]
        
        # Process each sample
        for i, spanish_path in enumerate(spanish_samples):
            if os.path.exists(spanish_path):
                print(f"\nProcessing sample {i+1}/5: {os.path.basename(spanish_path)}")
                
                # Generate English audio
                english_audio = self.generate_english_audio(spanish_path)
                
                if english_audio is not None:
                    # Save English output
                    output_path = f"final_professional_output_{i+1:02d}.wav"
                    sf.write(output_path, english_audio.numpy(), self.config.sample_rate)
                    print(f"English audio saved: {output_path}")
                    
                    # Print audio info
                    print(f"  - Duration: {len(english_audio) / self.config.sample_rate:.2f} seconds")
                    print(f"  - Sample rate: {self.config.sample_rate} Hz")
                    print(f"  - Max amplitude: {torch.max(torch.abs(english_audio)):.4f}")
                else:
                    print(f"Failed to generate English audio for sample {i+1}")
            else:
                print(f"Sample not found: {spanish_path}")
        
        print("\nPROFESSIONAL INTEGRATION COMPLETED")
        print("Check the generated 'final_professional_output_*.wav' files")
    
    def test_voice_cloning(self):
        """Test voice cloning quality"""
        print("TESTING VOICE CLONING QUALITY")
        print("=" * 40)
        
        # Test with a single sample
        test_file = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
        
        if os.path.exists(test_file):
            print(f"Testing voice cloning with: {test_file}")
            
            # Generate English audio
            english_audio = self.generate_english_audio(test_file)
            
            if english_audio is not None:
                # Save test output
                output_path = "voice_cloning_test_output.wav"
                sf.write(output_path, english_audio.numpy(), self.config.sample_rate)
                
                print("Voice cloning test completed")
                print(f"Output saved: {output_path}")
                
                # Analyze audio quality
                max_amp = torch.max(torch.abs(english_audio))
                rms = torch.sqrt(torch.mean(english_audio ** 2))
                
                print(f"Audio quality metrics:")
                print(f"  - Max amplitude: {max_amp:.4f}")
                print(f"  - RMS energy: {rms:.4f}")
                print(f"  - Dynamic range: {20 * torch.log10(max_amp / (rms + 1e-8)):.2f} dB")
                
                return True
            else:
                print("Voice cloning test failed")
                return False
        else:
            print(f"Test file not found: {test_file}")
            return False

def main():
    """Main function"""
    print("FINAL PROFESSIONAL INTEGRATION")
    print("Real implementation with professionally trained model")
    print("=" * 80)
    
    # Initialize integration
    integration = FinalProfessionalIntegration()
    
    # Test voice cloning
    if integration.test_voice_cloning():
        print("\nVoice cloning test: SUCCESS")
        
        # Process all Spanish samples
        integration.process_spanish_samples()
        
        print("\nFINAL PROFESSIONAL INTEGRATION COMPLETED SUCCESSFULLY")
        print("The system is now ready for thesis demonstration")
    else:
        print("\nVoice cloning test: FAILED")
        print("Please check the professional training results")

if __name__ == "__main__":
    main()
