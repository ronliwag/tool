#!/usr/bin/env python3
"""
PROFESSIONAL MODEL INTEGRATION
Integrates the professionally trained model with the correct architecture
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths for imports
sys.path.append('Original Streamspeech/Modified Streamspeech/models')
sys.path.append('Original Streamspeech/Modified Streamspeech/integration')

# Import the professional training generator
from professional_training_system import ProfessionalModifiedHiFiGANGenerator, TrainingConfig

class IntegratedStreamSpeechModifications:
    """Integrated system using the professionally trained model"""
    
    def __init__(self):
        print("INTEGRATING PROFESSIONAL TRAINING MODEL")
        
        # Configuration
        config = TrainingConfig()
        
        # Initialize the professional generator
        self.generator = ProfessionalModifiedHiFiGANGenerator(config)
        
        # Load the professional training checkpoint
        checkpoint_path = "D:/Thesis - Tool/checkpoints/professional_training_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load the generator state
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"Professional model loaded successfully")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Best loss: {checkpoint['losses']['g_total_loss']:.6f}")
        
        self.model_loaded_successfully = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_audio_with_modifications(self, spanish_audio_path):
        """Process Spanish audio with professional modifications"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            import numpy as np
            
            # Load Spanish audio
            spanish_audio, sr = torchaudio.load(spanish_audio_path)
            if sr != 22050:
                resampler = T.Resample(sr, 22050)
                spanish_audio = resampler(spanish_audio)
            
            # Convert to mel-spectrogram
            mel_transform = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=80
            )
            
            spanish_mel = mel_transform(spanish_audio.squeeze(0))
            spanish_mel = spanish_mel.squeeze(0) if spanish_mel.dim() == 3 else spanish_mel
            
            # Create embeddings
            speaker_embed = torch.randn(192)
            emotion_embed = torch.randn(256)
            
            # Generate English audio
            with torch.no_grad():
                generated_audio = self.generator(
                    spanish_mel.unsqueeze(0),
                    speaker_embed.unsqueeze(0),
                    emotion_embed.unsqueeze(0)
                )
            
            # Convert to numpy
            english_audio = generated_audio.squeeze(0).cpu().numpy()
            
            print(f"Generated English audio: {len(english_audio)} samples")
            print(f"Audio range: [{english_audio.min():.4f}, {english_audio.max():.4f}]")
            
            return english_audio
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

def test_integration():
    """Test the professional model integration"""
    print("TESTING PROFESSIONAL MODEL INTEGRATION")
    
    # Initialize integrated system
    integrated_system = IntegratedStreamSpeechModifications()
    
    if integrated_system.model_loaded_successfully:
        print("SUCCESS: Professional model integrated successfully")
        
        # Test with Spanish audio
        spanish_audio_path = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
        if os.path.exists(spanish_audio_path):
            print(f"Testing with: {spanish_audio_path}")
            english_audio = integrated_system.process_audio_with_modifications(spanish_audio_path)
            
            if english_audio is not None:
                print("SUCCESS: English audio generated successfully")
                print(f"Audio length: {len(english_audio)} samples")
                print(f"Audio duration: {len(english_audio) / 22050:.2f} seconds")
                
                # Save test output
                import soundfile as sf
                output_path = "professional_test_output.wav"
                sf.write(output_path, english_audio, 22050)
                print(f"Test output saved: {output_path}")
                
                return True
            else:
                print("ERROR: Failed to generate English audio")
                return False
        else:
            print(f"ERROR: Spanish audio file not found: {spanish_audio_path}")
            return False
    else:
        print("ERROR: Professional model integration failed")
        return False

if __name__ == "__main__":
    test_integration()

