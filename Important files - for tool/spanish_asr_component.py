#!/usr/bin/env python3
"""
Spanish Automatic Speech Recognition Component
For complete speech-to-speech translation pipeline
"""

import torch
import torchaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

class SpanishASR:
    """Spanish Automatic Speech Recognition using Whisper"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "openai/whisper-large-v3"
        self.sample_rate = 16000
        
        # Load Whisper model and processor
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Spanish ASR initialized: {self.model_name}")
        print(f"Device: {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio for ASR"""
        try:
            # Load audio using librosa for better compatibility
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def transcribe(self, audio_path):
        """Transcribe Spanish audio to text"""
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return None
            
            # Process with Whisper
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.get('input_features', inputs.get('input_values', list(inputs.values())[0])),
                    max_length=448,
                    num_beams=5,
                    early_stopping=True,
                    language="spanish",
                    task="transcribe"
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            print(f"Spanish transcription: {transcription}")
            return transcription.strip()
            
        except Exception as e:
            print(f"Error in Spanish transcription: {e}")
            return None
    
    def transcribe_audio_array(self, audio_array):
        """Transcribe audio array directly"""
        try:
            # Ensure proper format
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Process with Whisper
            inputs = self.processor(
                audio_array, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.get('input_features', inputs.get('input_values', list(inputs.values())[0])),
                    max_length=448,
                    num_beams=5,
                    early_stopping=True,
                    language="spanish",
                    task="transcribe"
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error in audio array transcription: {e}")
            return None

def main():
    """Test the Spanish ASR component"""
    print("Testing Spanish ASR Component")
    print("=" * 40)
    
    # Initialize ASR
    asr = SpanishASR()
    
    # Test with sample audio (if available)
    test_audio = input("Enter path to Spanish audio file for testing (or press Enter to skip): ").strip()
    
    if test_audio:
        transcription = asr.transcribe(test_audio)
        if transcription:
            print(f"Transcription successful: {transcription}")
        else:
            print("Transcription failed")
    else:
        print("Skipping ASR test")

if __name__ == "__main__":
    main()
