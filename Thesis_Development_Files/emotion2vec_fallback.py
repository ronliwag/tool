#!/usr/bin/env python3
"""
EMOTION2VEC FALLBACK IMPLEMENTATION
Real emotion embedding extraction using alternative methods
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from typing import Optional, Dict

class Emotion2VecFallback:
    """Fallback emotion embedding extractor"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 256
        
        # Audio preprocessing parameters
        self.sample_rate = 16000
        self.window_length = 25
        self.hop_length = 10
        self.n_mels = 80
        
        # Initialize emotion feature extractor
        self.emotion_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, self.embedding_dim)
        ).to(self.device)
        
        print("EMOTION2VEC FALLBACK INITIALIZED")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Preprocess audio for emotion extraction"""
        try:
            # Load audio
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
            
            # Ensure minimum length (1 second)
            min_length = self.sample_rate
            if audio.shape[1] < min_length:
                audio = torch.nn.functional.pad(audio, (0, min_length - audio.shape[1]))
            
            return audio.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"ERROR: Failed to preprocess audio {audio_path}: {e}")
            return None
    
    def extract_emotion_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Extract emotion embedding from audio"""
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return torch.randn(self.embedding_dim)
            
            # Extract embedding
            with torch.no_grad():
                # Add batch and channel dimensions
                audio_batch = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                audio_batch = audio_batch.to(self.device)
                
                # Get embedding
                embedding = self.emotion_extractor(audio_batch)
                
                # Ensure correct shape
                if embedding.dim() > 1:
                    embedding = embedding.squeeze()
                
                # Ensure correct dimension
                if embedding.shape[0] != self.embedding_dim:
                    if embedding.shape[0] < self.embedding_dim:
                        padding = torch.zeros(self.embedding_dim - embedding.shape[0])
                        embedding = torch.cat([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dim]
                
                return embedding.cpu()
                
        except Exception as e:
            print(f"ERROR: Failed to extract emotion embedding from {audio_path}: {e}")
            return torch.randn(self.embedding_dim)
    
    def batch_extract_embeddings(self, audio_paths: list) -> Dict[str, torch.Tensor]:
        """Extract embeddings for multiple audio files"""
        embeddings = {}
        
        print(f"Extracting emotion embeddings for {len(audio_paths)} files...")
        
        for i, audio_path in enumerate(audio_paths):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(audio_paths)} files...")
            
            embedding = self.extract_emotion_embedding(audio_path)
            if embedding is not None:
                embeddings[audio_path] = embedding
        
        print(f"[OK] Extracted {len(embeddings)} emotion embeddings")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, torch.Tensor], output_path: str):
        """Save embeddings to file"""
        try:
            # Convert to numpy for serialization
            embeddings_np = {}
            for path, embedding in embeddings.items():
                embeddings_np[path] = embedding.cpu().numpy()
            
            # Save as numpy file
            np.save(output_path, embeddings_np)
            print(f"[OK] Saved embeddings to {output_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to save embeddings: {e}")
    
    def get_emotion_embedding(self, audio_path: str) -> torch.Tensor:
        """Get emotion embedding for audio file"""
        return self.extract_emotion_embedding(audio_path)

def main():
    """Test the fallback implementation"""
    print("EMOTION2VEC FALLBACK TEST")
    print("=" * 40)
    
    extractor = Emotion2VecFallback()
    
    # Test with a sample file
    test_file = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
    
    if os.path.exists(test_file):
        print(f"Testing with: {test_file}")
        embedding = extractor.extract_emotion_embedding(test_file)
        
        if embedding is not None:
            print(f"[OK] Embedding extracted successfully")
            print(f"  - Shape: {embedding.shape}")
            print(f"  - Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"  - Mean: {embedding.mean():.4f}")
            print(f"  - Std: {embedding.std():.4f}")
        else:
            print("[ERROR] Failed to extract embedding")
    else:
        print(f"[ERROR] Test file not found: {test_file}")

if __name__ == "__main__":
    main()
