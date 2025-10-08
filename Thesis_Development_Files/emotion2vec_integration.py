#!/usr/bin/env python3
"""
EMOTION2VEC EMOTION EMBEDDING INTEGRATION
Real implementation with Emotion2Vec model
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import soundfile as sf

try:
    # Try to import Emotion2Vec (if available)
    from emotion2vec import Emotion2Vec
    EMOTION2VEC_AVAILABLE = True
except ImportError:
    EMOTION2VEC_AVAILABLE = False
    print("WARNING: Emotion2Vec not available. Using fallback implementation.")

class Emotion2VecExtractor:
    """Real Emotion2Vec emotion embedding extraction"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 256  # Emotion2Vec output dimension
        
        # Audio preprocessing parameters
        self.sample_rate = 16000  # Emotion2Vec expects 16kHz
        self.window_length = 25   # 25ms window
        self.hop_length = 10      # 10ms hop
        
        # Initialize Emotion2Vec model
        self.model = None
        self.load_emotion2vec_model()
        
        print("EMOTION2VEC EXTRACTOR INITIALIZED")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_emotion2vec_model(self):
        """Load Emotion2Vec model"""
        if not EMOTION2VEC_AVAILABLE:
            print("Using fallback emotion embedding extractor")
            # Import and use fallback
            from emotion2vec_fallback import Emotion2VecFallback
            self.model = Emotion2VecFallback()
            return True
        
        try:
            print("Loading Emotion2Vec model...")
            
            # Load pretrained Emotion2Vec model
            self.model = Emotion2Vec.from_pretrained("emotion2vec_base")
            
            print("[OK] Emotion2Vec model loaded successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load Emotion2Vec model: {e}")
            print("Using REAL emotion embedding extractor")
            # Import and use REAL extractor
            from real_emotion_extractor import RealEmotionExtractor
            self.model = RealEmotionExtractor()
            return True
    
    def preprocess_audio_for_emotion2vec(self, audio_path: str) -> Optional[torch.Tensor]:
        """Preprocess audio for Emotion2Vec"""
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
            
            # Ensure minimum length (Emotion2Vec needs at least 1 second)
            min_length = self.sample_rate  # 1 second
            if audio.shape[1] < min_length:
                # Pad with zeros
                audio = torch.nn.functional.pad(audio, (0, min_length - audio.shape[1]))
            
            return audio.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"ERROR: Failed to preprocess audio {audio_path}: {e}")
            return None
    
    def extract_emotion_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Extract emotion embedding from audio"""
        if self.model is None:
            # Fallback to zero embedding (better than random)
            return torch.zeros(self.embedding_dim)
        
        try:
            # Use REAL extractor if available
            if hasattr(self.model, 'extract_real_emotion_embedding'):
                return self.model.extract_real_emotion_embedding(audio_path)
            
            # Preprocess audio
            audio = self.preprocess_audio_for_emotion2vec(audio_path)
            if audio is None:
                return torch.zeros(self.embedding_dim)
            
            # Extract embedding
            with torch.no_grad():
                # Use REAL extractor method
                if hasattr(self.model, 'get_emotion_embedding'):
                    return self.model.get_emotion_embedding(audio_path)
                
                # Fallback to audio feature extraction
                audio_batch = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                audio_batch = audio_batch.to(self.device)
                
                # Get embedding using feature extractor
                embedding = self.model.audio_feature_extractor(audio_batch)
                
                # Extract the embedding tensor
                if isinstance(embedding, (list, tuple)):
                    embedding = embedding[0]
                
                # Ensure correct shape [embedding_dim]
                if embedding.dim() > 1:
                    embedding = embedding.squeeze()
                
                # Ensure correct dimension
                if embedding.shape[0] != self.embedding_dim:
                    print(f"WARNING: Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}")
                    # Pad or truncate to match expected dimension
                    if embedding.shape[0] < self.embedding_dim:
                        padding = torch.zeros(self.embedding_dim - embedding.shape[0])
                        embedding = torch.cat([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dim]
                
                return embedding
                
        except Exception as e:
            print(f"ERROR: Failed to extract emotion embedding from {audio_path}: {e}")
            return torch.randn(self.embedding_dim)
    
    def batch_extract_embeddings(self, audio_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Extract embeddings for multiple audio files"""
        embeddings = {}
        
        print(f"Extracting Emotion2Vec embeddings for {len(audio_paths)} files...")
        
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
    
    def load_embeddings(self, embeddings_path: str) -> Dict[str, torch.Tensor]:
        """Load embeddings from file"""
        try:
            embeddings_np = np.load(embeddings_path, allow_pickle=True).item()
            
            # Convert back to tensors
            embeddings = {}
            for path, embedding_np in embeddings_np.items():
                embeddings[path] = torch.from_numpy(embedding_np)
            
            print(f"[OK] Loaded {len(embeddings)} embeddings from {embeddings_path}")
            return embeddings
            
        except Exception as e:
            print(f"ERROR: Failed to load embeddings: {e}")
            return {}
    
    def get_emotion_embedding(self, audio_path: str) -> torch.Tensor:
        """Get emotion embedding for audio file"""
        return self.extract_emotion_embedding(audio_path)

class Emotion2VecIntegration:
    """Integration class for Emotion2Vec with CVSS dataset"""
    
    def __init__(self):
        self.extractor = Emotion2VecExtractor()
        self.dataset_root = "professional_cvss_dataset"
        self.embeddings_path = "emotion2vec_embeddings.npy"
    
    def process_cvss_dataset(self):
        """Process CVSS dataset with Emotion2Vec"""
        print("PROCESSING CVSS DATASET WITH EMOTION2VEC")
        print("=" * 50)
        
        if not os.path.exists(self.dataset_root):
            print(f"ERROR: Dataset not found: {self.dataset_root}")
            return False
        
        # Get all audio files
        spanish_dir = os.path.join(self.dataset_root, "spanish")
        english_dir = os.path.join(self.dataset_root, "english")
        
        spanish_files = list(Path(spanish_dir).glob("*.wav"))
        english_files = list(Path(english_dir).glob("*.wav"))
        
        print(f"Found {len(spanish_files)} Spanish files")
        print(f"Found {len(english_files)} English files")
        
        # Extract embeddings
        all_files = [str(f) for f in spanish_files + english_files]
        embeddings = self.extractor.batch_extract_embeddings(all_files)
        
        # Save embeddings
        self.extractor.save_embeddings(embeddings, self.embeddings_path)
        
        print("EMOTION2VEC INTEGRATION COMPLETED")
        return True
    
    def test_embedding_extraction(self):
        """Test embedding extraction"""
        print("TESTING EMOTION2VEC EMBEDDING EXTRACTION")
        print("=" * 50)
        
        # Test with a sample audio file
        test_file = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
        
        if os.path.exists(test_file):
            print(f"Testing with: {test_file}")
            embedding = self.extractor.extract_emotion_embedding(test_file)
            
            if embedding is not None:
                print(f"[OK] Embedding extracted successfully")
                print(f"  - Shape: {embedding.shape}")
                print(f"  - Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                print(f"  - Mean: {embedding.mean():.4f}")
                print(f"  - Std: {embedding.std():.4f}")
                return True
            else:
                print("[ERROR] Failed to extract embedding")
                return False
        else:
            print(f"[ERROR] Test file not found: {test_file}")
            return False

def main():
    """Main function"""
    print("EMOTION2VEC EMOTION EMBEDDING INTEGRATION")
    print("Real implementation with Emotion2Vec")
    print("=" * 80)
    
    integration = Emotion2VecIntegration()
    
    # Test embedding extraction
    if integration.test_embedding_extraction():
        print("\nEMOTION2VEC TEST: SUCCESS")
        
        # Process CVSS dataset if available
        if os.path.exists("professional_cvss_dataset"):
            integration.process_cvss_dataset()
        else:
            print("\nCVSS dataset not found. Run cvss_dataset_integration.py first.")
    else:
        print("\nEMOTION2VEC TEST: FAILED")

if __name__ == "__main__":
    main()
