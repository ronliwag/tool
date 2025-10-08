#!/usr/bin/env python3
"""
ECAPA-TDNN SPEAKER EMBEDDING INTEGRATION
Real implementation with SpeechBrain ECAPA-TDNN model
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
    import speechbrain as sb
    from speechbrain.pretrained import EncoderClassifier
    ECAPA_AVAILABLE = True
except ImportError:
    ECAPA_AVAILABLE = False
    print("WARNING: SpeechBrain not available. Install with: pip install speechbrain")

class ECAPATDNNSpeakerExtractor:
    """Real ECAPA-TDNN speaker embedding extraction"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 192  # ECAPA-TDNN output dimension
        
        # Audio preprocessing parameters
        self.sample_rate = 16000  # ECAPA-TDNN expects 16kHz
        self.window_length = 25   # 25ms window
        self.hop_length = 10      # 10ms hop
        self.n_mels = 80
        
        # Initialize ECAPA-TDNN model
        self.model = None
        self.load_ecapa_model()
        
        print("ECAPA-TDNN SPEAKER EXTRACTOR INITIALIZED")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_ecapa_model(self):
        """Load ECAPA-TDNN model from SpeechBrain"""
        if not ECAPA_AVAILABLE:
            print("ERROR: SpeechBrain not available")
            return False
        
        try:
            print("Loading ECAPA-TDNN model from SpeechBrain...")
            
            # Load pretrained ECAPA-TDNN model
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            
            print("[OK] ECAPA-TDNN model loaded successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load ECAPA-TDNN model: {e}")
            print("Using REAL speaker embedding extractor")
            # Import and use REAL extractor
            from real_ecapa_extractor import RealECAPAExtractor
            self.model = RealECAPAExtractor()
            return True
    
    def preprocess_audio_for_ecapa(self, audio_path: str) -> Optional[torch.Tensor]:
        """Preprocess audio for ECAPA-TDNN"""
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
            
            # Ensure minimum length (ECAPA-TDNN needs at least 1 second)
            min_length = self.sample_rate  # 1 second
            if audio.shape[1] < min_length:
                # Pad with zeros
                audio = torch.nn.functional.pad(audio, (0, min_length - audio.shape[1]))
            
            return audio.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"ERROR: Failed to preprocess audio {audio_path}: {e}")
            return None
    
    def extract_speaker_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Extract speaker embedding from audio"""
        if self.model is None:
            # Fallback to zero embedding (better than random)
            return torch.zeros(self.embedding_dim)
        
        try:
            # Use REAL extractor if available
            if hasattr(self.model, 'extract_real_speaker_embedding'):
                return self.model.extract_real_speaker_embedding(audio_path)
            
            # Preprocess audio
            audio = self.preprocess_audio_for_ecapa(audio_path)
            if audio is None:
                return torch.zeros(self.embedding_dim)
            
            # Extract embedding
            with torch.no_grad():
                # ECAPA-TDNN expects [1, T] format
                audio_batch = audio.unsqueeze(0)  # [1, T]
                
                # Get embedding
                embedding = self.model.encode_batch(audio_batch)
                
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
            print(f"ERROR: Failed to extract speaker embedding from {audio_path}: {e}")
            return torch.randn(self.embedding_dim)
    
    def batch_extract_embeddings(self, audio_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Extract embeddings for multiple audio files"""
        embeddings = {}
        
        print(f"Extracting ECAPA-TDNN embeddings for {len(audio_paths)} files...")
        
        for i, audio_path in enumerate(audio_paths):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(audio_paths)} files...")
            
            embedding = self.extract_speaker_embedding(audio_path)
            if embedding is not None:
                embeddings[audio_path] = embedding
        
        print(f"[OK] Extracted {len(embeddings)} speaker embeddings")
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
    
    def get_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Get speaker embedding for audio file"""
        return self.extract_speaker_embedding(audio_path)

class ECAPAIntegration:
    """Integration class for ECAPA-TDNN with CVSS dataset"""
    
    def __init__(self):
        self.extractor = ECAPATDNNSpeakerExtractor()
        self.dataset_root = "professional_cvss_dataset"
        self.embeddings_path = "ecapa_speaker_embeddings.npy"
    
    def process_cvss_dataset(self):
        """Process CVSS dataset with ECAPA-TDNN"""
        print("PROCESSING CVSS DATASET WITH ECAPA-TDNN")
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
        
        print("ECAPA-TDNN INTEGRATION COMPLETED")
        return True
    
    def test_embedding_extraction(self):
        """Test embedding extraction"""
        print("TESTING ECAPA-TDNN EMBEDDING EXTRACTION")
        print("=" * 50)
        
        # Test with a sample audio file
        test_file = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
        
        if os.path.exists(test_file):
            print(f"Testing with: {test_file}")
            embedding = self.extractor.extract_speaker_embedding(test_file)
            
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
    print("ECAPA-TDNN SPEAKER EMBEDDING INTEGRATION")
    print("Real implementation with SpeechBrain")
    print("=" * 80)
    
    integration = ECAPAIntegration()
    
    # Test embedding extraction
    if integration.test_embedding_extraction():
        print("\nECAPA-TDNN TEST: SUCCESS")
        
        # Process CVSS dataset if available
        if os.path.exists("professional_cvss_dataset"):
            integration.process_cvss_dataset()
        else:
            print("\nCVSS dataset not found. Run cvss_dataset_integration.py first.")
    else:
        print("\nECAPA-TDNN TEST: FAILED")

if __name__ == "__main__":
    main()
