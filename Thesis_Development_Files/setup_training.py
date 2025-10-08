"""
Setup script for REAL Modified HiFi-GAN Training
This script helps you prepare the training environment
"""

import os
import json
from pathlib import Path

def create_sample_metadata():
    """Create sample metadata file for CVSS-T dataset"""
    # This is a template - you'll need to replace with your actual CVSS-T data
    sample_metadata = [
        {
            "id": "sample_001",
            "split": "train",
            "spanish_audio": "spanish/sample_001.wav",
            "english_audio": "english/sample_001.wav",
            "speaker_id": "speaker_001",
            "emotion_label": "neutral",
            "duration": 3.5
        },
        {
            "id": "sample_002", 
            "split": "train",
            "spanish_audio": "spanish/sample_002.wav",
            "english_audio": "english/sample_002.wav",
            "speaker_id": "speaker_002",
            "emotion_label": "happy",
            "duration": 4.2
        },
        {
            "id": "sample_003",
            "split": "validation", 
            "spanish_audio": "spanish/sample_003.wav",
            "english_audio": "english/sample_003.wav",
            "speaker_id": "speaker_001",
            "emotion_label": "sad",
            "duration": 2.8
        }
    ]
    
    # Save metadata
    os.makedirs("D:/CVSS-T", exist_ok=True)
    with open("D:/CVSS-T/metadata.json", 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print("Sample metadata created at D:/CVSS-T/metadata.json")
    print("Replace with your actual CVSS-T dataset metadata!")

def create_training_config():
    """Create training configuration file"""
    config = {
        "cvss_t_root": "D:/CVSS-T",
        "spanish_audio_dir": "es",
        "english_audio_dir": "en", 
        "metadata_file": "metadata.json",
        "batch_size": 8,  # Reduced for GPU memory
        "learning_rate_g": 2e-4,
        "learning_rate_d": 1e-4,
        "beta1": 0.8,
        "beta2": 0.99,
        "num_epochs": 100,  # Start with fewer epochs
        "save_interval": 10,
        "log_interval": 5,
        "validation_interval": 5,
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "n_fft": 1024,
        "lambda_adv": 1.0,
        "lambda_perceptual": 1.0,
        "lambda_spectral": 1.0,
        "lambda_kl": 0.1,
        "speaker_embed_dim": 192,
        "emotion_embed_dim": 256,
        "checkpoint_dir": "D:/Thesis - Tool/checkpoints",
        "log_dir": "logs",
        "wandb_project": "modified-hifigan-training"
    }
    
    with open("training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training configuration created at training_config.json")

def create_requirements():
    """Create requirements.txt for training dependencies"""
    requirements = [
        "torch>=1.9.0",
        "torchaudio>=0.9.0", 
        "numpy>=1.21.0",
        "soundfile>=0.10.0",
        "librosa>=0.9.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "scipy>=1.7.0",
        "pathlib",
        "argparse",
        "dataclasses",
        "typing",
        "logging",
        "json",
        "os",
        "random",
        "warnings"
    ]
    
    with open("requirements_training.txt", 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("Requirements file created at requirements_training.txt")

def main():
    """Setup training environment"""
    print("Setting up REAL Modified HiFi-GAN Training Environment...")
    print("=" * 60)
    
    # Create sample metadata
    create_sample_metadata()
    
    # Create training config
    create_training_config()
    
    # Create requirements
    create_requirements()
    
    print("=" * 60)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements_training.txt")
    print("2. Prepare your REAL CVSS-T dataset at D:/CVSS-T/")
    print("3. Replace sample metadata with your actual dataset metadata")
    print("4. Run training: python train_modified_hifigan_real.py --config training_config.json")
    print("\nIMPORTANT:")
    print("- This is a REAL training system - no dummy data, no fake models")
    print("- Uses your GPU for acceleration")
    print("- Will train your full 279-key modified HiFi-GAN architecture")
    print("- Produces REAL checkpoints for your thesis demonstration")

if __name__ == "__main__":
    main()


