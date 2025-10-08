#!/usr/bin/env python3
"""
Download StreamSpeech models and vocoder from HuggingFace
"""

import os
import requests
from pathlib import Path
import json

def download_file(url, local_path):
    """Download a file from URL to local path"""
    print(f"Downloading {url} to {local_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded {local_path}")

def main():
    # Create pretrain_models directory
    pretrain_dir = Path("pretrain_models")
    pretrain_dir.mkdir(exist_ok=True)
    
    # StreamSpeech Models (Spanish to English only)
    models = {
        "streamspeech.simultaneous.es-en.pt": "https://huggingface.co/ICTNLP/StreamSpeech_Models/resolve/main/streamspeech.simultaneous.es-en.pt",
        "streamspeech.offline.es-en.pt": "https://huggingface.co/ICTNLP/StreamSpeech_Models/resolve/main/streamspeech.offline.es-en.pt",
        "unity.es-en.pt": "https://huggingface.co/ICTNLP/StreamSpeech_Models/resolve/main/unity.es-en.pt"
    }
    
    # Download StreamSpeech models
    for model_name, url in models.items():
        local_path = pretrain_dir / model_name
        if not local_path.exists():
            try:
                download_file(url, local_path)
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        else:
            print(f"✓ {model_name} already exists")
    
    # Download Unit-based HiFi-GAN Vocoder
    vocoder_dir = pretrain_dir / "unit-based_HiFi-GAN_vocoder" / "mHuBERT.layer11.km1000.en"
    vocoder_dir.mkdir(parents=True, exist_ok=True)
    
    vocoder_files = {
        "g_00500000": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000",
        "config.json": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json"
    }
    
    for filename, url in vocoder_files.items():
        local_path = vocoder_dir / filename
        if not local_path.exists():
            try:
                download_file(url, local_path)
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
        else:
            print(f"✓ {filename} already exists")
    
    print("\n🎉 Model download completed!")
    print("\nDownloaded models:")
    print("- StreamSpeech simultaneous model: pretrain_models/streamspeech.simultaneous.es-en.pt")
    print("- StreamSpeech offline model: pretrain_models/streamspeech.offline.es-en.pt")
    print("- UnitY model: pretrain_models/unity.es-en.pt")
    print("- Vocoder: pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/")
    
    print("\nYou can now run StreamSpeech with these models!")

if __name__ == "__main__":
    main()
