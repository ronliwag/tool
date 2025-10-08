import os
from pathlib import Path
import requests
import time

# Create directory for samples
output_dir = Path("Original Streamspeech/example/wavs")
output_dir.mkdir(parents=True, exist_ok=True)

print("Creating Spanish audio samples for testing...")

# Spanish audio samples with their translations
spanish_samples = [
    {
        "filename": "spanish_sample_1.wav",
        "text_es": "Hola, ¿cómo estás?",
        "text_en": "Hello, how are you?"
    },
    {
        "filename": "spanish_sample_2.wav", 
        "text_es": "Me gusta mucho la música",
        "text_en": "I really like music"
    },
    {
        "filename": "spanish_sample_3.wav",
        "text_es": "El clima está muy agradable hoy",
        "text_en": "The weather is very pleasant today"
    },
    {
        "filename": "spanish_sample_4.wav",
        "text_es": "Gracias por tu ayuda",
        "text_en": "Thank you for your help"
    },
    {
        "filename": "spanish_sample_5.wav",
        "text_es": "Buenos días a todos",
        "text_en": "Good morning everyone"
    }
]

# Create placeholder audio files (silent audio)
import numpy as np
import soundfile as sf

print("Creating placeholder audio files...")

for i, sample in enumerate(spanish_samples):
    # Create a short silent audio file (1 second, 16kHz)
    duration = 1.0  # 1 second
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Create a simple tone instead of silence for testing
    frequency = 440 + i * 100  # Different frequency for each file
    t = np.linspace(0, duration, samples, False)
    audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)  # Low volume sine wave
    
    # Save audio file
    audio_path = output_dir / sample["filename"]
    sf.write(audio_path, audio_data, sample_rate)
    
    print(f"Created: {sample['filename']}")
    print(f"Spanish: {sample['text_es']}")
    print(f"English: {sample['text_en']}")
    print("---")

# Create wav_list.txt
wav_list_path = Path("Original Streamspeech/example/wav_list.txt")
with open(wav_list_path, 'w', encoding='utf-8') as f:
    for sample in spanish_samples:
        f.write(f"example/wavs/{sample['filename']}\n")

# Create source.txt (Spanish)
source_path = Path("Original Streamspeech/example/source.txt")
with open(source_path, 'w', encoding='utf-8') as f:
    for sample in spanish_samples:
        f.write(f"{sample['text_es']}\n")

# Create target.txt (English)
target_path = Path("Original Streamspeech/example/target.txt")
with open(target_path, 'w', encoding='utf-8') as f:
    for sample in spanish_samples:
        f.write(f"{sample['text_en']}\n")

print("✓ All files created successfully!")
print(f"✓ Audio files saved to: {output_dir}")
print(f"✓ Text files updated: wav_list.txt, source.txt, target.txt")
print("\nYou can now test the StreamSpeech demo with these Spanish samples!")
print("\nNote: These are placeholder audio files for testing. For real Spanish audio,")
print("you can record your own or find Spanish audio samples online.")



