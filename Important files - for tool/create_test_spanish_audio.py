#!/usr/bin/env python3
"""
Create Test Spanish Audio for S2ST Pipeline Testing
"""

import numpy as np
import soundfile as sf
import os

def create_test_spanish_audio():
    """Create synthetic Spanish audio for testing"""
    
    # Audio parameters
    sample_rate = 22050
    duration = 3.0  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some variation to make it more speech-like
    audio += 0.1 * np.sin(2 * np.pi * 220 * t)  # Add harmonic
    audio += 0.05 * np.random.normal(0, 1, len(audio))  # Add noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save test audio
    output_path = "test_spanish_audio.wav"
    sf.write(output_path, audio, sample_rate)
    
    print(f"Test Spanish audio created: {output_path}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    return output_path

def main():
    """Create test audio file"""
    print("Creating Test Spanish Audio for S2ST Pipeline")
    print("=" * 50)
    
    test_audio_path = create_test_spanish_audio()
    
    print(f"\nTest audio ready: {test_audio_path}")
    print("You can now use this file to test the complete S2ST pipeline:")
    print("python complete_s2st_pipeline.py")
    print(f"Then enter: {test_audio_path}")

if __name__ == "__main__":
    main()

