#!/usr/bin/env python3
"""
Final Test: Improved Mel-Spectrogram Generation
==============================================

This script tests the improved mel-spectrogram generation that should
produce better audio characteristics with proper silent periods and
realistic speech patterns.
"""

import sys
import os
import numpy as np
import soundfile as sf
import torch

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_improved_mel_generation():
    """Test the improved mel-spectrogram generation"""
    try:
        print("Testing Improved Mel-Spectrogram Generation...")
        print("=" * 60)
        
        # Test the TTS component directly
        from streamspeech_modifications import EnglishTTSComponent
        
        print("\n1. Testing improved TTS component...")
        tts_component = EnglishTTSComponent()
        
        # Test with different text lengths
        test_texts = [
            "Hello",
            "This is a test",
            "This is a comprehensive test of our speech synthesis system"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: '{text}'")
            
            # Generate mel-spectrogram
            mel_spectrogram = tts_component.text_to_mel_spectrogram(text)
            
            print(f"  - Mel shape: {mel_spectrogram.shape}")
            print(f"  - Mel stats: min={mel_spectrogram.min():.4f}, max={mel_spectrogram.max():.4f}, mean={mel_spectrogram.mean():.4f}")
            print(f"  - Mel std: {mel_spectrogram.std():.4f}")
            
            # Check for proper characteristics
            if mel_spectrogram.shape[0] != 80:
                print(f"  - [ERROR] Wrong mel channels: {mel_spectrogram.shape[0]}")
                return False
            
            # Check for silent periods (should have some silence)
            silent_threshold = 0.1
            silent_ratio = np.mean(np.abs(mel_spectrogram) < silent_threshold)
            print(f"  - Silent periods: {silent_ratio:.2%}")
            
            if silent_ratio < 0.05:
                print(f"  - [WARNING] Very few silent periods")
            else:
                print(f"  - [OK] Good silent periods ratio")
            
            # Check for reasonable energy levels
            max_energy = np.max(mel_spectrogram)
            if max_energy > 2.0:
                print(f"  - [WARNING] Very high energy: {max_energy:.2f}")
            elif max_energy < 0.1:
                print(f"  - [WARNING] Very low energy: {max_energy:.2f}")
            else:
                print(f"  - [OK] Reasonable energy levels")
        
        print("\n2. Testing with HiFi-GAN vocoder...")
        
        # Load the trained model
        from integrate_trained_model import TrainedModelLoader
        
        model_loader = TrainedModelLoader()
        if not model_loader.initialize_full_system():
            print("[ERROR] Failed to initialize trained model system")
            return False
        
        print("[OK] HiFi-GAN vocoder loaded successfully")
        
        # Test with the longest text
        test_text = "This is a comprehensive test of our speech synthesis system"
        print(f"\nTesting with: '{test_text}'")
        
        # Generate mel-spectrogram
        mel_spectrogram = tts_component.text_to_mel_spectrogram(test_text)
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spectrogram).float().unsqueeze(0).to(model_loader.device)
        speaker_embed = torch.randn(1, 192).to(model_loader.device)
        emotion_embed = torch.randn(1, 256).to(model_loader.device)
        
        # Generate audio
        with torch.no_grad():
            generated_audio = model_loader.trained_model(mel_tensor, speaker_embed, emotion_embed)
        
        audio_numpy = generated_audio.squeeze().cpu().numpy()
        
        print(f"[OK] Generated audio: {len(audio_numpy)} samples")
        print(f"[OK] Audio stats: min={audio_numpy.min():.4f}, max={audio_numpy.max():.4f}, mean={audio_numpy.mean():.4f}")
        print(f"[OK] Audio std: {audio_numpy.std():.4f}")
        
        # Analyze audio quality
        max_amp = np.max(np.abs(audio_numpy))
        audio_std = np.std(audio_numpy)
        duration = len(audio_numpy) / 22050
        
        print(f"\n3. Audio Quality Analysis:")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Max amplitude: {max_amp:.4f}")
        print(f"  - Standard deviation: {audio_std:.4f}")
        
        # Check for silent periods in audio
        audio_abs = np.abs(audio_numpy)
        silent_threshold = 0.01
        silent_ratio = np.mean(audio_abs < silent_threshold)
        print(f"  - Silent periods ratio: {silent_ratio:.2%}")
        
        # Check for energy variation
        chunk_size = 1024
        chunks = [audio_numpy[i:i+chunk_size] for i in range(0, len(audio_numpy), chunk_size)]
        chunk_energies = [np.mean(np.abs(chunk)) for chunk in chunks if len(chunk) > 0]
        energy_variation = np.std(chunk_energies)
        print(f"  - Energy variation: {energy_variation:.4f}")
        
        # Overall assessment
        print(f"\n4. Quality Assessment:")
        
        if max_amp > 1.5:
            print("  - [WARNING] Audio amplitude very high - may be clipped/noisy")
        elif max_amp < 0.01:
            print("  - [WARNING] Audio amplitude very low - may be silent")
        else:
            print("  - [OK] Audio amplitude in good range")
        
        if audio_std > 0.5:
            print("  - [WARNING] Audio std very high - may be noisy")
        elif audio_std < 0.01:
            print("  - [WARNING] Audio std very low - may be flat/constant")
        else:
            print("  - [OK] Audio std in good range - has variation")
        
        if silent_ratio < 0.05:
            print("  - [WARNING] Very few silent periods - may be constant noise")
        else:
            print("  - [OK] Good silent periods ratio - speech-like")
        
        if energy_variation < 0.001:
            print("  - [WARNING] Very low energy variation - may be flat noise")
        else:
            print("  - [OK] Good energy variation - speech-like")
        
        # Save audio for manual verification
        output_path = "improved_mel_test_output.wav"
        sample_rate = 22050
        
        # Normalize audio before saving
        max_val = np.max(np.abs(audio_numpy))
        if max_val > 0:
            normalized_audio = audio_numpy / max_val * 0.8
        else:
            normalized_audio = audio_numpy
        
        sf.write(output_path, normalized_audio, sample_rate, subtype='PCM_16')
        print(f"\n5. Audio saved to: {output_path}")
        print(f"   - Sample rate: {sample_rate} Hz")
        print(f"   - Format: PCM_16")
        print(f"   - Duration: {len(normalized_audio)/sample_rate:.2f} seconds")
        
        # Final assessment
        print("\n" + "=" * 60)
        
        if (0.05 < max_amp < 1.2 and 
            0.02 < audio_std < 0.4 and 
            silent_ratio > 0.05 and 
            energy_variation > 0.001):
            print("[SUCCESS] IMPROVED MEL-SPECTROGRAM GENERATION TEST PASSED!")
            print("The mel-spectrogram improvements are working correctly.")
            print("Audio characteristics indicate speech-like output instead of flat noise.")
            return True
        else:
            print("[WARNING] IMPROVED MEL-SPECTROGRAM GENERATION NEEDS MORE WORK!")
            print("Some audio characteristics still indicate issues.")
            print("Check the generated audio file for manual verification.")
            return False
        
    except Exception as e:
        print(f"[ERROR] Improved mel generation test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Final Mel-Spectrogram Generation Test")
    print("====================================")
    print("This test verifies the improved mel-spectrogram generation")
    print("that should produce better audio characteristics.")
    print()
    
    success = test_improved_mel_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] FINAL MEL-SPECTROGRAM TEST PASSED!")
        print("The improved mel-spectrogram generation is working correctly.")
        print("Your system should now produce better audio output.")
        print("\nNext steps:")
        print("1. Run the desktop application to test the complete pipeline")
        print("2. Test with real Spanish audio files")
        print("3. Verify the audio quality in the desktop app")
    else:
        print("[WARNING] FINAL MEL-SPECTROGRAM TEST SHOWS ISSUES!")
        print("The mel-spectrogram generation still needs improvement.")
        print("\nNext steps:")
        print("1. Check the generated audio file manually")
        print("2. Further refine the mel-spectrogram generation parameters")
        print("3. Consider using a different approach for mel-spectrogram generation")
    print("=" * 60)

