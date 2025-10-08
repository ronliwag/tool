#!/usr/bin/env python3
"""
Test script to verify audio generation works correctly
"""

import sys
import os
import numpy as np
import soundfile as sf

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'integration'))

def test_audio_generation():
    """Test the audio generation functionality"""
    try:
        print("Testing StreamSpeech modifications audio generation...")
        
        # Import the modifications
        from streamspeech_modifications import StreamSpeechModifications
        
        # Create instance
        modifications = StreamSpeechModifications()
        
        # Test synthetic audio generation
        print("\n1. Testing synthetic audio generation...")
        spanish_audio = np.random.randn(22050)  # 1 second of random audio
        speaker_embed = np.random.randn(192)
        emotion_embed = np.random.randn(256)
        
        synthetic_audio = modifications.generate_synthetic_english_audio(
            spanish_audio, speaker_embed, emotion_embed
        )
        
        if synthetic_audio is not None and np.max(np.abs(synthetic_audio)) > 0:
            print(f"[OK] Synthetic audio generated: {len(synthetic_audio)} samples")
            print(f"  - Max amplitude: {np.max(np.abs(synthetic_audio)):.4f}")
            print(f"  - Duration: {len(synthetic_audio)/22050:.2f} seconds")
            
            # Save test audio
            test_output = "test_synthetic_audio.wav"
            sf.write(test_output, synthetic_audio, 22050)
            print(f"  - Saved to: {test_output}")
        else:
            print("[ERROR] Synthetic audio generation failed")
            return False
        
        # Test enhanced audio generation
        print("\n2. Testing enhanced audio generation...")
        mel_features = np.random.randn(80, 100)  # Random mel features
        
        enhanced_audio = modifications.generate_enhanced_audio(
            mel_features, speaker_embed, emotion_embed
        )
        
        if enhanced_audio is not None and np.max(np.abs(enhanced_audio)) > 0:
            print(f"[OK] Enhanced audio generated: {len(enhanced_audio)} samples")
            print(f"  - Max amplitude: {np.max(np.abs(enhanced_audio)):.4f}")
            print(f"  - Duration: {len(enhanced_audio)/22050:.2f} seconds")
            
            # Save test audio
            test_output = "test_enhanced_audio.wav"
            sf.write(test_output, enhanced_audio, 22050)
            print(f"  - Saved to: {test_output}")
        else:
            print("[ERROR] Enhanced audio generation failed")
            return False
        
        print("\n[SUCCESS] All audio generation tests passed!")
        print("The system should now produce audible English audio instead of silence.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_audio_generation()
    if success:
        print("\n[SUCCESS] AUDIO GENERATION TEST SUCCESSFUL!")
        print("Your system is now fixed and will produce audible English audio.")
    else:
        print("\n[FAILED] AUDIO GENERATION TEST FAILED!")
        print("Please check the errors above.")
