"""
Test Modifications Only - Verify Thesis Components Work Independently
====================================================================

This script tests the thesis modifications (ODConv, GRC+LoRA, FiLM) independently
without going through the entire StreamSpeech system.

Purpose: Verify that our modifications can generate audible English audio
from Spanish input for proper metrics comparison.
"""

import sys
import os
import numpy as np
import torch
import librosa
import soundfile as sf

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

def test_modifications_independently():
    """Test thesis modifications independently"""
    print("=" * 60)
    print("TESTING THESIS MODIFICATIONS INDEPENDENTLY")
    print("=" * 60)
    
    try:
        # Import our modifications
        from streamspeech_modifications import StreamSpeechModifications
        
        print("1. Initializing StreamSpeech Modifications...")
        ssm = StreamSpeechModifications()
        print("   - Initialization successful")
        
        # Load Spanish audio
        print("\n2. Loading Spanish audio...")
        spanish_file = "../../example/wavs/common_voice_es_18311412.mp3"
        if not os.path.exists(spanish_file):
            print(f"   - Spanish file not found: {spanish_file}")
            return False
        
        spanish_audio, sr = librosa.load(spanish_file, sr=22050)
        print(f"   - Spanish audio loaded: {len(spanish_audio)} samples, range: {spanish_audio.min():.4f} to {spanish_audio.max():.4f}")
        
        # Extract mel features
        print("\n3. Extracting mel features...")
        mel_spec = librosa.feature.melspectrogram(y=spanish_audio, sr=sr, n_mels=80, hop_length=276, n_fft=1024)
        mel_features = torch.from_numpy(mel_spec).unsqueeze(0)
        audio_tensor = torch.from_numpy(spanish_audio).unsqueeze(0)
        print(f"   - Mel features extracted: {mel_features.shape}")
        
        # Test direct English generation
        print("\n4. Testing direct English generation...")
        speaker_embed = torch.randn(1, 192)
        emotion_embed = torch.randn(1, 256)
        
        english_audio = ssm.generate_english_from_spanish_with_modifications(
            spanish_audio, mel_features, speaker_embed, emotion_embed
        )
        
        print(f"   - English audio generated: {len(english_audio)} samples")
        print(f"   - English audio range: {english_audio.min():.4f} to {english_audio.max():.4f}")
        
        # Check if audible
        is_audible = np.abs(english_audio).max() > 0.1
        print(f"   {'- AUDIBLE' if is_audible else '- NOT AUDIBLE'}: Max amplitude = {np.abs(english_audio).max():.4f}")
        
        # Test metrics calculation
        print("\n5. Testing metrics calculation...")
        metrics = ssm.calculate_voice_cloning_metrics(english_audio, spanish_audio, speaker_embed, emotion_embed)
        
        print(f"   - Speaker similarity: {metrics['speaker_similarity']:.3f}")
        print(f"   - Emotion preservation: {metrics['emotion_preservation']:.3f}")
        print(f"   - Quality score: {metrics['quality_score']:.3f}")
        print(f"   - Voice cloning success: {metrics['voice_cloning_success']}")
        
        # Save test output
        print("\n6. Saving test output...")
        output_file = "../../example/outputs/test_modifications_output.wav"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, english_audio, 22050)
        print(f"   - Test output saved: {output_file}")
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"- Modifications initialized: {ssm.modified_vocoder is not None}")
        print(f"- Spanish audio loaded: {len(spanish_audio)} samples")
        print(f"- English audio generated: {len(english_audio)} samples")
        print(f"- English audio audible: {is_audible}")
        print(f"- Metrics calculated: {metrics['voice_cloning_success']}")
        print(f"- Output saved: {output_file}")
        
        if is_audible and metrics['voice_cloning_success']:
            print("\nALL TESTS PASSED - Modifications working correctly!")
            return True
        else:
            print("\nSOME TESTS FAILED - Issues detected")
            return False
            
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_modifications_independently()
    exit(0 if success else 1)
