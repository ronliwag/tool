"""
Quick Defense Test - Verify Defense Pipeline Works
================================================

Quick test to verify the defense pipeline is working correctly.
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add paths
sys.path.append("Original Streamspeech/Modified Streamspeech/integration")

def test_defense_pipeline_quick():
    """Quick test of defense pipeline"""
    try:
        print("=" * 50)
        print("QUICK DEFENSE PIPELINE TEST")
        print("=" * 50)
        
        # Test 1: Import and initialize
        print("\n1. Importing defense pipeline...")
        from defense_streamspeech_pipeline import DefenseStreamSpeechPipeline
        print("[OK] Import successful")
        
        print("\n2. Creating defense pipeline...")
        pipeline = DefenseStreamSpeechPipeline()
        print("[OK] Pipeline created")
        
        print("\n3. Checking initialization status...")
        if hasattr(pipeline, 'is_initialized'):
            is_init = pipeline.is_initialized()
            print(f"[STATUS] Pipeline initialized: {is_init}")
        else:
            print("[STATUS] No is_initialized method")
        
        print("\n4. Testing initialize_models method...")
        result = pipeline.initialize_models()
        print(f"[RESULT] initialize_models returned: {result}")
        
        print("\n5. Checking components...")
        print(f"  ASR Model: {pipeline.asr_model is not None}")
        print(f"  Translation Model: {pipeline.translation_model is not None}")
        print(f"  TTS Model: {pipeline.tts_model is not None}")
        print(f"  Simplified Vocoder: {pipeline.simplified_vocoder is not None}")
        
        print("\n6. Testing audio processing...")
        # Create simple test audio
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)) * 0.5
        test_path = "quick_test.wav"
        sf.write(test_path, test_audio, 22050, subtype='PCM_16')
        
        # Test processing
        try:
            english_audio, results = pipeline.process_spanish_to_english(test_path)
            if english_audio is not None:
                print("[SUCCESS] Audio processing worked!")
                print(f"  Spanish text: {results.get('spanish_text', 'N/A')}")
                print(f"  English text: {results.get('english_text', 'N/A')}")
                print(f"  Audio length: {len(english_audio)} samples")
            else:
                print("[FAILED] Audio processing returned None")
        except Exception as e:
            print(f"[ERROR] Audio processing failed: {e}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print("\n" + "=" * 50)
        print("QUICK TEST COMPLETED")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_defense_pipeline_quick()

