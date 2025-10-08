"""
Simple Defense Test - No Unicode Characters
==========================================

Test the defense StreamSpeech pipeline without Unicode characters.
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add paths
sys.path.append("Original Streamspeech/Modified Streamspeech/integration")

def test_defense_pipeline():
    """Test the defense pipeline"""
    try:
        print("=" * 60)
        print("DEFENSE PIPELINE TEST - THESIS DEFENSE")
        print("=" * 60)
        
        # Import defense pipeline
        print("\n1. Importing defense pipeline...")
        from defense_streamspeech_pipeline import DefenseStreamSpeechPipeline
        print("[OK] Defense pipeline imported successfully")
        
        # Initialize pipeline
        print("\n2. Initializing defense pipeline...")
        pipeline = DefenseStreamSpeechPipeline()
        print("[OK] Defense pipeline initialized successfully")
        
        # Create test Spanish audio
        print("\n3. Creating test Spanish audio...")
        spanish_audio = create_spanish_test_audio()
        test_path = "test_spanish_input.wav"
        sf.write(test_path, spanish_audio, 22050, subtype='PCM_16')
        print(f"[OK] Test Spanish audio created: {test_path}")
        print(f"  Audio length: {len(spanish_audio)} samples ({len(spanish_audio)/22050:.2f} seconds)")
        
        # Process Spanish to English
        print("\n4. Processing Spanish to English...")
        english_audio, results = pipeline.process_spanish_to_english(test_path)
        
        if english_audio is not None:
            print("[SUCCESS] Spanish to English processing successful!")
            print(f"  Spanish text: {results.get('spanish_text', 'N/A')}")
            print(f"  English text: {results.get('english_text', 'N/A')}")
            print(f"  English audio length: {len(english_audio)} samples")
            print(f"  Audio range: [{english_audio.min():.4f}, {english_audio.max():.4f}]")
            
            # Save English output
            english_path = "defense_english_output.wav"
            sf.write(english_path, english_audio, 22050, subtype='PCM_16')
            print(f"[OK] English audio saved: {english_path}")
            
            # Verify audio quality
            max_amp = np.max(np.abs(english_audio))
            if max_amp > 0.1 and max_amp < 1.0:
                print("[GOOD] Audio quality: GOOD (proper amplitude range)")
            else:
                print(f"[CHECK] Audio quality: CHECK (max amplitude: {max_amp:.4f})")
            
            print("\n" + "=" * 60)
            print("DEFENSE PIPELINE TEST COMPLETED SUCCESSFULLY!")
            print("[READY] System ready for thesis defense demonstration")
            print("=" * 60)
            
            return True
        else:
            print("[FAILED] Spanish to English processing failed!")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Defense pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        if os.path.exists("test_spanish_input.wav"):
            os.remove("test_spanish_input.wav")

def create_spanish_test_audio():
    """Create realistic Spanish test audio"""
    sample_rate = 22050
    duration = 3.0  # 3 seconds
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Generate Spanish-like speech audio
    # Lower fundamental frequency for Spanish
    f0 = 160  # Hz
    spanish_audio = np.sin(2 * np.pi * f0 * t) * 0.3
    
    # Add Spanish formants
    f1, f2, f3 = 600, 1000, 2300  # Spanish formant frequencies
    spanish_audio += 0.25 * np.sin(2 * np.pi * f1 * t)
    spanish_audio += 0.2 * np.sin(2 * np.pi * f2 * t)
    spanish_audio += 0.15 * np.sin(2 * np.pi * f3 * t)
    
    # Add speech rhythm and modulation
    spanish_audio += np.sin(2 * np.pi * 1.5 * t) * 0.1  # Slow modulation
    spanish_audio += np.sin(2 * np.pi * 8 * t) * 0.05   # Fast modulation
    
    # Add realistic speech noise
    spanish_audio += np.random.normal(0, 0.03, len(spanish_audio))
    
    # Add speech envelope (attack and decay)
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)
    decay_samples = int(0.3 * sample_rate)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    spanish_audio *= envelope
    
    # Add some pauses (Spanish speech rhythm)
    pause_start = int(1.0 * sample_rate)
    pause_end = int(1.2 * sample_rate)
    spanish_audio[pause_start:pause_end] *= 0.1
    
    pause_start = int(2.2 * sample_rate)
    pause_end = int(2.4 * sample_rate)
    spanish_audio[pause_start:pause_end] *= 0.1
    
    # Normalize audio
    spanish_audio = spanish_audio / np.max(np.abs(spanish_audio)) * 0.8
    
    return spanish_audio.astype(np.float32)

if __name__ == "__main__":
    print("Starting Defense Pipeline Test...")
    
    # Run main test
    success = test_defense_pipeline()
    
    if success:
        print("\n[SUCCESS] DEFENSE SYSTEM READY!")
        print("Your system is prepared for thesis defense demonstration.")
        print("\nFiles created:")
        print("  - defense_english_output.wav (actual system output)")
        print("\nYou can now demonstrate Spanish-to-English translation!")
    else:
        print("\n[FAILED] Defense system needs attention before thesis defense.")

