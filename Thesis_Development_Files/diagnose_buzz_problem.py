#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT: BUZZ SOUNDS AND NO ENGLISH OUTPUT
Identifies the exact cause of buzz sounds in modified StreamSpeech
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

def diagnose_streamspeech_zeros():
    """Diagnose why StreamSpeech produces zeros"""
    print("=" * 60)
    print("DIAGNOSIS 1: STREAMSPEECH ZERO OUTPUT")
    print("=" * 60)
    
    try:
        # Check if StreamSpeech is properly initialized
        sys.path.append('Original Streamspeech/Modified Streamspeech/demo')
        
        # Check StreamSpeech model files
        streamspeech_path = "Original Streamspeech/Modified Streamspeech/demo/streamspeech"
        if os.path.exists(streamspeech_path):
            print(f"[OK] StreamSpeech path exists: {streamspeech_path}")
            
            # Check for model files
            model_files = []
            for root, dirs, files in os.walk(streamspeech_path):
                for file in files:
                    if file.endswith('.pt') or file.endswith('.pth'):
                        model_files.append(os.path.join(root, file))
            
            print(f"[INFO] Found {len(model_files)} model files:")
            for model_file in model_files:
                print(f"  - {model_file}")
                
            if len(model_files) == 0:
                print("[ERROR] No model files found - StreamSpeech not properly installed!")
                return False
                
        else:
            print(f"[ERROR] StreamSpeech path not found: {streamspeech_path}")
            return False
            
        # Check if we can import StreamSpeech
        try:
            import streamspeech
            print("[OK] StreamSpeech can be imported")
        except ImportError as e:
            print(f"[ERROR] Cannot import StreamSpeech: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] StreamSpeech diagnosis failed: {e}")
        return False

def diagnose_model_architecture_mismatch():
    """Diagnose model architecture mismatch"""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 2: MODEL ARCHITECTURE MISMATCH")
    print("=" * 60)
    
    try:
        # Check current model architecture
        sys.path.append('Original Streamspeech/Modified Streamspeech/models')
        
        # Import current model
        from professional_training_system import ProfessionalModifiedHiFiGANGenerator, TrainingConfig
        
        config = TrainingConfig()
        model = ProfessionalModifiedHiFiGANGenerator(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_keys = len(list(model.state_dict().keys()))
        
        print(f"[INFO] Current Model Architecture:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Parameter keys: {param_keys}")
        print(f"  - Architecture: ProfessionalModifiedHiFiGANGenerator")
        
        # Check for available checkpoints
        checkpoint_paths = [
            "professional_training_best.pt",
            "professional_training_epoch_9.pt",
            "professional_training_epoch_8.pt"
        ]
        
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                print(f"\n[INFO] Checking checkpoint: {checkpoint_path}")
                
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    if 'generator_state_dict' in checkpoint:
                        checkpoint_keys = len(checkpoint['generator_state_dict'].keys())
                        print(f"  - Checkpoint keys: {checkpoint_keys}")
                        
                        if checkpoint_keys != param_keys:
                            print(f"  [ERROR] MISMATCH: Model has {param_keys} keys, checkpoint has {checkpoint_keys}")
                            print(f"  [ERROR] Missing keys: {param_keys - checkpoint_keys}")
                            return False
                        else:
                            print(f"  [OK] Keys match: {checkpoint_keys}")
                    else:
                        print(f"  [ERROR] No 'generator_state_dict' in checkpoint")
                        return False
                        
                except Exception as e:
                    print(f"  [ERROR] Failed to load checkpoint: {e}")
                    return False
            else:
                print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Architecture diagnosis failed: {e}")
        return False

def diagnose_integration_pipeline():
    """Diagnose integration pipeline issues"""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 3: INTEGRATION PIPELINE")
    print("=" * 60)
    
    try:
        # Check if modified components are being used
        integration_file = "Original Streamspeech/Modified Streamspeech/integration/streamspeech_modifications.py"
        
        if os.path.exists(integration_file):
            print(f"[OK] Integration file exists: {integration_file}")
            
            # Read integration file to check for issues
            with open(integration_file, 'r') as f:
                content = f.read()
            
            # Check for critical issues
            issues = []
            
            if "SKIPPING modified vocoder due to architecture mismatch" in content:
                issues.append("Modified vocoder is being skipped due to architecture mismatch")
            
            if "StreamSpeech vocoder is producing ALL ZEROS" in content:
                issues.append("StreamSpeech is producing zeros")
            
            if "return np.zeros(22050)" in content:
                issues.append("System returns silence instead of processing")
            
            if issues:
                print("[ERROR] Integration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("[OK] No obvious integration issues found")
                
        else:
            print(f"[ERROR] Integration file not found: {integration_file}")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration diagnosis failed: {e}")
        return False

def diagnose_audio_processing():
    """Diagnose audio processing issues"""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 4: AUDIO PROCESSING")
    print("=" * 60)
    
    try:
        # Check if we can process audio
        test_audio_path = "test_spanish.wav"
        
        if os.path.exists(test_audio_path):
            print(f"[OK] Test audio exists: {test_audio_path}")
            
            # Load audio
            audio, sr = sf.read(test_audio_path)
            print(f"  - Sample rate: {sr}")
            print(f"  - Duration: {len(audio)/sr:.2f} seconds")
            print(f"  - Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
            
            # Check for silence
            if np.all(audio == 0):
                print("[ERROR] Test audio is completely silent!")
                return False
            elif np.max(np.abs(audio)) < 0.01:
                print("[WARNING] Test audio is very quiet")
            else:
                print("[OK] Test audio has proper amplitude")
                
        else:
            print(f"[WARNING] Test audio not found: {test_audio_path}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Audio processing diagnosis failed: {e}")
        return False

def main():
    """Run complete diagnosis"""
    print("BUZZ SOUND DIAGNOSIS - COMPLETE ANALYSIS")
    print("=" * 80)
    
    results = []
    
    # Run all diagnoses
    results.append(("StreamSpeech Zeros", diagnose_streamspeech_zeros()))
    results.append(("Architecture Mismatch", diagnose_model_architecture_mismatch()))
    results.append(("Integration Pipeline", diagnose_integration_pipeline()))
    results.append(("Audio Processing", diagnose_audio_processing()))
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[OK] All systems working correctly!")
        print("Buzz sounds may be caused by other factors.")
    else:
        print(f"\n[ERROR] {len(results) - passed} critical issues found!")
        print("These must be fixed before training in Colab.")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if not results[0][1]:  # StreamSpeech zeros
            print("1. Fix StreamSpeech model initialization")
            print("2. Verify StreamSpeech model files are present")
        if not results[1][1]:  # Architecture mismatch
            print("3. Fix model architecture mismatch")
            print("4. Ensure checkpoints match model architecture")
        if not results[2][1]:  # Integration issues
            print("5. Fix integration pipeline")
            print("6. Ensure modified components are used")
        if not results[3][1]:  # Audio processing
            print("7. Fix audio processing issues")
            print("8. Verify test audio files")

if __name__ == "__main__":
    main()
