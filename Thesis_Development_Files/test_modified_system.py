#!/usr/bin/env python3
"""
QUICK TEST SCRIPT FOR MODIFIED STREAMSPEECH SYSTEM
Tests the current state before Colab Pro training
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

def test_streamspeech_availability():
    """Test if StreamSpeech is properly available"""
    print("=" * 60)
    print("TEST 1: STREAMSPEECH AVAILABILITY")
    print("=" * 60)
    
    try:
        # Check StreamSpeech paths
        config_path = "Original Streamspeech/Modified Streamspeech/demo/config.json"
        
        if os.path.exists(config_path):
            print(f"[OK] StreamSpeech config found: {config_path}")
            
            # Check model files
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_path = config.get('model-path', '')
            if os.path.exists(model_path):
                print(f"[OK] StreamSpeech model found: {model_path}")
                return True
            else:
                print(f"[ERROR] StreamSpeech model missing: {model_path}")
                return False
        else:
            print(f"[ERROR] StreamSpeech config missing: {config_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] StreamSpeech availability test failed: {e}")
        return False

def test_enhanced_desktop_app():
    """Test if enhanced desktop app is available"""
    print("\n" + "=" * 60)
    print("TEST 2: ENHANCED DESKTOP APP")
    print("=" * 60)
    
    try:
        app_path = "Original Streamspeech/Modified Streamspeech/demo/enhanced_desktop_app.py"
        
        if os.path.exists(app_path):
            print(f"[OK] Enhanced desktop app found: {app_path}")
            
            # Check if it can be imported
            sys.path.append(os.path.dirname(app_path))
            try:
                import enhanced_desktop_app
                print("[OK] Enhanced desktop app can be imported")
                return True
            except Exception as e:
                print(f"[ERROR] Cannot import enhanced desktop app: {e}")
                return False
        else:
            print(f"[ERROR] Enhanced desktop app missing: {app_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Enhanced desktop app test failed: {e}")
        return False

def test_modified_components():
    """Test if modified components are available"""
    print("\n" + "=" * 60)
    print("TEST 3: MODIFIED COMPONENTS")
    print("=" * 60)
    
    try:
        # Check modified HiFi-GAN
        models_path = "Original Streamspeech/Modified Streamspeech/models"
        if os.path.exists(models_path):
            print(f"[OK] Models directory found: {models_path}")
            
            # Check for modified HiFi-GAN
            hifigan_path = os.path.join(models_path, "modified_hifigan.py")
            if os.path.exists(hifigan_path):
                print(f"[OK] Modified HiFi-GAN found: {hifigan_path}")
            else:
                print(f"[WARNING] Modified HiFi-GAN not found: {hifigan_path}")
            
            # Check for integration
            integration_path = "Original Streamspeech/Modified Streamspeech/integration"
            if os.path.exists(integration_path):
                print(f"[OK] Integration directory found: {integration_path}")
                
                # Check for StreamSpeech modifications
                modifications_path = os.path.join(integration_path, "streamspeech_modifications.py")
                if os.path.exists(modifications_path):
                    print(f"[OK] StreamSpeech modifications found: {modifications_path}")
                else:
                    print(f"[WARNING] StreamSpeech modifications not found: {modifications_path}")
            else:
                print(f"[WARNING] Integration directory not found: {integration_path}")
            
            return True
        else:
            print(f"[ERROR] Models directory missing: {models_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Modified components test failed: {e}")
        return False

def test_dataset_availability():
    """Test if datasets are available"""
    print("\n" + "=" * 60)
    print("TEST 4: DATASET AVAILABILITY")
    print("=" * 60)
    
    try:
        # Check for real dataset
        dataset_path = "professional_cvss_dataset"
        if os.path.exists(dataset_path):
            print(f"[OK] Professional CVSS dataset found: {dataset_path}")
            
            # Check for Spanish and English audio
            spanish_path = os.path.join(dataset_path, "spanish")
            english_path = os.path.join(dataset_path, "english")
            
            if os.path.exists(spanish_path):
                spanish_files = [f for f in os.listdir(spanish_path) if f.endswith('.wav')]
                print(f"[OK] Spanish audio files: {len(spanish_files)}")
            else:
                print(f"[WARNING] Spanish audio directory not found: {spanish_path}")
            
            if os.path.exists(english_path):
                english_files = [f for f in os.listdir(english_path) if f.endswith('.wav')]
                print(f"[OK] English audio files: {len(english_files)}")
            else:
                print(f"[WARNING] English audio directory not found: {english_path}")
            
            return True
        else:
            print(f"[WARNING] Professional CVSS dataset not found: {dataset_path}")
            
            # Check for smaller dataset
            small_dataset = "real_training_dataset"
            if os.path.exists(small_dataset):
                print(f"[OK] Small training dataset found: {small_dataset}")
                return True
            else:
                print(f"[WARNING] No training datasets found")
                return False
            
    except Exception as e:
        print(f"[ERROR] Dataset availability test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n" + "=" * 60)
    print("TEST 5: GPU AVAILABILITY")
    print("=" * 60)
    
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] GPU available: {gpu_name}")
            print(f"[OK] GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("[WARNING] No GPU available - will use CPU")
            return False
            
    except Exception as e:
        print(f"[ERROR] GPU availability test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MODIFIED STREAMSPEECH SYSTEM TEST")
    print("=" * 80)
    
    tests = []
    
    # Run all tests
    tests.append(("StreamSpeech Availability", test_streamspeech_availability()))
    tests.append(("Enhanced Desktop App", test_enhanced_desktop_app()))
    tests.append(("Modified Components", test_modified_components()))
    tests.append(("Dataset Availability", test_dataset_availability()))
    tests.append(("GPU Availability", test_gpu_availability()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in tests:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed >= 4:  # At least 4 out of 5 tests should pass
        print("\n[OK] System is ready for local testing!")
        print("\nNEXT STEPS:")
        print("1. Launch enhanced desktop app")
        print("2. Test with Spanish audio samples")
        print("3. Compare Original vs Modified modes")
        print("4. Fix any buzz sound issues")
        print("5. Prepare for Colab training")
    else:
        print(f"\n[WARNING] {len(tests) - passed} tests failed!")
        print("Some components may need attention before testing.")
    
    print("\nTo launch the enhanced desktop app:")
    print('cd "Original Streamspeech/Modified Streamspeech/demo"')
    print("python enhanced_desktop_app.py")

if __name__ == "__main__":
    main()
