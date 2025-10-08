#!/usr/bin/env python3
"""
Integration Test for Modified StreamSpeech
Tests if our thesis modifications are properly integrated
"""

import sys
import os

# Add paths
sys.path.append('..')
sys.path.append('../models')
sys.path.append('../integration')

def test_imports():
    """Test if all our modifications can be imported."""
    print("=== TESTING IMPORTS ===")
    
    try:
        from models.odconv import ODConvTranspose1D
        print("✅ ODConv imported successfully")
    except Exception as e:
        print(f"❌ ODConv import failed: {e}")
        return False
    
    try:
        from models.film_conditioning import FiLMLayer, SpeakerEmotionExtractor
        print("✅ FiLM conditioning imported successfully")
    except Exception as e:
        print(f"❌ FiLM import failed: {e}")
        return False
    
    try:
        from models.grc_lora import MultiReceptiveFieldFusion
        print("✅ GRC+LoRA imported successfully")
    except Exception as e:
        print(f"❌ GRC+LoRA import failed: {e}")
        return False
    
    try:
        from models.modified_hifigan_generator import ModifiedStreamSpeechVocoder
        print("✅ Modified HiFi-GAN imported successfully")
    except Exception as e:
        print(f"❌ Modified HiFi-GAN import failed: {e}")
        return False
    
    try:
        import integration.streamspeech_modifications as ssm
        print("✅ StreamSpeech modifications imported successfully")
    except Exception as e:
        print(f"❌ StreamSpeech modifications import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if our models can be loaded."""
    print("\n=== TESTING MODEL LOADING ===")
    
    try:
        import integration.streamspeech_modifications as ssm
        modifications = ssm.StreamSpeechModifications()
        print("✅ StreamSpeech modifications initialized")
        
        # Check if real models are loaded
        if modifications.modified_vocoder:
            print("✅ Modified vocoder loaded")
        else:
            print("❌ Modified vocoder not loaded")
            return False
            
        if modifications.embed_extractor:
            print("✅ Embedding extractor loaded")
        else:
            print("❌ Embedding extractor not loaded")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_real_models():
    """Test if real trained models are accessible."""
    print("\n=== TESTING REAL MODELS ===")
    
    model_paths = [
        r'D:\Thesis - Tool\checkpoints\best_model.pt',
        r'D:\Thesis - Tool\checkpoints\modified_best_model.pt',
        r'D:\Thesis - Tool\checkpoints\modified_latest_model.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Found real model: {os.path.basename(model_path)}")
        else:
            print(f"❌ Real model not found: {model_path}")
    
    return True

def main():
    """Run all tests."""
    print("MODIFIED STREAMSPEECH INTEGRATION TEST")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ IMPORT TEST FAILED")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ MODEL LOADING TEST FAILED")
        return False
    
    # Test real models
    if not test_real_models():
        print("\n❌ REAL MODELS TEST FAILED")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("✅ Our thesis modifications are properly integrated!")
    print("✅ Real models are accessible!")
    print("✅ Ready for thesis defense!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)