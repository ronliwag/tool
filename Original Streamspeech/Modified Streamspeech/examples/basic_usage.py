#!/usr/bin/env python3
"""
Basic usage example for Modified HiFi-GAN Vocoder with StreamSpeech Integration
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def basic_import_test():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core imports
        import torch
        import numpy as np
        import librosa
        print("✅ Core libraries imported successfully")
        
        # Test project imports
        from app.core.real_translation_engine import RealTranslationEngine
        print("✅ Translation engine imported successfully")
        
        # Test model imports
        from app.models.modified_hifigan import ModifiedHiFiGAN
        from app.models.ecapa_tdnn import ECAPATDNN
        from app.models.emotion2vec import Emotion2Vec
        print("✅ Model classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_translation_engine():
    """Test the translation engine"""
    print("\n🔍 Testing translation engine...")
    
    try:
        from app.core.real_translation_engine import RealTranslationEngine
        
        # Initialize engine
        engine = RealTranslationEngine()
        print("✅ Translation engine initialized")
        
        # Test basic functionality (without actual audio processing)
        print("✅ Translation engine ready")
        return True
        
    except Exception as e:
        print(f"❌ Translation engine error: {e}")
        return False

def test_web_interface():
    """Test web interface components"""
    print("\n🔍 Testing web interface...")
    
    try:
        from app.main import app
        print("✅ Web application imported successfully")
        
        # Test basic app configuration
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def main():
    """Main example function"""
    print("🚀 Modified HiFi-GAN Vocoder with StreamSpeech Integration")
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)
    
    # Test basic imports
    if not basic_import_test():
        print("\n❌ Basic import test failed")
        return False
    
    # Test translation engine
    if not test_translation_engine():
        print("\n❌ Translation engine test failed")
        return False
    
    # Test web interface
    if not test_web_interface():
        print("\n❌ Web interface test failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All basic tests passed!")
    print("\nNext steps:")
    print("1. Run: python scripts/launch_demo.py")
    print("2. Open browser to http://localhost:8000")
    print("3. Test real-time voice translation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
