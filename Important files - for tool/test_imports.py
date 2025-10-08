#!/usr/bin/env python3
"""
Test all imports to ensure they work before Colab upload
"""
import sys
import os

# Add the models directory to path
sys.path.append('Original Streamspeech/Modified Streamspeech/models')

print("="*60)
print("TESTING ALL IMPORTS FOR COLAB COMPATIBILITY")
print("="*60)

# Test 1: ODConv
try:
    from odconv import ODConvTranspose1D
    print("✅ ODConv import: SUCCESS")
except Exception as e:
    print(f"❌ ODConv import: FAILED - {e}")

# Test 2: FiLM Conditioning
try:
    from film_conditioning import FiLMLayer, SpeakerEmotionExtractor
    print("✅ FiLM import: SUCCESS")
except Exception as e:
    print(f"❌ FiLM import: FAILED - {e}")

# Test 3: GRC+LoRA
try:
    from grc_lora import MultiReceptiveFieldFusion
    print("✅ GRC+LoRA import: SUCCESS")
except Exception as e:
    print(f"❌ GRC+LoRA import: FAILED - {e}")

# Test 4: ModifiedStreamSpeechVocoder
try:
    from modified_hifigan_generator import ModifiedStreamSpeechVocoder
    print("✅ ModifiedStreamSpeechVocoder import: SUCCESS")
except Exception as e:
    print(f"❌ ModifiedStreamSpeechVocoder import: FAILED - {e}")

# Test 5: Real Extractors
try:
    sys.path.append('Thesis_Development_Files')
    from real_ecapa_extractor import RealECAPAExtractor
    print("✅ RealECAPAExtractor import: SUCCESS")
except Exception as e:
    print(f"❌ RealECAPAExtractor import: FAILED - {e}")

try:
    from real_emotion_extractor import RealEmotionExtractor
    print("✅ RealEmotionExtractor import: SUCCESS")
except Exception as e:
    print(f"❌ RealEmotionExtractor import: FAILED - {e}")

print("\n" + "="*60)
print("IMPORT TEST COMPLETE")
print("="*60)






