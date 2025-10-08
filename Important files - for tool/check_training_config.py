#!/usr/bin/env python3
"""
Check training configuration and mel parameters
"""
import pickle
import json
import os

# Check training config
config_path = "Original Streamspeech/Modified Streamspeech/training_results/Colab 1st Results/training_config.pkl"
logs_path = "Original Streamspeech/Modified Streamspeech/training_results/Colab 1st Results/training_logs.json"

print("="*60)
print("TRAINING CONFIGURATION ANALYSIS")
print("="*60)

# Load logs
try:
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    
    print("Training Logs:")
    print(f"  - Final Epoch: {logs.get('final_epoch', 'Unknown')}")
    print(f"  - Dataset Samples: {logs.get('dataset_samples', 'Unknown')}")
    print(f"  - Batch Size: {logs.get('batch_size', 'Unknown')}")
    print(f"  - Best Generator Loss: {logs.get('best_gen_loss', 'Unknown')}")
    print()
    
except Exception as e:
    print(f"Error loading logs: {e}")

# Load config
try:
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print("Training Configuration:")
    if isinstance(config, dict):
        for key, value in config.items():
            print(f"  - {key}: {value}")
    else:
        print(f"  - Config type: {type(config)}")
        print(f"  - Config: {config}")
    print()
    
except Exception as e:
    print(f"Error loading config: {e}")

print("="*60)
print("MEL EXTRACTION PARAMETERS")
print("="*60)

# Expected mel parameters based on the model
expected_mel_params = {
    'sr': 22050,
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'n_mels': 80,
    'fmin': 0,
    'fmax': 8000
}

print("Expected mel parameters for model:")
for key, value in expected_mel_params.items():
    print(f"  - {key}: {value}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print("❌ CHECKPOINT ARCHITECTURE MISMATCH")
print("   - Model: Full ODConv+GRC+FiLM architecture (279 params)")
print("   - Checkpoint: Basic HiFi-GAN only (44 params)")
print("   - Result: 100% random weights = buzzing sounds")
print()
print("✅ SOLUTION:")
print("   1. The checkpoint was trained on a DIFFERENT model architecture")
print("   2. Need to retrain with the EXACT current model architecture")
print("   3. Or use a checkpoint that matches the current model")