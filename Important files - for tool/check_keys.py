#!/usr/bin/env python3
"""
Check checkpoint keys vs model keys compatibility
"""
import torch
import json
import os
import sys

# Add the project paths
sys.path.append('Original Streamspeech/Modified Streamspeech')
sys.path.append('Original Streamspeech/Modified Streamspeech/models')
sys.path.append('Original Streamspeech/Modified Streamspeech/integration')

try:
    from models.modified_hifigan_generator import ModifiedStreamSpeechVocoder
    print("Successfully imported ModifiedStreamSpeechVocoder")
except ImportError as e:
    print(f"Import error: {e}")
    # Try alternative import
    try:
        from modified_hifigan_generator import ModifiedStreamSpeechVocoder
        print("Successfully imported ModifiedStreamSpeechVocoder (alternative)")
    except ImportError as e2:
        print(f"Alternative import error: {e2}")
        # Try direct import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("modified_hifigan_generator", "Original Streamspeech/Modified Streamspeech/models/modified_hifigan_generator.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ModifiedStreamSpeechVocoder = module.ModifiedStreamSpeechVocoder
            print("Successfully imported ModifiedStreamSpeechVocoder (direct)")
        except Exception as e3:
            print(f"Direct import error: {e3}")
            sys.exit(1)

# Checkpoint path - try multiple locations
ckpt_paths = [
    'Original Streamspeech/Modified Streamspeech/models/trained_generator.pth',
    'Original Streamspeech/Modified Streamspeech/training_results/Colab 1st Results/trained_generator.pth',
    'Original Streamspeech/Modified Streamspeech/models_backup/modified_best_model.pt'
]

ckpt_path = None
for path in ckpt_paths:
    if os.path.exists(path):
        ckpt_path = path
        print(f"Found checkpoint: {path}")
        break

if ckpt_path is None:
    print("No checkpoint found in expected locations!")
    print("Searched paths:")
    for path in ckpt_paths:
        print(f"  - {path}")
    sys.exit(1)

# Model configuration
config = {
    'mel_channels': 80,
    'audio_channels': 1,
    'upsample_rates': [8, 8, 2, 2],
    'upsample_kernel_sizes': [16, 16, 4, 4],
    'upsample_initial_channel': 512,
    'speaker_embed_dim': 192,
    'emotion_embed_dim': 256,
    'lora_rank': 4
}

print("Creating model with config:", config)
model = ModifiedStreamSpeechVocoder(config)
model_sd = model.generator.state_dict()

print("Model parameters:", len(model_sd))
print("Model keys sample:", list(model_sd.keys())[:5])

# Load checkpoint
print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"Checkpoint type: {type(ckpt)}")

# Extract state dict from checkpoint
possible = None
if isinstance(ckpt, dict):
    print("Checkpoint keys:", list(ckpt.keys()))
    for k in ['state_dict', 'generator', 'model', 'model_state', 'net', 'generator_state_dict']:
        if k in ckpt:
            possible = ckpt[k]
            print(f"Found state dict under key: {k}")
            break

if possible is None:
    possible = ckpt  # assume full state dict
    print("Using checkpoint as state dict directly")

ckpt_sd = possible if isinstance(possible, dict) else ckpt

if isinstance(ckpt_sd, dict):
    print("Checkpoint state dict keys sample:", list(ckpt_sd.keys())[:5])
else:
    print("Checkpoint is not a dict, type:", type(ckpt_sd))
    sys.exit(1)

# Compare keys
model_keys = set(model_sd.keys())
ckpt_keys = set(ckpt_sd.keys())

missing = sorted(list(model_keys - ckpt_keys))
unexpected = sorted(list(ckpt_keys - model_keys))

print("\n" + "="*60)
print("KEY COMPARISON RESULTS")
print("="*60)
print(f"Missing keys (model expects but ckpt lacks): {len(missing)}")
if missing:
    print("First 10 missing keys:", missing[:10])
    print("Last 10 missing keys:", missing[-10:])
else:
    print("No missing keys!")

print(f"\nUnexpected keys (ckpt has but model doesn't): {len(unexpected)}")
if unexpected:
    print("First 10 unexpected keys:", unexpected[:10])
    print("Last 10 unexpected keys:", unexpected[-10:])
else:
    print("No unexpected keys!")

# Check shape mismatches for intersecting keys
shape_mismatches = []
matching_keys = []
for k in model_keys & ckpt_keys:
    if model_sd[k].shape != ckpt_sd[k].shape:
        shape_mismatches.append((k, model_sd[k].shape, ckpt_sd[k].shape))
    else:
        matching_keys.append(k)

print(f"\nShape mismatches: {len(shape_mismatches)}")
if shape_mismatches:
    print("First 10 shape mismatches:")
    for i, (k, model_shape, ckpt_shape) in enumerate(shape_mismatches[:10]):
        print(f"  {k}: model {model_shape} vs ckpt {ckpt_shape}")
else:
    print("No shape mismatches!")

print(f"\nMatching keys: {len(matching_keys)}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
if missing or shape_mismatches:
    print("❌ CHECKPOINT INCOMPATIBLE - This explains the buzzing!")
    print("   - Missing keys or shape mismatches cause random weights")
    print("   - Solution: Retrain with exact model architecture")
else:
    print("✅ CHECKPOINT COMPATIBLE - Keys and shapes match")
    print("   - Issue is likely in mel preprocessing or audio pipeline")

print("\nCheckpoint summary:")
print(f"  - Model expects: {len(model_keys)} parameters")
print(f"  - Checkpoint has: {len(ckpt_keys)} parameters")
print(f"  - Matching: {len(matching_keys)} parameters")
print(f"  - Missing: {len(missing)} parameters")
print(f"  - Unexpected: {len(unexpected)} parameters")
print(f"  - Shape mismatches: {len(shape_mismatches)} parameters")
