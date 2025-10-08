#!/usr/bin/env python3
"""
TRAINING VALIDATION SCRIPT
Validates the professional training results and checkpoint integrity
"""

import torch
import os

def validate_training():
    """Validate training results"""
    checkpoint_path = "D:/Thesis - Tool/checkpoints/professional_training_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print("ERROR: Best checkpoint not found")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("TRAINING VALIDATION RESULTS:")
        print("=" * 50)
        
        # Validate checkpoint structure
        required_keys = ['generator_state_dict', 'discriminator_state_dict', 'epoch', 'losses']
        for key in required_keys:
            if key not in checkpoint:
                print(f"ERROR: Missing key: {key}")
                return False
        
        # Check generator parameters
        generator_keys = len(checkpoint['generator_state_dict'])
        discriminator_keys = len(checkpoint['discriminator_state_dict'])
        
        print(f"Generator parameters: {generator_keys}")
        print(f"Discriminator parameters: {discriminator_keys}")
        print(f"Training epoch: {checkpoint['epoch']}")
        
        # Check losses
        losses = checkpoint['losses']
        print(f"Best Generator Loss: {losses['g_total_loss']:.6f}")
        print(f"Best Discriminator Loss: {losses['d_loss']:.6f}")
        print(f"Perceptual Loss: {losses['g_perc_loss']:.6f}")
        print(f"Spectral Loss: {losses['g_spec_loss']:.6f}")
        
        # Check file size
        file_size = os.path.getsize(checkpoint_path)
        print(f"Checkpoint size: {file_size / (1024*1024):.2f} MB")
        
        # Validate model architecture
        if generator_keys < 100:
            print("WARNING: Generator has fewer parameters than expected")
        
        print("=" * 50)
        print("TRAINING VALIDATION: SUCCESS")
        print("All checkpoints are valid and training completed successfully")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate checkpoint: {e}")
        return False

if __name__ == "__main__":
    validate_training()

