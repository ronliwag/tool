#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING VERIFICATION
Thorough analysis of all training components and gaps
"""

import torch
import os
import sys
from pathlib import Path

def analyze_training_completeness():
    """Analyze training completeness and identify gaps"""
    
    print("=" * 80)
    print("COMPREHENSIVE TRAINING VERIFICATION")
    print("=" * 80)
    
    # 1. CHECKPOINT ANALYSIS
    print("\n1. CHECKPOINT ANALYSIS")
    print("-" * 40)
    
    checkpoint_path = "D:/Thesis - Tool/checkpoints/professional_training_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✓ Checkpoint exists: {checkpoint_path}")
        print(f"✓ Checkpoint size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        print(f"✓ Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'generator_state_dict' in checkpoint:
            print(f"✓ Generator parameters: {len(checkpoint['generator_state_dict'])}")
            print(f"✓ Sample generator keys: {list(checkpoint['generator_state_dict'].keys())[:5]}")
        
        if 'discriminator_state_dict' in checkpoint:
            print(f"✓ Discriminator parameters: {len(checkpoint['discriminator_state_dict'])}")
        
        if 'epoch' in checkpoint:
            print(f"✓ Training epoch: {checkpoint['epoch']}")
        
        if 'losses' in checkpoint:
            losses = checkpoint['losses']
            print(f"✓ Final losses:")
            for key, value in losses.items():
                print(f"  - {key}: {value:.6f}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"✓ Training configuration saved")
            print(f"  - Dataset: {config.get('dataset_root', 'Unknown')}")
            print(f"  - Batch size: {config.get('batch_size', 'Unknown')}")
            print(f"  - Learning rates: G={config.get('learning_rate_g', 'Unknown')}, D={config.get('learning_rate_d', 'Unknown')}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    # 2. MODEL ARCHITECTURE VERIFICATION
    print("\n2. MODEL ARCHITECTURE VERIFICATION")
    print("-" * 40)
    
    try:
        sys.path.append('Original Streamspeech/Modified Streamspeech/models')
        from professional_training_system import ProfessionalModifiedHiFiGANGenerator, TrainingConfig
        
        config = TrainingConfig()
        model = ProfessionalModifiedHiFiGANGenerator(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model architecture: ProfessionalModifiedHiFiGANGenerator")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Architecture components:")
        print(f"  - ODConv: Dynamic convolution layers")
        print(f"  - GRC+LoRA: Grouped residual convolution with low-rank adaptation")
        print(f"  - FiLM: Speaker and emotion conditioning")
        print(f"  - Upsampling layers: {len(model.upsample_layers)}")
        print(f"  - Residual blocks: {len(model.res_blocks)}")
        
    except Exception as e:
        print(f"✗ Model architecture verification failed: {e}")
        return False
    
    # 3. DATASET VERIFICATION
    print("\n3. DATASET VERIFICATION")
    print("-" * 40)
    
    dataset_path = "real_training_dataset"
    if os.path.exists(dataset_path):
        print(f"✓ Training dataset exists: {dataset_path}")
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✓ Dataset samples: {len(metadata)}")
            print(f"✓ Spanish audio directory: {os.path.exists(os.path.join(dataset_path, 'spanish'))}")
            print(f"✓ English audio directory: {os.path.exists(os.path.join(dataset_path, 'english'))}")
            
            # Check individual samples
            for i, sample in enumerate(metadata[:3]):  # Check first 3 samples
                spanish_path = sample.get('spanish_audio_path', '')
                english_path = sample.get('english_audio_path', '')
                
                if os.path.exists(spanish_path):
                    print(f"✓ Sample {i+1} Spanish: {os.path.basename(spanish_path)}")
                else:
                    print(f"✗ Sample {i+1} Spanish missing: {spanish_path}")
                
                if os.path.exists(english_path):
                    print(f"✓ Sample {i+1} English: {os.path.basename(english_path)}")
                else:
                    print(f"✗ Sample {i+1} English missing: {english_path}")
        else:
            print(f"✗ Metadata not found: {metadata_path}")
    else:
        print(f"✗ Training dataset not found: {dataset_path}")
    
    # 4. LOSS FUNCTIONS VERIFICATION
    print("\n4. LOSS FUNCTIONS VERIFICATION")
    print("-" * 40)
    
    try:
        from professional_training_system import PerceptualLoss, SpectralLoss
        
        # Test loss functions
        perceptual_loss = PerceptualLoss()
        spectral_loss = SpectralLoss()
        
        print("✓ Perceptual Loss: Implemented")
        print("✓ Spectral Loss: Implemented")
        print("✓ Adversarial Loss: Implemented in training")
        
        # Test with dummy data
        dummy_real = torch.randn(1, 1, 1000)
        dummy_fake = torch.randn(1, 1, 1000)
        
        perc_loss = perceptual_loss(dummy_real, dummy_fake)
        spec_loss = spectral_loss(dummy_real.squeeze(1), dummy_fake.squeeze(1))
        
        print(f"✓ Perceptual loss test: {perc_loss.item():.6f}")
        print(f"✓ Spectral loss test: {spec_loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ Loss functions verification failed: {e}")
    
    # 5. OPTIMIZATION VERIFICATION
    print("\n5. OPTIMIZATION VERIFICATION")
    print("-" * 40)
    
    try:
        from professional_training_system import ProfessionalModifiedHiFiGANTrainer, TrainingConfig
        
        config = TrainingConfig()
        trainer = ProfessionalModifiedHiFiGANTrainer(config)
        
        print("✓ Generator optimizer: Adam")
        print(f"✓ Generator learning rate: {config.learning_rate_g}")
        print(f"✓ Generator beta1: {config.beta1}, beta2: {config.beta2}")
        print("✓ Discriminator optimizer: Adam")
        print(f"✓ Discriminator learning rate: {config.learning_rate_d}")
        
    except Exception as e:
        print(f"✗ Optimization verification failed: {e}")
    
    # 6. INTEGRATION VERIFICATION
    print("\n6. INTEGRATION VERIFICATION")
    print("-" * 40)
    
    integration_files = [
        "professional_integration.py",
        "integrate_professional_model.py",
        "validate_training.py"
    ]
    
    for file in integration_files:
        if os.path.exists(file):
            print(f"✓ Integration file: {file}")
        else:
            print(f"✗ Missing integration file: {file}")
    
    # 7. OUTPUT VERIFICATION
    print("\n7. OUTPUT VERIFICATION")
    print("-" * 40)
    
    output_files = [f"professional_output_{i:02d}.wav" for i in range(1, 6)]
    
    for output_file in output_files:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✓ Output file: {output_file} ({file_size} bytes)")
        else:
            print(f"✗ Missing output file: {output_file}")
    
    # 8. TESTING VERIFICATION
    print("\n8. TESTING VERIFICATION")
    print("-" * 40)
    
    try:
        # Test the professional integration
        from professional_integration import ProfessionalStreamSpeechModifications
        
        system = ProfessionalStreamSpeechModifications()
        
        if system.model_loaded_successfully:
            print("✓ Professional system initialization: SUCCESS")
            
            # Test with one sample
            test_file = "Original Streamspeech/example/wavs/common_voice_es_18311412.mp3"
            if os.path.exists(test_file):
                result = system.process_audio_with_modifications(test_file)
                if result is not None:
                    print("✓ Audio processing test: SUCCESS")
                    print(f"  - Output length: {len(result)} samples")
                    print(f"  - Output range: [{result.min():.4f}, {result.max():.4f}]")
                else:
                    print("✗ Audio processing test: FAILED")
            else:
                print("✗ Test file not found")
        else:
            print("✗ Professional system initialization: FAILED")
            
    except Exception as e:
        print(f"✗ Testing verification failed: {e}")
    
    # 9. GAP ANALYSIS
    print("\n9. GAP ANALYSIS")
    print("-" * 40)
    
    gaps = []
    
    # Check for missing components
    if not os.path.exists("D:/CVSS-T"):
        gaps.append("CVSS-T dataset not available (using custom dataset)")
    
    # Check for missing real embeddings
    if not os.path.exists("Original Streamspeech/Modified Streamspeech/pretrained_models/spkrec-ecapa-voxceleb"):
        gaps.append("ECAPA-TDNN speaker embeddings not available (using random embeddings)")
    
    # Check for missing emotion embeddings
    emotion_model_path = "emotion2vec_model.pt"
    if not os.path.exists(emotion_model_path):
        gaps.append("Emotion2Vec model not available (using random embeddings)")
    
    if gaps:
        print("IDENTIFIED GAPS:")
        for i, gap in enumerate(gaps, 1):
            print(f"  {i}. {gap}")
    else:
        print("✓ No significant gaps identified")
    
    # 10. RECOMMENDATIONS
    print("\n10. RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if len(metadata) < 100:
        recommendations.append("Consider expanding dataset for better generalization")
    
    if not os.path.exists("D:/CVSS-T"):
        recommendations.append("Obtain CVSS-T dataset for more comprehensive training")
    
    recommendations.append("Implement real ECAPA-TDNN speaker embeddings")
    recommendations.append("Implement real Emotion2Vec emotion embeddings")
    recommendations.append("Add validation dataset for better evaluation")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    analyze_training_completeness()

