"""
Robust Mel-Spectrogram Conversion Utilities
==========================================

This module provides robust conversion functions for generating mel-spectrograms
that are compatible with the trained HiFi-GAN vocoder.
"""

import torch
import torchaudio
import json
import numpy as np
from pathlib import Path


def load_vocoder_config(path):
    """Load vocoder configuration from JSON file"""
    with open(path, 'r') as f:
        cfg = json.load(f)
    
    # Normalize key names if needed
    if 'training_config' in cfg:
        training_cfg = cfg['training_config']
        # Map training config to standard vocoder config format
        vocoder_cfg = {
            'sampling_rate': training_cfg.get('sample_rate', 22050),
            'n_mels': training_cfg.get('n_mels', 80),
            'filter_length': training_cfg.get('win_length', 1024),  # Use win_length as n_fft
            'n_fft': training_cfg.get('win_length', 1024),
            'win_length': training_cfg.get('win_length', 1024),
            'hop_length': training_cfg.get('hop_length', 256),
            'mel_fmin': 0,
            'mel_fmax': training_cfg.get('sample_rate', 22050) // 2,
            'speaker_dim': training_cfg.get('speaker_dim', 192),
            'emotion_dim': training_cfg.get('emotion_dim', 256)
        }
        return vocoder_cfg
    else:
        return cfg


def waveform_to_log_mel(waveform, cfg):
    """
    Convert waveform to log-mel spectrogram using exact vocoder training parameters
    
    Args:
        waveform: 1D numpy array or torch tensor (float32), sample rate = cfg['sampling_rate']
        cfg: vocoder configuration dictionary
        
    Returns:
        torch.Tensor: log-mel spectrogram [n_mels, frames]
    """
    # Create mel-spectrogram transform with exact vocoder parameters
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg['sampling_rate'],
        n_fft=cfg['filter_length'],
        win_length=cfg['win_length'],
        hop_length=cfg['hop_length'],
        n_mels=cfg['n_mels'],
        f_min=cfg.get('mel_fmin', 0),
        f_max=cfg.get('mel_fmax', cfg['sampling_rate'] // 2),
        power=1.0,  # Use magnitude instead of power
        normalized=False,
        center=True,
        pad_mode='reflect'
    )
    
    # Convert to tensor if needed
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Ensure correct shape [1, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Generate mel-spectrogram [1, n_mels, frames]
    mel = transform(waveform)
    
    # Convert to log scale (dB) - CRITICAL for HiFi-GAN
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, 
        multiplier=10.0, 
        amin=1e-10, 
        db_multiplier=0
    )
    
    # Clip to expected training range (common HiFi-GAN: -11.0 .. 2.0)
    mel_db = torch.clamp(mel_db, min=-11.5, max=2.0)
    
    return mel_db.squeeze(0)  # [n_mels, frames]


def validate_mel_format(mel_spectrogram, cfg):
    """
    Validate mel-spectrogram format and characteristics
    
    Args:
        mel_spectrogram: torch.Tensor [n_mels, frames]
        cfg: vocoder configuration dictionary
        
    Returns:
        dict: validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Check shape
    expected_n_mels = cfg['n_mels']
    if mel_spectrogram.shape[0] != expected_n_mels:
        results['valid'] = False
        results['issues'].append(f"Wrong mel channels: expected {expected_n_mels}, got {mel_spectrogram.shape[0]}")
    
    # Check for NaN/Inf
    if torch.isnan(mel_spectrogram).any():
        results['valid'] = False
        results['issues'].append("Contains NaN values")
    
    if torch.isinf(mel_spectrogram).any():
        results['valid'] = False
        results['issues'].append("Contains Inf values")
    
    # Check range
    mel_min = mel_spectrogram.min().item()
    mel_max = mel_spectrogram.max().item()
    mel_mean = mel_spectrogram.mean().item()
    mel_std = mel_spectrogram.std().item()
    
    results['stats'] = {
        'min': mel_min,
        'max': mel_max,
        'mean': mel_mean,
        'std': mel_std,
        'shape': list(mel_spectrogram.shape)
    }
    
    # Check reasonable range for log-mel
    if mel_min < -15.0 or mel_max > 5.0:
        results['issues'].append(f"Mel values out of reasonable range: [{mel_min:.2f}, {mel_max:.2f}]")
    
    if mel_std < 0.1:
        results['issues'].append(f"Very low variation (std={mel_std:.3f}), may be flat")
    
    return results


def compute_expected_waveform_length(mel_frames, hop_length):
    """
    Compute expected waveform length from mel-spectrogram frames
    
    Args:
        mel_frames: number of mel-spectrogram frames
        hop_length: hop length used in mel-spectrogram generation
        
    Returns:
        int: expected waveform length in samples
    """
    return mel_frames * hop_length


def save_mel_visualization(mel_spectrogram, output_path, title="Mel-Spectrogram"):
    """
    Save mel-spectrogram as PNG visualization
    
    Args:
        mel_spectrogram: torch.Tensor [n_mels, frames]
        output_path: path to save PNG file
        title: title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert to numpy if needed
        if isinstance(mel_spectrogram, torch.Tensor):
            mel_np = mel_spectrogram.detach().cpu().numpy()
        else:
            mel_np = mel_spectrogram
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.title(title)
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Channels')
        
        # Save with high DPI
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mel-spectrogram visualization saved to: {output_path}")
        
    except ImportError:
        print(f"Warning: matplotlib not available, skipping visualization for {output_path}")
    except Exception as e:
        print(f"Error saving mel visualization: {e}")


def create_synthetic_mel(n_mels=80, n_frames=100, sample_rate=22050):
    """
    Create a simple synthetic mel-spectrogram for testing
    
    Args:
        n_mels: number of mel channels
        n_frames: number of time frames
        sample_rate: sample rate (for frequency calculation)
        
    Returns:
        torch.Tensor: synthetic mel-spectrogram [n_mels, frames]
    """
    # Create a simple ramp/tone pattern
    mel = torch.zeros(n_mels, n_frames)
    
    # Add some basic formant-like structure
    for frame in range(n_frames):
        for mel_idx in range(n_mels):
            freq = 700 * (10**(mel_idx / 2595) - 1)  # mel to Hz conversion
            
            # Add basic formants
            if 600 <= freq <= 800:  # F1
                mel[mel_idx, frame] = 0.5
            elif 1000 <= freq <= 1200:  # F2
                mel[mel_idx, frame] = 0.3
            elif 2200 <= freq <= 2600:  # F3
                mel[mel_idx, frame] = 0.2
            else:
                mel[mel_idx, frame] = 0.1
    
    # Convert to log scale
    mel = torch.log(mel + 1e-8)
    
    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    
    # Clamp to reasonable range
    mel = torch.clamp(mel, -4.0, 4.0)
    
    return mel


if __name__ == "__main__":
    # Test the functions
    print("Testing mel_utils functions...")
    
    # Load config
    config_path = "diagnostics/vocoder_config.json"
    if Path(config_path).exists():
        cfg = load_vocoder_config(config_path)
        print(f"Loaded config: {cfg}")
        
        # Test synthetic mel
        synthetic_mel = create_synthetic_mel(cfg['n_mels'], 100, cfg['sampling_rate'])
        print(f"Created synthetic mel: {synthetic_mel.shape}")
        
        # Validate
        validation = validate_mel_format(synthetic_mel, cfg)
        print(f"Validation results: {validation}")
        
    else:
        print(f"Config file not found: {config_path}")


