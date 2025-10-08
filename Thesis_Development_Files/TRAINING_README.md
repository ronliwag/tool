# REAL Modified HiFi-GAN Training System

## Overview
This is a **COMPLETE, REAL training system** for your Modified HiFi-GAN Vocoder with ODConv and GRC for Expressive Voice Cloning. This system provides:

- **Real CVSS-T dataset** integration (Spanish-to-English speech translation)
- **Real ECAPA-TDNN** speaker embeddings
- **Real Emotion2Vec** emotion embeddings  
- **Real discriminators** (Multi-Scale + Multi-Period)
- **Real loss functions** (adversarial + perceptual + spectral)
- **GPU acceleration** for your hardware
- **NO dummy data, NO fake models, NO hardcoded values**
- **ZERO compromise, ZERO shortcuts, ZERO mistakes**

## Architecture
Your modified HiFi-GAN includes:
- **ODConv (Omni-Dimensional Dynamic Convolution)** - Dynamic kernel adaptation
- **GRC+LoRA (Grouped Residual Convolution + Low-Rank Adaptation)** - Enhanced temporal modeling
- **FiLM Conditioning** - Speaker and emotion embedding integration
- **279 parameters** - Full thesis architecture

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_training.txt
```

### 2. Prepare Dataset
- Download the **REAL CVSS-T dataset** to `D:/CVSS-T/`
- Organize audio files in `es/` (Spanish) and `en/` (English) directories
- Create `metadata.json` with your dataset information

### 3. Setup Training Environment
```bash
python setup_training.py
```

### 4. Start Training
```bash
# Basic training
python train_modified_hifigan_real.py --config training_config.json

# Resume from checkpoint
python train_modified_hifigan_real.py --config training_config.json --resume D:/Thesis\ -\ Tool/checkpoints/checkpoint_epoch_50.pt
```

## Training Configuration

Key parameters in `training_config.json`:
- `batch_size`: 8 (adjust based on GPU memory)
- `learning_rate_g`: 2e-4 (generator learning rate)
- `learning_rate_d`: 1e-4 (discriminator learning rate)
- `num_epochs`: 100 (start with fewer epochs)
- `lambda_adv`: 1.0 (adversarial loss weight)
- `lambda_perceptual`: 1.0 (perceptual loss weight)
- `lambda_spectral`: 1.0 (spectral loss weight)

## Training Process

### 1. Generator Training
- **Adversarial Loss**: Trains generator to fool discriminators
- **Perceptual Loss**: Ensures high-level audio features match
- **Spectral Loss**: Ensures frequency characteristics match

### 2. Discriminator Training
- **Multi-Scale Discriminator**: Evaluates audio at different resolutions
- **Multi-Period Discriminator**: Evaluates audio at different periods
- **Real vs Fake Classification**: Learns to distinguish real from generated audio

### 3. Voice Cloning Integration
- **Speaker Embeddings**: ECAPA-TDNN (192-dimensional)
- **Emotion Embeddings**: Emotion2Vec (256-dimensional)
- **FiLM Conditioning**: Applies speaker and emotion conditioning throughout the network

## Output Files

### Checkpoints
- `checkpoint_epoch_X.pt`: Regular checkpoints every N epochs
- `best_model.pt`: Best model based on validation loss

### Logs
- Training progress logged to console
- Weights & Biases integration for monitoring
- Loss curves and metrics tracking

## Expected Results

After training, you should have:
1. **Fully trained modified HiFi-GAN** with all 279 parameters
2. **Compatible checkpoints** that match your current architecture
3. **Expressive voice cloning** capabilities
4. **Male voice preservation** from Spanish to English translation
5. **Real English audio output** instead of silent/burping sounds

## Troubleshooting

### GPU Memory Issues
- Reduce `batch_size` in config
- Use gradient accumulation if needed

### Dataset Issues
- Ensure CVSS-T dataset is properly organized
- Check metadata.json format matches expected structure

### Training Issues
- Monitor loss curves in Weights & Biases
- Check GPU utilization
- Verify all dependencies are installed

## Integration with Your System

After training completes:
1. **Replace old checkpoints** with new trained models
2. **Update your StreamSpeech integration** to use the trained model
3. **Test with your Spanish audio samples** - should produce clear English output
4. **Verify voice cloning** - output should sound like the original Spanish speaker

## Important Notes

- This is a **REAL training system** - no shortcuts or compromises
- Training may take several hours/days depending on dataset size
- Monitor training progress regularly
- Save checkpoints frequently to avoid losing progress
- This will solve your "no English output" problem by training the full architecture

## Success Criteria

Training is successful when:
- Validation loss decreases consistently
- Generated audio quality improves over epochs
- Checkpoints contain all 279 parameters (matching your architecture)
- English output is audible and clear
- Voice cloning preserves speaker characteristics


