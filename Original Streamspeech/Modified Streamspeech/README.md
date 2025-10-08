# Modified StreamSpeech - Thesis Implementation
## A Modified HiFi-GAN Vocoder Using ODConv and GRC for Expressive Voice Cloning

This folder contains the modified StreamSpeech implementation with the following enhancements:

### Key Modifications
- **ODConv (Omni-Dimensional Dynamic Convolution)**: Replaces static convolutions with dynamic ones
- **GRC (Grouped Residual Convolution)**: Enhanced residual blocks for better temporal modeling
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning for speaker and emotion adaptation
- **FiLM Conditioning**: Integration of ECAPA-TDNN and Emotion2Vec embeddings

### Integration with Original StreamSpeech
- Original StreamSpeech remains completely untouched
- This modified version can be used alongside the original for comparison
- Desktop application allows switching between Original and Modified modes

### Thesis Defense Features
- Side-by-side comparison of Original vs Modified performance
- Real-time evaluation metrics (SIM, Average Lagging, ASR-BLEU)
- Professional interface for thesis demonstration
- Spanish to English translation focus

### Usage
1. Use the comparison desktop app to switch between modes
2. Process identical audio samples with both systems
3. Compare results for thesis defense evaluation
4. Document performance differences statistically

### Files Structure
- `demo/` - Desktop comparison application
- `models/` - Modified HiFi-GAN implementation
- `configs/` - Configuration files for modified system
- `evaluation/` - Thesis evaluation metrics and analysis
- `integration/` - Scripts to integrate with original StreamSpeech

---
**Thesis Defense Tool - Modified HiFi-GAN with ODConv, GRC, and LoRA**
*Prepared for: A MODIFIED HIFI-GAN VOCODER USING ODCONV AND GRC FOR EXPRESSIVE VOICE CLONING IN STREAMSPEECH'S REAL-TIME TRANSLATION*







