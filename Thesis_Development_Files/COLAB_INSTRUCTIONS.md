# GOOGLE COLAB PRO THESIS TRAINING INSTRUCTIONS

## Overview
This guide will help you run the complete thesis training in Google Colab Pro, avoiding all local hardware limitations while maintaining 100% real data and models.

## Prerequisites
- ✅ Google Colab Pro subscription (for better GPU)
- ✅ Your CVSS-T dataset (10k Spanish + 10k English samples)
- ✅ Internet connection

## Step-by-Step Instructions

### 1. Open Google Colab Pro
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Make sure you're using Colab Pro (better GPU allocation)

### 2. Upload the Setup Files
Upload these files to your Colab environment:
- `colab_setup.py` (complete setup script)
- `thesis_training_colab.ipynb` (notebook template)

### 3. Install Dependencies
Run this in the first cell:

```python
# Install required packages for Google Colab
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers speechbrain torch-audiomentations soundfile librosa
!pip install wandb  # For training monitoring

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 4. Upload Your Dataset
Run this in the second cell:

```python
# Upload your CVSS-T dataset
from google.colab import files
import zipfile

print("Please upload your CVSS-T dataset as a ZIP file:")
uploaded = files.upload()

# Extract the dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted {filename} successfully!")

# List extracted contents
!find . -name "*.wav" | head -10
```

### 5. Run the Complete Setup
Copy and paste the entire contents of `colab_setup.py` into a new cell and run it.

### 6. Expected Results
After running the setup, you should see:
- ✅ GPU detected and available
- ✅ REAL ECAPA-TDNN extractor initialized
- ✅ REAL emotion extractor initialized
- ✅ All dependencies installed successfully

## Advantages of Colab Pro Training

### Performance Benefits:
- **Better GPU**: Colab Pro provides better GPU allocation (T4, V100, A100)
- **More Memory**: Up to 25GB RAM vs your laptop's limitations
- **Faster Training**: No thermal throttling or power limitations
- **Parallel Processing**: Better multiprocessing support

### Training Configuration:
- **Batch Size**: 16 (vs 2 locally) - 8x faster training
- **Epochs**: 100 (vs 10 locally) - Complete training
- **Real Embeddings**: 100% real ECAPA-TDNN and Emotion2Vec
- **Real Dataset**: Your complete CVSS-T dataset

### No Compromises:
- ✅ 100% real data (no fake/dummy data)
- ✅ 100% real models (no fallbacks)
- ✅ 100% real embeddings (Hugging Face models)
- ✅ Professional training pipeline
- ✅ Complete voice cloning system

## Training Process
1. **Dataset Processing**: Real CVSS-T dataset integration
2. **Embedding Extraction**: Real speaker and emotion embeddings
3. **Model Training**: Professional Modified HiFi-GAN training
4. **Validation**: Voice cloning quality assessment
5. **Model Export**: Trained model for thesis demonstration

## Expected Training Time
- **Colab Pro**: 2-4 hours for complete training
- **Local (your laptop)**: 8-12 hours or impossible due to hardware limits

## Output Files
After training completes, you'll have:
- `professional_cvss_best.pt` - Best trained model
- `training_logs.json` - Training progress logs
- `validation_results.json` - Voice cloning quality metrics
- Sample output audio files for thesis demonstration

## Troubleshooting
- **GPU not available**: Make sure you're using Colab Pro
- **Out of memory**: Reduce batch size from 16 to 8
- **Dataset issues**: Ensure your ZIP file contains the correct structure

## Next Steps
1. Follow these instructions exactly
2. Upload your CVSS-T dataset
3. Run the complete training
4. Download the trained model
5. Use for thesis demonstration

**No compromises, no shortcuts, 100% real implementation!**
