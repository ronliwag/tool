# Important Files Excluded from GitHub Repository

These files were excluded from the GitHub repository due to size constraints. Upload these to cloud storage (Google Drive, Dropbox, etc.)

## Critical Model Files

### 1. Pretrained StreamSpeech Models (CRITICAL - Required for the tool to work)
- `Original Streamspeech/pretrain_models/streamspeech.simultaneous.es-en.pt` (821.96 MB)
- `Original Streamspeech/pretrain_models/streamspeech.offline.es-en.pt` (382.91 MB)
- `Original Streamspeech/pretrain_models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt` (309.85 MB)
- `Original Streamspeech/pretrain_models/mHuBERT/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin` (2.93 MB)

### 2. Trained Models (Your thesis work)
- `trained_models/hifigan_checkpoints/best_model.pth` (156.41 MB)
- `trained_models/hifigan_checkpoints/checkpoint_epoch_19.pth` (156.41 MB)
- `trained_models/hifigan_checkpoints/checkpoint_epoch_18.pth` (156.41 MB)
- `trained_models/model_config.json` (config file)

### 3. Modified StreamSpeech Trained Models
- `Original Streamspeech/Modified Streamspeech/models/trained_discriminator.pth` (292.94 MB)
- `Original Streamspeech/Modified Streamspeech/models/trained_generator.pth` (15.16 MB)
- `Original Streamspeech/Modified Streamspeech/models/trained_generator_backup.pth` (15.16 MB)

### 4. Training Results Backup
- `Original Streamspeech/Modified Streamspeech/training_results/Colab 1st Results/trained_discriminator.pth` (292.94 MB)
- `Original Streamspeech/Modified Streamspeech/training_results/Colab 1st Results/trained_generator.pth` (15.16 MB)

### 5. Preprocessing Models (Small but important)
- `Original Streamspeech/preprocess_scripts/mhubert.km1000.layer11.pt` (2.93 MB)

### 6. SentencePiece Models (Small - for tokenization)
- `Original Streamspeech/configs/es-en/src_unigram6000/spm_unigram_es.model` (0.33 MB)
- `Original Streamspeech/configs/es-en/tgt_unigram6000/spm_unigram_es.model` (0.33 MB)
- `Original Streamspeech/configs/de-en/src_unigram6000/spm_unigram_de.model` (0.33 MB)
- `Original Streamspeech/configs/de-en/tgt_unigram6000/spm_unigram_de.model` (0.33 MB)
- `Original Streamspeech/configs/fr-en/src_unigram6000/spm_unigram_fr.model` (0.33 MB)
- `Original Streamspeech/configs/fr-en/tgt_unigram6000/spm_unigram_fr.model` (0.33 MB)

## Datasets (Optional - only if you want to backup training data)

### 7. Training Dataset (ZIP archives)
- `Important files - for tool/professional_cvss_dataset.zip` (473.04 MB)
- `Thesis_Development_Files/professional_cvss_dataset.zip` (473.04 MB)

### 8. Dataset Folders
- `professional_cvss_dataset/` - Full training dataset with Spanish/English audio pairs
  - Contains 2651 Spanish .wav files
  - Contains 2651 English .wav files
  - Contains metadata.json
- `real_training_dataset/` - Smaller test dataset
  - Contains 5 Spanish .wav files
  - Contains 5 English .wav files
  - Contains metadata.json

### 9. Speaker Recognition Model
- `pretrained_models/spkrec-ecapa-voxceleb/` folder (check if exists)

## Total Size Breakdown

**Must Have (Critical for tool functionality):**
- Pretrained models: ~1.5 GB
- Your trained models: ~470 MB
- **Total Critical: ~2 GB**

**Nice to Have (Backup/Training):**
- Datasets: ~950 MB
- Backup checkpoints: ~470 MB
- **Total Backup: ~1.4 GB**

**Grand Total: ~3.4 GB**

## Recommendation for Cloud Storage Structure

```
Tool_v2_Models/
├── pretrained_models/
│   ├── streamspeech.simultaneous.es-en.pt
│   ├── streamspeech.offline.es-en.pt
│   └── mhubert_base_vp_en_es_fr_it3.pt
├── trained_models/
│   └── hifigan_checkpoints/
│       ├── best_model.pth
│       ├── checkpoint_epoch_19.pth
│       └── model_config.json
└── datasets/ (optional)
    ├── professional_cvss_dataset.zip
    └── real_training_dataset.zip
```

## Note
After uploading to cloud storage, create a README.md in your GitHub repo with download instructions for these models.

