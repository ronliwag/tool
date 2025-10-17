# Chipmunk Sound Troubleshooting Guide

## What Causes Chipmunk Sound?

Chipmunk sound (high-pitched, fast audio) is almost always caused by **sample rate mismatch**:
- Audio is generated at one sample rate (e.g., 22050 Hz)
- But played back at a different rate (e.g., 44100 Hz)
- This makes it play **2x faster** → chipmunk effect

## Quick Diagnosis

### Run the Diagnostic Script:
```bash
python diagnose_chipmunk_issue.py
```

This will:
1. ✅ Test each component individually
2. ✅ Save audio at each stage
3. ✅ Check sample rates everywhere
4. ✅ Verify translations are correct
5. ✅ Generate a detailed report

## Understanding the Output

### Diagnostic Files Created:
```
diagnostic_outputs/
├── 1_input_audio.wav          ← Original Spanish input (should sound normal)
├── 5_vocoder_output.wav       ← After vocoder processing
├── 6_full_pipeline_output.wav ← Final output from Modified mode
└── diagnostic_report.json     ← Detailed metrics
```

### Play Each File:
1. **1_input_audio.wav** - Should sound normal (Spanish)
2. **5_vocoder_output.wav** - If chipmunk HERE → vocoder issue
3. **6_full_pipeline_output.wav** - If chipmunk HERE → post-processing issue

## Common Issues & Fixes

### Issue 1: All outputs are chipmunk
**Cause**: Input audio has wrong sample rate metadata

**Fix**: Check the report for input sample rate
```json
{
  "sample_rates": {
    "input": 44100  ← Should be 22050 or 16000
  }
}
```

**Solution**: Resample input before processing

### Issue 2: Vocoder output is chipmunk
**Cause**: Vocoder generates audio at wrong sample rate

**Fix**: Check vocoder output sample rate in code
- Look in `integrate_trained_model.py`
- Verify the model was trained at 22050 Hz
- Check if model config has correct `sampling_rate`

**Location to check**:
```python
# In integrate_trained_model.py
# The model should output at 22050 Hz
# Check the training config
```

### Issue 3: Only final output is chipmunk
**Cause**: Post-processing resamples incorrectly

**Fix**: Check the playback code in the desktop app
- Look in `enhanced_desktop_app_streamlit_ui1.py`
- Find where audio is saved/played
- Verify sample rate is consistent

### Issue 4: Duration mismatch
If durations are different at each stage:
```
input: 5.0 seconds
vocoder_output: 2.5 seconds  ← PROBLEM! Half the duration
```

This means the vocoder is generating audio at the wrong rate.

## Step-by-Step Troubleshooting

### Step 1: Run Diagnostic
```bash
python diagnose_chipmunk_issue.py
```

### Step 2: Listen to Outputs
Play each file in `diagnostic_outputs/` folder:
- Which file first has chipmunk sound?
- Are durations correct (same as input)?

### Step 3: Check Sample Rates
Look at `diagnostic_report.json`:
```json
{
  "sample_rates": {
    "input": 16000,
    "mel_conversion": 22050,
    "vocoder_output": 22050,  ← Should all be 22050
    "pipeline_output": 22050
  }
}
```

### Step 4: Check Translations
Verify ASR and translation are working:
```json
{
  "translations": {
    "spanish": "hola mundo",  ← Should be actual Spanish text
    "english": "hello world"  ← Should be actual English translation
  }
}
```

## Specific Fixes

### Fix 1: Force Correct Sample Rate in Vocoder
Edit `integrate_trained_model.py`:
```python
# After generating audio from vocoder
audio = audio_tensor.squeeze(0).cpu().numpy()

# Ensure correct sample rate
EXPECTED_SR = 22050
# Audio should be generated at this rate by the model
```

### Fix 2: Check Model Training Config
Look at `trained_models/model_config.json`:
```json
{
  "training_config": {
    "sampling_rate": 22050,  ← Must be 22050!
    ...
  }
}
```

If this is wrong (e.g., 44100), the model was trained incorrectly.

### Fix 3: Verify Playback Sample Rate
In `enhanced_desktop_app_streamlit_ui1.py`, find where audio is saved:
```python
sf.write(output_path, audio, SAMPLE_RATE)  # SAMPLE_RATE must be 22050
```

### Fix 4: Check Mel-Spectrogram Conversion
In `streamspeech_modifications.py`:
```python
mel_spectrogram = librosa.feature.melspectrogram(
    y=waveform,
    sr=sample_rate,  # Must match model training (22050)
    n_mels=80,
    hop_length=256,
    n_fft=1024
)
```

## Expected Results

### ✅ Correct Configuration:
```
Sample rates: All 22050 Hz
Durations: All ~same (within 0.5s)
Translations: Correct Spanish → English
Audio: Normal sounding, not chipmunk
```

### ❌ Problematic Configuration:
```
Sample rates: Mixed (16000, 22050, 44100)
Durations: Varying significantly
Translations: Empty or incorrect
Audio: Chipmunk sound
```

## Quick Reference: Sample Rates

| Component | Expected Rate | Why |
|-----------|---------------|-----|
| Input Audio | 16000 Hz | Whisper model input |
| Resampled Audio | 22050 Hz | HiFi-GAN training rate |
| Mel-spectrogram | 22050 Hz | Must match vocoder training |
| Vocoder Output | 22050 Hz | Model generates at training rate |
| Final Output | 22050 Hz | Must match vocoder output |
| Playback | 22050 Hz | Must match final output |

**KEY RULE**: Once audio is resampled to 22050 Hz for the vocoder, it must stay at 22050 Hz throughout!

## Advanced Debugging

### Check Model Checkpoint:
```python
checkpoint = torch.load('trained_models/hifigan_checkpoints/best_model.pth')
print(checkpoint.keys())
print(checkpoint.get('config', {}))
```

### Manually Test Resampling:
```python
import librosa
import soundfile as sf

audio, sr = sf.read('input.wav')
audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050)
sf.write('resampled.wav', audio_22k, 22050)
# Play resampled.wav - should sound normal
```

### Check Vocoder Input/Output:
```python
# Add logging to integrate_trained_model.py
print(f"Vocoder input shape: {mel_tensor.shape}")
print(f"Vocoder output shape: {audio_tensor.shape}")
print(f"Expected samples: {mel_tensor.shape[-1] * hop_length}")
print(f"Actual samples: {audio_tensor.shape[-1]}")
```

## Need More Help?

1. **Run diagnostic script first**: `python diagnose_chipmunk_issue.py`
2. **Check the report**: `diagnostic_outputs/diagnostic_report.json`
3. **Listen to each output file** to isolate the problem
4. **Compare sample rates** in the report
5. **Check durations** for consistency

The diagnostic script will tell you exactly where the problem is!


