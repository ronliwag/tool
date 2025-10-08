# StreamSpeech Comparison Tool
## Thesis Defense Application

This application provides a desktop interface for comparing Original StreamSpeech with Modified StreamSpeech (featuring ODConv, GRC, and LoRA modifications) for thesis defense evaluation.

## Features

### System Modes
- **Original StreamSpeech**: Baseline system with standard HiFi-GAN vocoder
- **Modified StreamSpeech**: Enhanced system with ODConv, GRC, and LoRA modifications

### Key Capabilities
- Real-time audio processing and visualization
- Side-by-side comparison of results
- Professional evaluation metrics display
- Spanish to English translation focus
- Latency control (160ms - 640ms)
- Audio playback functionality

## Usage Instructions

### 1. Launch the Application
```bash
# Run the enhanced desktop application
run_enhanced_desktop.bat
```

### 2. Select System Mode
- Choose between "Original StreamSpeech" or "Modified StreamSpeech (ODConv+GRC+LoRA)"
- The status indicator shows the current mode

### 3. Load Audio File
- Click "Browse Audio File" to select a Spanish audio file
- Supported formats: WAV, MP3, FLAC
- Use the provided Spanish samples in `example/wavs/`

### 4. Configure Settings
- Adjust latency using the slider (160ms - 640ms)
- Lower latency = faster processing, higher latency = better quality
- Default: 320ms (optimal for real-time performance)

### 5. Process Audio
- Click "Process Audio" to start translation
- Monitor progress in the Processing Status section
- View real-time waveform visualizations

### 6. Review Results
- **Audio Visualization Tab**: View input and output waveforms
- **Recognition & Translation Tab**: See Spanish ASR and English translation text
- **Evaluation Metrics Tab**: Review performance metrics and thesis defense notes
- **Processing Log Tab**: Monitor detailed processing information

### 7. Compare Results
- Process the same audio file with both modes
- Compare the results side by side
- Document findings for thesis defense

## File Structure

```
demo/
├── enhanced_desktop_app.py          # Main comparison application
├── run_enhanced_desktop.bat         # Launch script
├── config.json                      # Original StreamSpeech config
├── config_modified.json             # Modified StreamSpeech config
├── COMPARISON_TOOL_README.md        # This file
└── example/
    ├── wavs/                        # Spanish audio samples
    └── outputs/                     # Generated outputs
```

## Configuration

### Original StreamSpeech
- Uses standard HiFi-GAN vocoder
- Configuration: `config.json`
- Model: `streamspeech.simultaneous.es-en.pt`

### Modified StreamSpeech
- Uses enhanced HiFi-GAN with ODConv, GRC, and LoRA
- Configuration: `config_modified.json`
- Same model but with modified processing pipeline

## Evaluation Metrics

The application displays key metrics for thesis defense:

1. **Real-time Performance**: Whether the system maintains real-time capability
2. **Processing Status**: Success/failure of audio processing
3. **Output Quality**: Whether translation was generated successfully
4. **Latency Analysis**: Performance at different latency settings

## Thesis Defense Usage

### For Research Question 1 (Original Performance)
1. Select "Original StreamSpeech" mode
2. Process Spanish audio samples
3. Record performance metrics
4. Document results

### For Research Question 2 (Modified Performance)
1. Select "Modified StreamSpeech" mode
2. Process same Spanish audio samples
3. Record performance metrics
4. Document results

### For Research Question 3 (Comparison)
1. Process identical audio with both modes
2. Compare side-by-side results
3. Analyze performance differences
4. Document statistical significance

## Troubleshooting

### Common Issues
- **"No output generated"**: Check audio file format and quality
- **"Agent initialization failed"**: Verify model files are present
- **"Audio playback error"**: Check pygame installation and audio drivers

### System Requirements
- Python 3.8+
- PyTorch 2.0+
- Required packages: soundfile, pygame, matplotlib, tkinter
- StreamSpeech models and dependencies

## Output Files

Generated outputs are saved to `example/outputs/` with naming convention:
- `original_output_[filename]`: Original StreamSpeech output
- `modified_output_[filename]`: Modified StreamSpeech output

## Support

For technical issues or questions about the thesis implementation, refer to:
- StreamSpeech documentation
- Thesis evaluation results in `D:/Thesis - Tool/`
- Processing logs in the application

---

**Thesis Defense Tool - Modified HiFi-GAN with ODConv, GRC, and LoRA**
*Prepared for: A MODIFIED HIFI-GAN VOCODER USING ODCONV AND GRC FOR EXPRESSIVE VOICE CLONING IN STREAMSPEECH'S REAL-TIME TRANSLATION*







