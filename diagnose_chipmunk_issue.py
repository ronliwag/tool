#!/usr/bin/env python3
"""
Diagnostic Script for Chipmunk Sound Issue
==========================================

This script tests each component of the Modified StreamSpeech pipeline
to identify where the chipmunk sound originates.

Chipmunk sound typically indicates:
1. Sample rate mismatch (audio played at wrong rate)
2. Audio resampling issues
3. Time-stretching artifacts
4. Wrong audio format/encoding
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Important files - for tool'))

# Create diagnostic output directory
DIAG_DIR = Path("diagnostic_outputs")
DIAG_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("CHIPMUNK SOUND DIAGNOSTIC TOOL")
print("=" * 80)
print()

class ChipmunkDiagnostic:
    """Diagnose the chipmunk sound issue step by step"""
    
    def __init__(self):
        self.results = {
            'sample_rates': {},
            'audio_durations': {},
            'translations': {},
            'errors': [],
            'mel_spectrograms': {}
        }
        self.mel_specs = {}  # Store mel specs for visualization
    
    def test_input_audio(self, audio_path):
        """Test 1: Verify input audio is correct"""
        print("\n" + "=" * 80)
        print("TEST 1: INPUT AUDIO VERIFICATION")
        print("=" * 80)
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            duration = len(audio) / sr
            
            print(f"✓ Input file: {audio_path}")
            print(f"✓ Sample rate: {sr} Hz")
            print(f"✓ Duration: {duration:.2f} seconds")
            print(f"✓ Samples: {len(audio)}")
            print(f"✓ Shape: {audio.shape}")
            print(f"✓ Data type: {audio.dtype}")
            print(f"✓ Min value: {audio.min():.6f}")
            print(f"✓ Max value: {audio.max():.6f}")
            print(f"✓ Mean value: {audio.mean():.6f}")
            
            # Store results
            self.results['sample_rates']['input'] = sr
            self.results['audio_durations']['input'] = duration
            
            # Save a copy for comparison
            output_path = DIAG_DIR / "1_input_audio.wav"
            sf.write(output_path, audio, sr)
            print(f"✓ Saved copy to: {output_path}")
            
            # Check if it's chipmunk already
            if sr != 22050 and sr != 16000:
                print(f"⚠ WARNING: Unusual sample rate {sr} Hz (expected 16000 or 22050)")
            
            return audio, sr
            
        except Exception as e:
            print(f"✗ ERROR in input audio: {e}")
            self.results['errors'].append(f"Input audio: {e}")
            return None, None
    
    def test_spanish_asr(self, audio_path):
        """Test 2: Spanish ASR component"""
        print("\n" + "=" * 80)
        print("TEST 2: SPANISH ASR (Speech Recognition)")
        print("=" * 80)
        
        try:
            from spanish_asr_component import SpanishASR
            
            print("Loading Spanish ASR model...")
            asr = SpanishASR()
            
            print("Transcribing Spanish audio...")
            spanish_text = asr.transcribe(audio_path)
            
            print(f"✓ Spanish transcription: '{spanish_text}'")
            self.results['translations']['spanish'] = spanish_text
            
            if not spanish_text or spanish_text == "":
                print("⚠ WARNING: Empty Spanish transcription!")
            
            return spanish_text
            
        except Exception as e:
            print(f"✗ ERROR in Spanish ASR: {e}")
            self.results['errors'].append(f"Spanish ASR: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def test_translation(self, spanish_text):
        """Test 3: Spanish to English Translation"""
        print("\n" + "=" * 80)
        print("TEST 3: SPANISH TO ENGLISH TRANSLATION")
        print("=" * 80)
        
        try:
            from spanish_english_translation import SpanishEnglishTranslator
            
            print("Loading translation model...")
            translator = SpanishEnglishTranslator()
            
            print(f"Translating: '{spanish_text}'")
            english_text = translator.translate(spanish_text)
            
            print(f"✓ English translation: '{english_text}'")
            self.results['translations']['english'] = english_text
            
            if not english_text or english_text == "":
                print("⚠ WARNING: Empty English translation!")
            
            return english_text
            
        except Exception as e:
            print(f"✗ ERROR in translation: {e}")
            self.results['errors'].append(f"Translation: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def test_mel_spectrogram(self, audio, sr):
        """Test 4: Mel-spectrogram conversion"""
        print("\n" + "=" * 80)
        print("TEST 4: MEL-SPECTROGRAM CONVERSION")
        print("=" * 80)
        
        try:
            print(f"Input audio shape: {audio.shape}")
            print(f"Input sample rate: {sr} Hz")
            
            # Resample to 22050 if needed
            if sr != 22050:
                print(f"Resampling from {sr} Hz to 22050 Hz...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
                print(f"✓ Resampled audio shape: {audio.shape}")
            
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,
                hop_length=256,
                n_fft=1024
            )
            
            # Convert to log scale
            mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
            
            print(f"✓ Mel-spectrogram shape: {mel_spec.shape}")
            print(f"✓ Mel-spectrogram range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
            print(f"✓ Mel-spectrogram mean: {mel_spec.mean():.2f}")
            
            self.results['sample_rates']['mel_conversion'] = sr
            self.mel_specs['input_mel'] = mel_spec
            
            # Save mel visualization
            self.visualize_mel_spectrogram(
                mel_spec, 
                "Input Mel-Spectrogram (Spanish Audio)",
                DIAG_DIR / "mel_1_input.png"
            )
            
            return mel_spec, sr
            
        except Exception as e:
            print(f"✗ ERROR in mel conversion: {e}")
            self.results['errors'].append(f"Mel conversion: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def test_vocoder(self, mel_spec, expected_sr=22050):
        """Test 5: Vocoder (Mel to Audio)"""
        print("\n" + "=" * 80)
        print("TEST 5: VOCODER (Mel-spectrogram to Audio)")
        print("=" * 80)
        
        try:
            print(f"Input mel shape: {mel_spec.shape}")
            print(f"Expected output sample rate: {expected_sr} Hz")
            
            # Try to load the trained vocoder
            from integrate_trained_model import TrainedModelLoader
            
            print("Loading trained vocoder model...")
            loader = TrainedModelLoader()
            
            if loader.initialize_full_system():
                print("✓ Trained model loaded successfully")
                
                # Convert mel to audio
                mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
                
                print("Generating audio from mel-spectrogram...")
                with torch.no_grad():
                    # Create dummy speaker and emotion embeddings
                    speaker_embed = torch.randn(1, 192)
                    emotion_embed = torch.randn(1, 256)
                    
                    audio_tensor = loader.trained_model(mel_tensor, speaker_embed, emotion_embed)
                
                audio = audio_tensor.squeeze(0).cpu().numpy()
                
                print(f"✓ Generated audio shape: {audio.shape}")
                print(f"✓ Generated audio range: [{audio.min():.6f}, {audio.max():.6f}]")
                print(f"✓ Generated audio mean: {audio.mean():.6f}")
                
                # Calculate duration
                duration = len(audio) / expected_sr
                print(f"✓ Generated audio duration: {duration:.2f} seconds")
                
                self.results['sample_rates']['vocoder_output'] = expected_sr
                self.results['audio_durations']['vocoder_output'] = duration
                
                # Save vocoder output
                output_path = DIAG_DIR / "5_vocoder_output.wav"
                sf.write(output_path, audio, expected_sr)
                print(f"✓ Saved vocoder output to: {output_path}")
                print(f"  ⚠ PLAY THIS FILE - does it sound like chipmunk?")
                
                return audio, expected_sr
            else:
                print("✗ Failed to load trained model")
                return None, None
                
        except Exception as e:
            print(f"✗ ERROR in vocoder: {e}")
            self.results['errors'].append(f"Vocoder: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def test_full_pipeline(self, audio_path):
        """Test 6: Full Modified StreamSpeech Pipeline"""
        print("\n" + "=" * 80)
        print("TEST 6: FULL MODIFIED STREAMSPEECH PIPELINE")
        print("=" * 80)
        
        try:
            from streamspeech_modifications import StreamSpeechModifications
            
            print("Initializing StreamSpeech modifications...")
            streamspeech = StreamSpeechModifications()
            
            print("Processing audio with full pipeline...")
            enhanced_audio, results = streamspeech.process_audio_with_modifications(audio_path=audio_path)
            
            if enhanced_audio is not None:
                print(f"✓ Pipeline completed successfully")
                print(f"✓ Output audio shape: {enhanced_audio.shape}")
                print(f"✓ Output audio range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
                
                if 'spanish_text' in results:
                    print(f"✓ Spanish text: '{results['spanish_text']}'")
                if 'english_text' in results:
                    print(f"✓ English text: '{results['english_text']}'")
                
                # Assume output is at 22050 Hz
                output_sr = 22050
                duration = len(enhanced_audio) / output_sr
                print(f"✓ Output duration: {duration:.2f} seconds")
                
                self.results['sample_rates']['pipeline_output'] = output_sr
                self.results['audio_durations']['pipeline_output'] = duration
                
                # Save pipeline output
                output_path = DIAG_DIR / "6_full_pipeline_output.wav"
                sf.write(output_path, enhanced_audio, output_sr)
                print(f"✓ Saved pipeline output to: {output_path}")
                print(f"  ⚠ PLAY THIS FILE - does it sound like chipmunk?")
                
                return enhanced_audio, output_sr
            else:
                print("✗ Pipeline returned None")
                return None, None
                
        except Exception as e:
            print(f"✗ ERROR in full pipeline: {e}")
            self.results['errors'].append(f"Full pipeline: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def visualize_mel_spectrogram(self, mel_spec, title, save_path):
        """Visualize a single mel-spectrogram"""
        try:
            plt.figure(figsize=(12, 4))
            plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved mel visualization to: {save_path}")
        except Exception as e:
            print(f"✗ Error visualizing mel: {e}")
    
    def test_original_mode(self, audio_path):
        """Test 7: Original StreamSpeech Mode"""
        print("\n" + "=" * 80)
        print("TEST 7: ORIGINAL STREAMSPEECH MODE")
        print("=" * 80)
        
        try:
            # Add Original StreamSpeech paths
            sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'demo'))
            sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'fairseq'))
            
            from app import run, reset, SAMPLE_RATE
            import app
            
            print("Processing with Original StreamSpeech...")
            reset()
            run(audio_path)
            
            if app.S2ST:
                # Convert to numpy array
                if isinstance(app.S2ST, list):
                    if len(app.S2ST) > 0 and isinstance(app.S2ST[0], list):
                        s2st_array = np.concatenate([np.array(chunk) for chunk in app.S2ST if chunk])
                    else:
                        s2st_array = np.array(app.S2ST, dtype=np.float32)
                else:
                    s2st_array = np.array(app.S2ST, dtype=np.float32)
                
                print(f"✓ Original mode output shape: {s2st_array.shape}")
                print(f"✓ Sample rate: {SAMPLE_RATE} Hz")
                
                # Save original mode output
                output_path = DIAG_DIR / "7_original_mode_output.wav"
                
                # Handle stereo if needed
                if len(s2st_array.shape) == 1:
                    s2st_array_stereo = np.column_stack((s2st_array, s2st_array))
                else:
                    s2st_array_stereo = s2st_array
                
                sf.write(output_path, s2st_array_stereo, SAMPLE_RATE)
                print(f"✓ Saved to: {output_path}")
                print(f"  ⚠ PLAY THIS FILE - compare with Modified mode")
                
                # Generate mel from output for visualization
                if len(s2st_array.shape) > 1:
                    audio_mono = s2st_array[:, 0]  # Use first channel
                else:
                    audio_mono = s2st_array
                
                mel_original = librosa.feature.melspectrogram(
                    y=audio_mono,
                    sr=SAMPLE_RATE,
                    n_mels=80,
                    hop_length=256,
                    n_fft=1024
                )
                mel_original = np.log(np.clip(mel_original, a_min=1e-5, a_max=None))
                
                self.mel_specs['original_output_mel'] = mel_original
                self.results['sample_rates']['original_output'] = SAMPLE_RATE
                self.results['audio_durations']['original_output'] = len(audio_mono) / SAMPLE_RATE
                
                # Visualize original output mel
                self.visualize_mel_spectrogram(
                    mel_original,
                    "Original Mode Output Mel-Spectrogram",
                    DIAG_DIR / "mel_7_original_output.png"
                )
                
                # Get translations
                spanish_text = ""
                if app.ASR:
                    max_key = max(app.ASR.keys())
                    spanish_text = app.ASR[max_key]
                    print(f"✓ Spanish (Original): {spanish_text}")
                
                english_text = ""
                if app.ST:
                    max_key = max(app.ST.keys())
                    english_text = app.ST[max_key]
                    print(f"✓ English (Original): {english_text}")
                
                self.results['translations']['original_spanish'] = spanish_text
                self.results['translations']['original_english'] = english_text
                
                return s2st_array, SAMPLE_RATE
            else:
                print("✗ Original mode returned no output")
                return None, None
                
        except Exception as e:
            print(f"✗ ERROR in original mode: {e}")
            self.results['errors'].append(f"Original mode: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def test_modified_mode(self, audio_path):
        """Test 8: Modified StreamSpeech Mode (using desktop app method)"""
        print("\n" + "=" * 80)
        print("TEST 8: MODIFIED STREAMSPEECH MODE")
        print("=" * 80)
        
        try:
            from streamspeech_modifications import StreamSpeechModifications
            
            print("Initializing Modified StreamSpeech...")
            streamspeech = StreamSpeechModifications()
            
            print("Processing audio with Modified mode...")
            enhanced_audio, results = streamspeech.process_audio_with_modifications(audio_path=audio_path)
            
            if enhanced_audio is not None:
                print(f"✓ Modified mode output shape: {enhanced_audio.shape}")
                
                # Assume 22050 Hz
                output_sr = 22050
                duration = len(enhanced_audio) / output_sr
                print(f"✓ Output duration: {duration:.2f} seconds")
                print(f"✓ Sample rate: {output_sr} Hz")
                
                self.results['sample_rates']['modified_output'] = output_sr
                self.results['audio_durations']['modified_output'] = duration
                
                # Save modified mode output
                output_path = DIAG_DIR / "8_modified_mode_output.wav"
                sf.write(output_path, enhanced_audio, output_sr)
                print(f"✓ Saved to: {output_path}")
                print(f"  ⚠ PLAY THIS FILE - compare with Original mode")
                
                # Generate mel from output for visualization
                mel_modified = librosa.feature.melspectrogram(
                    y=enhanced_audio,
                    sr=output_sr,
                    n_mels=80,
                    hop_length=256,
                    n_fft=1024
                )
                mel_modified = np.log(np.clip(mel_modified, a_min=1e-5, a_max=None))
                
                self.mel_specs['modified_output_mel'] = mel_modified
                
                # Visualize modified output mel
                self.visualize_mel_spectrogram(
                    mel_modified,
                    "Modified Mode Output Mel-Spectrogram",
                    DIAG_DIR / "mel_8_modified_output.png"
                )
                
                # Get translations
                if 'spanish_text' in results:
                    print(f"✓ Spanish (Modified): {results['spanish_text']}")
                    self.results['translations']['modified_spanish'] = results['spanish_text']
                if 'english_text' in results:
                    print(f"✓ English (Modified): {results['english_text']}")
                    self.results['translations']['modified_english'] = results['english_text']
                
                return enhanced_audio, output_sr
            else:
                print("✗ Modified mode returned None")
                return None, None
                
        except Exception as e:
            print(f"✗ ERROR in modified mode: {e}")
            self.results['errors'].append(f"Modified mode: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def compare_mel_spectrograms(self):
        """Test 9: Compare all mel-spectrograms side by side"""
        print("\n" + "=" * 80)
        print("TEST 9: MEL-SPECTROGRAM COMPARISON")
        print("=" * 80)
        
        try:
            # Create comprehensive comparison figure
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            plot_configs = [
                ('input_mel', 'Input Mel-Spectrogram\n(Spanish Audio)', gs[0, :]),
                ('original_output_mel', 'Original Mode Output\n(Unmodified StreamSpeech)', gs[1, 0]),
                ('modified_output_mel', 'Modified Mode Output\n(ODConv+GRC+LoRA)', gs[1, 1]),
            ]
            
            for mel_key, title, grid_spec in plot_configs:
                if mel_key in self.mel_specs:
                    ax = fig.add_subplot(grid_spec)
                    mel = self.mel_specs[mel_key]
                    
                    im = ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time Frames')
                    ax.set_ylabel('Mel Frequency Bins')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
                    cbar.set_label('Magnitude (dB)')
                    
                    # Add statistics
                    stats_text = f"Shape: {mel.shape}\nRange: [{mel.min():.1f}, {mel.max():.1f}]\nMean: {mel.mean():.1f}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8), fontsize=8)
            
            # Add difference plot if both outputs exist
            if 'original_output_mel' in self.mel_specs and 'modified_output_mel' in self.mel_specs:
                ax_diff = fig.add_subplot(gs[2, :])
                
                mel_orig = self.mel_specs['original_output_mel']
                mel_mod = self.mel_specs['modified_output_mel']
                
                # Resize to match if needed
                min_frames = min(mel_orig.shape[1], mel_mod.shape[1])
                mel_orig_crop = mel_orig[:, :min_frames]
                mel_mod_crop = mel_mod[:, :min_frames]
                
                diff = mel_mod_crop - mel_orig_crop
                
                im_diff = ax_diff.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
                ax_diff.set_title('Difference (Modified - Original)\nRed = Modified Louder, Blue = Original Louder', 
                                 fontsize=12, fontweight='bold')
                ax_diff.set_xlabel('Time Frames')
                ax_diff.set_ylabel('Mel Frequency Bins')
                
                cbar_diff = plt.colorbar(im_diff, ax=ax_diff, format='%+2.0f dB')
                cbar_diff.set_label('Difference (dB)')
                
                # Add statistics
                stats_text = f"Max diff: {diff.max():.1f} dB\nMin diff: {diff.min():.1f} dB\nMean |diff|: {np.abs(diff).mean():.1f} dB"
                ax_diff.text(0.02, 0.98, stats_text, transform=ax_diff.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                           facecolor='white', alpha=0.8), fontsize=8)
            
            plt.suptitle('Mel-Spectrogram Analysis: Original vs Modified StreamSpeech', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            comparison_path = DIAG_DIR / "mel_comparison_full.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved comprehensive comparison to: {comparison_path}")
            print("\n📊 MEL-SPECTROGRAM ANALYSIS:")
            print("   - Compare shapes: should be similar")
            print("   - Compare ranges: should be similar")
            print("   - Difference plot shows where Modified differs from Original")
            print("   - Large differences may indicate processing issues")
            
        except Exception as e:
            print(f"✗ ERROR in mel comparison: {e}")
            import traceback
            print(traceback.format_exc())
    
    def test_sample_rate_mismatch(self):
        """Test 10: Check for sample rate mismatches"""
        print("\n" + "=" * 80)
        print("TEST 10: SAMPLE RATE MISMATCH ANALYSIS")
        print("=" * 80)
        
        print("Checking sample rates at each stage:")
        for stage, sr in self.results['sample_rates'].items():
            print(f"  {stage:.<40} {sr} Hz")
        
        # Check if all sample rates match
        sample_rates = list(self.results['sample_rates'].values())
        if len(set(sample_rates)) > 1:
            print("\n⚠ WARNING: Sample rate mismatch detected!")
            print("  This is likely the cause of the chipmunk sound.")
            print("  All stages should use the same sample rate (22050 Hz).")
        else:
            print("\n✓ All sample rates are consistent")
        
        print("\nChecking audio durations:")
        for stage, duration in self.results['audio_durations'].items():
            print(f"  {stage:.<40} {duration:.2f} seconds")
        
        # Check duration consistency
        durations = list(self.results['audio_durations'].values())
        if len(durations) > 1:
            duration_variance = max(durations) - min(durations)
            if duration_variance > 0.5:
                print(f"\n⚠ WARNING: Significant duration variance ({duration_variance:.2f}s)")
                print("  This suggests time-stretching or resampling issues.")
    
    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC REPORT")
        print("=" * 80)
        
        print("\n1. TRANSLATIONS:")
        if 'spanish' in self.results['translations']:
            print(f"   Spanish: {self.results['translations']['spanish']}")
        if 'english' in self.results['translations']:
            print(f"   English: {self.results['translations']['english']}")
        
        print("\n2. SAMPLE RATES:")
        for stage, sr in self.results['sample_rates'].items():
            print(f"   {stage}: {sr} Hz")
        
        print("\n3. AUDIO DURATIONS:")
        for stage, duration in self.results['audio_durations'].items():
            print(f"   {stage}: {duration:.2f}s")
        
        if self.results['errors']:
            print("\n4. ERRORS ENCOUNTERED:")
            for error in self.results['errors']:
                print(f"   ✗ {error}")
        else:
            print("\n4. NO ERRORS ENCOUNTERED")
        
        print("\n5. DIAGNOSTIC FILES SAVED TO:")
        print(f"   {DIAG_DIR.absolute()}")
        print("\n   AUDIO FILES:")
        print("   - 1_input_audio.wav (original input)")
        print("   - 5_vocoder_output.wav (vocoder output)")
        print("   - 6_full_pipeline_output.wav (full pipeline output)")
        print("   - 7_original_mode_output.wav (Original StreamSpeech)")
        print("   - 8_modified_mode_output.wav (Modified StreamSpeech)")
        print("\n   MEL-SPECTROGRAM VISUALIZATIONS:")
        print("   - mel_1_input.png (input mel-spectrogram)")
        print("   - mel_7_original_output.png (Original mode output mel)")
        print("   - mel_8_modified_output.png (Modified mode output mel)")
        print("   - mel_comparison_full.png (side-by-side comparison)")
        print("\n   📊 OPEN mel_comparison_full.png for full visual comparison!")
        
        print("\n6. NEXT STEPS:")
        print("   1. 🎵 Play each audio file to identify where chipmunk starts:")
        print("      - 7_original_mode_output.wav (should sound normal)")
        print("      - 8_modified_mode_output.wav (check if chipmunk)")
        print("   2. 📊 Open mel_comparison_full.png to see visual comparison")
        print("   3. ⏱️  Compare durations - they should all be similar")
        print("   4. 🔊 Check sample rates - they should all be 22050 Hz")
        print("   5. 🔍 Look at mel-spectrograms:")
        print("      - Similar shapes = good")
        print("      - Similar patterns = good")
        print("      - Compressed time axis = sample rate mismatch!")
        print("   6. 📝 Compare translations between Original and Modified modes")
        
        # Save report to JSON
        report_path = DIAG_DIR / "diagnostic_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n   Report saved to: {report_path}")
        
        print("\n" + "=" * 80)

def main():
    """Run full diagnostic suite"""
    
    # Get input audio file
    print("Please provide a Spanish audio file to test.")
    print("Examples:")
    print("  - real_training_dataset/spanish/real_sample_001_spanish.wav")
    print("  - professional_cvss_dataset/spanish/[any file]")
    print()
    
    audio_file = input("Enter path to Spanish audio file (or press Enter for default): ").strip()
    
    if not audio_file:
        # Use default
        audio_file = "real_training_dataset/spanish/real_sample_001_spanish.wav"
        print(f"Using default: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        return
    
    print(f"\nStarting diagnostic on: {audio_file}")
    print()
    
    # Create diagnostic instance
    diag = ChipmunkDiagnostic()
    
    # Run tests sequentially
    audio, sr = diag.test_input_audio(audio_file)
    if audio is None:
        print("\n✗ Cannot proceed - input audio test failed")
        return
    
    spanish_text = diag.test_spanish_asr(audio_file)
    
    if spanish_text:
        english_text = diag.test_translation(spanish_text)
    
    mel_spec, mel_sr = diag.test_mel_spectrogram(audio, sr)
    
    if mel_spec is not None:
        vocoder_audio, vocoder_sr = diag.test_vocoder(mel_spec)
    
    pipeline_audio, pipeline_sr = diag.test_full_pipeline(audio_file)
    
    # Test Original mode
    original_audio, original_sr = diag.test_original_mode(audio_file)
    
    # Test Modified mode
    modified_audio, modified_sr = diag.test_modified_mode(audio_file)
    
    # Compare mel-spectrograms
    diag.compare_mel_spectrograms()
    
    # Analyze mismatches
    diag.test_sample_rate_mismatch()
    
    # Generate report
    diag.generate_report()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE!")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()


