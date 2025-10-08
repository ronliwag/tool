#!/usr/bin/env python3
"""
Complete Speech-to-Speech Translation Pipeline
Integrates ASR, Translation, and Modified HiFi-GAN Vocoder
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import time
from pathlib import Path

# Add model paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'integration'))

# Import components
from spanish_asr_component import SpanishASR
from spanish_english_translation import SpanishEnglishTranslator
from integrate_trained_model import TrainedModelLoader

class CompleteS2STPipeline:
    """Complete Speech-to-Speech Translation Pipeline"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.asr = None
        self.translator = None
        self.vocoder = None
        self.components_initialized = False
        
        print("Initializing Complete S2ST Pipeline")
        print(f"Device: {self.device}")
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            print("\n1. Initializing Spanish ASR...")
            self.asr = SpanishASR(self.device)
            print("ASR initialized successfully")
            
            print("\n2. Initializing Spanish-English Translator...")
            self.translator = SpanishEnglishTranslator(self.device)
            print("Translator initialized successfully")
            
            print("\n3. Initializing Modified HiFi-GAN Vocoder...")
            self.vocoder = TrainedModelLoader()
            if not self.vocoder.initialize_full_system():
                raise Exception("Failed to initialize trained model")
            print("Vocoder initialized successfully")
            
            print("\nAll components initialized successfully!")
            self.components_initialized = True
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            self.components_initialized = False
            raise
    
    def process_audio_file(self, spanish_audio_path, output_dir=None):
        """Process Spanish audio file through complete pipeline"""
        try:
            start_time = time.time()
            
            print(f"\nProcessing: {spanish_audio_path}")
            print("=" * 60)
            
            # Step 1: Spanish ASR
            print("Step 1: Spanish Speech Recognition...")
            spanish_text = self.asr.transcribe(spanish_audio_path)
            if not spanish_text:
                raise Exception("Failed to transcribe Spanish audio")
            
            print(f"Spanish text: {spanish_text}")
            
            # Step 2: Spanish to English Translation
            print("\nStep 2: Spanish to English Translation...")
            english_text = self.translator.translate(spanish_text)
            if not english_text:
                raise Exception("Failed to translate Spanish text")
            
            print(f"English text: {english_text}")
            
            # Step 3: Generate English Audio using Modified HiFi-GAN
            print("\nStep 3: English Audio Generation with Modified HiFi-GAN...")
            english_audio_path = self.vocoder.process_audio(spanish_audio_path, output_dir)
            if not english_audio_path:
                raise Exception("Failed to generate English audio")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Results summary
            results = {
                'spanish_audio_path': spanish_audio_path,
                'spanish_text': spanish_text,
                'english_text': english_text,
                'english_audio_path': english_audio_path,
                'processing_time': processing_time,
                'success': True
            }
            
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Spanish text: {spanish_text}")
            print(f"English text: {english_text}")
            print(f"English audio: {english_audio_path}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {
                'spanish_audio_path': spanish_audio_path,
                'spanish_text': None,
                'english_text': None,
                'english_audio_path': None,
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def process_audio_array(self, spanish_audio_array, sample_rate=22050):
        """Process Spanish audio array through complete pipeline"""
        try:
            start_time = time.time()
            
            print("\nProcessing audio array")
            print("=" * 40)
            
            # Step 1: Spanish ASR
            print("Step 1: Spanish Speech Recognition...")
            spanish_text = self.asr.transcribe_audio_array(spanish_audio_array)
            if not spanish_text:
                raise Exception("Failed to transcribe Spanish audio")
            
            print(f"Spanish text: {spanish_text}")
            
            # Step 2: Spanish to English Translation
            print("\nStep 2: Spanish to English Translation...")
            english_text = self.translator.translate(spanish_text)
            if not english_text:
                raise Exception("Failed to translate Spanish text")
            
            print(f"English text: {english_text}")
            
            # Step 3: Generate English Audio using Modified HiFi-GAN
            print("\nStep 3: English Audio Generation with Modified HiFi-GAN...")
            
            # For audio array, we need to save it temporarily and process
            temp_path = "temp_spanish_audio.wav"
            sf.write(temp_path, spanish_audio_array, sample_rate)
            
            try:
                english_audio_path = self.vocoder.process_audio(temp_path)
                if not english_audio_path:
                    raise Exception("Failed to generate English audio")
                
                # Load the generated English audio
                english_audio, _ = librosa.load(english_audio_path, sr=sample_rate)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Results summary
            results = {
                'spanish_text': spanish_text,
                'english_text': english_text,
                'english_audio': english_audio,
                'processing_time': processing_time,
                'success': True
            }
            
            print("\n" + "=" * 40)
            print("PROCESSING COMPLETE")
            print("=" * 40)
            print(f"Spanish text: {spanish_text}")
            print(f"English text: {english_text}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error processing audio array: {e}")
            return {
                'spanish_text': None,
                'english_text': None,
                'english_audio': None,
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def batch_process(self, spanish_audio_paths, output_dir=None):
        """Process multiple Spanish audio files"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, audio_path in enumerate(spanish_audio_paths):
            print(f"\nProcessing file {i+1}/{len(spanish_audio_paths)}")
            result = self.process_audio_file(audio_path, output_dir)
            results.append(result)
        
        return results
    
    def get_pipeline_info(self):
        """Get information about the pipeline components"""
        info = {
            'device': str(self.device),
            'asr_model': self.asr.model_name if self.asr else None,
            'translation_model': self.translator.model_name if self.translator else None,
            'vocoder_model': 'Modified HiFi-GAN with ODConv and GRC+LoRA',
            'components_initialized': all([self.asr, self.translator, self.vocoder])
        }
        
        return info

def main():
    """Test the complete S2ST pipeline"""
    print("COMPLETE SPEECH-TO-SPEECH TRANSLATION PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = CompleteS2STPipeline()
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        print("\nPipeline Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test with audio file
        test_audio = input("\nEnter path to Spanish audio file for testing (or press Enter to skip): ").strip()
        
        if test_audio and os.path.exists(test_audio):
            results = pipeline.process_audio_file(test_audio)
            
            if results['success']:
                print(f"\nSUCCESS! English audio generated: {results['english_audio_path']}")
            else:
                print(f"\nFAILED: {results.get('error', 'Unknown error')}")
        else:
            print("Skipping audio file test")
        
        # Test with audio array
        print("\nTesting with synthetic Spanish audio...")
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate a simple sine wave as test audio
        test_audio_array = np.sin(2 * np.pi * 440 * t) * 0.3
        test_audio_array = test_audio_array.astype(np.float32)
        
        results = pipeline.process_audio_array(test_audio_array, sample_rate)
        
        if results['success']:
            print("Synthetic audio test completed successfully")
        else:
            print(f"Synthetic audio test failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Pipeline test failed: {e}")

if __name__ == "__main__":
    main()
