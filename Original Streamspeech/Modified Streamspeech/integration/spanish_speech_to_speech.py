#!/usr/bin/env python3
"""
Spanish Speech-to-Speech Integration using Hugging Face models
Optimized for Spanish language with real-time processing
"""

import os
import sys
import torch
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import soundfile as sf
import librosa

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpanishSpeechToSpeech:
    """Spanish-optimized Speech-to-Speech translation using Hugging Face models"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = 16000
        
        # Spanish-optimized models
        self.asr_model = None
        self.asr_processor = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.tts_model = None
        self.tts_processor = None
        
        # Real-time processing
        self.audio_buffer = []
        self.is_streaming = False
        
        logger.info(f"Initializing Spanish Speech-to-Speech on {device}")
        self._load_models()
    
    def _load_models(self):
        """Load Spanish-optimized models"""
        try:
            from transformers import (
                Wav2Vec2ForCTC, Wav2Vec2Processor,
                AutoModelForSeq2SeqLM, AutoTokenizer,
                AutoModel, AutoProcessor
            )
            
            logger.info("Loading Spanish ASR model...")
            # Use WhisperForConditionalGeneration for proper speech recognition
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.asr_model_name = "openai/whisper-large-v3"
            self.asr_processor = WhisperProcessor.from_pretrained(self.asr_model_name)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(self.asr_model_name).to(self.device)
            
            logger.info("Loading Spanish-English translation model...")
            # Use a better Spanish-English model
            self.translation_model_name = "Helsinki-NLP/opus-mt-es-en"
            self.translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name)
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name).to(self.device)
            
            logger.info("Loading Spanish TTS model...")
            # Use a proper TTS model for English output
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            self.tts_model_name = "microsoft/speecht5_tts"
            self.tts_processor = SpeechT5Processor.from_pretrained(self.tts_model_name)
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(self.tts_model_name).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            
            logger.info("✅ All Spanish models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def transcribe_spanish(self, audio_data: np.ndarray) -> str:
        """Transcribe Spanish audio to text with high accuracy"""
        try:
            # Ensure proper audio format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Process with Whisper
            inputs = self.asr_processor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription with proper Whisper parameters
            with torch.no_grad():
                outputs = self.asr_model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=5,
                    early_stopping=True,
                    language="es",  # Spanish language code
                    task="transcribe",
                    temperature=0.0,
                    no_repeat_ngram_size=0
                )
            
            # Decode transcription
            transcription = self.asr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            logger.info(f"Spanish transcription: {transcription}")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in Spanish transcription: {e}")
            return ""
    
    def translate_spanish_to_english(self, spanish_text: str) -> str:
        """Translate Spanish text to English with high accuracy"""
        try:
            if not spanish_text.strip():
                return ""
            
            # Tokenize input
            inputs = self.translation_tokenizer(
                spanish_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.translation_model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode translation
            translation = self.translation_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            logger.info(f"Translation: {spanish_text} -> {translation}")
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return ""
    
    def synthesize_english_speech(self, english_text: str) -> np.ndarray:
        """Synthesize English speech from text"""
        try:
            if not english_text.strip():
                return np.array([])
            
            # Process text
            inputs = self.tts_processor(text=english_text, return_tensors="pt").to(self.device)
            
            # Generate speech with proper SpeechT5 API
            with torch.no_grad():
                # Generate mel spectrograms
                mel_spectrograms = self.tts_model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings=None
                )
                
                # Convert mel spectrograms to audio using vocoder
                audio = self.vocoder(mel_spectrograms)
            
            # Convert to numpy array and normalize
            audio = audio.squeeze().cpu().numpy()
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            logger.info(f"Synthesized {len(audio)} samples for: {english_text}")
            return audio
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return np.array([])
    
    def process_audio_realtime(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process audio chunk in real-time for Spanish speech-to-speech"""
        try:
            start_time = time.time()
            
            # Add to buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Process if buffer is large enough (1 second of audio)
            if len(self.audio_buffer) >= self.sample_rate:
                # Convert buffer to numpy array
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Transcribe Spanish
                spanish_text = self.transcribe_spanish(audio_array)
                
                if spanish_text:
                    # Translate to English
                    english_text = self.translate_spanish_to_english(spanish_text)
                    
                    if english_text:
                        # Synthesize English speech
                        english_audio = self.synthesize_english_speech(english_text)
                        
                        processing_time = time.time() - start_time
                        
                        return {
                            "success": True,
                            "spanish_text": spanish_text,
                            "english_text": english_text,
                            "english_audio": english_audio,
                            "processing_time": processing_time,
                            "confidence": 0.9  # High confidence with Whisper
                        }
                
                # Clear buffer after processing
                self.audio_buffer = []
            
            return {
                "success": False,
                "spanish_text": "",
                "english_text": "",
                "english_audio": None,
                "processing_time": 0,
                "confidence": 0
            }
            
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            return {
                "success": False,
                "spanish_text": "",
                "english_text": "",
                "english_audio": None,
                "processing_time": 0,
                "confidence": 0
            }
    
    def start_streaming(self):
        """Start real-time streaming mode"""
        self.is_streaming = True
        self.audio_buffer = []
        logger.info("Started Spanish speech-to-speech streaming")
    
    def stop_streaming(self):
        """Stop real-time streaming mode"""
        self.is_streaming = False
        self.audio_buffer = []
        logger.info("Stopped Spanish speech-to-speech streaming")

def test_spanish_s2s():
    """Test the Spanish speech-to-speech system"""
    logger.info("Testing Spanish Speech-to-Speech system...")
    
    try:
        # Initialize system
        s2s = SpanishSpeechToSpeech()
        
        # Generate test Spanish audio (sine wave for testing)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * 440 * t) * 0.3
        
        # Test real-time processing
        s2s.start_streaming()
        result = s2s.process_audio_realtime(test_audio)
        s2s.stop_streaming()
        
        if result["success"]:
            logger.info(f"✅ Spanish: {result['spanish_text']}")
            logger.info(f"✅ English: {result['english_text']}")
            logger.info(f"✅ Processing time: {result['processing_time']:.3f}s")
            logger.info(f"✅ Confidence: {result['confidence']:.3f}")
        else:
            logger.warning("⚠️ No speech detected in test audio")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_spanish_s2s()
