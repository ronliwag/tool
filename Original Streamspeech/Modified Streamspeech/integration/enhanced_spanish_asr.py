#!/usr/bin/env python3
"""
Enhanced Spanish ASR Model for Thesis System
Replaces the current ASR while keeping all other thesis components
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSpanishASR:
    """Enhanced Spanish ASR using Whisper while keeping thesis components"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = 16000
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing Enhanced Spanish ASR on {device}")
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model optimized for Spanish"""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            
            logger.info("Loading Whisper model for Spanish ASR...")
            model_name = "openai/whisper-large-v3"
            
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Enhanced Spanish ASR loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading ASR model: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe Spanish audio to text with high accuracy
        
        Args:
            audio_data: Audio array (float32, [-1, 1])
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed Spanish text
        """
        try:
            # Ensure proper audio format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Process with Whisper
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription with Spanish language specification
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=5,
                    early_stopping=True,
                    language="es",  # Spanish
                    task="transcribe",
                    temperature=0.0,
                    no_repeat_ngram_size=0
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            logger.info(f"Spanish transcription: {transcription}")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in Spanish transcription: {e}")
            return ""
    
    def transcribe_realtime(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Real-time transcription for streaming audio"""
        try:
            # Transcribe the chunk
            text = self.transcribe(audio_chunk)
            
            return {
                "success": bool(text),
                "text": text,
                "confidence": 0.9 if text else 0.0,
                "is_partial": False  # Whisper gives final results
            }
            
        except Exception as e:
            logger.error(f"Error in real-time transcription: {e}")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "is_partial": False
            }
