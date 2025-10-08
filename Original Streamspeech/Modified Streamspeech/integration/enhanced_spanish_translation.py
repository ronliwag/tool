#!/usr/bin/env python3
"""
Enhanced Spanish-English Translation for Thesis System
Uses better translation models while keeping thesis components
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSpanishTranslation:
    """Enhanced Spanish-English translation using better models"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing Enhanced Spanish Translation on {device}")
        self._load_model()
    
    def _load_model(self):
        """Load better Spanish-English translation model"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            logger.info("Loading enhanced Spanish-English translation model...")
            model_name = "Helsinki-NLP/opus-mt-es-en"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Enhanced Spanish translation loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading translation model: {e}")
            raise
    
    def translate(self, spanish_text: str) -> str:
        """
        Translate Spanish text to English with high accuracy
        
        Args:
            spanish_text: Spanish text to translate
            
        Returns:
            English translation
        """
        try:
            if not spanish_text.strip():
                return ""
            
            # Tokenize input
            inputs = self.tokenizer(
                spanish_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    temperature=0.7
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            logger.info(f"Translation: {spanish_text} -> {translation}")
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return ""
    
    def translate_realtime(self, spanish_text: str) -> Dict[str, Any]:
        """Real-time translation for streaming text"""
        try:
            # Translate the text
            english_text = self.translate(spanish_text)
            
            return {
                "success": bool(english_text),
                "spanish_text": spanish_text,
                "english_text": english_text,
                "confidence": 0.9 if english_text else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in real-time translation: {e}")
            return {
                "success": False,
                "spanish_text": spanish_text,
                "english_text": "",
                "confidence": 0.0
            }
