#!/usr/bin/env python3
"""
Web interface integration for Spanish Speech-to-Speech
Replaces the current ASR/Translation/TTS with Hugging Face models
"""

import os
import sys
import asyncio
import json
import base64
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append('.')

from spanish_speech_to_speech import SpanishSpeechToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpanishWebIntegration:
    """Web interface integration for Spanish Speech-to-Speech"""
    
    def __init__(self):
        self.s2s = SpanishSpeechToSpeech()
        self.is_streaming = False
        logger.info("Spanish Web Integration initialized")
    
    async def process_audio_chunk(self, audio_data: np.ndarray, client_id: str):
        """Process audio chunk from web interface"""
        try:
            if not self.is_streaming:
                self.s2s.start_streaming()
                self.is_streaming = True
            
            # Process with Spanish S2S
            result = self.s2s.process_audio_realtime(audio_data)
            
            if result["success"]:
                # Send real-time ASR result
                asr_response = {
                    "type": "asr_result",
                    "source_text": result["spanish_text"],
                    "is_partial": False,
                    "confidence": result["confidence"]
                }
                
                # Send translation result
                translation_response = {
                    "type": "translation_result",
                    "source_text": result["spanish_text"],
                    "translated_text": result["english_text"],
                    "confidence": result["confidence"]
                }
                
                # Send TTS result
                if result["english_audio"] is not None:
                    audio_b64 = base64.b64encode(
                        (result["english_audio"] * 32767).astype(np.int16).tobytes()
                    ).decode('utf-8')
                    
                    tts_response = {
                        "type": "tts_result",
                        "audio_data": audio_b64,
                        "text": result["english_text"]
                    }
                    
                    return [asr_response, translation_response, tts_response]
                
                return [asr_response, translation_response]
            
            return []
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return []
    
    def stop_streaming(self):
        """Stop streaming mode"""
        if self.is_streaming:
            self.s2s.stop_streaming()
            self.is_streaming = False

# Global instance
spanish_web_integration = SpanishWebIntegration()

async def process_spanish_audio(audio_data: np.ndarray, client_id: str):
    """Process Spanish audio through web interface"""
    return await spanish_web_integration.process_audio_chunk(audio_data, client_id)
