"""
Evaluation metrics for speech-to-speech translation
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    cosine_similarity: float
    average_lagging: float
    asr_bleu: float
    latency: float
    wer: float  # Word Error Rate
    bleu_score: float
    met: float  # Mean Evaluation Time
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'cosine_similarity': self.cosine_similarity,
            'average_lagging': self.average_lagging,
            'asr_bleu': self.asr_bleu,
            'latency': self.latency,
            'wer': self.wer,
            'bleu_score': self.bleu_score,
            'met': self.met
        }

class SpeechTranslationEvaluator:
    """Evaluator for speech-to-speech translation quality"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def compute_cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are on the same device
            embedding1 = embedding1.to(self.device)
            embedding2 = embedding2.to(self.device)
            
            # Normalize embeddings
            norm1 = torch.norm(embedding1, dim=-1, keepdim=True)
            norm2 = torch.norm(embedding2, dim=-1, keepdim=True)
            
            # Avoid division by zero
            norm1 = torch.clamp(norm1, min=1e-8)
            norm2 = torch.clamp(norm2, min=1e-8)
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Compute cosine similarity
            similarity = torch.sum(embedding1_norm * embedding2_norm, dim=-1)
            return float(torch.mean(similarity).item())
            
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def compute_average_lagging(self, source_length: int, target_length: int, 
                              translation_times: List[float]) -> float:
        """Compute average lagging for real-time translation"""
        try:
            if not translation_times:
                return 0.0
                
            # Calculate lagging as the delay between source and target processing
            total_lag = sum(translation_times)
            avg_lag = total_lag / len(translation_times)
            
            # Normalize by source length
            normalized_lag = avg_lag / max(source_length, 1)
            return float(normalized_lag)
            
        except Exception as e:
            logger.error(f"Error computing average lagging: {e}")
            return 0.0
    
    def compute_asr_bleu(self, reference_text: str, predicted_text: str) -> float:
        """Compute ASR-BLEU score"""
        try:
            # Simple BLEU-like score for ASR evaluation
            ref_words = reference_text.lower().split()
            pred_words = predicted_text.lower().split()
            
            if not ref_words:
                return 0.0
            
            # Count matches
            matches = 0
            for word in pred_words:
                if word in ref_words:
                    matches += 1
                    
            # Calculate precision
            precision = matches / max(len(pred_words), 1)
            
            # Calculate recall
            recall = matches / max(len(ref_words), 1)
            
            # Calculate F1 score as BLEU approximation
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
    
        except Exception as e:
            logger.error(f"Error computing ASR-BLEU: {e}")
            return 0.0
    
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate"""
        try:
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            
            if not ref_words:
                return 0.0 if not hyp_words else 1.0
                
            # Simple edit distance calculation
            m, n = len(ref_words), len(hyp_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                        
            wer = dp[m][n] / max(len(ref_words), 1)
            return float(wer)
            
        except Exception as e:
            logger.error(f"Error computing WER: {e}")
            return 1.0
    
    def compute_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score for translation quality"""
        try:
            # Simple BLEU-1 score
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            
            if not ref_words or not hyp_words:
                return 0.0
                
            # Count n-gram matches
            ref_ngrams = set(ref_words)
            hyp_ngrams = set(hyp_words)
            
            matches = len(ref_ngrams.intersection(hyp_ngrams))
            precision = matches / max(len(hyp_ngrams), 1)
            
            # Brevity penalty
            bp = min(1.0, len(hyp_words) / max(len(ref_words), 1))
            
            bleu = bp * precision
            return float(bleu)
            
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0
    
    def evaluate_translation(self, 
                           source_embedding: torch.Tensor,
                           target_embedding: torch.Tensor,
                           reference_text: str,
                           predicted_text: str,
                           translation_times: List[float],
                           source_length: int,
                           target_length: int) -> EvaluationResult:
        """Comprehensive evaluation of translation quality"""
        
        try:
            # Compute all metrics
            cosine_sim = self.compute_cosine_similarity(source_embedding, target_embedding)
            avg_lag = self.compute_average_lagging(source_length, target_length, translation_times)
            asr_bleu = self.compute_asr_bleu(reference_text, predicted_text)
            wer = self.compute_wer(reference_text, predicted_text)
            bleu = self.compute_bleu_score(reference_text, predicted_text)
            
            # Calculate latency (average translation time)
            latency = np.mean(translation_times) if translation_times else 0.0
            
            # Mean evaluation time
            met = latency + avg_lag
            
            return EvaluationResult(
                cosine_similarity=cosine_sim,
                average_lagging=avg_lag,
                asr_bleu=asr_bleu,
                latency=latency,
                wer=wer,
                bleu_score=bleu,
                met=met
            )
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return EvaluationResult(
                cosine_similarity=0.0,
                average_lagging=0.0,
                asr_bleu=0.0,
                latency=0.0,
                wer=1.0,
                bleu_score=0.0,
                met=0.0
            )

def create_evaluator(device: str = "cuda") -> SpeechTranslationEvaluator:
    """Factory function to create evaluator"""
    return SpeechTranslationEvaluator(device=device)
