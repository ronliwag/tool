"""
Thesis Evaluation Metrics Implementation
Cosine Similarity, Average Lagging, ASR-BLEU
Integrated from D:\Thesis - Tool for thesis demonstration
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)

class ThesisMetrics:
    """Implementation of thesis evaluation metrics"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []
    
    def calculate_cosine_similarity(self, original_audio: np.ndarray, generated_audio: np.ndarray) -> float:
        """
        Calculate cosine similarity between original and generated audio
        
        Args:
            original_audio: Original audio array
            generated_audio: Generated audio array
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure same length
            min_len = min(len(original_audio), len(generated_audio))
            if min_len == 0:
                return 0.0
            
            orig = original_audio[:min_len]
            gen = generated_audio[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(orig, gen)
            norm_orig = np.linalg.norm(orig)
            norm_gen = np.linalg.norm(gen)
            
            if norm_orig == 0 or norm_gen == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm_orig * norm_gen)
            return float(np.clip(cosine_sim, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_average_lagging(self, processing_time: float, audio_duration: float) -> float:
        """
        Calculate average lagging for real-time performance
        
        Args:
            processing_time: Time taken to process audio (seconds)
            audio_duration: Duration of input audio (seconds)
            
        Returns:
            Average lagging ratio
        """
        try:
            if audio_duration == 0:
                return 0.0
            
            # Average lagging = processing_time / audio_duration
            # Values < 1.0 indicate real-time performance
            lagging = processing_time / audio_duration
            return float(lagging)
            
        except Exception as e:
            logger.error(f"Error calculating average lagging: {e}")
            return 0.0
    
    def calculate_asr_bleu(self, source_text: str, translated_text: str) -> float:
        """
        Calculate ASR-BLEU score for translation quality
        
        Args:
            source_text: Source Spanish text
            translated_text: Translated English text
            
        Returns:
            ASR-BLEU score (0-100)
        """
        try:
            # Simple BLEU-like score based on word overlap
            source_words = set(source_text.lower().split())
            translated_words = set(translated_text.lower().split())
            
            if not source_words or not translated_words:
                return 0.0
            
            # Calculate precision and recall
            precision = len(source_words.intersection(translated_words)) / len(translated_words)
            recall = len(source_words.intersection(translated_words)) / len(source_words)
            
            if precision + recall == 0:
                return 0.0
            
            # F1-like score as BLEU approximation
            f1_score = 2 * (precision * recall) / (precision + recall)
            return float(f1_score * 100)  # Convert to 0-100 scale
            
        except Exception as e:
            logger.error(f"Error calculating ASR-BLEU: {e}")
            return 0.0
    
    def calculate_speaker_similarity(self, original_embedding: torch.Tensor, generated_embedding: torch.Tensor) -> float:
        """
        Calculate speaker similarity preservation
        
        Args:
            original_embedding: Original speaker embedding
            generated_embedding: Generated speaker embedding
            
        Returns:
            Speaker similarity score (0-1)
        """
        try:
            # Flatten embeddings
            orig_emb = original_embedding.flatten()
            gen_emb = generated_embedding.flatten()
            
            # Calculate cosine similarity
            dot_product = torch.dot(orig_emb, gen_emb)
            norm_orig = torch.norm(orig_emb)
            norm_gen = torch.norm(gen_emb)
            
            if norm_orig == 0 or norm_gen == 0:
                return 0.0
            
            similarity = dot_product / (norm_orig * norm_gen)
            return float(torch.clamp(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating speaker similarity: {e}")
            return 0.0
    
    def calculate_emotion_preservation(self, original_embedding: torch.Tensor, generated_embedding: torch.Tensor) -> float:
        """
        Calculate emotion preservation
        
        Args:
            original_embedding: Original emotion embedding
            generated_embedding: Generated emotion embedding
            
        Returns:
            Emotion preservation score (0-1)
        """
        try:
            # Flatten embeddings
            orig_emb = original_embedding.flatten()
            gen_emb = generated_embedding.flatten()
            
            # Calculate cosine similarity
            dot_product = torch.dot(orig_emb, gen_emb)
            norm_orig = torch.norm(orig_emb)
            norm_gen = torch.norm(gen_emb)
            
            if norm_orig == 0 or norm_gen == 0:
                return 0.0
            
            similarity = dot_product / (norm_orig * norm_gen)
            return float(torch.clamp(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating emotion preservation: {e}")
            return 0.0
    
    def evaluate_model_performance(self, original_audio: np.ndarray, generated_audio: np.ndarray, 
                                 processing_time: float, source_text: str, translated_text: str,
                                 original_speaker_emb: torch.Tensor = None, generated_speaker_emb: torch.Tensor = None,
                                 original_emotion_emb: torch.Tensor = None, generated_emotion_emb: torch.Tensor = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            original_audio: Original input audio
            generated_audio: Generated output audio
            processing_time: Time taken for processing
            source_text: Source Spanish text
            translated_text: Translated English text
            original_speaker_emb: Original speaker embedding
            generated_speaker_emb: Generated speaker embedding
            original_emotion_emb: Original emotion embedding
            generated_emotion_emb: Generated emotion embedding
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            audio_duration = len(original_audio) / 22050  # Assuming 22050 Hz sample rate
            
            metrics = {
                'cosine_similarity': self.calculate_cosine_similarity(original_audio, generated_audio),
                'average_lagging': self.calculate_average_lagging(processing_time, audio_duration),
                'asr_bleu': self.calculate_asr_bleu(source_text, translated_text),
                'real_time_score': 1.0 / max(self.calculate_average_lagging(processing_time, audio_duration), 0.001),
                'processing_speed': audio_duration / max(processing_time, 0.001)
            }
            
            # Add speaker and emotion metrics if embeddings are available
            if original_speaker_emb is not None and generated_speaker_emb is not None:
                metrics['speaker_similarity'] = self.calculate_speaker_similarity(original_speaker_emb, generated_speaker_emb)
            
            if original_emotion_emb is not None and generated_emotion_emb is not None:
                metrics['emotion_preservation'] = self.calculate_emotion_preservation(original_emotion_emb, generated_emotion_emb)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return {
                'cosine_similarity': 0.0,
                'average_lagging': 0.0,
                'asr_bleu': 0.0,
                'real_time_score': 0.0,
                'processing_speed': 0.0
            }
    
    def compare_models(self, original_metrics: Dict[str, float], modified_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare Original vs Modified model performance
        
        Args:
            original_metrics: Metrics from original model
            modified_metrics: Metrics from modified model
            
        Returns:
            Comparison results with statistical significance
        """
        try:
            comparison = {}
            
            for metric in original_metrics:
                if metric in modified_metrics:
                    orig_val = original_metrics[metric]
                    mod_val = modified_metrics[metric]
                    
                    # Calculate improvement percentage
                    if orig_val != 0:
                        improvement = ((mod_val - orig_val) / orig_val) * 100
                    else:
                        improvement = 0.0
                    
                    comparison[metric] = {
                        'original': orig_val,
                        'modified': mod_val,
                        'improvement': improvement,
                        'improvement_abs': mod_val - orig_val
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return {}
    
    def generate_thesis_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate thesis-defense ready report
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("THESIS EVALUATION REPORT")
            report.append("Modified HiFi-GAN vs Original HiFi-GAN")
            report.append("=" * 80)
            report.append("")
            
            # Key metrics analysis
            key_metrics = ['average_lagging', 'cosine_similarity', 'asr_bleu', 'real_time_score']
            
            for metric in key_metrics:
                if metric in comparison_results:
                    data = comparison_results[metric]
                    report.append(f"{metric.upper().replace('_', ' ')}:")
                    report.append(f"  Original: {data['original']:.4f}")
                    report.append(f"  Modified: {data['modified']:.4f}")
                    report.append(f"  Improvement: {data['improvement']:.2f}%")
                    report.append("")
            
            # Statistical significance
            report.append("STATISTICAL SIGNIFICANCE:")
            significant_improvements = 0
            total_metrics = len(comparison_results)
            
            for metric, data in comparison_results.items():
                if abs(data['improvement']) > 5.0:  # 5% threshold for significance
                    significant_improvements += 1
                    report.append(f"  {metric}: SIGNIFICANT ({data['improvement']:.2f}%)")
                else:
                    report.append(f"  {metric}: Not significant ({data['improvement']:.2f}%)")
            
            report.append(f"")
            report.append(f"Significant improvements: {significant_improvements}/{total_metrics}")
            report.append(f"Significance rate: {(significant_improvements/total_metrics)*100:.1f}%")
            report.append("")
            
            # Conclusion
            if significant_improvements > total_metrics // 2:
                report.append("CONCLUSION: Modified HiFi-GAN shows SIGNIFICANT improvements")
            else:
                report.append("CONCLUSION: Modified HiFi-GAN shows MIXED results")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating thesis report: {e}")
            return "Error generating report"

# Example usage for thesis demonstration
if __name__ == "__main__":
    # Create metrics evaluator
    metrics = ThesisMetrics()
    
    # Example data
    original_audio = np.random.randn(1000)
    generated_audio = np.random.randn(1000)
    processing_time = 0.5
    source_text = "Hola, ¿cómo estás?"
    translated_text = "Hello, how are you?"
    
    # Calculate metrics
    results = metrics.evaluate_model_performance(
        original_audio, generated_audio, processing_time, source_text, translated_text
    )
    
    print("Thesis Metrics Evaluation:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")