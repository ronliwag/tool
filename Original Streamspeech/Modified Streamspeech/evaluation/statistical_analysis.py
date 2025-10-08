#!/usr/bin/env python3
"""
Statistical Analysis Module for Thesis Evaluation
Implements paired two-tailed t-test (α = 0.05) for model comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    metric_name: str
    baseline_mean: float
    modified_mean: float
    mean_difference: float
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str

class ThesisStatisticalAnalyzer:
    """
    Statistical analyzer for thesis model comparison
    Implements paired two-tailed t-test (α = 0.05) as required
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
        self.results = []
        self.baseline_data = {}
        self.modified_data = {}
        
    def add_baseline_result(self, metric_name: str, value: float, sample_id: str = None):
        """Add baseline model result"""
        if metric_name not in self.baseline_data:
            self.baseline_data[metric_name] = []
        self.baseline_data[metric_name].append(value)
        
    def add_modified_result(self, metric_name: str, value: float, sample_id: str = None):
        """Add modified model result"""
        if metric_name not in self.modified_data:
            self.modified_data[metric_name] = []
        self.modified_data[metric_name].append(value)
    
    def add_comparison_result(self, metric_name: str, baseline_value: float, modified_value: float, sample_id: str = None):
        """Add paired comparison result"""
        self.add_baseline_result(metric_name, baseline_value, sample_id)
        self.add_modified_result(metric_name, modified_value, sample_id)
    
    def run_paired_t_test(self, metric_name: str) -> StatisticalResult:
        """
        Run paired two-tailed t-test for a specific metric
        
        Args:
            metric_name: Name of the metric to test
            
        Returns:
            StatisticalResult object with test results
        """
        try:
            if metric_name not in self.baseline_data or metric_name not in self.modified_data:
                raise ValueError(f"No data available for metric: {metric_name}")
            
            baseline_values = np.array(self.baseline_data[metric_name])
            modified_values = np.array(self.modified_data[metric_name])
            
            # Ensure same length
            min_len = min(len(baseline_values), len(modified_values))
            baseline_values = baseline_values[:min_len]
            modified_values = modified_values[:min_len]
            
            if len(baseline_values) < 2:
                raise ValueError(f"Insufficient data for t-test: {len(baseline_values)} samples")
            
            # Calculate differences
            differences = modified_values - baseline_values
            
            # Run paired t-test
            t_stat, p_value = stats.ttest_rel(modified_values, baseline_values)
            
            # Calculate descriptive statistics
            baseline_mean = np.mean(baseline_values)
            modified_mean = np.mean(modified_values)
            mean_difference = modified_mean - baseline_mean
            
            # Degrees of freedom
            df = len(differences) - 1
            
            # Check significance
            is_significant = p_value < self.alpha
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(modified_values)) / 2)
            effect_size = mean_difference / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate confidence interval for mean difference
            se_diff = np.std(differences) / np.sqrt(len(differences))
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_error = t_critical * se_diff
            ci_lower = mean_difference - margin_error
            ci_upper = mean_difference + margin_error
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                metric_name, baseline_mean, modified_mean, 
                mean_difference, p_value, is_significant, effect_size
            )
            
            result = StatisticalResult(
                metric_name=metric_name,
                baseline_mean=baseline_mean,
                modified_mean=modified_mean,
                mean_difference=mean_difference,
                t_statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=df,
                is_significant=is_significant,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation
            )
            
            self.results.append(result)
            logger.info(f"Paired t-test completed for {metric_name}: p={p_value:.4f}, significant={is_significant}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running paired t-test for {metric_name}: {e}")
            raise e
    
    def run_all_tests(self) -> List[StatisticalResult]:
        """Run paired t-tests for all available metrics"""
        results = []
        
        # Get all available metrics
        available_metrics = set(self.baseline_data.keys()) & set(self.modified_data.keys())
        
        for metric_name in available_metrics:
            try:
                result = self.run_paired_t_test(metric_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run t-test for {metric_name}: {e}")
        
        return results
    
    def _generate_interpretation(self, metric_name: str, baseline_mean: float, modified_mean: float, 
                               mean_difference: float, p_value: float, is_significant: bool, 
                               effect_size: float) -> str:
        """Generate human-readable interpretation of results"""
        
        direction = "improved" if mean_difference > 0 else "degraded"
        significance = "significantly" if is_significant else "not significantly"
        
        # Effect size interpretation
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        interpretation = (
            f"The modified model {significance} {direction} {metric_name} "
            f"(baseline: {baseline_mean:.4f}, modified: {modified_mean:.4f}, "
            f"difference: {mean_difference:.4f}, p={p_value:.4f}). "
            f"The effect size is {effect_interpretation} (Cohen's d={effect_size:.3f})."
        )
        
        return interpretation
    
    def generate_report(self) -> str:
        """Generate comprehensive statistical report"""
        if not self.results:
            return "No statistical tests have been run yet."
        
        report = []
        report.append("=" * 80)
        report.append("THESIS STATISTICAL ANALYSIS REPORT")
        report.append("Paired Two-Tailed t-test (α = 0.05)")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("SUMMARY OF RESULTS:")
        report.append("-" * 80)
        report.append(f"{'Metric':<25} {'Baseline':<10} {'Modified':<10} {'Difference':<12} {'p-value':<10} {'Significant':<12}")
        report.append("-" * 80)
        
        for result in self.results:
            report.append(
                f"{result.metric_name:<25} "
                f"{result.baseline_mean:<10.4f} "
                f"{result.modified_mean:<10.4f} "
                f"{result.mean_difference:<12.4f} "
                f"{result.p_value:<10.4f} "
                f"{'Yes' if result.is_significant else 'No':<12}"
            )
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)
        
        for result in self.results:
            report.append(f"\n{result.metric_name.upper()}:")
            report.append(f"  Baseline Mean: {result.baseline_mean:.4f}")
            report.append(f"  Modified Mean: {result.modified_mean:.4f}")
            report.append(f"  Mean Difference: {result.mean_difference:.4f}")
            report.append(f"  t-statistic: {result.t_statistic:.4f}")
            report.append(f"  p-value: {result.p_value:.4f}")
            report.append(f"  Degrees of Freedom: {result.degrees_of_freedom}")
            report.append(f"  Significant (α=0.05): {'Yes' if result.is_significant else 'No'}")
            report.append(f"  Effect Size (Cohen's d): {result.effect_size:.4f}")
            report.append(f"  95% CI for Difference: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append(f"  Interpretation: {result.interpretation}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "thesis_statistical_results.json"):
        """Save statistical results to JSON file"""
        try:
            results_data = []
            for result in self.results:
                results_data.append({
                    "metric_name": result.metric_name,
                    "baseline_mean": result.baseline_mean,
                    "modified_mean": result.modified_mean,
                    "mean_difference": result.mean_difference,
                    "t_statistic": result.t_statistic,
                    "p_value": result.p_value,
                    "degrees_of_freedom": result.degrees_of_freedom,
                    "is_significant": result.is_significant,
                    "effect_size": result.effect_size,
                    "confidence_interval": list(result.confidence_interval),
                    "interpretation": result.interpretation
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Statistical results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, input_file: str):
        """Load statistical results from JSON file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            self.results = []
            for data in results_data:
                result = StatisticalResult(
                    metric_name=data["metric_name"],
                    baseline_mean=data["baseline_mean"],
                    modified_mean=data["modified_mean"],
                    mean_difference=data["mean_difference"],
                    t_statistic=data["t_statistic"],
                    p_value=data["p_value"],
                    degrees_of_freedom=data["degrees_of_freedom"],
                    is_significant=data["is_significant"],
                    effect_size=data["effect_size"],
                    confidence_interval=tuple(data["confidence_interval"]),
                    interpretation=data["interpretation"]
                )
                self.results.append(result)
            
            logger.info(f"Statistical results loaded from {input_file}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics"""
        if not self.results:
            return {"message": "No results available"}
        
        summary = {
            "total_metrics": len(self.results),
            "significant_metrics": sum(1 for r in self.results if r.is_significant),
            "non_significant_metrics": sum(1 for r in self.results if not r.is_significant),
            "average_effect_size": np.mean([abs(r.effect_size) for r in self.results]),
            "metrics": {}
        }
        
        for result in self.results:
            summary["metrics"][result.metric_name] = {
                "significant": result.is_significant,
                "effect_size": result.effect_size,
                "p_value": result.p_value,
                "mean_difference": result.mean_difference
            }
        
        return summary

def create_statistical_analyzer(alpha: float = 0.05) -> ThesisStatisticalAnalyzer:
    """Factory function to create statistical analyzer"""
    return ThesisStatisticalAnalyzer(alpha=alpha)

if __name__ == "__main__":
    # Test the statistical analyzer
    analyzer = create_statistical_analyzer()
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 30
    
    # Simulate baseline and modified model results
    baseline_cosine = np.random.normal(0.75, 0.05, n_samples)
    modified_cosine = np.random.normal(0.80, 0.05, n_samples)
    
    baseline_lagging = np.random.normal(1.2, 0.1, n_samples)
    modified_lagging = np.random.normal(1.0, 0.1, n_samples)
    
    baseline_bleu = np.random.normal(0.70, 0.03, n_samples)
    modified_bleu = np.random.normal(0.75, 0.03, n_samples)
    
    # Add data to analyzer
    for i in range(n_samples):
        analyzer.add_comparison_result("cosine_similarity", baseline_cosine[i], modified_cosine[i])
        analyzer.add_comparison_result("average_lagging", baseline_lagging[i], modified_lagging[i])
        analyzer.add_comparison_result("asr_bleu", baseline_bleu[i], modified_bleu[i])
    
    # Run statistical tests
    results = analyzer.run_all_tests()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save results
    analyzer.save_results("test_statistical_results.json")
    
    print("Statistical analysis completed successfully!")