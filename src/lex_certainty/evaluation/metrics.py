"""
Legal Metrics and Evaluation

Implements comprehensive evaluation metrics for legal AI systems with
specialized metrics for abstention, compliance, and legal document analysis.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy import stats

from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    CLASSIFICATION = "classification"
    ABSTENTION = "abstention"
    COMPLIANCE = "compliance"
    CONFIDENCE = "confidence"
    LEGAL_QUALITY = "legal_quality"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class MetricResult:
    """Individual metric result."""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.confidence_interval:
            return f"{self.name}: {self.value:.3f} (CI: {self.confidence_interval[0]:.3f}-{self.confidence_interval[1]:.3f})"
        return f"{self.name}: {self.value:.3f}"


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    metrics: List[MetricResult]
    overall_score: float
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metrics": {m.name: m.value for m in self.metrics},
            "confidence_intervals": {m.name: m.confidence_interval for m in self.metrics if m.confidence_interval},
            "overall_score": self.overall_score,
            "metadata": self.evaluation_metadata
        }


class AbstractionMetrics:
    """
    Metrics specialized for abstention evaluation in legal AI systems.
    
    Implements mathematical abstention evaluation based on EDFL principles
    and legal domain-specific considerations.
    """
    
    def __init__(self, config: Optional[LexCertaintyConfig] = None):
        """Initialize abstention metrics."""
        self.config = config or LexCertaintyConfig()
    
    def evaluate_abstention_quality(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        abstention_decisions: List[bool],
        confidence_scores: List[float],
        costs: Optional[Dict[str, float]] = None
    ) -> EvaluationResult:
        """
        Evaluate quality of abstention decisions.
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            abstention_decisions: Boolean list indicating abstention
            confidence_scores: Confidence scores for each prediction
            costs: Cost structure for different outcomes
            
        Returns:
            EvaluationResult with abstention-specific metrics
        """
        costs = costs or {"abstention": 0.3, "error": 1.0, "correct": 0.0}
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        abstentions = np.array(abstention_decisions)
        confidences = np.array(confidence_scores)
        
        # Split into abstained and non-abstained
        abstained_indices = abstentions
        non_abstained_indices = ~abstentions
        
        metrics = []
        
        # Abstention rate
        abstention_rate = np.mean(abstentions)
        metrics.append(MetricResult(
            name="abstention_rate",
            value=abstention_rate,
            metadata={"total_samples": len(predictions), "abstained": np.sum(abstentions)}
        ))
        
        # Accuracy on non-abstained predictions
        if np.any(non_abstained_indices):
            non_abs_predictions = predictions[non_abstained_indices]
            non_abs_truth = ground_truth[non_abstained_indices]
            
            accuracy = accuracy_score(non_abs_truth, non_abs_predictions)
            metrics.append(MetricResult(
                name="non_abstained_accuracy",
                value=accuracy,
                metadata={"samples_evaluated": np.sum(non_abstained_indices)}
            ))
        else:
            accuracy = 0.0
            metrics.append(MetricResult(
                name="non_abstained_accuracy",
                value=0.0,
                metadata={"samples_evaluated": 0}
            ))
        
        # Selective accuracy (accuracy considering abstentions as correct decisions when appropriate)
        selective_accuracy = self._calculate_selective_accuracy(
            predictions, ground_truth, abstentions, confidences
        )
        metrics.append(MetricResult(
            name="selective_accuracy",
            value=selective_accuracy
        ))
        
        # Coverage-accuracy trade-off
        coverage = 1 - abstention_rate  # Coverage = 1 - abstention rate
        metrics.append(MetricResult(
            name="coverage",
            value=coverage
        ))
        
        # Abstention quality metrics
        abstention_quality = self._evaluate_abstention_decisions(
            predictions, ground_truth, abstentions, confidences
        )
        metrics.extend(abstention_quality)
        
        # Cost-benefit analysis
        total_cost = self._calculate_total_cost(
            predictions, ground_truth, abstentions, costs
        )
        metrics.append(MetricResult(
            name="total_cost",
            value=total_cost,
            metadata=costs
        ))
        
        # Overall abstention score (combines accuracy, coverage, and cost)
        overall_score = self._calculate_abstention_score(
            accuracy, coverage, total_cost, costs
        )
        
        return EvaluationResult(
            metrics=metrics,
            overall_score=overall_score,
            evaluation_metadata={
                "evaluation_type": "abstention",
                "total_samples": len(predictions),
                "abstention_strategy": "confidence_based"
            }
        )
    
    def _calculate_selective_accuracy(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        abstentions: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """Calculate selective accuracy considering abstention quality."""
        if len(predictions) == 0:
            return 0.0
        
        # For non-abstained predictions, use normal accuracy
        non_abstained = ~abstentions
        correct_predictions = 0
        
        if np.any(non_abstained):
            correct_predictions = np.sum(
                predictions[non_abstained] == ground_truth[non_abstained]
            )
        
        # For abstained predictions, consider them "correct" if confidence was low
        # and the model would have been wrong, or if uncertainty was genuinely high
        abstained_indices = abstentions
        correct_abstentions = 0
        
        if np.any(abstained_indices):
            # Estimate if abstentions were justified
            abs_confidences = confidences[abstained_indices]
            abs_predictions = predictions[abstained_indices]
            abs_truth = ground_truth[abstained_indices]
            
            # Abstention is considered "correct" if:
            # 1. Confidence was below threshold, OR
            # 2. Model would have been wrong and confidence was not high
            confidence_threshold = 0.7
            
            for i, (conf, pred, truth) in enumerate(zip(abs_confidences, abs_predictions, abs_truth)):
                if conf < confidence_threshold:
                    correct_abstentions += 1  # Low confidence abstention
                elif pred != truth and conf < 0.9:
                    correct_abstentions += 1  # Would have been wrong, reasonable abstention
        
        total_correct = correct_predictions + correct_abstentions
        return total_correct / len(predictions)
    
    def _evaluate_abstention_decisions(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        abstentions: np.ndarray,
        confidences: np.ndarray
    ) -> List[MetricResult]:
        """Evaluate quality of individual abstention decisions."""
        metrics = []
        
        # Correlation between confidence and correctness
        non_abstained = ~abstentions
        if np.any(non_abstained):
            correctness = (predictions[non_abstained] == ground_truth[non_abstained]).astype(float)
            conf_corr = np.corrcoef(confidences[non_abstained], correctness)[0, 1]
            if not np.isnan(conf_corr):
                metrics.append(MetricResult(
                    name="confidence_correctness_correlation",
                    value=conf_corr,
                    metadata={"interpretation": "Higher is better (confidence aligns with correctness)"}
                ))
        
        # Abstention precision: fraction of abstentions that were justified
        abstained_indices = abstentions
        if np.any(abstained_indices):
            # Check if abstained samples would have been errors
            abs_predictions = predictions[abstained_indices]
            abs_truth = ground_truth[abstained_indices]
            abs_confidences = confidences[abstained_indices]
            
            # Justified abstentions: low confidence OR would have been wrong
            justified = (abs_confidences < 0.7) | (abs_predictions != abs_truth)
            abstention_precision = np.mean(justified)
            
            metrics.append(MetricResult(
                name="abstention_precision",
                value=abstention_precision,
                metadata={"justified_abstentions": np.sum(justified), "total_abstentions": len(justified)}
            ))
        
        # Missed abstentions: incorrect predictions with low confidence
        if np.any(non_abstained):
            non_abs_correct = predictions[non_abstained] == ground_truth[non_abstained]
            non_abs_conf = confidences[non_abstained]
            
            # Should have abstained: incorrect and low confidence
            should_abstain = (~non_abs_correct) & (non_abs_conf < 0.6)
            missed_abstentions = np.mean(should_abstain)
            
            metrics.append(MetricResult(
                name="missed_abstention_rate",
                value=missed_abstentions,
                metadata={"should_have_abstained": np.sum(should_abstain)}
            ))
        
        return metrics
    
    def _calculate_total_cost(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        abstentions: np.ndarray,
        costs: Dict[str, float]
    ) -> float:
        """Calculate total cost considering abstentions, errors, and correct predictions."""
        total_cost = 0.0
        
        # Cost of abstentions
        total_cost += np.sum(abstentions) * costs["abstention"]
        
        # Cost of predictions (correct and incorrect)
        non_abstained = ~abstentions
        if np.any(non_abstained):
            correct = predictions[non_abstained] == ground_truth[non_abstained]
            total_cost += np.sum(correct) * costs["correct"]
            total_cost += np.sum(~correct) * costs["error"]
        
        # Normalize by total number of samples
        return total_cost / len(predictions)
    
    def _calculate_abstention_score(
        self,
        accuracy: float,
        coverage: float,
        total_cost: float,
        costs: Dict[str, float]
    ) -> float:
        """Calculate overall abstention quality score."""
        # Weighted combination of metrics
        # Higher accuracy and coverage, lower cost = better score
        
        accuracy_weight = 0.4
        coverage_weight = 0.3
        cost_weight = 0.3
        
        # Normalize cost (lower is better)
        max_possible_cost = costs["error"]  # If all predictions were wrong
        normalized_cost = 1.0 - (total_cost / max_possible_cost)
        normalized_cost = max(0.0, min(1.0, normalized_cost))
        
        score = (
            accuracy_weight * accuracy +
            coverage_weight * coverage +
            cost_weight * normalized_cost
        )
        
        return score


class LegalMetrics:
    """
    Comprehensive metrics for legal AI evaluation including domain-specific
    legal quality metrics, compliance assessment, and risk evaluation.
    """
    
    def __init__(self, config: Optional[LexCertaintyConfig] = None):
        """Initialize legal metrics."""
        self.config = config or LexCertaintyConfig()
        self.abstention_metrics = AbstractionMetrics(config)
    
    def evaluate_legal_analysis(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        evaluation_criteria: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate legal analysis quality across multiple dimensions.
        
        Args:
            predictions: Model predictions with legal analysis
            ground_truth: Ground truth legal analysis
            evaluation_criteria: Specific criteria for evaluation
            
        Returns:
            EvaluationResult with legal-specific metrics
        """
        criteria = evaluation_criteria or self._get_default_legal_criteria()
        metrics = []
        
        # Legal accuracy metrics
        legal_accuracy = self._evaluate_legal_accuracy(predictions, ground_truth)
        metrics.extend(legal_accuracy)
        
        # Compliance detection metrics
        compliance_metrics = self._evaluate_compliance_detection(predictions, ground_truth)
        metrics.extend(compliance_metrics)
        
        # Risk assessment metrics
        risk_metrics = self._evaluate_risk_assessment(predictions, ground_truth)
        metrics.extend(risk_metrics)
        
        # Legal reasoning quality
        reasoning_metrics = self._evaluate_legal_reasoning(predictions, ground_truth)
        metrics.extend(reasoning_metrics)
        
        # Citation and reference accuracy
        citation_metrics = self._evaluate_citations(predictions, ground_truth)
        metrics.extend(citation_metrics)
        
        # Calculate overall legal quality score
        overall_score = self._calculate_legal_quality_score(metrics)
        
        return EvaluationResult(
            metrics=metrics,
            overall_score=overall_score,
            evaluation_metadata={
                "evaluation_type": "legal_analysis",
                "criteria": criteria,
                "total_cases": len(predictions)
            }
        )
    
    def evaluate_compliance_assessment(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate compliance assessment accuracy."""
        metrics = []
        
        # Extract compliance scores and statuses
        pred_scores = [p.get("compliance_score", 0.0) for p in predictions]
        true_scores = [gt.get("compliance_score", 0.0) for gt in ground_truth]
        
        pred_status = [p.get("compliance_status", "unknown") for p in predictions]
        true_status = [gt.get("compliance_status", "unknown") for gt in ground_truth]
        
        # Score correlation
        if len(pred_scores) > 1 and len(true_scores) > 1:
            score_correlation = np.corrcoef(pred_scores, true_scores)[0, 1]
            if not np.isnan(score_correlation):
                metrics.append(MetricResult(
                    name="compliance_score_correlation",
                    value=score_correlation
                ))
        
        # Status classification accuracy
        if len(set(true_status)) > 1:  # Multiple classes present
            status_accuracy = accuracy_score(true_status, pred_status)
            metrics.append(MetricResult(
                name="compliance_status_accuracy", 
                value=status_accuracy
            ))
            
            # Status-specific F1 scores
            try:
                f1_macro = f1_score(true_status, pred_status, average='macro')
                metrics.append(MetricResult(
                    name="compliance_status_f1_macro",
                    value=f1_macro
                ))
            except Exception as e:
                logger.warning(f"Could not calculate F1 score: {e}")
        
        # Issue detection metrics
        issue_metrics = self._evaluate_issue_detection(predictions, ground_truth)
        metrics.extend(issue_metrics)
        
        overall_score = np.mean([m.value for m in metrics]) if metrics else 0.0
        
        return EvaluationResult(
            metrics=metrics,
            overall_score=overall_score,
            evaluation_metadata={
                "evaluation_type": "compliance_assessment",
                "total_assessments": len(predictions)
            }
        )
    
    def _get_default_legal_criteria(self) -> Dict[str, Any]:
        """Get default legal evaluation criteria."""
        return {
            "accuracy_weight": 0.3,
            "completeness_weight": 0.2,
            "legal_reasoning_weight": 0.2,
            "compliance_detection_weight": 0.15,
            "citation_accuracy_weight": 0.15
        }
    
    def _evaluate_legal_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate accuracy of legal conclusions and findings."""
        metrics = []
        
        # Legal conclusion accuracy
        pred_conclusions = [p.get("legal_conclusion", "") for p in predictions]
        true_conclusions = [gt.get("legal_conclusion", "") for gt in ground_truth]
        
        # Semantic similarity of conclusions (simplified)
        conclusion_matches = [
            self._semantic_similarity(pred, true) > 0.8
            for pred, true in zip(pred_conclusions, true_conclusions)
        ]
        
        conclusion_accuracy = np.mean(conclusion_matches) if conclusion_matches else 0.0
        metrics.append(MetricResult(
            name="legal_conclusion_accuracy",
            value=conclusion_accuracy,
            metadata={"matches": sum(conclusion_matches), "total": len(conclusion_matches)}
        ))
        
        # Legal finding accuracy (key findings identification)
        finding_accuracy = self._evaluate_legal_findings(predictions, ground_truth)
        metrics.append(MetricResult(
            name="legal_findings_accuracy",
            value=finding_accuracy
        ))
        
        return metrics
    
    def _evaluate_compliance_detection(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate compliance issue detection."""
        metrics = []
        
        # Extract compliance flags
        pred_flags = [set(p.get("compliance_flags", [])) for p in predictions]
        true_flags = [set(gt.get("compliance_flags", [])) for gt in ground_truth]
        
        # Flag detection metrics
        if pred_flags and true_flags:
            precisions = []
            recalls = []
            f1s = []
            
            for pred_set, true_set in zip(pred_flags, true_flags):
                if len(true_set) == 0 and len(pred_set) == 0:
                    precisions.append(1.0)
                    recalls.append(1.0)
                    f1s.append(1.0)
                elif len(pred_set) == 0:
                    precisions.append(1.0)  # No false positives
                    recalls.append(0.0)
                    f1s.append(0.0)
                elif len(true_set) == 0:
                    precisions.append(0.0)
                    recalls.append(1.0)  # No missed flags (vacuously true)
                    f1s.append(0.0)
                else:
                    intersection = len(pred_set & true_set)
                    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
                    recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
            
            metrics.extend([
                MetricResult(name="compliance_flag_precision", value=np.mean(precisions)),
                MetricResult(name="compliance_flag_recall", value=np.mean(recalls)),
                MetricResult(name="compliance_flag_f1", value=np.mean(f1s))
            ])
        
        return metrics
    
    def _evaluate_risk_assessment(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate risk assessment accuracy."""
        metrics = []
        
        # Risk level classification
        pred_risk = [p.get("risk_level", "unknown") for p in predictions]
        true_risk = [gt.get("risk_level", "unknown") for gt in ground_truth]
        
        if len(set(true_risk)) > 1:
            risk_accuracy = accuracy_score(true_risk, pred_risk)
            metrics.append(MetricResult(
                name="risk_level_accuracy",
                value=risk_accuracy
            ))
        
        # Risk score correlation
        pred_risk_scores = [p.get("risk_score", 0.0) for p in predictions]
        true_risk_scores = [gt.get("risk_score", 0.0) for gt in ground_truth]
        
        if len(pred_risk_scores) > 1 and np.std(true_risk_scores) > 0:
            risk_correlation = np.corrcoef(pred_risk_scores, true_risk_scores)[0, 1]
            if not np.isnan(risk_correlation):
                metrics.append(MetricResult(
                    name="risk_score_correlation",
                    value=risk_correlation
                ))
        
        return metrics
    
    def _evaluate_legal_reasoning(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate quality of legal reasoning."""
        metrics = []
        
        # Reasoning completeness (presence of key reasoning elements)
        reasoning_elements = ["legal_basis", "analysis", "conclusion"]
        
        completeness_scores = []
        for pred in predictions:
            reasoning = pred.get("legal_reasoning", {})
            present_elements = sum(1 for elem in reasoning_elements if reasoning.get(elem))
            completeness = present_elements / len(reasoning_elements)
            completeness_scores.append(completeness)
        
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
        metrics.append(MetricResult(
            name="reasoning_completeness",
            value=avg_completeness
        ))
        
        # Logical consistency (simplified check)
        consistency_scores = []
        for pred in predictions:
            # Check for logical consistency indicators
            reasoning_text = str(pred.get("legal_reasoning", "")).lower()
            
            # Look for contradictions or inconsistencies (simplified heuristic)
            contradiction_indicators = ["however", "but", "contradicts", "inconsistent"]
            consistency_indicators = ["therefore", "consequently", "follows", "consistent"]
            
            contradictions = sum(1 for ind in contradiction_indicators if ind in reasoning_text)
            consistencies = sum(1 for ind in consistency_indicators if ind in reasoning_text)
            
            # Simple consistency score
            if contradictions + consistencies > 0:
                consistency = consistencies / (contradictions + consistencies)
            else:
                consistency = 0.5  # Neutral if no indicators
            
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        metrics.append(MetricResult(
            name="reasoning_consistency",
            value=avg_consistency
        ))
        
        return metrics
    
    def _evaluate_citations(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate citation and legal reference accuracy."""
        metrics = []
        
        # Citation accuracy
        pred_citations = [set(p.get("legal_citations", [])) for p in predictions]
        true_citations = [set(gt.get("legal_citations", [])) for gt in ground_truth]
        
        if pred_citations and true_citations:
            citation_precisions = []
            citation_recalls = []
            
            for pred_cites, true_cites in zip(pred_citations, true_citations):
                if len(pred_cites) == 0 and len(true_cites) == 0:
                    citation_precisions.append(1.0)
                    citation_recalls.append(1.0)
                elif len(pred_cites) == 0:
                    citation_precisions.append(1.0)
                    citation_recalls.append(0.0)
                elif len(true_cites) == 0:
                    citation_precisions.append(0.0)
                    citation_recalls.append(1.0)
                else:
                    intersection = len(pred_cites & true_cites)
                    precision = intersection / len(pred_cites)
                    recall = intersection / len(true_cites)
                    citation_precisions.append(precision)
                    citation_recalls.append(recall)
            
            metrics.extend([
                MetricResult(name="citation_precision", value=np.mean(citation_precisions)),
                MetricResult(name="citation_recall", value=np.mean(citation_recalls))
            ])
        
        return metrics
    
    def _evaluate_legal_findings(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """Evaluate accuracy of key legal findings."""
        if not predictions or not ground_truth:
            return 0.0
        
        finding_matches = 0
        total_findings = 0
        
        for pred, truth in zip(predictions, ground_truth):
            pred_findings = set(pred.get("key_findings", []))
            true_findings = set(truth.get("key_findings", []))
            
            if len(true_findings) > 0:
                intersection = len(pred_findings & true_findings)
                total_possible = len(true_findings)
                finding_matches += intersection
                total_findings += total_possible
        
        return finding_matches / total_findings if total_findings > 0 else 0.0
    
    def _evaluate_issue_detection(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """Evaluate compliance issue detection accuracy."""
        metrics = []
        
        pred_issues = [p.get("issues", []) for p in predictions]
        true_issues = [gt.get("issues", []) for gt in ground_truth]
        
        if pred_issues and true_issues:
            # Issue type detection accuracy
            pred_issue_types = [set(issue.get("type", "") for issue in issues) 
                              for issues in pred_issues]
            true_issue_types = [set(issue.get("type", "") for issue in issues) 
                              for issues in true_issues]
            
            type_precisions = []
            type_recalls = []
            
            for pred_types, true_types in zip(pred_issue_types, true_issue_types):
                if len(pred_types) == 0 and len(true_types) == 0:
                    type_precisions.append(1.0)
                    type_recalls.append(1.0)
                elif len(pred_types) == 0:
                    type_precisions.append(1.0)
                    type_recalls.append(0.0)
                elif len(true_types) == 0:
                    type_precisions.append(0.0)
                    type_recalls.append(1.0)
                else:
                    intersection = len(pred_types & true_types)
                    precision = intersection / len(pred_types)
                    recall = intersection / len(true_types)
                    type_precisions.append(precision)
                    type_recalls.append(recall)
            
            metrics.extend([
                MetricResult(name="issue_type_precision", value=np.mean(type_precisions)),
                MetricResult(name="issue_type_recall", value=np.mean(type_recalls))
            ])
        
        return metrics
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simplified)."""
        # Simplified semantic similarity using word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_legal_quality_score(self, metrics: List[MetricResult]) -> float:
        """Calculate overall legal quality score from individual metrics."""
        if not metrics:
            return 0.0
        
        # Weight different types of metrics
        weights = {
            "legal_conclusion_accuracy": 0.25,
            "legal_findings_accuracy": 0.20,
            "compliance_flag_f1": 0.20,
            "risk_level_accuracy": 0.15,
            "reasoning_completeness": 0.10,
            "citation_precision": 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.name, 0.05)  # Default small weight
            weighted_score += metric.value * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0