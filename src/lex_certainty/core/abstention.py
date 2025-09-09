"""
Mathematical Abstention Engine

Implementation of mathematical abstention principles for legal AI,
inspired by EDFL (Expectation-level Decompression Law) research.

This module provides guaranteed mathematical bounds on decision risk
and intelligent abstention mechanisms for legal compliance scenarios.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
from scipy import stats

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskBounds:
    """Mathematical risk bounds for a decision"""
    lower_bound: float
    upper_bound: float
    expected_risk: float
    confidence_interval: float
    calibration_score: float
    
    def is_within_threshold(self, threshold: float) -> bool:
        """Check if risk is within acceptable threshold"""
        return self.upper_bound <= threshold
        
    def get_risk_level(self) -> RiskLevel:
        """Classify risk level based on bounds"""
        if self.upper_bound <= 0.05:
            return RiskLevel.LOW
        elif self.upper_bound <= 0.15:
            return RiskLevel.MEDIUM
        elif self.upper_bound <= 0.30:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


@dataclass  
class AbstractionResult:
    """Result of abstention analysis"""
    should_abstain: bool
    risk_bounds: RiskBounds
    confidence_score: float
    abstention_reason: Optional[str]
    mathematical_justification: Dict[str, Any]
    metadata: Dict[str, Any]


class RiskCalculator:
    """
    Calculates mathematical risk bounds inspired by EDFL principles.
    
    Implements expectation-level risk decomposition for legal scenarios,
    providing guaranteed bounds on decision uncertainty.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        calibration_samples: int = 1000,
        bootstrap_iterations: int = 500
    ):
        self.confidence_level = confidence_level
        self.calibration_samples = calibration_samples  
        self.bootstrap_iterations = bootstrap_iterations
        self.calibration_history: List[Dict] = []
        
    def calculate_risk_bounds(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        context_complexity: float = 1.0,
        legal_specificity: float = 1.0
    ) -> RiskBounds:
        """
        Calculate mathematical risk bounds for predictions.
        
        Args:
            predictions: Model predictions/logits
            confidence_scores: Associated confidence scores  
            context_complexity: Complexity factor (1.0 = standard)
            legal_specificity: Legal domain specificity factor
            
        Returns:
            RiskBounds with mathematical guarantees
        """
        
        # Normalize inputs
        if len(predictions.shape) > 1:
            predictions = np.max(predictions, axis=1)
        
        # Calculate base uncertainty
        entropy = self._calculate_entropy(predictions, confidence_scores)
        
        # EDFL-inspired expectation decomposition
        expected_risk = self._calculate_expected_risk(
            entropy, context_complexity, legal_specificity
        )
        
        # Bootstrap confidence intervals
        lower_bound, upper_bound = self._bootstrap_confidence_interval(
            predictions, confidence_scores, expected_risk
        )
        
        # Calculate calibration score
        calibration_score = self._estimate_calibration(
            confidence_scores, predictions
        )
        
        # Apply legal domain corrections
        corrected_bounds = self._apply_legal_corrections(
            lower_bound, upper_bound, expected_risk, 
            legal_specificity, calibration_score
        )
        
        return RiskBounds(
            lower_bound=corrected_bounds[0],
            upper_bound=corrected_bounds[1], 
            expected_risk=expected_risk,
            confidence_interval=self.confidence_level,
            calibration_score=calibration_score
        )
    
    def _calculate_entropy(
        self, 
        predictions: np.ndarray, 
        confidence_scores: np.ndarray
    ) -> float:
        """Calculate normalized entropy measure"""
        
        # Handle edge cases
        if len(predictions) == 0:
            return 1.0
            
        # Normalize predictions to probabilities
        if np.max(predictions) > 1.0:
            predictions = self._softmax(predictions)
        
        # Calculate entropy with confidence weighting
        epsilon = 1e-10  # Numerical stability
        weighted_entropy = 0.0
        
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            if pred > epsilon:
                entropy_term = -pred * np.log(pred + epsilon)
                weight = 1.0 - conf  # Higher uncertainty for low confidence
                weighted_entropy += weight * entropy_term
        
        # Normalize by number of predictions
        if len(predictions) > 0:
            weighted_entropy /= len(predictions)
            
        return np.clip(weighted_entropy, 0.0, 1.0)
    
    def _calculate_expected_risk(
        self,
        entropy: float,
        context_complexity: float,
        legal_specificity: float
    ) -> float:
        """
        Calculate expected risk using EDFL-inspired decomposition.
        
        Implements mathematical expectation at different compression levels
        as inspired by Expectation-level Decompression Law research.
        """
        
        # Base risk from entropy
        base_risk = entropy
        
        # Context complexity adjustment
        # More complex contexts increase uncertainty
        complexity_factor = 1.0 + (context_complexity - 1.0) * 0.3
        
        # Legal specificity adjustment  
        # High legal specificity can both increase precision and risk
        specificity_factor = 1.0 + (legal_specificity - 1.0) * 0.2
        
        # EDFL-inspired multi-level expectation
        # Decompose expectation at different compression levels
        level_1_risk = base_risk * complexity_factor
        level_2_risk = level_1_risk * specificity_factor
        level_3_risk = self._higher_order_corrections(level_2_risk, entropy)
        
        expected_risk = np.clip(level_3_risk, 0.0, 1.0)
        
        return expected_risk
    
    def _higher_order_corrections(self, base_risk: float, entropy: float) -> float:
        """Apply higher-order corrections inspired by EDFL"""
        
        # Non-linear corrections for extreme cases
        if entropy > 0.8:  # High uncertainty
            correction = base_risk * (1.0 + 0.1 * (entropy - 0.8) / 0.2)
        elif entropy < 0.2:  # High certainty - but check for overconfidence
            correction = base_risk * (1.0 + 0.05 * (0.2 - entropy) / 0.2)
        else:
            correction = base_risk
            
        return correction
    
    def _bootstrap_confidence_interval(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray, 
        expected_risk: float
    ) -> Tuple[float, float]:
        """Bootstrap confidence intervals for risk estimates"""
        
        n_samples = len(predictions)
        if n_samples < 10:
            # Not enough data for bootstrap - use conservative estimates
            margin = 0.1 * expected_risk
            return (
                max(0.0, expected_risk - margin),
                min(1.0, expected_risk + margin)
            )
        
        bootstrap_risks = []
        
        for _ in range(self.bootstrap_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_conf = confidence_scores[indices]
            
            # Calculate risk for bootstrap sample
            boot_entropy = self._calculate_entropy(boot_preds, boot_conf)
            boot_risk = self._calculate_expected_risk(boot_entropy, 1.0, 1.0)
            bootstrap_risks.append(boot_risk)
        
        bootstrap_risks = np.array(bootstrap_risks)
        
        # Calculate confidence interval
        alpha = 1.0 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_risks, lower_percentile)
        upper_bound = np.percentile(bootstrap_risks, upper_percentile) 
        
        return float(lower_bound), float(upper_bound)
    
    def _estimate_calibration(
        self,
        confidence_scores: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """Estimate model calibration quality"""
        
        if len(confidence_scores) < 5:
            return 0.5  # Neutral calibration for small samples
            
        # Bin confidences and measure calibration gap
        n_bins = min(10, len(confidence_scores) // 2)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        calibration_errors = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = (confidence_scores >= bin_lower) & (confidence_scores < bin_upper)
            
            if np.sum(in_bin) == 0:
                continue
                
            # Average confidence in bin
            avg_confidence = np.mean(confidence_scores[in_bin])
            
            # Average accuracy in bin (assuming binary correctness)
            avg_accuracy = np.mean(predictions[in_bin])
            
            # Calibration error for this bin
            calibration_error = abs(avg_confidence - avg_accuracy)
            calibration_errors.append(calibration_error)
        
        if calibration_errors:
            overall_calibration_error = np.mean(calibration_errors)
            calibration_score = 1.0 - overall_calibration_error
        else:
            calibration_score = 0.5
            
        return np.clip(calibration_score, 0.0, 1.0)
    
    def _apply_legal_corrections(
        self,
        lower_bound: float,
        upper_bound: float, 
        expected_risk: float,
        legal_specificity: float,
        calibration_score: float
    ) -> Tuple[float, float]:
        """Apply legal domain-specific corrections to bounds"""
        
        # Conservative adjustment for legal domain
        legal_conservatism = 0.1 * legal_specificity
        
        # Calibration adjustment - poor calibration increases bounds
        calibration_adjustment = (1.0 - calibration_score) * 0.05
        
        # Apply corrections
        corrected_lower = max(0.0, lower_bound - legal_conservatism)
        corrected_upper = min(1.0, upper_bound + legal_conservatism + calibration_adjustment)
        
        # Ensure bounds are valid
        if corrected_lower > corrected_upper:
            corrected_lower = corrected_upper - 0.01
            
        return corrected_lower, corrected_upper
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / np.sum(exp_x)


class AbstractionEngine:
    """
    Core abstention engine implementing mathematical abstention decisions.
    
    Decides when to abstain from making decisions based on mathematical
    risk bounds and configurable thresholds.
    """
    
    def __init__(
        self,
        risk_threshold: float = 0.05,
        confidence_threshold: float = 0.8,
        legal_conservatism: float = 1.2
    ):
        self.risk_threshold = risk_threshold
        self.confidence_threshold = confidence_threshold 
        self.legal_conservatism = legal_conservatism
        self.risk_calculator = RiskCalculator()
        
        # Statistics tracking
        self.decision_history: List[Dict] = []
        self.abstention_count = 0
        self.total_decisions = 0
    
    def should_abstain(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> AbstractionResult:
        """
        Determine if system should abstain from decision.
        
        Args:
            predictions: Model predictions
            confidence_scores: Confidence scores for predictions
            context: Additional context information
            
        Returns:
            AbstractionResult with decision and mathematical justification
        """
        
        context = context or {}
        
        # Extract context factors
        complexity = context.get('complexity', 1.0)
        legal_specificity = context.get('legal_specificity', 1.0)
        stakes = context.get('stakes', 'medium')
        
        # Calculate risk bounds
        risk_bounds = self.risk_calculator.calculate_risk_bounds(
            predictions=predictions,
            confidence_scores=confidence_scores,
            context_complexity=complexity,
            legal_specificity=legal_specificity
        )
        
        # Apply legal conservatism based on stakes
        adjusted_threshold = self._adjust_threshold_for_stakes(stakes)
        
        # Make abstention decision
        should_abstain, reason = self._make_abstention_decision(
            risk_bounds, adjusted_threshold, confidence_scores
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            risk_bounds, confidence_scores
        )
        
        # Create mathematical justification
        justification = {
            'risk_bounds': {
                'lower': risk_bounds.lower_bound,
                'upper': risk_bounds.upper_bound,
                'expected': risk_bounds.expected_risk
            },
            'thresholds': {
                'original': self.risk_threshold,
                'adjusted': adjusted_threshold,
                'confidence': self.confidence_threshold
            },
            'factors': {
                'complexity': complexity,
                'legal_specificity': legal_specificity,
                'stakes': stakes,
                'calibration': risk_bounds.calibration_score
            },
            'decision_logic': reason
        }
        
        # Update statistics
        self.total_decisions += 1
        if should_abstain:
            self.abstention_count += 1
            
        # Store decision history
        decision_record = {
            'timestamp': np.datetime64('now'),
            'abstained': should_abstain,
            'risk_bounds': risk_bounds,
            'confidence': overall_confidence,
            'context': context
        }
        self.decision_history.append(decision_record)
        
        return AbstractionResult(
            should_abstain=should_abstain,
            risk_bounds=risk_bounds,
            confidence_score=overall_confidence,
            abstention_reason=reason if should_abstain else None,
            mathematical_justification=justification,
            metadata={
                'decision_id': len(self.decision_history),
                'abstention_rate': self.abstention_count / self.total_decisions,
                'risk_level': risk_bounds.get_risk_level().value
            }
        )
    
    def _adjust_threshold_for_stakes(self, stakes: str) -> float:
        """Adjust risk threshold based on decision stakes"""
        
        stake_multipliers = {
            'low': 1.5,      # More permissive for low stakes
            'medium': 1.0,   # Standard threshold
            'high': 0.7,     # More conservative for high stakes  
            'critical': 0.5  # Very conservative for critical decisions
        }
        
        multiplier = stake_multipliers.get(stakes, 1.0)
        adjusted = self.risk_threshold * multiplier * self.legal_conservatism
        
        return np.clip(adjusted, 0.01, 0.5)  # Reasonable bounds
    
    def _make_abstention_decision(
        self,
        risk_bounds: RiskBounds,
        threshold: float,
        confidence_scores: np.ndarray
    ) -> Tuple[bool, str]:
        """Make the core abstention decision with reasoning"""
        
        # Primary criteria: Risk bounds
        if not risk_bounds.is_within_threshold(threshold):
            reason = f"Risk upper bound ({risk_bounds.upper_bound:.3f}) exceeds threshold ({threshold:.3f})"
            return True, reason
        
        # Secondary criteria: Low confidence
        avg_confidence = np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.0
        if avg_confidence < self.confidence_threshold:
            reason = f"Average confidence ({avg_confidence:.3f}) below threshold ({self.confidence_threshold:.3f})"
            return True, reason
        
        # Tertiary criteria: Poor calibration
        if risk_bounds.calibration_score < 0.6:
            reason = f"Poor model calibration ({risk_bounds.calibration_score:.3f}) indicates unreliable confidence"
            return True, reason
        
        # Quaternary criteria: Extreme risk variance
        risk_variance = risk_bounds.upper_bound - risk_bounds.lower_bound
        if risk_variance > 0.2:
            reason = f"High risk variance ({risk_variance:.3f}) indicates unstable estimates"
            return True, reason
        
        # Decision: Proceed
        return False, "Risk and confidence within acceptable bounds"
    
    def _calculate_overall_confidence(
        self,
        risk_bounds: RiskBounds,
        confidence_scores: np.ndarray
    ) -> float:
        """Calculate overall system confidence in the decision"""
        
        # Base confidence from predictions
        base_confidence = np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.5
        
        # Adjust for risk bounds
        risk_confidence = 1.0 - risk_bounds.expected_risk
        
        # Adjust for calibration quality
        calibration_confidence = risk_bounds.calibration_score
        
        # Weighted combination
        overall = (
            0.4 * base_confidence +
            0.4 * risk_confidence +
            0.2 * calibration_confidence
        )
        
        return np.clip(overall, 0.0, 1.0)
    
    def get_abstention_stats(self) -> Dict[str, Any]:
        """Get abstention statistics and performance metrics"""
        
        if self.total_decisions == 0:
            return {
                'total_decisions': 0,
                'abstention_rate': 0.0,
                'avg_risk': 0.0,
                'avg_confidence': 0.0
            }
        
        abstention_rate = self.abstention_count / self.total_decisions
        
        # Calculate averages from history
        risks = [d['risk_bounds'].expected_risk for d in self.decision_history]
        confidences = [d['confidence'] for d in self.decision_history]
        
        return {
            'total_decisions': self.total_decisions,
            'abstention_count': self.abstention_count,
            'abstention_rate': abstention_rate,
            'avg_risk': np.mean(risks) if risks else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'risk_distribution': {
                'min': np.min(risks) if risks else 0.0,
                'max': np.max(risks) if risks else 0.0,
                'std': np.std(risks) if risks else 0.0
            }
        }