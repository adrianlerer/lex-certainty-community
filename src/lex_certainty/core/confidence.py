"""
Confidence Estimation for Legal AI

Implements sophisticated confidence estimation methods for legal document
analysis and compliance decision making.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats

logger = logging.getLogger(__name__)


class ConfidenceMethod(Enum):
    """Different confidence estimation methods"""
    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    BAYESIAN_APPROXIMATION = "bayesian_approximation"
    LEGAL_HEURISTIC = "legal_heuristic"


@dataclass
class ConfidenceResult:
    """Result of confidence estimation"""
    confidence_score: float
    calibrated_score: float
    uncertainty_type: str
    estimation_method: ConfidenceMethod
    metadata: Dict[str, Any]


class ConfidenceEstimator:
    """
    Advanced confidence estimation for legal AI decisions.
    
    Implements multiple methods for estimating and calibrating confidence
    scores, with special considerations for legal domain requirements.
    """
    
    def __init__(
        self,
        default_method: ConfidenceMethod = ConfidenceMethod.LEGAL_HEURISTIC,
        calibration_enabled: bool = True
    ):
        self.default_method = default_method
        self.calibration_enabled = calibration_enabled
        
        # Calibration data storage
        self.calibration_history: List[Dict] = []
        self.temperature_param: float = 1.0
        self.platt_params: Dict[str, float] = {'A': 1.0, 'B': 0.0}
        
        logger.info(f"Initialized ConfidenceEstimator with method: {default_method.value}")
    
    def estimate_confidence(
        self,
        predictions: np.ndarray,
        logits: Optional[np.ndarray] = None,
        additional_features: Optional[Dict[str, Any]] = None,
        method: Optional[ConfidenceMethod] = None
    ) -> ConfidenceResult:
        """
        Estimate confidence for predictions using specified method.
        
        Args:
            predictions: Model predictions (probabilities or logits)
            logits: Raw model logits (if available)
            additional_features: Additional features for confidence estimation
            method: Specific confidence estimation method to use
            
        Returns:
            ConfidenceResult with estimated confidence and metadata
        """
        
        method = method or self.default_method
        additional_features = additional_features or {}
        
        # Convert predictions to probabilities if needed
        if np.any(predictions > 1.0) or np.any(predictions < 0.0):
            probs = self._softmax(predictions)
        else:
            probs = predictions.copy()
        
        # Estimate confidence using selected method
        if method == ConfidenceMethod.TEMPERATURE_SCALING:
            result = self._temperature_scaling_confidence(probs, logits, additional_features)
        elif method == ConfidenceMethod.PLATT_SCALING:
            result = self._platt_scaling_confidence(probs, additional_features)
        elif method == ConfidenceMethod.ENSEMBLE_VARIANCE:
            result = self._ensemble_variance_confidence(probs, additional_features)
        elif method == ConfidenceMethod.BAYESIAN_APPROXIMATION:
            result = self._bayesian_approximation_confidence(probs, additional_features)
        elif method == ConfidenceMethod.LEGAL_HEURISTIC:
            result = self._legal_heuristic_confidence(probs, additional_features)
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        # Apply calibration if enabled
        if self.calibration_enabled:
            result.calibrated_score = self._apply_calibration(
                result.confidence_score, method, additional_features
            )
        else:
            result.calibrated_score = result.confidence_score
        
        # Store for future calibration
        self._update_calibration_history(result, additional_features)
        
        return result
    
    def _temperature_scaling_confidence(
        self,
        probs: np.ndarray,
        logits: Optional[np.ndarray],
        features: Dict[str, Any]
    ) -> ConfidenceResult:
        """Estimate confidence using temperature scaling"""
        
        if logits is not None:
            # Apply temperature scaling to logits
            calibrated_logits = logits / self.temperature_param
            calibrated_probs = self._softmax(calibrated_logits)
            
            # Confidence is max probability after temperature scaling
            confidence = np.max(calibrated_probs)
            
            metadata = {
                'temperature': self.temperature_param,
                'original_max_prob': np.max(probs),
                'calibrated_max_prob': confidence,
                'entropy': -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-10))
            }
        else:
            # Fallback when logits not available
            confidence = np.max(probs)
            metadata = {
                'temperature': None,
                'max_prob': confidence,
                'entropy': -np.sum(probs * np.log(probs + 1e-10))
            }
        
        return ConfidenceResult(
            confidence_score=confidence,
            calibrated_score=confidence,
            uncertainty_type="aleatoric",
            estimation_method=ConfidenceMethod.TEMPERATURE_SCALING,
            metadata=metadata
        )
    
    def _platt_scaling_confidence(
        self,
        probs: np.ndarray,
        features: Dict[str, Any]
    ) -> ConfidenceResult:
        """Estimate confidence using Platt scaling"""
        
        # Get max probability and apply Platt scaling
        max_prob = np.max(probs)
        
        # Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
        # Where f is the decision function value
        f = np.log(max_prob / (1 - max_prob + 1e-10))  # Convert prob to logit
        
        A, B = self.platt_params['A'], self.platt_params['B']
        calibrated_prob = 1.0 / (1.0 + np.exp(A * f + B))
        
        confidence = calibrated_prob
        
        metadata = {
            'platt_A': A,
            'platt_B': B,
            'original_max_prob': max_prob,
            'decision_function': f,
            'calibrated_prob': calibrated_prob
        }
        
        return ConfidenceResult(
            confidence_score=confidence,
            calibrated_score=confidence,
            uncertainty_type="epistemic",
            estimation_method=ConfidenceMethod.PLATT_SCALING,
            metadata=metadata
        )
    
    def _ensemble_variance_confidence(
        self,
        probs: np.ndarray,
        features: Dict[str, Any]
    ) -> ConfidenceResult:
        """Estimate confidence using ensemble variance approximation"""
        
        # Simulate ensemble predictions by adding noise
        n_ensemble = features.get('ensemble_size', 10)
        noise_level = features.get('noise_level', 0.1)
        
        ensemble_predictions = []
        for _ in range(n_ensemble):
            noise = np.random.normal(0, noise_level, len(probs))
            noisy_probs = np.clip(probs + noise, 0.01, 0.99)
            noisy_probs = noisy_probs / np.sum(noisy_probs)  # Renormalize
            ensemble_predictions.append(noisy_probs)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate variance across ensemble
        prediction_variance = np.var(ensemble_predictions, axis=0)
        mean_variance = np.mean(prediction_variance)
        
        # Confidence is inversely related to variance
        confidence = 1.0 - np.clip(mean_variance * 10, 0.0, 1.0)
        
        metadata = {
            'ensemble_size': n_ensemble,
            'prediction_variance': prediction_variance.tolist(),
            'mean_variance': mean_variance,
            'max_variance': np.max(prediction_variance),
            'ensemble_mean': np.mean(ensemble_predictions, axis=0).tolist()
        }
        
        return ConfidenceResult(
            confidence_score=confidence,
            calibrated_score=confidence,
            uncertainty_type="epistemic",
            estimation_method=ConfidenceMethod.ENSEMBLE_VARIANCE,
            metadata=metadata
        )
    
    def _bayesian_approximation_confidence(
        self,
        probs: np.ndarray,
        features: Dict[str, Any]
    ) -> ConfidenceResult:
        """Estimate confidence using Bayesian approximation"""
        
        # Use Dirichlet distribution as Bayesian approximation
        # Convert probabilities to Dirichlet parameters
        alpha = probs * features.get('pseudo_count', 10) + 1
        
        # Sample from Dirichlet to estimate uncertainty
        n_samples = features.get('bayesian_samples', 100)
        dirichlet_samples = np.random.dirichlet(alpha, n_samples)
        
        # Calculate prediction uncertainty
        prediction_variance = np.var(dirichlet_samples, axis=0)
        total_variance = np.sum(prediction_variance)
        
        # Estimate epistemic uncertainty
        expected_entropy = np.mean([
            -np.sum(sample * np.log(sample + 1e-10)) 
            for sample in dirichlet_samples
        ])
        
        # Confidence based on predictive uncertainty
        confidence = 1.0 - np.clip(expected_entropy / np.log(len(probs)), 0.0, 1.0)
        
        metadata = {
            'dirichlet_alpha': alpha.tolist(),
            'prediction_variance': prediction_variance.tolist(),
            'total_variance': total_variance,
            'expected_entropy': expected_entropy,
            'aleatoric_uncertainty': expected_entropy,
            'epistemic_uncertainty': total_variance
        }
        
        return ConfidenceResult(
            confidence_score=confidence,
            calibrated_score=confidence,
            uncertainty_type="both",
            estimation_method=ConfidenceMethod.BAYESIAN_APPROXIMATION,
            metadata=metadata
        )
    
    def _legal_heuristic_confidence(
        self,
        probs: np.ndarray,
        features: Dict[str, Any]
    ) -> ConfidenceResult:
        """
        Estimate confidence using legal domain-specific heuristics.
        
        This method considers legal-specific factors that affect confidence
        in compliance decisions.
        """
        
        # Base confidence from prediction strength
        max_prob = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        normalized_entropy = entropy / np.log(len(probs))
        
        base_confidence = max_prob * (1.0 - normalized_entropy)
        
        # Legal domain adjustments
        adjustments = []
        
        # 1. Regulatory clarity adjustment
        regulatory_clarity = features.get('regulatory_clarity', 0.5)  # 0=unclear, 1=clear
        clarity_adjustment = (regulatory_clarity - 0.5) * 0.2
        adjustments.append(('regulatory_clarity', clarity_adjustment))
        
        # 2. Legal precedent availability  
        precedent_strength = features.get('precedent_strength', 0.5)
        precedent_adjustment = (precedent_strength - 0.5) * 0.15
        adjustments.append(('precedent_strength', precedent_adjustment))
        
        # 3. Cultural context clarity
        cultural_clarity = features.get('cultural_clarity', 0.5)
        cultural_adjustment = (cultural_clarity - 0.5) * 0.1
        adjustments.append(('cultural_clarity', cultural_adjustment))
        
        # 4. Scenario complexity penalty
        scenario_complexity = features.get('scenario_complexity', 1.0)
        complexity_penalty = -0.1 * max(0, scenario_complexity - 1.0)
        adjustments.append(('complexity_penalty', complexity_penalty))
        
        # 5. Stakes-based adjustment (higher stakes = more conservative confidence)
        stakes = features.get('stakes', 'medium')
        stakes_map = {'low': 0.05, 'medium': 0.0, 'high': -0.1, 'critical': -0.2}
        stakes_adjustment = stakes_map.get(stakes, 0.0)
        adjustments.append(('stakes_adjustment', stakes_adjustment))
        
        # Apply all adjustments
        total_adjustment = sum(adj[1] for adj in adjustments)
        adjusted_confidence = np.clip(base_confidence + total_adjustment, 0.0, 1.0)
        
        # Conservative bias for legal domain
        legal_conservatism = features.get('legal_conservatism', 0.9)
        final_confidence = adjusted_confidence * legal_conservatism
        
        metadata = {
            'base_confidence': base_confidence,
            'max_prob': max_prob,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'adjustments': dict(adjustments),
            'total_adjustment': total_adjustment,
            'legal_conservatism': legal_conservatism,
            'regulatory_factors': {
                'regulatory_clarity': regulatory_clarity,
                'precedent_strength': precedent_strength,
                'cultural_clarity': cultural_clarity
            },
            'scenario_factors': {
                'complexity': scenario_complexity,
                'stakes': stakes
            }
        }
        
        return ConfidenceResult(
            confidence_score=final_confidence,
            calibrated_score=final_confidence,
            uncertainty_type="legal_contextual",
            estimation_method=ConfidenceMethod.LEGAL_HEURISTIC,
            metadata=metadata
        )
    
    def _apply_calibration(
        self,
        raw_confidence: float,
        method: ConfidenceMethod,
        features: Dict[str, Any]
    ) -> float:
        """Apply calibration to raw confidence score"""
        
        if len(self.calibration_history) < 10:
            # Not enough calibration data
            return raw_confidence
        
        # Simple reliability diagram-based calibration
        try:
            confidences = [h['confidence'] for h in self.calibration_history[-100:]]
            accuracies = [h.get('ground_truth_accuracy', 0.5) for h in self.calibration_history[-100:]]
            
            if len(confidences) >= 10:
                # Bin-based calibration
                n_bins = min(10, len(confidences) // 2)
                
                # Find which bin this confidence falls into
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_idx = np.digitize(raw_confidence, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                
                # Get calibration for this bin
                bin_lower = bin_edges[bin_idx]
                bin_upper = bin_edges[bin_idx + 1]
                
                in_bin_mask = (np.array(confidences) >= bin_lower) & (np.array(confidences) < bin_upper)
                
                if np.sum(in_bin_mask) > 0:
                    bin_accuracy = np.mean(np.array(accuracies)[in_bin_mask])
                    # Linear interpolation between raw confidence and bin accuracy
                    calibrated = 0.7 * raw_confidence + 0.3 * bin_accuracy
                    return np.clip(calibrated, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
        
        return raw_confidence
    
    def _update_calibration_history(
        self,
        result: ConfidenceResult,
        features: Dict[str, Any]
    ):
        """Update calibration history for future use"""
        
        record = {
            'confidence': result.confidence_score,
            'method': result.estimation_method.value,
            'uncertainty_type': result.uncertainty_type,
            'features': features.copy(),
            'timestamp': np.datetime64('now')
        }
        
        self.calibration_history.append(record)
        
        # Keep history manageable
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-1000:]
    
    def update_calibration_with_ground_truth(
        self,
        confidence_scores: List[float],
        ground_truth_accuracies: List[float],
        method: Optional[ConfidenceMethod] = None
    ):
        """
        Update calibration parameters with ground truth feedback.
        
        Args:
            confidence_scores: Historical confidence predictions
            ground_truth_accuracies: Corresponding ground truth accuracies
            method: Specific method to calibrate (None for default)
        """
        
        method = method or self.default_method
        
        if len(confidence_scores) != len(ground_truth_accuracies):
            raise ValueError("Confidence scores and accuracies must have same length")
        
        if len(confidence_scores) < 10:
            logger.warning("Insufficient data for calibration update")
            return
        
        conf_array = np.array(confidence_scores)
        acc_array = np.array(ground_truth_accuracies)
        
        try:
            if method == ConfidenceMethod.TEMPERATURE_SCALING:
                # Optimize temperature parameter
                def temperature_loss(temp):
                    if temp <= 0:
                        return float('inf')
                    # Negative log likelihood
                    probs = 1.0 / (1.0 + np.exp(-conf_array / temp))
                    loss = -np.sum(acc_array * np.log(probs + 1e-10) + 
                                  (1 - acc_array) * np.log(1 - probs + 1e-10))
                    return loss
                
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
                if result.success:
                    self.temperature_param = result.x
                    logger.info(f"Updated temperature parameter: {self.temperature_param:.3f}")
            
            elif method == ConfidenceMethod.PLATT_SCALING:
                # Fit Platt scaling parameters using logistic regression
                # Convert confidences to logits
                logits = np.log(conf_array / (1 - conf_array + 1e-10))
                
                # Fit A and B parameters
                from scipy.optimize import minimize
                
                def platt_loss(params):
                    A, B = params
                    probs = 1.0 / (1.0 + np.exp(A * logits + B))
                    loss = -np.sum(acc_array * np.log(probs + 1e-10) + 
                                  (1 - acc_array) * np.log(1 - probs + 1e-10))
                    return loss
                
                result = minimize(platt_loss, [1.0, 0.0], method='BFGS')
                if result.success:
                    self.platt_params['A'], self.platt_params['B'] = result.x
                    logger.info(f"Updated Platt parameters: A={result.x[0]:.3f}, B={result.x[1]:.3f}")
            
            # Update calibration history with ground truth
            for conf, acc in zip(confidence_scores, ground_truth_accuracies):
                if self.calibration_history:
                    # Find matching record and update
                    for record in reversed(self.calibration_history):
                        if abs(record['confidence'] - conf) < 1e-6:
                            record['ground_truth_accuracy'] = acc
                            break
            
        except Exception as e:
            logger.error(f"Calibration update failed for {method.value}: {e}")
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics and performance metrics"""
        
        if len(self.calibration_history) < 5:
            return {
                'total_records': len(self.calibration_history),
                'calibration_error': None,
                'reliability_data': None
            }
        
        # Extract confidence and accuracy data
        records_with_gt = [r for r in self.calibration_history if 'ground_truth_accuracy' in r]
        
        if len(records_with_gt) < 5:
            return {
                'total_records': len(self.calibration_history),
                'records_with_ground_truth': len(records_with_gt),
                'calibration_error': None
            }
        
        confidences = np.array([r['confidence'] for r in records_with_gt])
        accuracies = np.array([r['ground_truth_accuracy'] for r in records_with_gt])
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        reliability_data = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                bin_size = np.sum(in_bin)
                
                reliability_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_confidence': avg_confidence,
                    'avg_accuracy': avg_accuracy,
                    'bin_size': int(bin_size),
                    'calibration_error': abs(avg_confidence - avg_accuracy)
                })
                
                ece += (bin_size / len(confidences)) * abs(avg_confidence - avg_accuracy)
        
        return {
            'total_records': len(self.calibration_history),
            'records_with_ground_truth': len(records_with_gt),
            'expected_calibration_error': ece,
            'reliability_data': reliability_data,
            'calibration_parameters': {
                'temperature': self.temperature_param,
                'platt_A': self.platt_params['A'],
                'platt_B': self.platt_params['B']
            }
        }
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Numerically stable softmax with temperature"""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / np.sum(exp_x)