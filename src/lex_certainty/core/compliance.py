"""
Certified Compliance Engine

Main interface for legal compliance analysis using mathematical abstention.
Combines risk-aware decision making with legal domain expertise.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

from .abstention import AbstractionEngine, AbstractionResult
from .confidence import ConfidenceEstimator
from ..legal.context import LegalContext, ArgentineLegalContext
from ..legal.prompter import LegalPromptEngine
from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class ComplianceDecision(Enum):
    """Possible compliance decisions"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"  
    UNCLEAR = "unclear"
    ABSTAIN = "abstain"


class ComplexityLevel(Enum):
    """Scenario complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ComplianceScenario:
    """
    Legal compliance scenario for analysis
    """
    id: str
    title: str
    description: str
    regulatory_context: str
    complexity: Union[str, ComplexityLevel] = ComplexityLevel.MEDIUM
    cultural_specificity: Union[str, ComplexityLevel] = ComplexityLevel.MEDIUM
    stakes: str = "medium"  # low, medium, high, critical
    expected_outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Convert string values to enums if needed"""
        if isinstance(self.complexity, str):
            self.complexity = ComplexityLevel(self.complexity)
        if isinstance(self.cultural_specificity, str):
            self.cultural_specificity = ComplexityLevel(self.cultural_specificity)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComplianceResult:
    """
    Result of compliance analysis with mathematical guarantees
    """
    scenario_id: str
    timestamp: datetime
    
    # Primary decision
    decision: ComplianceDecision
    confidence: float
    reasoning: str
    
    # Mathematical guarantees
    abstention_result: Optional[AbstractionResult] = None
    risk_bounds: Optional[Dict[str, float]] = None
    
    # Legal analysis
    legal_factors: Optional[List[str]] = None
    regulatory_references: Optional[List[str]] = None
    cultural_considerations: Optional[List[str]] = None
    
    # Audit trail
    analysis_metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        result_dict = asdict(self)
        result_dict['decision'] = self.decision.value
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict
    
    @property
    def abstained(self) -> bool:
        """Check if system abstained from decision"""
        return self.decision == ComplianceDecision.ABSTAIN


class CertifiedComplianceEngine:
    """
    Main compliance engine with mathematical abstention guarantees.
    
    Provides certified compliance analysis for legal scenarios with
    quantified risk bounds and intelligent abstention capabilities.
    """
    
    def __init__(
        self,
        config: Optional[LexCertaintyConfig] = None,
        legal_context: Optional[LegalContext] = None,
        risk_threshold: float = 0.05,
        confidence_threshold: float = 0.8
    ):
        self.config = config or LexCertaintyConfig()
        self.legal_context = legal_context or ArgentineLegalContext()
        
        # Initialize core components
        self.abstention_engine = AbstractionEngine(
            risk_threshold=risk_threshold,
            confidence_threshold=confidence_threshold,
            legal_conservatism=self.legal_context.get_conservatism_factor()
        )
        
        self.confidence_estimator = ConfidenceEstimator()
        self.prompt_engine = LegalPromptEngine(legal_context=self.legal_context)
        
        # Analysis statistics
        self.analysis_count = 0
        self.abstention_count = 0
        self.analysis_history: List[ComplianceResult] = []
        self.start_time = datetime.now()
        
        logger.info(f"Initialized CertifiedComplianceEngine with {self.legal_context.__class__.__name__}")
    
    async def analyze_scenario(
        self,
        scenario: ComplianceScenario,
        use_abstention: bool = True
    ) -> ComplianceResult:
        """
        Analyze compliance scenario with mathematical guarantees.
        
        Args:
            scenario: Compliance scenario to analyze
            use_abstention: Whether to use mathematical abstention (recommended)
            
        Returns:
            ComplianceResult with decision and mathematical justification
        """
        
        start_time = datetime.now()
        logger.info(f"Analyzing scenario: {scenario.id} - {scenario.title}")
        
        try:
            # Step 1: Generate structured legal prompt
            prompt_result = await self.prompt_engine.generate_compliance_prompt(scenario)
            
            # Step 2: Get model predictions (simulated for now - would call actual LLM)
            predictions, confidence_scores = await self._get_model_predictions(
                prompt_result.prompt, scenario
            )
            
            # Step 3: Apply mathematical abstention if enabled
            abstention_result = None
            if use_abstention:
                abstention_result = self.abstention_engine.should_abstain(
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    context={
                        'complexity': self._complexity_to_float(scenario.complexity),
                        'legal_specificity': self._complexity_to_float(scenario.cultural_specificity),
                        'stakes': scenario.stakes,
                        'regulatory_context': scenario.regulatory_context
                    }
                )
            
            # Step 4: Make final decision
            final_decision, reasoning = self._make_final_decision(
                predictions, confidence_scores, abstention_result, scenario
            )
            
            # Step 5: Extract legal factors and analysis
            legal_analysis = self._extract_legal_analysis(scenario, prompt_result)
            
            # Step 6: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                confidence_scores, abstention_result
            )
            
            # Step 7: Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ComplianceResult(
                scenario_id=scenario.id,
                timestamp=start_time,
                decision=final_decision,
                confidence=overall_confidence,
                reasoning=reasoning,
                abstention_result=abstention_result,
                risk_bounds=self._extract_risk_bounds(abstention_result),
                legal_factors=legal_analysis.get('factors', []),
                regulatory_references=legal_analysis.get('references', []),
                cultural_considerations=legal_analysis.get('cultural', []),
                analysis_metadata={
                    'prompt_tokens': prompt_result.metadata.get('token_count', 0),
                    'model_version': self.config.model_version,
                    'legal_context': self.legal_context.__class__.__name__,
                    'abstention_enabled': use_abstention,
                    'complexity_factors': {
                        'scenario': scenario.complexity.value,
                        'cultural': scenario.cultural_specificity.value,
                        'stakes': scenario.stakes
                    }
                },
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_statistics(result)
            
            logger.info(f"Analysis complete for {scenario.id}: {final_decision.value} (confidence: {overall_confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for scenario {scenario.id}: {str(e)}")
            
            # Return error result with abstention
            return ComplianceResult(
                scenario_id=scenario.id,
                timestamp=start_time,
                decision=ComplianceDecision.ABSTAIN,
                confidence=0.0,
                reasoning=f"Analysis failed due to error: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _get_model_predictions(
        self,
        prompt: str,
        scenario: ComplianceScenario
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for the scenario.
        
        Note: This is a simplified implementation. In production, this would
        call actual LLM APIs (OpenAI, etc.) and parse responses.
        """
        
        # Simulate model response based on scenario characteristics
        # In production: would call LLM API and parse structured response
        
        # Simulate different prediction patterns based on complexity
        complexity_factor = self._complexity_to_float(scenario.complexity)
        cultural_factor = self._complexity_to_float(scenario.cultural_specificity)
        
        # Base predictions for compliance decision
        # [compliant_prob, non_compliant_prob, unclear_prob]
        if "gift" in scenario.description.lower() and "licitación" in scenario.description.lower():
            # High-risk scenario - likely non-compliant
            base_probs = np.array([0.15, 0.75, 0.10])
        elif "consulta" in scenario.description.lower():
            # Medium-risk - likely compliant with consultation
            base_probs = np.array([0.60, 0.25, 0.15])
        else:
            # Default uncertain distribution
            base_probs = np.array([0.40, 0.30, 0.30])
        
        # Add noise based on complexity
        noise_level = 0.1 * complexity_factor
        noise = np.random.normal(0, noise_level, 3)
        predictions = np.clip(base_probs + noise, 0.01, 0.99)
        
        # Normalize to sum to 1
        predictions = predictions / np.sum(predictions)
        
        # Generate confidence scores
        # Lower confidence for higher complexity
        base_confidence = 0.9 - 0.3 * complexity_factor - 0.2 * cultural_factor
        confidence_noise = np.random.normal(0, 0.1)
        confidence = np.clip(base_confidence + confidence_noise, 0.1, 0.95)
        
        # Create confidence array for each prediction
        confidence_scores = np.array([confidence, confidence * 0.9, confidence * 0.8])
        
        logger.debug(f"Model predictions: {predictions}, confidences: {confidence_scores}")
        
        return predictions, confidence_scores
    
    def _make_final_decision(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        abstention_result: Optional[AbstractionResult],
        scenario: ComplianceScenario
    ) -> tuple[ComplianceDecision, str]:
        """Make final compliance decision based on all inputs"""
        
        # Check abstention first
        if abstention_result and abstention_result.should_abstain:
            return ComplianceDecision.ABSTAIN, abstention_result.abstention_reason
        
        # Get highest confidence prediction
        max_idx = np.argmax(predictions)
        max_confidence = confidence_scores[max_idx] if len(confidence_scores) > max_idx else 0.5
        
        # Map prediction index to decision
        decision_map = [
            ComplianceDecision.COMPLIANT,
            ComplianceDecision.NON_COMPLIANT, 
            ComplianceDecision.UNCLEAR
        ]
        
        predicted_decision = decision_map[max_idx]
        
        # Check if confidence is sufficient for the prediction
        min_confidence_thresholds = {
            ComplianceDecision.COMPLIANT: 0.7,
            ComplianceDecision.NON_COMPLIANT: 0.8,  # Higher bar for risky decision
            ComplianceDecision.UNCLEAR: 0.6
        }
        
        required_confidence = min_confidence_thresholds[predicted_decision]
        
        if max_confidence < required_confidence:
            return ComplianceDecision.UNCLEAR, (
                f"Insufficient confidence ({max_confidence:.1%}) for {predicted_decision.value} "
                f"decision (requires {required_confidence:.1%})"
            )
        
        # Create reasoning
        reasoning = (
            f"Analysis indicates {predicted_decision.value} with {max_confidence:.1%} confidence. "
            f"Prediction probabilities: compliant={predictions[0]:.1%}, "
            f"non-compliant={predictions[1]:.1%}, unclear={predictions[2]:.1%}"
        )
        
        return predicted_decision, reasoning
    
    def _extract_legal_analysis(
        self,
        scenario: ComplianceScenario,
        prompt_result: Any
    ) -> Dict[str, List[str]]:
        """Extract legal factors, references, and cultural considerations"""
        
        # In production: would parse LLM response for specific legal elements
        # For now: simulate based on scenario content
        
        factors = []
        references = []
        cultural = []
        
        # Analyze scenario content for key factors
        description_lower = scenario.description.lower()
        
        # Legal factors
        if "regalo" in description_lower or "gift" in description_lower:
            factors.append("Entrega de obsequios a funcionarios públicos")
        if "licitación" in description_lower or "bidding" in description_lower:
            factors.append("Proceso de contratación pública")
        if "funcionario" in description_lower or "official" in description_lower:
            factors.append("Interacción con funcionarios públicos")
        
        # Regulatory references
        if "ley 27401" in scenario.regulatory_context.lower():
            references.append("Ley 27401 - Responsabilidad Penal Empresaria")
            references.append("Artículo 1 - Hechos punibles de personas humanas")
        
        # Cultural considerations
        if scenario.cultural_specificity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]:
            cultural.append("Contexto cultural argentino de relaciones comerciales")
            cultural.append("Expectativas de cortesía empresarial vs cumplimiento legal")
        
        return {
            'factors': factors,
            'references': references, 
            'cultural': cultural
        }
    
    def _extract_risk_bounds(
        self,
        abstention_result: Optional[AbstractionResult]
    ) -> Optional[Dict[str, float]]:
        """Extract risk bounds from abstention result"""
        
        if not abstention_result or not abstention_result.risk_bounds:
            return None
            
        bounds = abstention_result.risk_bounds
        return {
            'lower_bound': bounds.lower_bound,
            'upper_bound': bounds.upper_bound,
            'expected_risk': bounds.expected_risk,
            'calibration_score': bounds.calibration_score
        }
    
    def _calculate_overall_confidence(
        self,
        confidence_scores: np.ndarray,
        abstention_result: Optional[AbstractionResult]
    ) -> float:
        """Calculate overall confidence in the analysis"""
        
        # Base confidence from model
        base_confidence = np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.5
        
        # Adjust based on abstention analysis
        if abstention_result:
            # If abstained, confidence is in the abstention decision
            if abstention_result.should_abstain:
                return abstention_result.confidence_score
            else:
                # If not abstained, boost confidence slightly
                base_confidence = min(0.95, base_confidence * 1.1)
        
        return base_confidence
    
    def _complexity_to_float(self, complexity: ComplexityLevel) -> float:
        """Convert complexity enum to numerical factor"""
        complexity_map = {
            ComplexityLevel.LOW: 0.7,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 1.3
        }
        return complexity_map.get(complexity, 1.0)
    
    def _update_statistics(self, result: ComplianceResult):
        """Update internal statistics tracking"""
        
        self.analysis_count += 1
        if result.abstained:
            self.abstention_count += 1
            
        self.analysis_history.append(result)
        
        # Keep history manageable (last 1000 analyses)
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance and statistics report"""
        
        if self.analysis_count == 0:
            return {
                'total_analyses': 0,
                'abstention_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_processing_time': 0.0
            }
        
        # Calculate statistics
        abstention_rate = self.abstention_count / self.analysis_count
        
        confidences = [r.confidence for r in self.analysis_history if r.confidence is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        processing_times = [r.processing_time for r in self.analysis_history if r.processing_time is not None]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Decision distribution
        decisions = [r.decision for r in self.analysis_history]
        decision_counts = {}
        for decision in ComplianceDecision:
            decision_counts[decision.value] = sum(1 for d in decisions if d == decision)
        
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        
        return {
            'total_analyses': self.analysis_count,
            'abstention_count': self.abstention_count,
            'abstention_rate': abstention_rate,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'decision_distribution': decision_counts,
            'system_info': {
                'uptime_hours': uptime,
                'legal_context': self.legal_context.__class__.__name__,
                'risk_threshold': self.abstention_engine.risk_threshold,
                'confidence_threshold': self.abstention_engine.confidence_threshold
            },
            'abstention_stats': self.abstention_engine.get_abstention_stats()
        }
    
    def batch_analyze(
        self,
        scenarios: List[ComplianceScenario],
        max_concurrent: int = 5
    ) -> List[ComplianceResult]:
        """
        Analyze multiple scenarios concurrently.
        
        Args:
            scenarios: List of scenarios to analyze
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of ComplianceResults in same order as input
        """
        
        async def analyze_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def analyze_with_semaphore(scenario):
                async with semaphore:
                    return await self.analyze_scenario(scenario)
            
            tasks = [analyze_with_semaphore(scenario) for scenario in scenarios]
            return await asyncio.gather(*tasks)
        
        # Run the batch analysis
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from within async context, create a task
            return asyncio.create_task(analyze_batch())
        else:
            # If called from sync context, run directly
            return loop.run_until_complete(analyze_batch())