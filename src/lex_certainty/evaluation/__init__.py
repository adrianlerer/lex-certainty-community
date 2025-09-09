"""
Evaluation and benchmarking for LexCertainty.

This module provides comprehensive evaluation metrics and benchmarking
capabilities for legal AI systems with abstention support.
"""

from .metrics import (
    LegalMetrics,
    AbstractionMetrics, 
    MetricResult,
    EvaluationResult,
    MetricType
)

__all__ = [
    "LegalMetrics",
    "AbstractionMetrics",
    "MetricResult", 
    "EvaluationResult",
    "MetricType",
]