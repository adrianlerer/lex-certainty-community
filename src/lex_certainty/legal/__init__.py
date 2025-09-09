"""
Legal processing components for LexCertainty.

This module provides legal domain-specific processing capabilities including
document analysis, Argentine legal context, and structured prompting.
"""

from .document_processor import LegalDocumentProcessor, ProcessedDocument, DocumentType, ProcessingMode
from .context import (
    LegalContext, 
    ArgentineLegalContext, 
    ComplianceAssessment, 
    ComplianceIssue,
    ComplianceStatus,
    RiskLevel
)
from .prompter import (
    LegalPromptEngine,
    PromptContext,
    PromptType,
    PromptComplexity
)

__all__ = [
    # Document processor
    "LegalDocumentProcessor",
    "ProcessedDocument", 
    "DocumentType",
    "ProcessingMode",
    
    # Legal context
    "LegalContext",
    "ArgentineLegalContext",
    "ComplianceAssessment",
    "ComplianceIssue",
    "ComplianceStatus", 
    "RiskLevel",
    
    # Prompt engineering
    "LegalPromptEngine",
    "PromptContext",
    "PromptType",
    "PromptComplexity",
]