"""
Document chunking and context augmentation for LexCertainty.

This module provides sophisticated chunking strategies and context augmentation
based on the "LLMs for LLMs" academic methodology.
"""

from .chunker import (
    DocumentChunker,
    DocumentChunk,
    ChunkingStrategy,
    ChunkType,
    LegalStructureDetector,
    SemanticBoundaryDetector
)
from .augmentation import (
    ContextAugmenter,
    AugmentationStrategy,
    ContextElement,
    ContextType,
    AugmentationResult,
    LegalKnowledgeBase
)

__all__ = [
    # Chunking
    "DocumentChunker",
    "DocumentChunk", 
    "ChunkingStrategy",
    "ChunkType",
    "LegalStructureDetector",
    "SemanticBoundaryDetector",
    
    # Augmentation
    "ContextAugmenter",
    "AugmentationStrategy",
    "ContextElement",
    "ContextType", 
    "AugmentationResult",
    "LegalKnowledgeBase",
]