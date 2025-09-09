"""
LexCertainty Community Edition

Mathematical abstention framework for legal AI with guaranteed risk bounds.

This package implements mathematical abstention principles for legal document
analysis and compliance certification, inspired by EDFL research and optimized
for legal domain applications.
"""

__version__ = "0.1.0"
__author__ = "Adrian Lerer and Contributors"
__email__ = "research@lexcertainty.com"
__license__ = "MIT"

# Core components
from .core.abstention import AbstractionEngine, RiskCalculator
from .core.compliance import CertifiedComplianceEngine, ComplianceScenario
from .core.confidence import ConfidenceEstimator

# Legal processing
from .legal.document_processor import LegalDocumentProcessor
from .legal.context import ArgentineLegalContext, LegalContext
from .legal.prompter import LegalPromptEngine

# Chunking and document handling
from .chunking.chunker import DocumentChunker, ChunkingStrategy
from .chunking.augmentation import ContextAugmenter

# Evaluation and benchmarks
from .evaluation.metrics import LegalMetrics, AbstractionMetrics
from .evaluation.benchmarks import CUADBenchmark, BenchmarkSuite

# Utilities
from .utils.config import LexCertaintyConfig
from .utils.logging import setup_logging

__all__ = [
    # Core
    "AbstractionEngine",
    "RiskCalculator", 
    "CertifiedComplianceEngine",
    "ComplianceScenario",
    "ConfidenceEstimator",
    
    # Legal
    "LegalDocumentProcessor",
    "ArgentineLegalContext",
    "LegalContext", 
    "LegalPromptEngine",
    
    # Chunking
    "DocumentChunker",
    "ChunkingStrategy",
    "ContextAugmenter",
    
    # Evaluation
    "LegalMetrics",
    "AbstractionMetrics", 
    "CUADBenchmark",
    "BenchmarkSuite",
    
    # Utilities
    "LexCertaintyConfig",
    "setup_logging",
]

# Package metadata
DESCRIPTION = "Mathematical abstention framework for legal AI"
LONG_DESCRIPTION = __doc__

# Supported Python versions
PYTHON_REQUIRES = ">=3.8"

# Core dependencies  
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "transformers>=4.20.0",
    "torch>=1.12.0",
    "openai>=1.0.0",
    "tiktoken>=0.4.0",
    "pydantic>=1.8.0",
    "python-dotenv>=0.19.0",
    "tqdm>=4.62.0",
    "click>=8.0.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-asyncio>=0.18.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0", 
    "mypy>=0.950",
    "pre-commit>=2.17.0",
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
]

# Research dependencies
RESEARCH_REQUIRES = [
    "jupyter>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.6.0",
    "datasets>=2.0.0",
    "evaluate>=0.2.0",
]

def get_version():
    """Get package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "lex-certainty-community",
        "version": __version__,
        "author": __author__,
        "email": __email__, 
        "license": __license__,
        "description": DESCRIPTION,
        "python_requires": PYTHON_REQUIRES,
    }