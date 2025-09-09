"""
Configuration management for LexCertainty.

Provides centralized configuration for all LexCertainty components
with environment variable support and validation.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LexCertaintyConfig:
    """
    Central configuration for LexCertainty Community Edition.
    
    Manages all configuration parameters with defaults optimized
    for legal document processing and mathematical abstention.
    """
    
    # Core processing settings
    legal_chunk_size: int = 1000
    legal_chunk_overlap: int = 100
    confidence_threshold: float = 0.75
    abstention_threshold: float = 0.8
    
    # Model settings
    default_model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Legal domain settings
    jurisdiction: str = "argentina"
    default_language: str = "es"
    legal_frameworks: List[str] = field(default_factory=lambda: ["ley_27401"])
    
    # Risk assessment settings
    risk_tolerance: str = "medium"  # low, medium, high
    enable_abstention: bool = True
    enable_mathematical_guarantees: bool = True
    
    # Processing settings
    parallel_processing: bool = True
    max_concurrent_chunks: int = 5
    cache_enabled: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_audit_log: bool = True
    
    # API settings
    api_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Performance settings
    batch_size: int = 10
    memory_limit_mb: int = 1024
    
    def __post_init__(self):
        """Post-initialization validation and environment variable loading."""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Core settings
        self.legal_chunk_size = int(os.getenv("LEX_CHUNK_SIZE", self.legal_chunk_size))
        self.legal_chunk_overlap = int(os.getenv("LEX_CHUNK_OVERLAP", self.legal_chunk_overlap))
        self.confidence_threshold = float(os.getenv("LEX_CONFIDENCE_THRESHOLD", self.confidence_threshold))
        self.abstention_threshold = float(os.getenv("LEX_ABSTENTION_THRESHOLD", self.abstention_threshold))
        
        # Model settings
        self.default_model = os.getenv("LEX_DEFAULT_MODEL", self.default_model)
        self.temperature = float(os.getenv("LEX_TEMPERATURE", self.temperature))
        self.max_tokens = int(os.getenv("LEX_MAX_TOKENS", self.max_tokens))
        
        # Legal settings
        self.jurisdiction = os.getenv("LEX_JURISDICTION", self.jurisdiction)
        self.default_language = os.getenv("LEX_LANGUAGE", self.default_language)
        
        if frameworks_env := os.getenv("LEX_LEGAL_FRAMEWORKS"):
            self.legal_frameworks = frameworks_env.split(",")
        
        # Risk settings
        self.risk_tolerance = os.getenv("LEX_RISK_TOLERANCE", self.risk_tolerance)
        self.enable_abstention = os.getenv("LEX_ENABLE_ABSTENTION", "true").lower() == "true"
        self.enable_mathematical_guarantees = os.getenv("LEX_ENABLE_MATH_GUARANTEES", "true").lower() == "true"
        
        # Performance settings
        self.parallel_processing = os.getenv("LEX_PARALLEL_PROCESSING", "true").lower() == "true"
        self.max_concurrent_chunks = int(os.getenv("LEX_MAX_CONCURRENT", self.max_concurrent_chunks))
        self.cache_enabled = os.getenv("LEX_CACHE_ENABLED", "true").lower() == "true"
        
        # Logging
        self.log_level = os.getenv("LEX_LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("LEX_LOG_FILE", self.log_file)
        self.enable_audit_log = os.getenv("LEX_AUDIT_LOG", "true").lower() == "true"
        
        # API settings
        self.api_timeout = int(os.getenv("LEX_API_TIMEOUT", self.api_timeout))
        self.retry_attempts = int(os.getenv("LEX_RETRY_ATTEMPTS", self.retry_attempts))
        self.retry_delay = float(os.getenv("LEX_RETRY_DELAY", self.retry_delay))
        
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate ranges
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {self.confidence_threshold}")
        
        if not 0.0 <= self.abstention_threshold <= 1.0:
            raise ValueError(f"abstention_threshold must be between 0 and 1, got {self.abstention_threshold}")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        
        # Validate chunk size relationships
        if self.legal_chunk_overlap >= self.legal_chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Validate risk tolerance
        if self.risk_tolerance not in ["low", "medium", "high"]:
            raise ValueError(f"risk_tolerance must be 'low', 'medium', or 'high', got {self.risk_tolerance}")
        
        # Validate jurisdiction
        supported_jurisdictions = ["argentina", "colombia", "mexico", "chile", "peru"]
        if self.jurisdiction not in supported_jurisdictions:
            logger.warning(f"Jurisdiction '{self.jurisdiction}' not officially supported. Supported: {supported_jurisdictions}")
        
        # Validate language
        supported_languages = ["es", "en", "pt"]
        if self.default_language not in supported_languages:
            logger.warning(f"Language '{self.default_language}' not officially supported. Supported: {supported_languages}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            "model": self.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.api_timeout
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking-specific configuration."""
        return {
            "chunk_size": self.legal_chunk_size,
            "overlap_size": self.legal_chunk_overlap,
            "parallel_processing": self.parallel_processing,
            "max_concurrent": self.max_concurrent_chunks
        }
    
    def get_abstention_config(self) -> Dict[str, Any]:
        """Get abstention-specific configuration."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "abstention_threshold": self.abstention_threshold,
            "enable_abstention": self.enable_abstention,
            "enable_guarantees": self.enable_mathematical_guarantees,
            "risk_tolerance": self.risk_tolerance
        }
    
    def get_legal_config(self) -> Dict[str, Any]:
        """Get legal domain-specific configuration."""
        return {
            "jurisdiction": self.jurisdiction,
            "language": self.default_language,
            "legal_frameworks": self.legal_frameworks,
            "risk_tolerance": self.risk_tolerance
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "legal_chunk_size": self.legal_chunk_size,
            "legal_chunk_overlap": self.legal_chunk_overlap,
            "confidence_threshold": self.confidence_threshold,
            "abstention_threshold": self.abstention_threshold,
            "default_model": self.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "jurisdiction": self.jurisdiction,
            "default_language": self.default_language,
            "legal_frameworks": self.legal_frameworks,
            "risk_tolerance": self.risk_tolerance,
            "enable_abstention": self.enable_abstention,
            "enable_mathematical_guarantees": self.enable_mathematical_guarantees,
            "parallel_processing": self.parallel_processing,
            "max_concurrent_chunks": self.max_concurrent_chunks,
            "cache_enabled": self.cache_enabled,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_audit_log": self.enable_audit_log,
            "api_timeout": self.api_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "batch_size": self.batch_size,
            "memory_limit_mb": self.memory_limit_mb
        }
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "LexCertaintyConfig":
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        import json
        
        try:
            if config_path.suffix.lower() in ['.json']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            config = cls()
            config.update_from_dict(config_data)
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"LexCertaintyConfig(jurisdiction={self.jurisdiction}, frameworks={self.legal_frameworks})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"LexCertaintyConfig({self.to_dict()})"