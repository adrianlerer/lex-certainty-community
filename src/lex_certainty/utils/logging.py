"""
Logging configuration for LexCertainty.

Provides structured logging setup for legal AI processing with
audit trail support and security considerations.
"""

import os
import sys
import logging
import logging.config
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_audit_log: bool = True,
    log_format: Optional[str] = None,
    disable_existing_loggers: bool = False
) -> None:
    """
    Set up structured logging for LexCertainty.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_audit_log: Whether to enable audit logging
        log_format: Custom log format string
        disable_existing_loggers: Whether to disable existing loggers
    """
    
    # Default log format with timestamp and legal context
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
    
    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": disable_existing_loggers,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "audit": {
                "format": "%(asctime)s - AUDIT - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JSONFormatter,
                "timestamp": True,
                "legal_context": True
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "lex_certainty": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "": {  # Root logger
                "level": "WARNING",
                "handlers": ["console"]
            }
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "standard",
            "filename": str(log_path),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        }
        
        # Add file handler to lex_certainty logger
        config["loggers"]["lex_certainty"]["handlers"].append("file")
    
    # Add audit log handler if enabled
    if enable_audit_log:
        audit_log_path = _get_audit_log_path(log_file)
        
        config["handlers"]["audit"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "audit",
            "filename": str(audit_log_path),
            "maxBytes": 50 * 1024 * 1024,  # 50MB
            "backupCount": 10,
            "encoding": "utf-8"
        }
        
        # Create audit logger
        config["loggers"]["lex_certainty.audit"] = {
            "level": "INFO",
            "handlers": ["audit"],
            "propagate": False
        }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log configuration completion
    logger = logging.getLogger("lex_certainty.logging")
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}, Audit: {enable_audit_log}")


def _get_audit_log_path(log_file: Optional[str]) -> Path:
    """Generate audit log file path."""
    if log_file:
        log_path = Path(log_file)
        audit_path = log_path.parent / f"{log_path.stem}_audit{log_path.suffix}"
    else:
        # Default audit log location
        logs_dir = Path.home() / ".lex_certainty" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        audit_path = logs_dir / "lex_certainty_audit.log"
    
    return audit_path


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with legal context.
    """
    
    def __init__(self, timestamp: bool = True, legal_context: bool = True):
        """Initialize JSON formatter."""
        super().__init__()
        self.timestamp = timestamp
        self.legal_context = legal_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json
        
        # Base log data
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add timestamp if enabled
        if self.timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
        
        # Add legal context if available and enabled
        if self.legal_context and hasattr(record, 'legal_context'):
            log_data["legal_context"] = record.legal_context
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'message', 'exc_info', 'exc_text', 
                          'stack_info', 'legal_context']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class AuditLogger:
    """
    Specialized audit logger for legal AI operations.
    
    Provides structured audit logging for compliance and traceability
    in legal document processing.
    """
    
    def __init__(self, logger_name: str = "lex_certainty.audit"):
        """Initialize audit logger."""
        self.logger = logging.getLogger(logger_name)
    
    def log_document_processing(
        self,
        document_id: str,
        document_type: str,
        user_id: Optional[str] = None,
        processing_mode: str = "standard",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log document processing event."""
        audit_data = {
            "event_type": "document_processing",
            "document_id": document_id,
            "document_type": document_type,
            "processing_mode": processing_mode,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.logger.info(self._format_audit_message("Document processing started", audit_data))
    
    def log_compliance_assessment(
        self,
        assessment_id: str,
        legal_frameworks: list,
        compliance_score: float,
        risk_level: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log compliance assessment event."""
        audit_data = {
            "event_type": "compliance_assessment",
            "assessment_id": assessment_id,
            "legal_frameworks": legal_frameworks,
            "compliance_score": compliance_score,
            "risk_level": risk_level,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.logger.info(self._format_audit_message("Compliance assessment completed", audit_data))
    
    def log_abstention_decision(
        self,
        decision_id: str,
        should_abstain: bool,
        confidence_score: float,
        abstention_reason: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log abstention decision event."""
        audit_data = {
            "event_type": "abstention_decision",
            "decision_id": decision_id,
            "should_abstain": should_abstain,
            "confidence_score": confidence_score,
            "abstention_reason": abstention_reason,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.logger.info(self._format_audit_message("Abstention decision made", audit_data))
    
    def log_risk_assessment(
        self,
        assessment_id: str,
        risk_level: str,
        risk_factors: list,
        mitigation_recommendations: list,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log risk assessment event."""
        audit_data = {
            "event_type": "risk_assessment",
            "assessment_id": assessment_id,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_recommendations": mitigation_recommendations,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.logger.info(self._format_audit_message("Risk assessment completed", audit_data))
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log general system event."""
        audit_data = {
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(self._format_audit_message(description, audit_data))
    
    def _format_audit_message(self, message: str, audit_data: Dict[str, Any]) -> str:
        """Format audit message with structured data."""
        import json
        return f"{message} | {json.dumps(audit_data, ensure_ascii=False, default=str)}"


def get_logger(name: str, legal_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional legal context.
    
    Args:
        name: Logger name
        legal_context: Optional legal context to attach to log records
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if legal_context:
        # Create custom adapter to add legal context
        class LegalContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add legal context to extra
                if 'extra' not in kwargs:
                    kwargs['extra'] = {}
                kwargs['extra']['legal_context'] = self.extra['legal_context']
                return msg, kwargs
        
        logger = LegalContextAdapter(logger, {'legal_context': legal_context})
    
    return logger


def configure_third_party_logging() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce verbosity of common third-party libraries
    third_party_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "openai",
        "httpx",
        "httpcore",
        "tiktoken"
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Special handling for specific libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.WARNING)


# Initialize audit logger instance
audit_logger = AuditLogger()


def setup_default_logging() -> None:
    """Set up default logging configuration for LexCertainty."""
    log_level = os.getenv("LEX_LOG_LEVEL", "INFO")
    log_file = os.getenv("LEX_LOG_FILE")
    enable_audit = os.getenv("LEX_AUDIT_LOG", "true").lower() == "true"
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_audit_log=enable_audit
    )
    
    configure_third_party_logging()


# Auto-setup logging when module is imported
if not logging.getLogger("lex_certainty").handlers:
    setup_default_logging()