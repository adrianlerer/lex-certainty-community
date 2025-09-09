"""
Legal Document Processor

Implements structured legal document processing with chunking, augmentation,
and domain-specific handling for legal texts. Based on methodologies from
the "LLMs for LLMs" academic paper for enhanced legal document understanding.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import tiktoken

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from ..core.abstention import AbstractionEngine, AbstractionResult
from ..core.confidence import ConfidenceEstimator, ConfidenceResult
from ..chunking.chunker import DocumentChunker, ChunkingStrategy, DocumentChunk
from ..chunking.augmentation import ContextAugmenter, AugmentationStrategy
from ..legal.context import LegalContext, ArgentineLegalContext
from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Legal document type classification."""
    CONTRACT = "contract"
    POLICY = "policy" 
    REGULATION = "regulation"
    COMPLIANCE_REPORT = "compliance_report"
    LEGAL_MEMO = "legal_memo"
    LITIGATION = "litigation"
    CORPORATE_GOVERNANCE = "corporate_governance"
    EMPLOYMENT = "employment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    UNKNOWN = "unknown"


class ProcessingMode(Enum):
    """Document processing mode."""
    FAST = "fast"           # Basic chunking and analysis
    STANDARD = "standard"   # Full processing with augmentation
    DEEP = "deep"          # Comprehensive analysis with cross-references
    AUDIT = "audit"        # Full audit trail and certification


@dataclass
class DocumentMetadata:
    """Document metadata for legal processing."""
    document_id: str
    document_type: DocumentType
    jurisdiction: str = "argentina"
    language: str = "es"
    creation_date: Optional[str] = None
    last_modified: Optional[str] = None
    author: Optional[str] = None
    legal_framework: List[str] = field(default_factory=list)
    sensitivity_level: str = "standard"  # low, standard, high, confidential
    checksum: Optional[str] = None


class ProcessedDocument(BaseModel):
    """Processed legal document with analysis results."""
    
    metadata: Dict[str, Any] = Field(description="Document metadata")
    chunks: List[Dict[str, Any]] = Field(description="Document chunks with analysis")
    legal_analysis: Dict[str, Any] = Field(description="Legal analysis results")
    compliance_flags: List[Dict[str, Any]] = Field(description="Compliance issues identified")
    confidence_scores: Dict[str, float] = Field(description="Confidence in analysis")
    abstention_results: Dict[str, Any] = Field(description="Abstention analysis")
    processing_stats: Dict[str, Any] = Field(description="Processing statistics")
    audit_trail: List[Dict[str, Any]] = Field(description="Audit trail")
    
    class Config:
        arbitrary_types_allowed = True


class LegalDocumentProcessor:
    """
    Main legal document processor implementing structured processing
    with mathematical abstention principles.
    
    Implements methodology from "LLMs for LLMs" paper:
    - Distribution-based localization for legal contexts
    - Inverse cardinality weighting for legal concepts
    - Structured chunking with legal domain awareness
    - Confidence-aware processing with abstention
    """
    
    def __init__(
        self,
        config: Optional[LexCertaintyConfig] = None,
        legal_context: Optional[LegalContext] = None,
        abstention_engine: Optional[AbstractionEngine] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
    ):
        """Initialize legal document processor."""
        self.config = config or LexCertaintyConfig()
        self.legal_context = legal_context or ArgentineLegalContext()
        
        # Initialize core engines
        self.abstention_engine = abstention_engine or AbstractionEngine(
            config=self.config,
            domain="legal"
        )
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator(
            config=self.config,
            domain="legal"
        )
        
        # Initialize processing components
        self.chunker = DocumentChunker(
            strategy=ChunkingStrategy.LEGAL_SEMANTIC,
            max_chunk_size=self.config.legal_chunk_size,
            overlap_size=self.config.legal_chunk_overlap,
            legal_context=self.legal_context
        )
        
        self.augmenter = ContextAugmenter(
            strategy=AugmentationStrategy.LEGAL_ENHANCED,
            legal_context=self.legal_context
        )
        
        # Tokenizer for text analysis
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "abstentions": 0,
            "high_risk_flags": 0,
            "processing_time": 0.0
        }
        
        logger.info(f"Initialized LegalDocumentProcessor with {self.legal_context.__class__.__name__}")
    
    async def process_document(
        self,
        document: str,
        metadata: Optional[DocumentMetadata] = None,
        mode: ProcessingMode = ProcessingMode.STANDARD,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Process a legal document with comprehensive analysis.
        
        Args:
            document: Document text content
            metadata: Document metadata
            mode: Processing mode (fast, standard, deep, audit)
            custom_context: Additional context for processing
            
        Returns:
            ProcessedDocument with complete analysis
        """
        import time
        start_time = time.time()
        
        try:
            # Generate metadata if not provided
            if metadata is None:
                metadata = await self._generate_metadata(document)
            
            # Validate and prepare document
            document = self._preprocess_document(document)
            
            # Initialize audit trail
            audit_trail = [
                {
                    "timestamp": time.time(),
                    "action": "processing_started", 
                    "mode": mode.value,
                    "document_id": metadata.document_id,
                    "metadata": metadata.__dict__
                }
            ]
            
            # Document chunking with legal semantics
            chunks = await self._chunk_document(document, metadata, mode)
            audit_trail.append({
                "timestamp": time.time(),
                "action": "chunking_completed",
                "chunks_created": len(chunks)
            })
            
            # Process chunks with augmentation
            processed_chunks = []
            for chunk in chunks:
                processed_chunk = await self._process_chunk(chunk, metadata, mode)
                processed_chunks.append(processed_chunk)
            
            audit_trail.append({
                "timestamp": time.time(),
                "action": "chunk_processing_completed",
                "processed_chunks": len(processed_chunks)
            })
            
            # Legal analysis and compliance checking
            legal_analysis = await self._perform_legal_analysis(
                processed_chunks, metadata, mode
            )
            
            # Identify compliance flags
            compliance_flags = await self._identify_compliance_flags(
                processed_chunks, legal_analysis, metadata
            )
            
            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                processed_chunks, legal_analysis
            )
            
            # Perform abstention analysis  
            abstention_results = await self._perform_abstention_analysis(
                processed_chunks, confidence_scores, metadata
            )
            
            # Generate processing statistics
            processing_time = time.time() - start_time
            processing_stats = {
                "processing_time": processing_time,
                "total_tokens": len(self.tokenizer.encode(document)),
                "chunks_processed": len(processed_chunks),
                "mode": mode.value,
                "abstentions": len([r for r in abstention_results.get("chunk_results", []) 
                                 if r.get("should_abstain", False)]),
                "high_confidence_chunks": len([c for c in processed_chunks 
                                             if c.get("confidence", 0) > 0.8]),
                "compliance_flags_count": len(compliance_flags)
            }
            
            # Update global stats
            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] += len(processed_chunks)
            self.stats["processing_time"] += processing_time
            
            audit_trail.append({
                "timestamp": time.time(),
                "action": "processing_completed",
                "stats": processing_stats
            })
            
            return ProcessedDocument(
                metadata=metadata.__dict__,
                chunks=processed_chunks,
                legal_analysis=legal_analysis,
                compliance_flags=compliance_flags,
                confidence_scores=confidence_scores,
                abstention_results=abstention_results,
                processing_stats=processing_stats,
                audit_trail=audit_trail
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            audit_trail.append({
                "timestamp": time.time(),
                "action": "processing_failed",
                "error": str(e)
            })
            raise
    
    async def _generate_metadata(self, document: str) -> DocumentMetadata:
        """Generate document metadata from content analysis."""
        # Generate document ID from content hash
        document_id = hashlib.sha256(document.encode()).hexdigest()[:16]
        
        # Classify document type using legal patterns
        document_type = self._classify_document_type(document)
        
        # Detect language and jurisdiction
        language = self._detect_language(document)
        jurisdiction = self._detect_jurisdiction(document)
        
        # Extract legal frameworks mentioned
        legal_framework = self._extract_legal_frameworks(document)
        
        # Calculate document checksum
        checksum = hashlib.md5(document.encode()).hexdigest()
        
        return DocumentMetadata(
            document_id=document_id,
            document_type=document_type,
            language=language,
            jurisdiction=jurisdiction,
            legal_framework=legal_framework,
            checksum=checksum
        )
    
    def _classify_document_type(self, document: str) -> DocumentType:
        """Classify legal document type based on content patterns."""
        document_lower = document.lower()
        
        # Define legal document patterns
        patterns = {
            DocumentType.CONTRACT: [
                r"\bcontrat[oa]\b", r"\bacuerdo\b", r"\bconvenio\b",
                r"\bcláusul[as]?\b", r"\bobligacion[es]?\b"
            ],
            DocumentType.POLICY: [
                r"\bpolític[as]?\b", r"\bprocedimiento[s]?\b", 
                r"\bnorma[s]?\b", r"\bdirectriz[as]?\b"
            ],
            DocumentType.REGULATION: [
                r"\bley\b", r"\bdecreto\b", r"\bresolución\b",
                r"\breglament[oa]\b", r"\bnormativ[oa]\b"
            ],
            DocumentType.COMPLIANCE_REPORT: [
                r"\bcumplimiento\b", r"\binforme\b", r"\bauditoría\b",
                r"\bverificación\b", r"\bmonitoreo\b"
            ]
        }
        
        # Score each document type
        type_scores = {}
        for doc_type, type_patterns in patterns.items():
            score = 0
            for pattern in type_patterns:
                matches = len(re.findall(pattern, document_lower))
                score += matches
            type_scores[doc_type] = score
        
        # Return highest scoring type or UNKNOWN
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return DocumentType.UNKNOWN
    
    def _detect_language(self, document: str) -> str:
        """Detect document language."""
        # Simple Spanish detection based on common words
        spanish_indicators = [
            "el", "la", "de", "que", "y", "en", "un", "es", "se", "no",
            "te", "lo", "le", "da", "su", "por", "son", "con", "para"
        ]
        
        words = document.lower().split()[:100]  # Check first 100 words
        spanish_count = sum(1 for word in words if word in spanish_indicators)
        
        return "es" if spanish_count > len(words) * 0.3 else "en"
    
    def _detect_jurisdiction(self, document: str) -> str:
        """Detect legal jurisdiction from document content."""
        argentina_indicators = [
            r"\bargentina\b", r"\bbsas\b", r"\bbuenos aires\b",
            r"\bcaba\b", r"\bpba\b", r"\bley\s+\d+", 
            r"\bdecreto\s+\d+", r"\bcjn\b"
        ]
        
        document_lower = document.lower()
        for pattern in argentina_indicators:
            if re.search(pattern, document_lower):
                return "argentina"
        
        return "unknown"
    
    def _extract_legal_frameworks(self, document: str) -> List[str]:
        """Extract legal framework references from document."""
        frameworks = []
        
        # Common Argentine legal frameworks
        framework_patterns = {
            "Ley 27401": r"\bley\s+27\.?401\b",
            "Código Civil": r"\bcódigo\s+civil\b",
            "Código Penal": r"\bcódigo\s+penal\b", 
            "Ley de Sociedades": r"\bley\s+(?:de\s+)?sociedades\b",
            "Ley de Contratos": r"\bley\s+(?:de\s+)?contratos\b"
        }
        
        document_lower = document.lower()
        for framework, pattern in framework_patterns.items():
            if re.search(pattern, document_lower):
                frameworks.append(framework)
        
        return frameworks
    
    def _preprocess_document(self, document: str) -> str:
        """Preprocess document text for analysis."""
        # Remove excessive whitespace
        document = re.sub(r'\s+', ' ', document.strip())
        
        # Normalize legal citations
        document = re.sub(r'\bArt\.?\s*(\d+)', r'Artículo \1', document)
        document = re.sub(r'\bInc\.?\s*(\d+)', r'Inciso \1', document)
        
        return document
    
    async def _chunk_document(
        self,
        document: str,
        metadata: DocumentMetadata,
        mode: ProcessingMode
    ) -> List[DocumentChunk]:
        """Chunk document with legal semantic awareness."""
        # Adjust chunking strategy based on processing mode
        if mode == ProcessingMode.FAST:
            self.chunker.max_chunk_size = self.config.legal_chunk_size // 2
        elif mode == ProcessingMode.DEEP:
            self.chunker.max_chunk_size = self.config.legal_chunk_size * 2
        
        # Perform chunking with legal context
        chunks = await self.chunker.chunk_document(
            document=document,
            metadata=metadata.__dict__,
            preserve_legal_structure=True
        )
        
        return chunks
    
    async def _process_chunk(
        self,
        chunk: DocumentChunk,
        metadata: DocumentMetadata,
        mode: ProcessingMode
    ) -> Dict[str, Any]:
        """Process individual chunk with augmentation and analysis."""
        # Apply context augmentation
        if mode in [ProcessingMode.STANDARD, ProcessingMode.DEEP, ProcessingMode.AUDIT]:
            augmented_chunk = await self.augmenter.augment_chunk(
                chunk=chunk,
                context={
                    "document_type": metadata.document_type.value,
                    "jurisdiction": metadata.jurisdiction,
                    "legal_frameworks": metadata.legal_framework
                }
            )
        else:
            augmented_chunk = chunk
        
        # Calculate chunk-level confidence
        confidence_result = await self.confidence_estimator.estimate_confidence(
            text=augmented_chunk.content,
            context={
                "chunk_type": "legal",
                "document_type": metadata.document_type.value,
                "legal_frameworks": metadata.legal_framework
            }
        )
        
        # Identify legal entities and concepts
        legal_entities = self._extract_legal_entities(augmented_chunk.content)
        legal_concepts = self._extract_legal_concepts(augmented_chunk.content)
        
        # Risk assessment for chunk
        risk_indicators = self._assess_chunk_risk(augmented_chunk.content, metadata)
        
        return {
            "chunk_id": augmented_chunk.id,
            "content": augmented_chunk.content,
            "original_content": chunk.content,
            "start_position": augmented_chunk.start_position,
            "end_position": augmented_chunk.end_position,
            "chunk_type": augmented_chunk.chunk_type,
            "confidence": confidence_result.confidence_score,
            "confidence_method": confidence_result.method_used,
            "legal_entities": legal_entities,
            "legal_concepts": legal_concepts,
            "risk_indicators": risk_indicators,
            "augmentation_applied": mode != ProcessingMode.FAST,
            "processing_metadata": {
                "tokens": len(self.tokenizer.encode(augmented_chunk.content)),
                "semantic_density": self._calculate_semantic_density(augmented_chunk.content),
                "legal_complexity": self._assess_legal_complexity(augmented_chunk.content)
            }
        }
    
    def _extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text."""
        entities = {
            "persons": [],
            "organizations": [],  
            "legal_references": [],
            "dates": [],
            "amounts": []
        }
        
        # Legal reference patterns
        legal_ref_patterns = [
            r'\bLey\s+\d+(?:\.\d+)*\b',
            r'\bDecreto\s+\d+(?:/\d+)?\b',
            r'\bResolución\s+\d+(?:/\d+)?\b',
            r'\bArtículo\s+\d+\b',
            r'\bInciso\s+[a-z]\)\b'
        ]
        
        for pattern in legal_ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["legal_references"].extend(matches)
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Amount patterns (monetary)
        amount_patterns = [
            r'\$\s*[\d.,]+',
            r'\b\d+(?:\.\d+)?\s*(?:pesos|dólares|euros)\b'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["amounts"].extend(matches)
        
        return entities
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts and terminology."""
        legal_concepts = []
        
        # Legal concept patterns for Argentine law
        concept_patterns = [
            r'\b(?:responsabilidad\s+penal|criminal\s+liability)\b',
            r'\b(?:due\s+diligence|debida\s+diligencia)\b',
            r'\b(?:compliance|cumplimiento)\b',
            r'\b(?:governance|gobierno\s+corporativo)\b',
            r'\b(?:integridad|integrity)\b',
            r'\b(?:transparencia|transparency)\b',
            r'\b(?:anticorrupción|anti-corruption)\b',
            r'\b(?:código\s+de\s+(?:ética|conducta)|code\s+of\s+(?:ethics|conduct))\b'
        ]
        
        text_lower = text.lower()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            legal_concepts.extend(matches)
        
        return list(set(legal_concepts))  # Remove duplicates
    
    def _assess_chunk_risk(self, text: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Assess legal risk indicators in chunk."""
        risk_indicators = {
            "corruption_risk": 0.0,
            "compliance_risk": 0.0,
            "regulatory_risk": 0.0,
            "contractual_risk": 0.0,
            "risk_keywords": []
        }
        
        # High-risk keywords for Argentine legal context
        risk_keywords = {
            "corruption": ["soborno", "coima", "corrupción", "kickback", "facilitating payment"],
            "compliance": ["incumplimiento", "violación", "infracción", "breach", "non-compliance"],
            "regulatory": ["sanción", "multa", "penalty", "fine", "enforcement action"],
            "contractual": ["incumplimiento contractual", "breach of contract", "default", "termination"]
        }
        
        text_lower = text.lower()
        
        for risk_category, keywords in risk_keywords.items():
            category_score = 0.0
            found_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    category_score += 1.0
                    found_keywords.append(keyword)
            
            # Normalize by text length
            category_score = min(category_score / (len(text.split()) / 100), 1.0)
            
            risk_indicators[f"{risk_category}_risk"] = category_score
            if found_keywords:
                risk_indicators["risk_keywords"].extend(found_keywords)
        
        return risk_indicators
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of legal text."""
        words = text.split()
        if not words:
            return 0.0
        
        # Legal terms that indicate high semantic density
        legal_terms = [
            "artículo", "inciso", "párrafo", "establecer", "determinar",
            "obligación", "derecho", "responsabilidad", "sanción",
            "cumplimiento", "violación", "infracción"
        ]
        
        legal_word_count = sum(1 for word in words 
                              if word.lower().rstrip('.,;:') in legal_terms)
        
        return min(legal_word_count / len(words), 1.0)
    
    def _assess_legal_complexity(self, text: str) -> float:
        """Assess legal complexity of text."""
        # Simple complexity metrics
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Legal references increase complexity
        legal_refs = len(re.findall(r'\b(?:Ley|Decreto|Artículo|Inciso)\s+\d+', text))
        
        # Nested clauses increase complexity  
        nested_clauses = text.count('(') + text.count('[')
        
        # Normalize complexity score
        complexity = (avg_sentence_length / 20.0 + legal_refs / 5.0 + nested_clauses / 10.0) / 3.0
        
        return min(complexity, 1.0)
    
    async def _perform_legal_analysis(
        self,
        chunks: List[Dict[str, Any]],
        metadata: DocumentMetadata,
        mode: ProcessingMode
    ) -> Dict[str, Any]:
        """Perform comprehensive legal analysis on processed chunks."""
        analysis = {
            "document_summary": {},
            "legal_issues": [],
            "compliance_assessment": {},
            "risk_analysis": {},
            "recommendations": []
        }
        
        # Aggregate chunk-level metrics
        total_confidence = np.mean([chunk["confidence"] for chunk in chunks])
        total_risk = np.mean([
            sum(chunk["risk_indicators"].values()) / 4.0 
            for chunk in chunks
        ])
        
        # Legal issue identification
        legal_issues = []
        for chunk in chunks:
            if chunk["risk_indicators"]["corruption_risk"] > 0.3:
                legal_issues.append({
                    "issue_type": "corruption_risk",
                    "severity": "high" if chunk["risk_indicators"]["corruption_risk"] > 0.7 else "medium",
                    "chunk_id": chunk["chunk_id"],
                    "description": "Potential corruption-related content identified"
                })
        
        # Compliance assessment using legal context
        compliance_assessment = await self.legal_context.assess_compliance(
            document_type=metadata.document_type.value,
            content_chunks=[chunk["content"] for chunk in chunks],
            legal_frameworks=metadata.legal_framework
        )
        
        analysis.update({
            "document_summary": {
                "total_chunks": len(chunks),
                "avg_confidence": total_confidence,
                "overall_risk": total_risk,
                "legal_complexity": np.mean([
                    chunk["processing_metadata"]["legal_complexity"] 
                    for chunk in chunks
                ]),
                "semantic_density": np.mean([
                    chunk["processing_metadata"]["semantic_density"]
                    for chunk in chunks
                ])
            },
            "legal_issues": legal_issues,
            "compliance_assessment": compliance_assessment,
            "risk_analysis": {
                "corruption_risk": np.mean([
                    chunk["risk_indicators"]["corruption_risk"] 
                    for chunk in chunks
                ]),
                "compliance_risk": np.mean([
                    chunk["risk_indicators"]["compliance_risk"]
                    for chunk in chunks  
                ]),
                "overall_risk": total_risk
            }
        })
        
        return analysis
    
    async def _identify_compliance_flags(
        self,
        chunks: List[Dict[str, Any]],
        legal_analysis: Dict[str, Any],
        metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Identify compliance flags and issues."""
        flags = []
        
        # High-risk chunks
        for chunk in chunks:
            overall_risk = sum(chunk["risk_indicators"].values()) / 4.0
            
            if overall_risk > 0.6:
                flags.append({
                    "flag_type": "high_risk_content",
                    "severity": "high",
                    "chunk_id": chunk["chunk_id"],
                    "description": f"High-risk content detected (risk score: {overall_risk:.2f})",
                    "risk_score": overall_risk,
                    "risk_breakdown": chunk["risk_indicators"]
                })
        
        # Low confidence flags
        low_confidence_chunks = [c for c in chunks if c["confidence"] < 0.5]
        if low_confidence_chunks:
            flags.append({
                "flag_type": "low_confidence",
                "severity": "medium",
                "chunk_count": len(low_confidence_chunks),
                "description": f"Low confidence in analysis for {len(low_confidence_chunks)} chunks",
                "affected_chunks": [c["chunk_id"] for c in low_confidence_chunks]
            })
        
        # Legal framework compliance
        compliance_issues = legal_analysis.get("compliance_assessment", {}).get("issues", [])
        for issue in compliance_issues:
            flags.append({
                "flag_type": "compliance_violation",
                "severity": issue.get("severity", "medium"),
                "framework": issue.get("framework"),
                "description": issue.get("description"),
                "remediation": issue.get("remediation")
            })
        
        return flags
    
    async def _calculate_confidence_scores(
        self,
        chunks: List[Dict[str, Any]], 
        legal_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive confidence scores."""
        chunk_confidences = [chunk["confidence"] for chunk in chunks]
        
        return {
            "overall_confidence": np.mean(chunk_confidences),
            "min_chunk_confidence": min(chunk_confidences),
            "max_chunk_confidence": max(chunk_confidences),
            "confidence_variance": np.var(chunk_confidences),
            "legal_analysis_confidence": legal_analysis.get("document_summary", {}).get("avg_confidence", 0.0),
            "high_confidence_ratio": len([c for c in chunk_confidences if c > 0.7]) / len(chunk_confidences)
        }
    
    async def _perform_abstention_analysis(
        self,
        chunks: List[Dict[str, Any]],
        confidence_scores: Dict[str, float],
        metadata: DocumentMetadata  
    ) -> Dict[str, Any]:
        """Perform abstention analysis using mathematical principles."""
        chunk_results = []
        
        for chunk in chunks:
            # Prepare abstention context
            abstention_context = {
                "confidence": chunk["confidence"],
                "risk_indicators": chunk["risk_indicators"],
                "legal_complexity": chunk["processing_metadata"]["legal_complexity"],
                "document_type": metadata.document_type.value,
                "jurisdiction": metadata.jurisdiction
            }
            
            # Get abstention decision
            abstention_result = await self.abstention_engine.should_abstain(
                confidence=chunk["confidence"],
                context=abstention_context,
                domain="legal"
            )
            
            chunk_results.append({
                "chunk_id": chunk["chunk_id"],
                "should_abstain": abstention_result.should_abstain,
                "abstention_reason": abstention_result.reason,
                "confidence_threshold": abstention_result.confidence_threshold,
                "risk_bounds": abstention_result.risk_bounds.__dict__ if abstention_result.risk_bounds else None
            })
        
        # Overall abstention recommendation
        abstention_count = len([r for r in chunk_results if r["should_abstain"]])
        abstention_ratio = abstention_count / len(chunk_results)
        
        # Document-level abstention decision
        overall_confidence = confidence_scores["overall_confidence"]
        overall_abstention = await self.abstention_engine.should_abstain(
            confidence=overall_confidence,
            context={
                "document_type": metadata.document_type.value,
                "jurisdiction": metadata.jurisdiction,
                "chunk_abstention_ratio": abstention_ratio,
                "legal_frameworks": metadata.legal_framework
            },
            domain="legal"
        )
        
        return {
            "chunk_results": chunk_results,
            "overall_abstention": {
                "should_abstain": overall_abstention.should_abstain,
                "reason": overall_abstention.reason,
                "confidence": overall_confidence,
                "abstention_ratio": abstention_ratio
            },
            "abstention_statistics": {
                "total_chunks": len(chunks),
                "abstained_chunks": abstention_count,
                "abstention_ratio": abstention_ratio,
                "confidence_below_threshold": len([c for c in chunks if c["confidence"] < 0.5])
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0, 
            "abstentions": 0,
            "high_risk_flags": 0,
            "processing_time": 0.0
        }