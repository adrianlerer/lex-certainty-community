"""
Context Augmentation for Legal Documents

Implements context augmentation strategies based on "LLMs for LLMs" methodology
with distribution-based localization and inverse cardinality weighting for
enhanced legal document understanding.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
from pydantic import BaseModel, Field

from ..chunking.chunker import DocumentChunk
from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class AugmentationStrategy(Enum):
    """Context augmentation strategies."""
    NONE = "none"                       # No augmentation
    BASIC = "basic"                     # Basic context enrichment
    LEGAL_ENHANCED = "legal_enhanced"   # Legal-specific augmentation
    ACADEMIC = "academic"               # Academic paper methodology
    ADAPTIVE = "adaptive"               # Adaptive based on content


class ContextType(Enum):
    """Types of context for augmentation."""
    LEGAL_FRAMEWORK = "legal_framework"
    JURISDICTION = "jurisdiction"
    DOCUMENT_TYPE = "document_type"
    CROSS_REFERENCE = "cross_reference"
    PRECEDENT = "precedent"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    COMPLIANCE_RULE = "compliance_rule"


@dataclass
class ContextElement:
    """Individual context element for augmentation."""
    context_type: ContextType
    content: str
    relevance_score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relevance score."""
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class AugmentationResult:
    """Result of context augmentation."""
    original_chunk: DocumentChunk
    augmented_content: str
    context_elements: List[ContextElement]
    augmentation_metadata: Dict[str, Any]
    confidence_boost: float = 0.0


class LegalKnowledgeBase:
    """
    Legal knowledge base for context enrichment.
    
    Implements distribution-based localization and inverse cardinality
    weighting as described in "LLMs for LLMs" paper.
    """
    
    def __init__(self, jurisdiction: str = "argentina"):
        """Initialize legal knowledge base."""
        self.jurisdiction = jurisdiction
        self.legal_frameworks = self._load_legal_frameworks()
        self.definitions = self._load_legal_definitions()
        self.procedures = self._load_legal_procedures()
        self.compliance_rules = self._load_compliance_rules()
        
        # Distribution statistics for weighting
        self.term_distributions = self._calculate_term_distributions()
        self.concept_frequencies = self._calculate_concept_frequencies()
        
    def _load_legal_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Load legal framework information."""
        if self.jurisdiction == "argentina":
            return {
                "ley_27401": {
                    "title": "Ley 27401 - Responsabilidad Penal Empresaria",
                    "scope": "corporate criminal liability",
                    "key_concepts": [
                        "programa de integridad", "due diligence", "compliance",
                        "responsabilidad penal", "persona jurídica"
                    ],
                    "articles": {
                        "1": "Objeto y ámbito de aplicación",
                        "7": "Programa de Integridad", 
                        "8": "Elementos del Programa de Integridad",
                        "9": "Persona Responsable del Programa"
                    },
                    "penalties": "multa de 2 a 5 veces del beneficio indebido"
                },
                "codigo_civil": {
                    "title": "Código Civil y Comercial de la Nación",
                    "scope": "civil and commercial law",
                    "key_concepts": [
                        "persona jurídica", "capacidad", "representación",
                        "responsabilidad", "contrato"
                    ]
                },
                "codigo_penal": {
                    "title": "Código Penal de la Nación",
                    "scope": "criminal law",
                    "key_concepts": [
                        "delito", "culpabilidad", "responsabilidad penal",
                        "sanción", "imputabilidad"
                    ]
                }
            }
        return {}
    
    def _load_legal_definitions(self) -> Dict[str, str]:
        """Load legal term definitions."""
        return {
            "programa de integridad": (
                "Conjunto de acciones, mecanismos y procedimientos internos "
                "de promoción de la integridad, supervisión y control, orientados "
                "a prevenir, detectar y corregir irregularidades y actos ilícitos."
            ),
            "due diligence": (
                "Proceso de investigación y análisis de una empresa o persona "
                "para evaluar su situación legal, financiera y operacional."
            ),
            "compliance": (
                "Conjunto de procedimientos y buenas prácticas adoptados por "
                "las organizaciones para identificar y clasificar los riesgos "
                "operativos y legales."
            ),
            "responsabilidad penal empresaria": (
                "Responsabilidad de las personas jurídicas por delitos cometidos "
                "en su nombre, interés o beneficio por sus directivos o empleados."
            ),
            "persona jurídica": (
                "Entidad de existencia ideal dotada de personalidad jurídica "
                "propia y distinta de sus miembros."
            )
        }
    
    def _load_legal_procedures(self) -> Dict[str, Dict[str, Any]]:
        """Load legal procedure information."""
        return {
            "implementacion_programa_integridad": {
                "steps": [
                    "1. Mapeo de riesgos específicos",
                    "2. Desarrollo de código de ética",
                    "3. Capacitación del personal",
                    "4. Implementación de canales de denuncia",
                    "5. Monitoreo y auditoría continua",
                    "6. Investigación y remediation"
                ],
                "requirements": [
                    "Mapeo de procesos y riesgos",
                    "Código de ética y conducta",  
                    "Capacitación y comunicación",
                    "Canal de consultas y denuncias",
                    "Evaluación de terceros",
                    "Monitoreo continuo",
                    "Investigación y remediation"
                ],
                "documentation": [
                    "Manual de programa de integridad",
                    "Matriz de riesgos",
                    "Registros de capacitación", 
                    "Reportes de monitoreo"
                ]
            }
        }
    
    def _load_compliance_rules(self) -> Dict[str, List[str]]:
        """Load compliance rules and requirements."""
        return {
            "ley_27401_requirements": [
                "Mapeo de procesos y identificación de riesgos",
                "Desarrollo e implementación de un código de ética",
                "Implementación de políticas y procedimientos",
                "Identificación y evaluación de terceros",
                "Capacitación periódica en materia de integridad",
                "Canales internos de denuncia",
                "Protección del denunciante", 
                "Evaluación de riesgos y monitoreo del programa",
                "Investigación interna y debida diligencia",
                "Verificación y mejora continua del programa"
            ],
            "governance_principles": [
                "Transparencia en la gestión",
                "Rendición de cuentas",
                "Separación de funciones",
                "Control interno eficaz",
                "Gestión de riesgos",
                "Integridad y valores éticos"
            ]
        }
    
    def _calculate_term_distributions(self) -> Dict[str, float]:
        """Calculate term distribution statistics for weighting."""
        # Simulate term frequency distributions in legal corpus
        # In real implementation, this would be calculated from actual corpus
        return {
            "programa de integridad": 0.15,
            "responsabilidad penal": 0.08,
            "compliance": 0.12,
            "due diligence": 0.06,
            "código de ética": 0.09,
            "persona jurídica": 0.25,
            "contrato": 0.45,
            "artículo": 0.60,
            "ley": 0.40
        }
    
    def _calculate_concept_frequencies(self) -> Dict[str, int]:
        """Calculate concept frequencies for inverse cardinality weighting."""
        # Simulate concept occurrence frequencies
        return {
            "legal_framework": 1500,
            "compliance_requirement": 800,
            "corporate_governance": 600,
            "criminal_liability": 400,
            "integrity_program": 300,
            "due_diligence": 250,
            "risk_assessment": 450,
            "internal_control": 650
        }
    
    def get_context_elements(
        self, 
        chunk: DocumentChunk,
        context_request: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get relevant context elements for a chunk."""
        elements = []
        content = chunk.content.lower()
        
        # Legal framework context
        elements.extend(self._get_framework_context(content, context_request))
        
        # Definition context
        elements.extend(self._get_definition_context(content))
        
        # Procedure context  
        elements.extend(self._get_procedure_context(content, context_request))
        
        # Compliance rule context
        elements.extend(self._get_compliance_context(content, context_request))
        
        # Apply distribution-based weighting
        elements = self._apply_distribution_weighting(elements)
        
        # Sort by relevance and return top elements
        elements.sort(key=lambda x: x.relevance_score, reverse=True)
        return elements[:10]  # Limit to top 10 most relevant
    
    def _get_framework_context(
        self,
        content: str,
        context_request: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get legal framework context."""
        elements = []
        
        for framework_id, framework_info in self.legal_frameworks.items():
            # Check for framework relevance
            relevance = 0.0
            
            # Direct mentions
            if framework_id.replace("_", " ") in content:
                relevance += 0.8
            
            # Key concept mentions
            concept_matches = 0
            for concept in framework_info.get("key_concepts", []):
                if concept in content:
                    concept_matches += 1
                    relevance += 0.1
            
            if relevance > 0.1:
                context_content = f"Marco Legal: {framework_info['title']}\n"
                context_content += f"Ámbito: {framework_info['scope']}\n"
                
                if "articles" in framework_info:
                    context_content += "Artículos relevantes:\n"
                    for art_num, art_title in framework_info["articles"].items():
                        context_content += f"- Art. {art_num}: {art_title}\n"
                
                element = ContextElement(
                    context_type=ContextType.LEGAL_FRAMEWORK,
                    content=context_content,
                    relevance_score=relevance,
                    source=framework_id,
                    metadata={
                        "framework": framework_id,
                        "concept_matches": concept_matches
                    }
                )
                elements.append(element)
        
        return elements
    
    def _get_definition_context(self, content: str) -> List[ContextElement]:
        """Get definition context for legal terms."""
        elements = []
        
        for term, definition in self.definitions.items():
            if term in content:
                # Calculate relevance based on term importance and context
                base_relevance = 0.6
                
                # Boost for specialized terms (lower frequency = higher importance)
                term_freq = self.term_distributions.get(term, 0.5)
                importance_boost = (1.0 - term_freq) * 0.3
                
                relevance = min(base_relevance + importance_boost, 1.0)
                
                context_content = f"Definición de '{term}': {definition}"
                
                element = ContextElement(
                    context_type=ContextType.DEFINITION,
                    content=context_content,
                    relevance_score=relevance,
                    source="legal_definitions",
                    metadata={"term": term, "frequency": term_freq}
                )
                elements.append(element)
        
        return elements
    
    def _get_procedure_context(
        self,
        content: str,
        context_request: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get procedure context."""
        elements = []
        
        # Check for procedure-related keywords
        procedure_keywords = [
            "implementación", "proceso", "procedimiento", "paso",
            "requirement", "requisito", "etapa"
        ]
        
        has_procedure_context = any(keyword in content for keyword in procedure_keywords)
        
        if has_procedure_context:
            for proc_id, proc_info in self.procedures.items():
                relevance = 0.4
                
                # Check for specific procedure mentions
                if any(step.lower() in content for step in proc_info.get("steps", [])):
                    relevance += 0.3
                
                if any(req.lower() in content for req in proc_info.get("requirements", [])):
                    relevance += 0.2
                
                if relevance > 0.5:
                    context_content = f"Procedimiento: {proc_id.replace('_', ' ').title()}\n"
                    
                    if "steps" in proc_info:
                        context_content += "Pasos:\n"
                        for step in proc_info["steps"]:
                            context_content += f"- {step}\n"
                    
                    if "requirements" in proc_info:
                        context_content += "Requisitos:\n"
                        for req in proc_info["requirements"]:
                            context_content += f"- {req}\n"
                    
                    element = ContextElement(
                        context_type=ContextType.PROCEDURE,
                        content=context_content,
                        relevance_score=relevance,
                        source=proc_id,
                        metadata={"procedure": proc_id}
                    )
                    elements.append(element)
        
        return elements
    
    def _get_compliance_context(
        self,
        content: str,
        context_request: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get compliance rule context."""
        elements = []
        
        compliance_keywords = [
            "cumplimiento", "compliance", "requisito", "obligación",
            "deber", "norma", "regla"
        ]
        
        has_compliance_context = any(keyword in content for keyword in compliance_keywords)
        
        if has_compliance_context:
            for rule_category, rules in self.compliance_rules.items():
                relevance = 0.3
                
                # Check for specific rule mentions
                matching_rules = [rule for rule in rules if any(
                    word in content for word in rule.lower().split()[:3]
                )]
                
                if matching_rules:
                    relevance += len(matching_rules) * 0.1
                    relevance = min(relevance, 1.0)
                    
                    context_content = f"Reglas de Cumplimiento - {rule_category.replace('_', ' ').title()}:\n"
                    for rule in matching_rules[:5]:  # Limit to top 5
                        context_content += f"- {rule}\n"
                    
                    element = ContextElement(
                        context_type=ContextType.COMPLIANCE_RULE,
                        content=context_content,
                        relevance_score=relevance,
                        source=rule_category,
                        metadata={
                            "rule_category": rule_category,
                            "matching_rules_count": len(matching_rules)
                        }
                    )
                    elements.append(element)
        
        return elements
    
    def _apply_distribution_weighting(
        self,
        elements: List[ContextElement]
    ) -> List[ContextElement]:
        """Apply distribution-based weighting as per academic methodology."""
        for element in elements:
            # Inverse cardinality weighting
            concept_type = element.context_type.value
            concept_freq = self.concept_frequencies.get(concept_type, 1000)
            
            # Inverse cardinality: less frequent = more important
            ic_weight = 1.0 / (1.0 + np.log(concept_freq))
            
            # Distribution-based localization
            # Boost elements that are contextually relevant but not overly common
            if element.context_type in [ContextType.DEFINITION, ContextType.PROCEDURE]:
                distribution_boost = ic_weight * 0.3
            else:
                distribution_boost = ic_weight * 0.1
            
            # Apply weighting
            element.relevance_score = min(
                element.relevance_score + distribution_boost,
                1.0
            )
        
        return elements


class ContextAugmenter:
    """
    Context augmenter implementing "LLMs for LLMs" methodology.
    
    Provides sophisticated context enrichment for legal documents using:
    - Distribution-based localization
    - Inverse cardinality weighting  
    - Legal domain-specific knowledge
    """
    
    def __init__(
        self,
        strategy: AugmentationStrategy = AugmentationStrategy.LEGAL_ENHANCED,
        legal_context: Optional[Any] = None,
        config: Optional[LexCertaintyConfig] = None
    ):
        """Initialize context augmenter."""
        self.strategy = strategy
        self.legal_context = legal_context
        self.config = config or LexCertaintyConfig()
        
        # Initialize knowledge base
        jurisdiction = getattr(legal_context, 'jurisdiction', 'argentina') if legal_context else 'argentina'
        self.knowledge_base = LegalKnowledgeBase(jurisdiction)
        
        # Augmentation statistics
        self.stats = {
            "chunks_augmented": 0,
            "context_elements_added": 0,
            "avg_relevance_score": 0.0,
            "augmentation_time": 0.0
        }
        
        logger.info(f"Initialized ContextAugmenter with {strategy.value} strategy")
    
    async def augment_chunk(
        self,
        chunk: DocumentChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Augment a document chunk with relevant context.
        
        Args:
            chunk: Document chunk to augment
            context: Additional context information
            
        Returns:
            Augmented DocumentChunk
        """
        import time
        start_time = time.time()
        
        try:
            if self.strategy == AugmentationStrategy.NONE:
                return chunk
            
            context = context or {}
            
            # Get relevant context elements
            context_elements = self.knowledge_base.get_context_elements(chunk, context)
            
            # Apply augmentation strategy
            if self.strategy == AugmentationStrategy.BASIC:
                augmented_content = await self._basic_augmentation(chunk, context_elements)
            elif self.strategy == AugmentationStrategy.LEGAL_ENHANCED:
                augmented_content = await self._legal_enhanced_augmentation(
                    chunk, context_elements, context
                )
            elif self.strategy == AugmentationStrategy.ACADEMIC:
                augmented_content = await self._academic_augmentation(
                    chunk, context_elements, context
                )
            elif self.strategy == AugmentationStrategy.ADAPTIVE:
                augmented_content = await self._adaptive_augmentation(
                    chunk, context_elements, context
                )
            else:
                augmented_content = chunk.content
            
            # Create augmented chunk
            augmented_chunk = DocumentChunk(
                id=f"{chunk.id}_augmented",
                content=augmented_content,
                start_position=chunk.start_position,
                end_position=chunk.end_position,
                chunk_type=chunk.chunk_type,
                legal_elements=chunk.legal_elements.copy(),
                semantic_score=chunk.semantic_score,
                parent_chunk=chunk.id,
                child_chunks=chunk.child_chunks.copy(),
                cross_references=chunk.cross_references.copy()
            )
            
            # Add augmentation metadata
            augmented_chunk.legal_elements["augmentation"] = {
                "strategy": self.strategy.value,
                "context_elements_count": len(context_elements),
                "avg_relevance": np.mean([e.relevance_score for e in context_elements]) if context_elements else 0.0,
                "processing_time": time.time() - start_time
            }
            
            # Update statistics
            self.stats["chunks_augmented"] += 1
            self.stats["context_elements_added"] += len(context_elements)
            if context_elements:
                current_avg = self.stats["avg_relevance_score"]
                new_avg = np.mean([e.relevance_score for e in context_elements])
                self.stats["avg_relevance_score"] = (current_avg + new_avg) / 2
            self.stats["augmentation_time"] += time.time() - start_time
            
            return augmented_chunk
            
        except Exception as e:
            logger.error(f"Augmentation failed for chunk {chunk.id}: {str(e)}")
            return chunk  # Return original chunk on failure
    
    async def _basic_augmentation(
        self,
        chunk: DocumentChunk,
        context_elements: List[ContextElement]
    ) -> str:
        """Basic context augmentation."""
        if not context_elements:
            return chunk.content
        
        # Simple context prepending
        context_text = "\n\n--- Contexto Legal Relevante ---\n"
        
        for element in context_elements[:3]:  # Limit to top 3
            if element.relevance_score > 0.5:
                context_text += f"\n{element.content}\n"
        
        context_text += "--- Fin del Contexto ---\n\n"
        
        return context_text + chunk.content
    
    async def _legal_enhanced_augmentation(
        self,
        chunk: DocumentChunk,
        context_elements: List[ContextElement],
        context: Dict[str, Any]
    ) -> str:
        """Legal-enhanced augmentation with structured context."""
        if not context_elements:
            return chunk.content
        
        # Categorize context elements
        categorized = defaultdict(list)
        for element in context_elements:
            if element.relevance_score > 0.4:
                categorized[element.context_type].append(element)
        
        # Build structured context
        context_sections = []
        
        # Legal frameworks
        if ContextType.LEGAL_FRAMEWORK in categorized:
            context_sections.append("=== MARCOS LEGALES APLICABLES ===")
            for element in categorized[ContextType.LEGAL_FRAMEWORK][:2]:
                context_sections.append(element.content)
        
        # Definitions
        if ContextType.DEFINITION in categorized:
            context_sections.append("\n=== DEFINICIONES LEGALES ===")
            for element in categorized[ContextType.DEFINITION][:3]:
                context_sections.append(element.content)
        
        # Procedures
        if ContextType.PROCEDURE in categorized:
            context_sections.append("\n=== PROCEDIMIENTOS RELEVANTES ===")
            for element in categorized[ContextType.PROCEDURE][:2]:
                context_sections.append(element.content)
        
        # Compliance rules
        if ContextType.COMPLIANCE_RULE in categorized:
            context_sections.append("\n=== REGLAS DE CUMPLIMIENTO ===")
            for element in categorized[ContextType.COMPLIANCE_RULE][:2]:
                context_sections.append(element.content)
        
        if context_sections:
            enhanced_context = "\n".join(context_sections)
            enhanced_context += "\n\n" + "="*50 + "\n"
            enhanced_context += "CONTENIDO DEL DOCUMENTO:\n"
            enhanced_context += "="*50 + "\n\n"
            
            return enhanced_context + chunk.content
        
        return chunk.content
    
    async def _academic_augmentation(
        self,
        chunk: DocumentChunk,
        context_elements: List[ContextElement],
        context: Dict[str, Any]
    ) -> str:
        """Academic methodology augmentation with distribution weighting."""
        if not context_elements:
            return chunk.content
        
        # Apply academic methodology principles:
        # 1. Distribution-based localization
        # 2. Inverse cardinality weighting
        # 3. Contextual relevance scoring
        
        # Filter high-relevance elements
        high_relevance = [e for e in context_elements if e.relevance_score > 0.6]
        medium_relevance = [e for e in context_elements if 0.4 <= e.relevance_score <= 0.6]
        
        context_text = "--- CONTEXTO ENRIQUECIDO (Metodología LLMs for LLMs) ---\n"
        
        # High-priority context (distribution-weighted)
        if high_relevance:
            context_text += "\n[CONTEXTO PRINCIPAL - Alta Relevancia]\n"
            for element in high_relevance[:2]:
                weight_indicator = f"(Peso: {element.relevance_score:.2f})"
                context_text += f"{element.content} {weight_indicator}\n\n"
        
        # Supporting context (inverse cardinality weighted)
        if medium_relevance:
            context_text += "[CONTEXTO COMPLEMENTARIO - Relevancia Media]\n"
            for element in medium_relevance[:2]:
                weight_indicator = f"(Peso: {element.relevance_score:.2f})"
                context_text += f"{element.content} {weight_indicator}\n\n"
        
        # Document type and jurisdiction context
        doc_type = context.get("document_type", "desconocido")
        jurisdiction = context.get("jurisdiction", "argentina")
        
        context_text += f"[METADATOS CONTEXTUALES]\n"
        context_text += f"Tipo de documento: {doc_type}\n"
        context_text += f"Jurisdicción: {jurisdiction}\n"
        
        if context.get("legal_frameworks"):
            frameworks = ", ".join(context["legal_frameworks"])
            context_text += f"Marcos legales identificados: {frameworks}\n"
        
        context_text += "\n--- FIN DEL CONTEXTO ---\n\n"
        
        return context_text + chunk.content
    
    async def _adaptive_augmentation(
        self,
        chunk: DocumentChunk,
        context_elements: List[ContextElement],
        context: Dict[str, Any]
    ) -> str:
        """Adaptive augmentation based on chunk characteristics."""
        if not context_elements:
            return chunk.content
        
        # Analyze chunk characteristics
        chunk_analysis = self._analyze_chunk_characteristics(chunk)
        
        # Determine optimal augmentation approach
        if chunk_analysis["legal_density"] > 0.7:
            # High legal density - use academic approach
            return await self._academic_augmentation(chunk, context_elements, context)
        elif chunk_analysis["complexity"] > 0.6:
            # High complexity - use legal enhanced approach
            return await self._legal_enhanced_augmentation(chunk, context_elements, context)
        else:
            # Standard content - use basic approach
            return await self._basic_augmentation(chunk, context_elements)
    
    def _analyze_chunk_characteristics(self, chunk: DocumentChunk) -> Dict[str, float]:
        """Analyze chunk characteristics for adaptive augmentation."""
        content = chunk.content.lower()
        
        # Legal term density
        legal_terms = [
            "artículo", "inciso", "cláusula", "ley", "decreto", "resolución",
            "compliance", "integridad", "responsabilidad", "obligación",
            "derecho", "sanción", "multa", "penalidad"
        ]
        
        words = content.split()
        legal_word_count = sum(1 for word in words if any(term in word for term in legal_terms))
        legal_density = legal_word_count / len(words) if words else 0.0
        
        # Complexity indicators
        complexity_indicators = [
            content.count('('),  # Parentheses
            content.count('['),  # Brackets
            len(re.findall(r'\d+[º°]?\.', content)),  # Numbered items
            content.count(';'),  # Semicolons
            content.count(':')   # Colons
        ]
        
        complexity = sum(complexity_indicators) / (len(words) / 10) if words else 0.0
        complexity = min(complexity, 1.0)
        
        # Reference density
        references = len(re.findall(
            r'\b(?:artículo|art\.?|ley|decreto)\s+\d+',
            content
        ))
        reference_density = references / (len(words) / 50) if words else 0.0
        reference_density = min(reference_density, 1.0)
        
        return {
            "legal_density": legal_density,
            "complexity": complexity,
            "reference_density": reference_density,
            "word_count": len(words)
        }
    
    async def batch_augment(
        self,
        chunks: List[DocumentChunk],
        global_context: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Batch augment multiple chunks with shared context optimization."""
        global_context = global_context or {}
        
        # Pre-calculate shared context elements for efficiency
        shared_elements = []
        if global_context.get("legal_frameworks"):
            for framework in global_context["legal_frameworks"]:
                shared_elements.extend(
                    self.knowledge_base._get_framework_context("", {"framework": framework})
                )
        
        augmented_chunks = []
        
        for chunk in chunks:
            # Combine chunk-specific and shared context
            chunk_elements = self.knowledge_base.get_context_elements(chunk, global_context)
            
            # Merge and deduplicate
            all_elements = chunk_elements + shared_elements
            unique_elements = []
            seen_content = set()
            
            for element in all_elements:
                content_hash = hashlib.md5(element.content.encode()).hexdigest()
                if content_hash not in seen_content:
                    unique_elements.append(element)
                    seen_content.add(content_hash)
            
            # Sort by relevance and augment
            unique_elements.sort(key=lambda x: x.relevance_score, reverse=True)
            augmented_chunk = await self.augment_chunk(chunk, global_context)
            augmented_chunks.append(augmented_chunk)
        
        return augmented_chunks
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset augmentation statistics."""
        self.stats = {
            "chunks_augmented": 0,
            "context_elements_added": 0,
            "avg_relevance_score": 0.0,
            "augmentation_time": 0.0
        }