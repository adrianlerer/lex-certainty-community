"""
Legal Context Framework

Provides legal context and jurisdiction-specific knowledge for document
processing and compliance analysis, with specialized implementation for
Argentine legal framework including Ley 27401.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from datetime import datetime, date

import numpy as np
from pydantic import BaseModel, Field

from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANCE = "partial_compliance"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_INFO = "insufficient_info"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceIssue:
    """Individual compliance issue."""
    issue_type: str
    description: str
    severity: RiskLevel
    framework: str
    article_reference: Optional[str] = None
    remediation: Optional[str] = None
    deadline: Optional[date] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Comprehensive compliance assessment result."""
    overall_status: ComplianceStatus
    risk_level: RiskLevel
    compliance_score: float  # 0.0 to 1.0
    issues: List[ComplianceIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    framework_coverage: Dict[str, float] = field(default_factory=dict)
    assessment_metadata: Dict[str, Any] = field(default_factory=dict)


class LegalContext(ABC):
    """
    Abstract base class for legal context providers.
    
    Defines interface for jurisdiction-specific legal knowledge
    and compliance assessment capabilities.
    """
    
    def __init__(
        self,
        jurisdiction: str,
        language: str = "es",
        config: Optional[LexCertaintyConfig] = None
    ):
        """Initialize legal context."""
        self.jurisdiction = jurisdiction
        self.language = language
        self.config = config or LexCertaintyConfig()
        
    @abstractmethod
    async def assess_compliance(
        self,
        document_type: str,
        content_chunks: List[str],
        legal_frameworks: Optional[List[str]] = None
    ) -> ComplianceAssessment:
        """Assess compliance for document content."""
        pass
    
    @abstractmethod
    def get_legal_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Get available legal frameworks for jurisdiction."""
        pass
    
    @abstractmethod
    def get_compliance_requirements(self, framework: str) -> List[Dict[str, Any]]:
        """Get compliance requirements for specific framework."""
        pass
    
    @abstractmethod
    def classify_document_risk(
        self,
        document_type: str,
        content: str
    ) -> Tuple[RiskLevel, List[str]]:
        """Classify document risk level and identify risk factors."""
        pass


class ArgentineLegalContext(LegalContext):
    """
    Argentine legal context implementation with comprehensive support
    for Argentine legal framework, especially Ley 27401 (Corporate Criminal Liability).
    """
    
    def __init__(self, config: Optional[LexCertaintyConfig] = None):
        """Initialize Argentine legal context."""
        super().__init__(jurisdiction="argentina", language="es", config=config)
        
        self.legal_frameworks = self._initialize_frameworks()
        self.compliance_matrices = self._initialize_compliance_matrices()
        self.risk_indicators = self._initialize_risk_indicators()
        
        logger.info("Initialized Argentine legal context with Ley 27401 support")
    
    def _initialize_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Argentine legal frameworks."""
        return {
            "ley_27401": {
                "title": "Ley 27401 - Régimen de Responsabilidad Penal Empresaria",
                "enacted": "2017-12-01",
                "scope": "Corporate criminal liability for legal entities",
                "key_articles": {
                    "1": {
                        "title": "Objeto",
                        "content": "Establecer el régimen de responsabilidad penal aplicable a las personas jurídicas privadas",
                        "keywords": ["responsabilidad penal", "personas jurídicas", "régimen"]
                    },
                    "7": {
                        "title": "Eximición de responsabilidad",  
                        "content": "La persona jurídica quedará exenta de responsabilidad cuando hubiere implementado un programa de integridad",
                        "keywords": ["programa de integridad", "eximición", "responsabilidad"]
                    },
                    "8": {
                        "title": "Elementos del programa de integridad",
                        "content": "El programa de integridad deberá contemplar los elementos establecidos en el artículo",
                        "keywords": ["elementos", "programa", "integridad", "requisitos"]
                    },
                    "9": {
                        "title": "Responsable del programa",
                        "content": "Designación de un responsable del programa de integridad",
                        "keywords": ["responsable", "designación", "programa"]
                    }
                },
                "penalties": {
                    "multa": "De 2 a 5 veces del beneficio indebido obtenido",
                    "suspension": "Suspensión total o parcial de actividades hasta 10 años",
                    "cancelacion": "Cancelación de la personería jurídica",
                    "prohibicion": "Prohibición de contratar con el Estado"
                },
                "applicable_crimes": [
                    "cohecho nacional",
                    "cohecho transnacional", 
                    "tráfico de influencias",
                    "negociaciones incompatibles",
                    "concusión",
                    "enriquecimiento ilícito",
                    "balances e informes falsos"
                ]
            },
            "codigo_civil_comercial": {
                "title": "Código Civil y Comercial de la Nación",
                "enacted": "2015-08-01",
                "scope": "Civil and commercial law framework",
                "key_sections": {
                    "personas_juridicas": "Libro Primero - Parte General - Título II",
                    "contratos": "Libro Tercero - Derechos Personales - Título II",
                    "responsabilidad": "Libro Tercero - Derechos Personales - Título V"
                }
            },
            "codigo_penal": {
                "title": "Código Penal de la Nación",
                "scope": "Criminal law framework",
                "relevant_sections": {
                    "delitos_funcionarios": "Título XI - Capítulo VI",
                    "delitos_administracion": "Título XII"
                }
            },
            "ley_sociedades": {
                "title": "Ley General de Sociedades 19.550",
                "scope": "Corporate governance and company law",
                "key_aspects": [
                    "corporate governance",
                    "director responsibilities", 
                    "shareholders rights",
                    "corporate compliance"
                ]
            }
        }
    
    def _initialize_compliance_matrices(self) -> Dict[str, Dict[str, Any]]:
        """Initialize compliance requirement matrices."""
        return {
            "ley_27401_programa_integridad": {
                "requirements": [
                    {
                        "id": "mapeo_riesgos",
                        "title": "Mapeo de procesos y identificación de riesgos",
                        "description": "Identificación y análisis de riesgos específicos de corrupción",
                        "mandatory": True,
                        "verification_criteria": [
                            "Matriz de riesgos documentada",
                            "Mapeo de procesos críticos",
                            "Identificación de controles",
                            "Evaluación periódica de riesgos"
                        ],
                        "evidence_required": [
                            "Matriz de riesgos",
                            "Documentación de procesos",
                            "Reportes de evaluación"
                        ]
                    },
                    {
                        "id": "codigo_etica",
                        "title": "Desarrollo e implementación de código de ética",
                        "description": "Código de ética y conducta que incluya políticas anti-corrupción",
                        "mandatory": True,
                        "verification_criteria": [
                            "Código de ética documentado",
                            "Políticas anti-corrupción específicas",
                            "Procedimientos de escalamiento",
                            "Comunicación a toda la organización"
                        ],
                        "evidence_required": [
                            "Código de ética aprobado",
                            "Registros de comunicación",
                            "Confirmaciones de recepción"
                        ]
                    },
                    {
                        "id": "politicas_procedimientos",
                        "title": "Implementación de políticas y procedimientos",
                        "description": "Políticas específicas para prevención de corrupción",
                        "mandatory": True,
                        "verification_criteria": [
                            "Políticas documentadas",
                            "Procedimientos operativos",
                            "Controles internos",
                            "Segregación de funciones"
                        ]
                    },
                    {
                        "id": "evaluacion_terceros", 
                        "title": "Identificación y evaluación de terceros",
                        "description": "Due diligence de socios comerciales y terceros",
                        "mandatory": True,
                        "verification_criteria": [
                            "Procedimiento de due diligence",
                            "Base de datos de terceros",
                            "Evaluaciones periódicas",
                            "Criterios de aprobación"
                        ]
                    },
                    {
                        "id": "capacitacion",
                        "title": "Capacitación periódica en integridad",
                        "description": "Programa de capacitación en ética y anti-corrupción",
                        "mandatory": True,
                        "verification_criteria": [
                            "Plan de capacitación documentado",
                            "Material de capacitación",
                            "Registros de asistencia",
                            "Evaluaciones de conocimiento"
                        ]
                    },
                    {
                        "id": "canal_denuncias",
                        "title": "Canales internos de denuncia",
                        "description": "Mecanismos seguros y confidenciales para denuncias",
                        "mandatory": True,
                        "verification_criteria": [
                            "Canal de denuncias disponible",
                            "Procedimiento de gestión",
                            "Protección del denunciante",
                            "Registro de denuncias"
                        ]
                    },
                    {
                        "id": "proteccion_denunciante",
                        "title": "Protección del denunciante",
                        "description": "Medidas de protección contra represalias",
                        "mandatory": True,
                        "verification_criteria": [
                            "Política de no represalias",
                            "Mecanismos de protección",
                            "Investigación de represalias",
                            "Sanciones por represalias"
                        ]
                    },
                    {
                        "id": "monitoreo_evaluacion",
                        "title": "Evaluación de riesgos y monitoreo",
                        "description": "Sistema de monitoreo continuo del programa",
                        "mandatory": True,
                        "verification_criteria": [
                            "Sistema de monitoreo",
                            "Indicadores de desempeño",
                            "Reportes periódicos",
                            "Revisión por dirección"
                        ]
                    },
                    {
                        "id": "investigacion_interna",
                        "title": "Investigación interna y debida diligencia",
                        "description": "Procedimientos de investigación interna",
                        "mandatory": True,
                        "verification_criteria": [
                            "Procedimiento de investigación",
                            "Equipo de investigación",
                            "Documentación de casos",
                            "Medidas correctivas"
                        ]
                    },
                    {
                        "id": "mejora_continua",
                        "title": "Verificación y mejora continua",
                        "description": "Proceso de mejora continua del programa",
                        "mandatory": True,
                        "verification_criteria": [
                            "Auditorías internas",
                            "Revisiones periódicas",
                            "Planes de mejora",
                            "Actualización de políticas"
                        ]
                    }
                ]
            }
        }
    
    def _initialize_risk_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize risk indicators for different contexts."""
        return {
            "corruption_indicators": [
                {
                    "keyword": "soborno",
                    "weight": 0.9,
                    "context": "direct corruption reference",
                    "risk_level": "critical"
                },
                {
                    "keyword": "coima",
                    "weight": 0.9, 
                    "context": "direct corruption reference (Argentine slang)",
                    "risk_level": "critical"
                },
                {
                    "keyword": "facilitating payment",
                    "weight": 0.8,
                    "context": "facilitation payments",
                    "risk_level": "high"
                },
                {
                    "keyword": "kickback",
                    "weight": 0.8,
                    "context": "kickback schemes",
                    "risk_level": "high"
                },
                {
                    "keyword": "comisión irregular",
                    "weight": 0.7,
                    "context": "irregular commissions",
                    "risk_level": "high"
                }
            ],
            "governance_indicators": [
                {
                    "keyword": "conflicto de interés",
                    "weight": 0.6,
                    "context": "conflict of interest",
                    "risk_level": "medium"
                },
                {
                    "keyword": "due diligence",
                    "weight": -0.3,  # Negative weight indicates good practice
                    "context": "risk mitigation",
                    "risk_level": "low"
                },
                {
                    "keyword": "programa de integridad",
                    "weight": -0.4,
                    "context": "compliance program",
                    "risk_level": "low"
                }
            ],
            "regulatory_indicators": [
                {
                    "keyword": "sanción",
                    "weight": 0.5,
                    "context": "regulatory penalties",
                    "risk_level": "medium"
                },
                {
                    "keyword": "incumplimiento",
                    "weight": 0.6,
                    "context": "non-compliance",
                    "risk_level": "medium"
                },
                {
                    "keyword": "violación",
                    "weight": 0.7,
                    "context": "violation",
                    "risk_level": "high"
                }
            ]
        }
    
    async def assess_compliance(
        self,
        document_type: str,
        content_chunks: List[str],
        legal_frameworks: Optional[List[str]] = None
    ) -> ComplianceAssessment:
        """
        Assess compliance for document content against Argentine legal frameworks.
        
        Args:
            document_type: Type of document being assessed
            content_chunks: List of document content chunks
            legal_frameworks: Specific frameworks to assess against
            
        Returns:
            ComplianceAssessment with detailed analysis
        """
        # Combine all content for analysis
        full_content = " ".join(content_chunks)
        
        # Determine applicable frameworks
        if legal_frameworks is None:
            legal_frameworks = self._determine_applicable_frameworks(document_type, full_content)
        
        issues = []
        recommendations = []
        framework_coverage = {}
        
        # Assess each framework
        for framework in legal_frameworks:
            if framework == "ley_27401":
                framework_issues, coverage = await self._assess_ley_27401_compliance(
                    full_content, document_type
                )
                issues.extend(framework_issues)
                framework_coverage[framework] = coverage
                
                # Add Ley 27401 specific recommendations
                if coverage < 0.8:
                    recommendations.extend(self._get_ley_27401_recommendations(coverage))
            
            elif framework in self.legal_frameworks:
                # Generic framework assessment
                framework_issues, coverage = await self._assess_generic_framework(
                    framework, full_content, document_type
                )
                issues.extend(framework_issues)
                framework_coverage[framework] = coverage
        
        # Calculate overall metrics
        overall_status = self._determine_overall_status(issues, framework_coverage)
        risk_level = self._calculate_risk_level(issues)
        compliance_score = self._calculate_compliance_score(framework_coverage, issues)
        
        # Add general recommendations
        recommendations.extend(self._get_general_recommendations(
            document_type, issues, framework_coverage
        ))
        
        return ComplianceAssessment(
            overall_status=overall_status,
            risk_level=risk_level,
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations,
            framework_coverage=framework_coverage,
            assessment_metadata={
                "document_type": document_type,
                "frameworks_assessed": legal_frameworks,
                "content_length": len(full_content),
                "chunks_analyzed": len(content_chunks),
                "assessment_timestamp": datetime.now().isoformat()
            }
        )
    
    async def _assess_ley_27401_compliance(
        self,
        content: str,
        document_type: str
    ) -> Tuple[List[ComplianceIssue], float]:
        """Assess compliance specifically against Ley 27401."""
        issues = []
        requirements = self.compliance_matrices["ley_27401_programa_integridad"]["requirements"]
        
        content_lower = content.lower()
        
        # Track requirement coverage
        covered_requirements = 0
        total_requirements = len(requirements)
        
        for req in requirements:
            req_id = req["id"]
            req_title = req["title"]
            
            # Check for requirement coverage in content
            coverage_score = 0.0
            
            # Check verification criteria
            criteria_met = 0
            for criterion in req["verification_criteria"]:
                if self._check_criterion_coverage(criterion, content_lower):
                    criteria_met += 1
            
            coverage_score = criteria_met / len(req["verification_criteria"])
            
            if coverage_score >= 0.7:
                covered_requirements += 1
            elif coverage_score >= 0.3:
                # Partial compliance issue
                issues.append(ComplianceIssue(
                    issue_type="partial_compliance",
                    description=f"Cumplimiento parcial del requisito: {req_title}",
                    severity=RiskLevel.MEDIUM,
                    framework="ley_27401",
                    article_reference="Art. 8",
                    remediation=f"Completar implementación de: {req_title}",
                    metadata={
                        "requirement_id": req_id,
                        "coverage_score": coverage_score,
                        "missing_criteria": len(req["verification_criteria"]) - criteria_met
                    }
                ))
            else:
                # Non-compliance issue
                issues.append(ComplianceIssue(
                    issue_type="non_compliance",
                    description=f"Requisito no cumplido: {req_title}",
                    severity=RiskLevel.HIGH,
                    framework="ley_27401",
                    article_reference="Art. 8",
                    remediation=f"Implementar completamente: {req_title}",
                    metadata={
                        "requirement_id": req_id,
                        "coverage_score": coverage_score
                    }
                ))
        
        # Overall coverage score
        coverage = covered_requirements / total_requirements
        
        # Check for specific corruption risks
        corruption_issues = self._detect_corruption_risks(content_lower)
        issues.extend(corruption_issues)
        
        return issues, coverage
    
    def _check_criterion_coverage(self, criterion: str, content: str) -> bool:
        """Check if a specific criterion is covered in the content."""
        criterion_lower = criterion.lower()
        
        # Define keyword mappings for criteria
        criterion_keywords = {
            "matriz de riesgos": ["matriz", "riesgo", "evaluación"],
            "código de ética": ["código", "ética", "conducta"],
            "procedimiento": ["procedimiento", "proceso", "metodología"],
            "capacitación": ["capacitación", "entrenamiento", "formación"],
            "canal de denuncias": ["canal", "denuncia", "reporte"],
            "due diligence": ["due diligence", "debida diligencia", "verificación"],
            "monitoreo": ["monitoreo", "seguimiento", "supervisión"],
            "auditoría": ["auditoría", "revisión", "evaluación"]
        }
        
        # Check for direct mention
        if criterion_lower in content:
            return True
        
        # Check for keyword coverage
        for key_phrase, keywords in criterion_keywords.items():
            if key_phrase in criterion_lower:
                keyword_matches = sum(1 for kw in keywords if kw in content)
                if keyword_matches >= 2:  # Require at least 2 related keywords
                    return True
        
        return False
    
    def _detect_corruption_risks(self, content: str) -> List[ComplianceIssue]:
        """Detect corruption-related risks in content."""
        issues = []
        
        for indicator_category, indicators in self.risk_indicators.items():
            for indicator in indicators:
                keyword = indicator["keyword"]
                weight = indicator["weight"]
                
                if keyword in content and weight > 0:  # Positive weight indicates risk
                    severity = RiskLevel.CRITICAL if weight >= 0.8 else (
                        RiskLevel.HIGH if weight >= 0.6 else RiskLevel.MEDIUM
                    )
                    
                    issues.append(ComplianceIssue(
                        issue_type="corruption_risk",
                        description=f"Indicador de riesgo detectado: {keyword}",
                        severity=severity,
                        framework="ley_27401", 
                        remediation="Revisar contexto y implementar controles adicionales",
                        metadata={
                            "indicator": keyword,
                            "weight": weight,
                            "context": indicator["context"]
                        }
                    ))
        
        return issues
    
    async def _assess_generic_framework(
        self,
        framework: str,
        content: str,
        document_type: str
    ) -> Tuple[List[ComplianceIssue], float]:
        """Assess compliance against generic framework."""
        issues = []
        
        if framework not in self.legal_frameworks:
            return issues, 0.0
        
        framework_info = self.legal_frameworks[framework]
        content_lower = content.lower()
        
        # Basic assessment based on framework coverage
        coverage_score = 0.0
        
        if "key_articles" in framework_info:
            article_matches = 0
            total_articles = len(framework_info["key_articles"])
            
            for article_num, article_info in framework_info["key_articles"].items():
                keywords = article_info.get("keywords", [])
                keyword_matches = sum(1 for kw in keywords if kw in content_lower)
                
                if keyword_matches > 0:
                    article_matches += 1
            
            coverage_score = article_matches / total_articles if total_articles > 0 else 0.0
        
        # Generate generic compliance issues for low coverage
        if coverage_score < 0.5:
            issues.append(ComplianceIssue(
                issue_type="insufficient_coverage",
                description=f"Cobertura insuficiente del marco legal: {framework_info['title']}",
                severity=RiskLevel.MEDIUM,
                framework=framework,
                remediation=f"Revisar cumplimiento de {framework_info['title']}"
            ))
        
        return issues, coverage_score
    
    def _determine_applicable_frameworks(
        self,
        document_type: str,
        content: str
    ) -> List[str]:
        """Determine which legal frameworks are applicable."""
        applicable = []
        content_lower = content.lower()
        
        # Always consider basic frameworks
        applicable.extend(["codigo_civil_comercial", "codigo_penal"])
        
        # Check for specific framework indicators
        if any(keyword in content_lower for keyword in [
            "responsabilidad penal", "programa de integridad", "ley 27401",
            "corporate liability", "compliance program"
        ]):
            applicable.append("ley_27401")
        
        if any(keyword in content_lower for keyword in [
            "sociedad", "directorio", "accionista", "governance"
        ]):
            applicable.append("ley_sociedades")
        
        # Document type specific frameworks
        if document_type in ["compliance_report", "policy"]:
            if "ley_27401" not in applicable:
                applicable.append("ley_27401")
        
        return applicable
    
    def _determine_overall_status(
        self,
        issues: List[ComplianceIssue],
        framework_coverage: Dict[str, float]
    ) -> ComplianceStatus:
        """Determine overall compliance status."""
        if not issues:
            return ComplianceStatus.COMPLIANT
        
        critical_issues = [i for i in issues if i.severity == RiskLevel.CRITICAL]
        high_issues = [i for i in issues if i.severity == RiskLevel.HIGH]
        
        if critical_issues:
            return ComplianceStatus.NON_COMPLIANT
        
        if high_issues or any(coverage < 0.5 for coverage in framework_coverage.values()):
            return ComplianceStatus.PARTIAL_COMPLIANCE
        
        if any(coverage < 0.8 for coverage in framework_coverage.values()):
            return ComplianceStatus.REQUIRES_REVIEW
        
        return ComplianceStatus.COMPLIANT
    
    def _calculate_risk_level(self, issues: List[ComplianceIssue]) -> RiskLevel:
        """Calculate overall risk level from issues."""
        if not issues:
            return RiskLevel.LOW
        
        max_severity = max(issue.severity for issue in issues)
        return max_severity
    
    def _calculate_compliance_score(
        self,
        framework_coverage: Dict[str, float],
        issues: List[ComplianceIssue]
    ) -> float:
        """Calculate overall compliance score (0.0 to 1.0)."""
        if not framework_coverage:
            return 0.0
        
        # Base score from framework coverage
        coverage_score = np.mean(list(framework_coverage.values()))
        
        # Penalty for issues
        issue_penalty = 0.0
        for issue in issues:
            if issue.severity == RiskLevel.CRITICAL:
                issue_penalty += 0.3
            elif issue.severity == RiskLevel.HIGH:
                issue_penalty += 0.2
            elif issue.severity == RiskLevel.MEDIUM:
                issue_penalty += 0.1
            else:
                issue_penalty += 0.05
        
        # Final score with penalties
        final_score = max(0.0, coverage_score - issue_penalty)
        return min(final_score, 1.0)
    
    def _get_ley_27401_recommendations(self, coverage: float) -> List[str]:
        """Get specific recommendations for Ley 27401 compliance."""
        recommendations = []
        
        if coverage < 0.3:
            recommendations.extend([
                "Implementar un programa de integridad completo según Art. 8 de Ley 27401",
                "Designar un responsable del programa de integridad (Art. 9)",
                "Desarrollar matriz de riesgos específica para prevención de corrupción"
            ])
        elif coverage < 0.6:
            recommendations.extend([
                "Completar elementos faltantes del programa de integridad",
                "Reforzar capacitación en ética y anti-corrupción",
                "Mejorar sistema de monitoreo y evaluación continua"
            ])
        elif coverage < 0.8:
            recommendations.extend([
                "Optimizar procedimientos de due diligence de terceros",
                "Fortalecer canal de denuncias y protección del denunciante",
                "Implementar auditorías periódicas del programa"
            ])
        
        return recommendations
    
    def _get_general_recommendations(
        self,
        document_type: str,
        issues: List[ComplianceIssue],
        framework_coverage: Dict[str, float]
    ) -> List[str]:
        """Get general compliance recommendations."""
        recommendations = []
        
        # Document type specific recommendations
        if document_type == "policy":
            recommendations.append(
                "Asegurar que las políticas estén alineadas con marcos legales vigentes"
            )
        elif document_type == "compliance_report":
            recommendations.append(
                "Incluir análisis de gaps y plan de remediación en reportes de cumplimiento"
            )
        
        # Issue-based recommendations
        if any(issue.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH] for issue in issues):
            recommendations.append(
                "Priorizar la resolución inmediata de issues críticos y de alto riesgo"
            )
        
        # Coverage-based recommendations
        low_coverage_frameworks = [
            fw for fw, coverage in framework_coverage.items() if coverage < 0.6
        ]
        
        if low_coverage_frameworks:
            recommendations.append(
                f"Mejorar cobertura en marcos legales: {', '.join(low_coverage_frameworks)}"
            )
        
        return recommendations
    
    def get_legal_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Get available legal frameworks for Argentina."""
        return self.legal_frameworks.copy()
    
    def get_compliance_requirements(self, framework: str) -> List[Dict[str, Any]]:
        """Get compliance requirements for specific framework."""
        if framework == "ley_27401":
            return self.compliance_matrices["ley_27401_programa_integridad"]["requirements"]
        
        return []
    
    def classify_document_risk(
        self,
        document_type: str,
        content: str
    ) -> Tuple[RiskLevel, List[str]]:
        """Classify document risk level and identify risk factors."""
        content_lower = content.lower()
        risk_factors = []
        risk_scores = []
        
        # Analyze risk indicators
        for category, indicators in self.risk_indicators.items():
            category_score = 0.0
            category_factors = []
            
            for indicator in indicators:
                keyword = indicator["keyword"]
                weight = indicator["weight"]
                
                if keyword in content_lower:
                    category_score += abs(weight)  # Use absolute value for risk scoring
                    if weight > 0:  # Positive weight indicates actual risk
                        category_factors.append(f"{keyword} ({indicator['context']})")
            
            if category_factors:
                risk_factors.extend(category_factors)
                risk_scores.append(category_score)
        
        # Calculate overall risk level
        if not risk_scores:
            return RiskLevel.LOW, risk_factors
        
        max_risk_score = max(risk_scores)
        
        if max_risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif max_risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif max_risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return risk_level, risk_factors