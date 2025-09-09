"""
Legal Prompt Engineering System

Implements structured prompting for legal AI with domain-specific templates
and intelligent prompt construction based on document type and context.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from pydantic import BaseModel, Field

from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of legal prompts."""
    ANALYSIS = "analysis"
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    DOCUMENT_REVIEW = "document_review"
    CONTRACT_ANALYSIS = "contract_analysis"
    REGULATORY_CHECK = "regulatory_check"
    DUE_DILIGENCE = "due_diligence"
    ABSTENTION_DECISION = "abstention_decision"


class PromptComplexity(Enum):
    """Complexity levels for prompts."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class PromptTemplate:
    """Legal prompt template structure."""
    name: str
    prompt_type: PromptType
    complexity: PromptComplexity
    template: str
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    domain_specific: bool = True
    language: str = "es"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptContext:
    """Context for prompt generation."""
    document_type: str
    jurisdiction: str = "argentina"
    legal_frameworks: List[str] = field(default_factory=list)
    user_role: str = "legal_analyst"
    analysis_depth: str = "standard"
    specific_requirements: List[str] = field(default_factory=list)
    risk_tolerance: str = "medium"
    language: str = "es"
    custom_context: Dict[str, Any] = field(default_factory=dict)


class LegalPromptEngine:
    """
    Legal prompt engineering system with structured templates and
    intelligent prompt construction for legal domain applications.
    """
    
    def __init__(
        self,
        default_language: str = "es",
        jurisdiction: str = "argentina",
        config: Optional[LexCertaintyConfig] = None
    ):
        """Initialize legal prompt engine."""
        self.default_language = default_language
        self.jurisdiction = jurisdiction
        self.config = config or LexCertaintyConfig()
        
        # Initialize prompt templates
        self.templates = self._initialize_templates()
        
        # Prompt generation statistics
        self.stats = {
            "prompts_generated": 0,
            "templates_used": {},
            "avg_prompt_length": 0,
            "complexity_distribution": {level.value: 0 for level in PromptComplexity}
        }
        
        logger.info(f"Initialized LegalPromptEngine for {jurisdiction} jurisdiction")
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize legal prompt templates."""
        templates = {}
        
        # Document Analysis Templates
        templates["legal_document_analysis"] = PromptTemplate(
            name="legal_document_analysis",
            prompt_type=PromptType.ANALYSIS,
            complexity=PromptComplexity.INTERMEDIATE,
            template="""
Eres un experto analista legal especializado en el sistema jurídico argentino. Tu tarea es realizar un análisis exhaustivo del siguiente documento legal.

**CONTEXTO JURÍDICO:**
- Jurisdicción: {jurisdiction}
- Tipo de documento: {document_type}
- Marcos legales aplicables: {legal_frameworks}

**INSTRUCCIONES DE ANÁLISIS:**

1. **Estructura y Forma:**
   - Identificar el tipo y naturaleza del documento
   - Evaluar la estructura formal y completitud
   - Verificar elementos obligatorios según normativa aplicable

2. **Contenido Sustantivo:**
   - Analizar cláusulas, artículos o disposiciones principales
   - Identificar derechos, obligaciones y responsabilidades
   - Detectar posibles inconsistencias o ambigüedades

3. **Cumplimiento Legal:**
   - Verificar conformidad con marcos legales identificados
   - Identificar riesgos de incumplimiento
   - Evaluar necesidad de cláusulas o elementos adicionales

4. **Análisis de Riesgos:**
   - Identificar riesgos legales, regulatorios y operacionales
   - Evaluar exposición a responsabilidad civil o penal
   - Detectar indicadores de riesgo de corrupción (si aplica)

**DOCUMENTO A ANALIZAR:**
{document_content}

**FORMATO DE RESPUESTA:**
Proporciona un análisis estructurado en formato JSON con las siguientes secciones:
- resumen_ejecutivo
- analisis_formal
- analisis_sustantivo
- cumplimiento_legal
- riesgos_identificados
- recomendaciones
- nivel_confianza (0-1)

Sé preciso, objetivo y fundamenta tus conclusiones en el marco legal aplicable.
""",
            required_context=["jurisdiction", "document_type", "document_content"],
            optional_context=["legal_frameworks"],
            output_format="json",
            language="es"
        )
        
        # Compliance Assessment Template
        templates["compliance_assessment"] = PromptTemplate(
            name="compliance_assessment",
            prompt_type=PromptType.COMPLIANCE,
            complexity=PromptComplexity.ADVANCED,
            template="""
Eres un especialista en cumplimiento normativo con expertise en la legislación argentina, especialmente en Ley 27401 de Responsabilidad Penal Empresaria.

**CONTEXTO DE EVALUACIÓN:**
- Marco legal: {legal_framework}
- Tipo de evaluación: {assessment_type}
- Entidad evaluada: {entity_type}

**CRITERIOS DE EVALUACIÓN LEY 27401:**

1. **Programa de Integridad (Art. 8):**
   - Mapeo de procesos y identificación de riesgos
   - Desarrollo e implementación de código de ética
   - Implementación de políticas y procedimientos
   - Identificación y evaluación de terceros
   - Capacitación periódica en integridad
   - Canales internos de denuncia
   - Protección del denunciante
   - Evaluación de riesgos y monitoreo
   - Investigación interna y debida diligencia
   - Verificación y mejora continua

2. **Elementos de Evaluación:**
   - Documentación existente
   - Implementación efectiva
   - Monitoreo y supervisión
   - Capacitación y comunicación
   - Investigación y remediation

**CONTENIDO A EVALUAR:**
{content}

**INSTRUCCIONES:**
1. Evalúa cada elemento del programa de integridad
2. Identifica gaps y deficiencias
3. Califica el nivel de implementación (0-100%)
4. Proporciona recomendaciones específicas y priorizadas
5. Estima el riesgo residual

**FORMATO DE RESPUESTA:**
```json
{
  "evaluacion_general": {
    "puntuacion_total": number,
    "nivel_implementacion": "string",
    "riesgo_residual": "bajo|medio|alto|critico"
  },
  "elementos_evaluados": [
    {
      "elemento": "string",
      "puntuacion": number,
      "estado": "implementado|parcial|faltante",
      "observaciones": "string"
    }
  ],
  "gaps_identificados": ["string"],
  "recomendaciones": [
    {
      "prioridad": "alta|media|baja",
      "descripcion": "string",
      "plazo_sugerido": "string"
    }
  ],
  "nivel_confianza": number
}
```
""",
            required_context=["legal_framework", "content"],
            optional_context=["assessment_type", "entity_type"],
            output_format="json",
            language="es"
        )
        
        # Risk Assessment Template
        templates["risk_assessment"] = PromptTemplate(
            name="risk_assessment",
            prompt_type=PromptType.RISK_ASSESSMENT,
            complexity=PromptComplexity.EXPERT,
            template="""
Eres un especialista en evaluación de riesgos legales y de cumplimiento con experiencia en el contexto regulatorio argentino.

**PARÁMETROS DE EVALUACIÓN:**
- Tipo de riesgo: {risk_type}
- Contexto: {risk_context}
- Tolerancia al riesgo: {risk_tolerance}

**METODOLOGÍA DE EVALUACIÓN:**

1. **Identificación de Riesgos:**
   - Riesgos legales y regulatorios
   - Riesgos reputacionales
   - Riesgos operacionales
   - Riesgos de terceros

2. **Análisis de Probabilidad:**
   - Factores de riesgo presentes
   - Controles existentes
   - Experiencia previa en el sector

3. **Análisis de Impacto:**
   - Consecuencias legales (sanciones, multas)
   - Impacto reputacional
   - Costos operacionales
   - Interrupción del negocio

4. **Matriz de Riesgo:**
   - Probabilidad × Impacto
   - Clasificación: Bajo, Medio, Alto, Crítico

**INFORMACIÓN PARA EVALUAR:**
{content}

**FACTORES ESPECÍFICOS A CONSIDERAR:**
{risk_factors}

**FORMATO DE RESPUESTA:**
Proporciona una evaluación estructurada que incluya:

```json
{
  "resumen_riesgos": {
    "nivel_riesgo_general": "bajo|medio|alto|critico",
    "riesgos_criticos": number,
    "riesgos_altos": number,
    "puntuacion_riesgo": number
  },
  "riesgos_identificados": [
    {
      "id": "string",
      "categoria": "legal|regulatorio|reputacional|operacional",
      "descripcion": "string",
      "probabilidad": number,
      "impacto": number,
      "nivel_riesgo": "bajo|medio|alto|critico",
      "controles_existentes": ["string"],
      "controles_recomendados": ["string"]
    }
  ],
  "plan_mitigacion": [
    {
      "riesgo_id": "string",
      "acciones": ["string"],
      "responsable": "string",
      "plazo": "string",
      "costo_estimado": "string"
    }
  ],
  "monitoreo": {
    "indicadores": ["string"],
    "frecuencia_revision": "string"
  },
  "nivel_confianza": number
}
```

Fundamenta tu evaluación en marcos regulatorios vigentes y mejores prácticas del sector.
""",
            required_context=["content", "risk_type"],
            optional_context=["risk_context", "risk_tolerance", "risk_factors"],
            output_format="json",
            language="es"
        )
        
        # Contract Analysis Template
        templates["contract_analysis"] = PromptTemplate(
            name="contract_analysis",
            prompt_type=PromptType.CONTRACT_ANALYSIS,
            complexity=PromptComplexity.ADVANCED,
            template="""
Eres un abogado especializado en derecho contractual argentino con expertise en análisis de contratos comerciales y compliance.

**CONTEXTO DEL ANÁLISIS:**
- Tipo de contrato: {contract_type}
- Partes involucradas: {parties}
- Legislación aplicable: Código Civil y Comercial, {additional_laws}

**ELEMENTOS DE ANÁLISIS:**

1. **Aspectos Formales:**
   - Capacidad de las partes
   - Consentimiento y vicios
   - Objeto lícito y determinado
   - Forma requerida por ley

2. **Cláusulas Principales:**
   - Obligaciones de cada parte
   - Condiciones y términos
   - Plazos y vencimientos
   - Contraprestaciones

3. **Cláusulas de Protección:**
   - Limitación de responsabilidad
   - Fuerza mayor
   - Resolución de conflictos
   - Confidencialidad

4. **Aspectos de Compliance:**
   - Cumplimiento normativo
   - Anti-corrupción (si aplica)
   - Due diligence de terceros
   - Cláusulas éticas

**CONTRATO A ANALIZAR:**
{contract_content}

**PUNTOS ESPECÍFICOS DE INTERÉS:**
{specific_points}

**INSTRUCCIONES DE ANÁLISIS:**
1. Revisa la validez formal del contrato
2. Identifica cláusulas problemáticas o faltantes
3. Evalúa riesgos legales para cada parte
4. Verifica cumplimiento de normativa aplicable
5. Propón mejoras o modificaciones

**FORMATO DE RESPUESTA:**
```json
{
  "validez_formal": {
    "es_valido": boolean,
    "observaciones": ["string"]
  },
  "analisis_clausulas": [
    {
      "clausula": "string",
      "tipo": "principal|proteccion|administrativa",
      "evaluacion": "adecuada|deficiente|faltante|problematica",
      "observaciones": "string",
      "recomendaciones": "string"
    }
  ],
  "riesgos_identificados": [
    {
      "riesgo": "string",
      "parte_afectada": "string",
      "severidad": "baja|media|alta",
      "mitigacion": "string"
    }
  ],
  "cumplimiento_normativo": {
    "conforme": boolean,
    "gaps_identificados": ["string"],
    "normativa_aplicable": ["string"]
  },
  "recomendaciones_generales": ["string"],
  "nivel_confianza": number
}
```
""",
            required_context=["contract_content", "contract_type"],
            optional_context=["parties", "additional_laws", "specific_points"],
            output_format="json",
            language="es"
        )
        
        # Abstention Decision Template
        templates["abstention_decision"] = PromptTemplate(
            name="abstention_decision",
            prompt_type=PromptType.ABSTENTION_DECISION,
            complexity=PromptComplexity.EXPERT,
            template="""
Eres un sistema de evaluación de abstención para análisis legal con principios matemáticos rigurosos basados en la teoría EDFL (Expectation-level Decompression Law).

**CONTEXTO DE DECISIÓN:**
- Nivel de confianza actual: {confidence_level}
- Umbral de abstención: {abstention_threshold}
- Dominio: Legal - {legal_domain}
- Complejidad del análisis: {analysis_complexity}

**FACTORES DE EVALUACIÓN:**

1. **Confianza Estadística:**
   - Métrica de confianza: {confidence_score}
   - Intervalos de bootstrap: {confidence_intervals}
   - Varianza del modelo: {model_variance}

2. **Riesgo Legal:**
   - Severidad de error potencial: {error_severity}
   - Consecuencias de decisión incorrecta: {error_consequences}
   - Reversibilidad de la decisión: {reversibility}

3. **Complejidad del Dominio:**
   - Ambigüedad jurídica: {legal_ambiguity}
   - Precedentes disponibles: {precedent_availability}
   - Consenso doctrinal: {doctrinal_consensus}

**CONTENIDO PARA EVALUACIÓN:**
{content}

**PRINCIPIOS DE ABSTENCIÓN:**
1. Abstener cuando el riesgo de error supere el beneficio de la decisión
2. Considerar el costo de abstención vs. costo de error
3. Evaluar si información adicional puede mejorar significativamente la confianza
4. Aplicar principio de precaución en temas de alto impacto legal

**FORMATO DE DECISIÓN:**
```json
{
  "decision_abstention": {
    "debe_abstenerse": boolean,
    "razon_principal": "string",
    "nivel_confianza_requerido": number,
    "nivel_confianza_actual": number,
    "gap_confianza": number
  },
  "analisis_factores": {
    "confianza_estadistica": {
      "puntuacion": number,
      "evaluacion": "suficiente|insuficiente|marginal"
    },
    "riesgo_legal": {
      "puntuacion": number,
      "evaluacion": "bajo|medio|alto|critico"
    },
    "complejidad_dominio": {
      "puntuacion": number,
      "evaluacion": "simple|moderada|compleja|experta"
    }
  },
  "recomendaciones": {
    "si_no_abstencion": ["string"],
    "si_abstencion": ["string"],
    "informacion_adicional": ["string"]
  },
  "limites_matematicos": {
    "bound_inferior": number,
    "bound_superior": number,
    "intervalo_confianza": "string"
  }
}
```

Base tu decisión en principios matemáticos rigurosos y considera las implicaciones legales específicas del contexto argentino.
""",
            required_context=["content", "confidence_level", "confidence_score"],
            optional_context=[
                "abstention_threshold", "legal_domain", "analysis_complexity",
                "confidence_intervals", "model_variance", "error_severity"
            ],
            output_format="json",
            complexity=PromptComplexity.EXPERT,
            language="es"
        )
        
        return templates
    
    def generate_prompt(
        self,
        prompt_type: PromptType,
        context: PromptContext,
        content: str,
        template_name: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate a legal prompt based on type and context.
        
        Args:
            prompt_type: Type of legal prompt to generate
            context: Context information for prompt generation
            content: Content to include in prompt
            template_name: Specific template name (optional)
            custom_instructions: Additional custom instructions
            
        Returns:
            Generated prompt string
        """
        # Select appropriate template
        template = self._select_template(prompt_type, template_name, context)
        
        if not template:
            raise ValueError(f"No template found for prompt type: {prompt_type}")
        
        # Prepare context variables
        prompt_variables = self._prepare_context_variables(context, content)
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt_variables["custom_instructions"] = custom_instructions
        
        # Generate prompt from template
        try:
            generated_prompt = template.template.format(**prompt_variables)
        except KeyError as e:
            logger.error(f"Missing required context variable: {e}")
            # Fill missing variables with defaults
            missing_vars = self._get_missing_variables(template, prompt_variables)
            prompt_variables.update(missing_vars)
            generated_prompt = template.template.format(**prompt_variables)
        
        # Add custom instructions if not already included
        if custom_instructions and "custom_instructions" not in template.template:
            generated_prompt += f"\n\n**INSTRUCCIONES ADICIONALES:**\n{custom_instructions}"
        
        # Update statistics
        self._update_stats(template)
        
        return generated_prompt
    
    def _select_template(
        self,
        prompt_type: PromptType,
        template_name: Optional[str],
        context: PromptContext
    ) -> Optional[PromptTemplate]:
        """Select appropriate template based on criteria."""
        if template_name and template_name in self.templates:
            return self.templates[template_name]
        
        # Find templates matching the prompt type
        matching_templates = [
            template for template in self.templates.values()
            if template.prompt_type == prompt_type
        ]
        
        if not matching_templates:
            return None
        
        # Select best template based on context
        if len(matching_templates) == 1:
            return matching_templates[0]
        
        # Prioritize by complexity and domain specificity
        best_template = max(
            matching_templates,
            key=lambda t: (
                t.domain_specific,
                t.complexity.value == context.analysis_depth,
                t.language == context.language
            )
        )
        
        return best_template
    
    def _prepare_context_variables(
        self,
        context: PromptContext,
        content: str
    ) -> Dict[str, str]:
        """Prepare context variables for template formatting."""
        variables = {
            "document_type": context.document_type,
            "jurisdiction": context.jurisdiction,
            "legal_frameworks": ", ".join(context.legal_frameworks) if context.legal_frameworks else "Marco legal general",
            "document_content": content,
            "content": content,
            "user_role": context.user_role,
            "analysis_depth": context.analysis_depth,
            "risk_tolerance": context.risk_tolerance,
            "language": context.language
        }
        
        # Add specific context based on document type
        if context.document_type == "contract":
            variables.update({
                "contract_type": context.custom_context.get("contract_type", "comercial"),
                "contract_content": content,
                "parties": context.custom_context.get("parties", "no especificado"),
                "additional_laws": context.custom_context.get("additional_laws", ""),
                "specific_points": context.custom_context.get("specific_points", "")
            })
        
        # Add compliance-specific context
        if "ley_27401" in context.legal_frameworks:
            variables.update({
                "legal_framework": "Ley 27401",
                "assessment_type": context.custom_context.get("assessment_type", "programa_integridad"),
                "entity_type": context.custom_context.get("entity_type", "persona_juridica")
            })
        
        # Add risk assessment context
        variables.update({
            "risk_type": context.custom_context.get("risk_type", "legal"),
            "risk_context": context.custom_context.get("risk_context", "cumplimiento_normativo"),
            "risk_factors": context.custom_context.get("risk_factors", "")
        })
        
        # Add abstention decision context
        variables.update({
            "confidence_level": str(context.custom_context.get("confidence_level", 0.7)),
            "confidence_score": str(context.custom_context.get("confidence_score", 0.7)),
            "abstention_threshold": str(context.custom_context.get("abstention_threshold", 0.8)),
            "legal_domain": context.custom_context.get("legal_domain", "general"),
            "analysis_complexity": context.custom_context.get("analysis_complexity", "intermediate"),
            "confidence_intervals": context.custom_context.get("confidence_intervals", "[0.6, 0.8]"),
            "model_variance": str(context.custom_context.get("model_variance", 0.1)),
            "error_severity": context.custom_context.get("error_severity", "medium"),
            "error_consequences": context.custom_context.get("error_consequences", "moderate"),
            "reversibility": context.custom_context.get("reversibility", "partial"),
            "legal_ambiguity": context.custom_context.get("legal_ambiguity", "medium"),
            "precedent_availability": context.custom_context.get("precedent_availability", "limited"),
            "doctrinal_consensus": context.custom_context.get("doctrinal_consensus", "partial")
        })
        
        # Add any additional custom context
        variables.update(context.custom_context)
        
        return variables
    
    def _get_missing_variables(
        self,
        template: PromptTemplate,
        variables: Dict[str, str]
    ) -> Dict[str, str]:
        """Get default values for missing template variables."""
        defaults = {
            "jurisdiction": self.jurisdiction,
            "language": self.default_language,
            "document_type": "documento_legal",
            "legal_frameworks": "Marco legal general",
            "content": "",
            "document_content": "",
            "contract_content": "",
            "contract_type": "comercial",
            "parties": "no especificado",
            "additional_laws": "",
            "specific_points": "",
            "legal_framework": "Marco legal general",
            "assessment_type": "evaluacion_general",
            "entity_type": "organizacion",
            "risk_type": "legal",
            "risk_context": "general",
            "risk_factors": "",
            "confidence_level": "0.7",
            "confidence_score": "0.7",
            "abstention_threshold": "0.8",
            "legal_domain": "general",
            "analysis_complexity": "intermediate"
        }
        
        # Extract variables referenced in template
        template_vars = re.findall(r'\{(\w+)\}', template.template)
        missing_vars = {}
        
        for var in template_vars:
            if var not in variables and var in defaults:
                missing_vars[var] = defaults[var]
            elif var not in variables:
                missing_vars[var] = f"[{var}]"  # Placeholder
        
        return missing_vars
    
    def _update_stats(self, template: PromptTemplate) -> None:
        """Update prompt generation statistics."""
        self.stats["prompts_generated"] += 1
        
        template_name = template.name
        if template_name not in self.stats["templates_used"]:
            self.stats["templates_used"][template_name] = 0
        self.stats["templates_used"][template_name] += 1
        
        self.stats["complexity_distribution"][template.complexity.value] += 1
        
        # Update average prompt length (estimate)
        current_avg = self.stats["avg_prompt_length"]
        estimated_length = len(template.template)
        new_count = self.stats["prompts_generated"]
        self.stats["avg_prompt_length"] = (current_avg * (new_count - 1) + estimated_length) / new_count
    
    def create_custom_template(
        self,
        name: str,
        prompt_type: PromptType,
        template_content: str,
        complexity: PromptComplexity = PromptComplexity.INTERMEDIATE,
        required_context: Optional[List[str]] = None,
        language: str = "es"
    ) -> PromptTemplate:
        """Create a custom prompt template."""
        custom_template = PromptTemplate(
            name=name,
            prompt_type=prompt_type,
            complexity=complexity,
            template=template_content,
            required_context=required_context or [],
            language=language,
            metadata={"custom": True, "created": datetime.now().isoformat()}
        )
        
        self.templates[name] = custom_template
        logger.info(f"Created custom template: {name}")
        
        return custom_template
    
    def get_available_templates(
        self,
        prompt_type: Optional[PromptType] = None,
        complexity: Optional[PromptComplexity] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """Get list of available template names with optional filtering."""
        templates = self.templates.values()
        
        if prompt_type:
            templates = [t for t in templates if t.prompt_type == prompt_type]
        
        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        return [t.name for t in templates]
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific template."""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        return {
            "name": template.name,
            "type": template.prompt_type.value,
            "complexity": template.complexity.value,
            "required_context": template.required_context,
            "optional_context": template.optional_context,
            "output_format": template.output_format,
            "language": template.language,
            "domain_specific": template.domain_specific,
            "metadata": template.metadata
        }
    
    def validate_context(
        self,
        template_name: str,
        context: PromptContext
    ) -> Tuple[bool, List[str]]:
        """Validate that context provides required variables for template."""
        if template_name not in self.templates:
            return False, [f"Template '{template_name}' not found"]
        
        template = self.templates[template_name]
        variables = self._prepare_context_variables(context, "")
        
        missing = []
        for required_var in template.required_context:
            if required_var not in variables or not variables[required_var]:
                missing.append(required_var)
        
        return len(missing) == 0, missing
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prompt generation statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset prompt generation statistics."""
        self.stats = {
            "prompts_generated": 0,
            "templates_used": {},
            "avg_prompt_length": 0,
            "complexity_distribution": {level.value: 0 for level in PromptComplexity}
        }