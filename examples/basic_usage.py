"""
LexCertainty Community Edition - Basic Usage Examples

This example demonstrates the key features of LexCertainty for legal document
processing with mathematical abstention principles.
"""

import asyncio
import logging
from pathlib import Path

# Import LexCertainty components
from lex_certainty import (
    # Core components
    LegalDocumentProcessor,
    AbstractionEngine,
    CertifiedComplianceEngine,
    ConfidenceEstimator,
    
    # Legal context
    ArgentineLegalContext,
    
    # Processing modes
    ProcessingMode,
    DocumentType,
    
    # Configuration
    LexCertaintyConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_document_processing_example():
    """
    Example 1: Basic legal document processing with abstention.
    """
    logger.info("=== Basic Document Processing Example ===")
    
    # Sample legal document (compliance policy)
    sample_document = """
    POLÍTICA DE INTEGRIDAD CORPORATIVA
    
    1. OBJETIVO
    La presente política tiene como objetivo establecer los principios y 
    procedimientos para prevenir actos de corrupción en nuestra organización,
    en cumplimiento de la Ley 27401 de Responsabilidad Penal Empresaria.
    
    2. PROGRAMA DE INTEGRIDAD
    Nuestra empresa implementa un programa de integridad que incluye:
    
    a) Mapeo de procesos y identificación de riesgos específicos
    b) Código de ética y conducta empresarial
    c) Capacitación periódica en materia de integridad
    d) Canal de denuncias interno y confidencial
    e) Protección del denunciante
    f) Monitoreo y evaluación continua
    
    3. RESPONSABILIDADES
    La alta dirección es responsable de la implementación y supervisión
    del programa de integridad, designando un responsable específico
    según lo establecido en el artículo 9 de la Ley 27401.
    
    4. DUE DILIGENCE DE TERCEROS
    Todos los socios comerciales serán evaluados mediante un proceso
    de due diligence que incluye verificación de antecedentes y 
    evaluación de riesgos de integridad.
    """
    
    # Initialize configuration
    config = LexCertaintyConfig()
    config.legal_chunk_size = 500
    config.confidence_threshold = 0.75
    
    # Initialize legal context for Argentina
    legal_context = ArgentineLegalContext()
    
    # Initialize document processor
    processor = LegalDocumentProcessor(
        config=config,
        legal_context=legal_context
    )
    
    try:
        # Process the document
        result = await processor.process_document(
            document=sample_document,
            mode=ProcessingMode.STANDARD
        )
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Document type: {result.metadata['document_type']}")
        logger.info(f"Total chunks: {len(result.chunks)}")
        logger.info(f"Overall confidence: {result.confidence_scores['overall_confidence']:.3f}")
        logger.info(f"Compliance flags: {len(result.compliance_flags)}")
        
        # Check abstention recommendations
        overall_abstention = result.abstention_results["overall_abstention"]
        if overall_abstention["should_abstain"]:
            logger.warning(f"🚨 Abstention recommended: {overall_abstention['reason']}")
        else:
            logger.info("✅ Analysis confidence is sufficient to proceed")
        
        # Display compliance assessment
        compliance_score = result.legal_analysis.get("compliance_assessment", {}).get("score", 0.0)
        logger.info(f"Compliance score: {compliance_score:.2f}")
        
        # Show high-risk flags if any
        high_risk_flags = [flag for flag in result.compliance_flags 
                          if flag.get("severity") == "high"]
        if high_risk_flags:
            logger.warning(f"High-risk issues found: {len(high_risk_flags)}")
            for flag in high_risk_flags[:3]:  # Show first 3
                logger.warning(f"  - {flag['description']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


async def compliance_engine_example():
    """
    Example 2: Using the Certified Compliance Engine.
    """
    logger.info("\n=== Certified Compliance Engine Example ===")
    
    # Sample compliance scenario
    scenario_content = """
    Nuestra empresa ha implementado las siguientes medidas de compliance:
    
    1. Código de Ética: Aprobado por el directorio en enero 2024
    2. Capacitación: Programa anual de 8 horas para todo el personal
    3. Canal de Denuncias: Línea telefónica y portal web disponible 24/7
    4. Due Diligence: Evaluación de todos los proveedores críticos
    5. Monitoreo: Auditorías internas trimestrales
    
    Sin embargo, aún no hemos implementado:
    - Matriz de riesgos específica para corrupción
    - Procedimiento formal de investigación interna
    - Evaluación de efectividad del programa
    """
    
    # Initialize compliance engine
    config = LexCertaintyConfig()
    legal_context = ArgentineLegalContext()
    
    compliance_engine = CertifiedComplianceEngine(
        config=config,
        legal_context=legal_context
    )
    
    try:
        # Analyze compliance scenario
        result = await compliance_engine.analyze_scenario(
            scenario_content=scenario_content,
            legal_frameworks=["ley_27401"],
            risk_tolerance="medium"
        )
        
        logger.info(f"Compliance analysis completed!")
        logger.info(f"Certified result: {result.is_certified}")
        logger.info(f"Confidence level: {result.confidence:.3f}")
        logger.info(f"Risk level: {result.risk_level}")
        
        if result.mathematical_guarantees:
            guarantees = result.mathematical_guarantees
            logger.info(f"Mathematical guarantees:")
            logger.info(f"  - Risk bounds: [{guarantees['risk_bounds']['lower']:.3f}, {guarantees['risk_bounds']['upper']:.3f}]")
            logger.info(f"  - Confidence interval: {guarantees['confidence_interval']}")
        
        # Show compliance gaps
        if result.compliance_gaps:
            logger.warning(f"Compliance gaps identified:")
            for gap in result.compliance_gaps[:5]:  # Show first 5
                logger.warning(f"  - {gap}")
        
        # Show recommendations
        if result.recommendations:
            logger.info(f"Recommendations:")
            for rec in result.recommendations[:3]:  # Show first 3
                logger.info(f"  - {rec}")
        
        return result
        
    except Exception as e:
        logger.error(f"Compliance analysis failed: {str(e)}")
        raise


async def abstention_analysis_example():
    """
    Example 3: Mathematical abstention analysis.
    """
    logger.info("\n=== Mathematical Abstention Analysis Example ===")
    
    # Sample ambiguous legal text
    ambiguous_text = """
    La cláusula establece que "las partes deberán cumplir con todas las 
    obligaciones aplicables, incluyendo pero no limitándose a las 
    regulaciones vigentes que pudieran corresponder según el caso".
    
    Esta redacción podría interpretarse de múltiples maneras en cuanto
    a qué regulaciones específicas son aplicables y el alcance exacto
    de las obligaciones.
    """
    
    # Initialize abstention engine
    config = LexCertaintyConfig()
    abstention_engine = AbstractionEngine(
        config=config,
        domain="legal"
    )
    
    # Initialize confidence estimator
    confidence_estimator = ConfidenceEstimator(
        config=config,
        domain="legal"
    )
    
    try:
        # Estimate confidence for the text
        confidence_result = await confidence_estimator.estimate_confidence(
            text=ambiguous_text,
            context={
                "document_type": "contract",
                "analysis_type": "clause_interpretation"
            }
        )
        
        logger.info(f"Confidence estimation:")
        logger.info(f"  - Score: {confidence_result.confidence_score:.3f}")
        logger.info(f"  - Method: {confidence_result.method_used}")
        logger.info(f"  - Uncertainty: {confidence_result.uncertainty:.3f}")
        
        # Perform abstention analysis
        abstention_result = await abstention_engine.should_abstain(
            confidence=confidence_result.confidence_score,
            context={
                "text_ambiguity": "high",
                "legal_complexity": "medium",
                "stakes": "medium"
            },
            domain="legal"
        )
        
        logger.info(f"\nAbstention analysis:")
        logger.info(f"  - Should abstain: {abstention_result.should_abstain}")
        logger.info(f"  - Reason: {abstention_result.reason}")
        logger.info(f"  - Confidence threshold: {abstention_result.confidence_threshold:.3f}")
        
        if abstention_result.risk_bounds:
            bounds = abstention_result.risk_bounds
            logger.info(f"  - Risk bounds: [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
        
        # Provide recommendations
        if abstention_result.should_abstain:
            logger.warning("🚨 Abstention recommended!")
            logger.info("Recommendations:")
            logger.info("  - Seek additional legal expert review")
            logger.info("  - Clarify ambiguous terms with counterparty")
            logger.info("  - Consider adding specific definitions")
        else:
            logger.info("✅ Confidence sufficient for analysis")
        
        return abstention_result
        
    except Exception as e:
        logger.error(f"Abstention analysis failed: {str(e)}")
        raise


async def main():
    """
    Run all examples sequentially.
    """
    logger.info("Starting LexCertainty Community Edition Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Basic document processing
        doc_result = await basic_document_processing_example()
        
        # Example 2: Compliance engine
        compliance_result = await compliance_engine_example()
        
        # Example 3: Abstention analysis
        abstention_result = await abstention_analysis_example()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! ✅")
        
        # Summary
        logger.info("\nSUMMARY:")
        logger.info(f"Documents processed: 1")
        logger.info(f"Compliance scenarios analyzed: 1")
        logger.info(f"Abstention analyses performed: 1")
        
        return {
            "document_processing": doc_result,
            "compliance_analysis": compliance_result,
            "abstention_analysis": abstention_result
        }
        
    except Exception as e:
        logger.error(f"Examples failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the examples
    results = asyncio.run(main())
    
    print("\n🎉 LexCertainty examples completed!")
    print("Check the logs above for detailed results.")
    print("\nNext steps:")
    print("- Try processing your own legal documents")
    print("- Explore advanced configuration options")  
    print("- Check out the enterprise features at: https://github.com/adrianlerer/lex-certainty-enterprise")