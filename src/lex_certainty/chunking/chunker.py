"""
Legal Document Chunker

Implements sophisticated chunking strategies for legal documents with
semantic awareness and legal structure preservation.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import tiktoken

import numpy as np
from pydantic import BaseModel, Field

from ..utils.config import LexCertaintyConfig

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Document chunking strategies."""
    SIMPLE = "simple"                    # Fixed-size chunks
    SEMANTIC = "semantic"                # Semantic boundary awareness  
    LEGAL_SEMANTIC = "legal_semantic"    # Legal-specific semantic chunking
    HIERARCHICAL = "hierarchical"        # Preserve legal hierarchy
    CLAUSE_BASED = "clause_based"        # Contract clause boundaries
    ARTICLE_BASED = "article_based"      # Legal article boundaries


class ChunkType(Enum):
    """Types of legal document chunks."""
    HEADER = "header"
    BODY = "body"
    CLAUSE = "clause"
    ARTICLE = "article"
    FOOTNOTE = "footnote"
    APPENDIX = "appendix"
    REFERENCE = "reference"
    SIGNATURE = "signature"


@dataclass
class DocumentChunk:
    """Legal document chunk with metadata."""
    id: str
    content: str
    start_position: int
    end_position: int
    chunk_type: ChunkType = ChunkType.BODY
    legal_elements: Dict[str, Any] = field(default_factory=dict)
    semantic_score: float = 0.0
    parent_chunk: Optional[str] = None
    child_chunks: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if not self.id:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
            self.id = f"chunk_{self.start_position}_{content_hash}"


class LegalStructureDetector:
    """Detects legal document structure and boundaries."""
    
    def __init__(self):
        """Initialize legal structure patterns."""
        # Article patterns for different legal systems
        self.article_patterns = [
            r'^ARTÍ?CULO\s+\d+[º°]?\.?\s*[-–—]?\s*',  # Spanish: ARTÍCULO 1° - 
            r'^Art\.?\s*\d+[º°]?\.?\s*[-–—]?\s*',      # Abbreviated: Art. 1° -
            r'^Article\s+\d+\.?\s*[-–—]?\s*',          # English: Article 1 -
            r'^\d+\.\s*[-–—]?\s*',                     # Numbered: 1. -
        ]
        
        # Clause patterns
        self.clause_patterns = [
            r'^CLÁUSULA\s+\d+[º°]?\.?\s*[-–—]?\s*',    # Spanish: CLÁUSULA 1° -
            r'^Clause\s+\d+\.?\s*[-–—]?\s*',           # English: Clause 1 -
            r'^[A-Z][).]?\s+[-–—]?\s*',                # Letter enumeration: A) -
        ]
        
        # Section patterns  
        self.section_patterns = [
            r'^(?:CAPÍ?TULO|TÍTULO|SECCIÓN)\s+[IVXLC]+\.?\s*[-–—]?\s*',  # Roman numerals
            r'^(?:CHAPTER|TITLE|SECTION)\s+[IVXLC]+\.?\s*[-–—]?\s*',     # English
            r'^(?:CAPÍ?TULO|TÍTULO|SECCIÓN)\s+\d+\.?\s*[-–—]?\s*',       # Numbers
        ]
        
        # Paragraph patterns
        self.paragraph_patterns = [
            r'^\s*\(\w+\)\s*',                        # (a), (1), etc.
            r'^\s*\w+\)\s*',                          # a), 1), etc.
            r'^\s*[ivxlc]+\)\s*',                     # Roman numerals i), ii), etc.
        ]
        
    def detect_legal_boundaries(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Detect legal structure boundaries in text.
        
        Returns list of (start_pos, end_pos, element_type, element_id).
        """
        boundaries = []
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Check for articles
            for pattern in self.article_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    article_id = self._extract_numbering(line, "article")
                    boundaries.append((line_start, line_end, "article", article_id))
                    break
            
            # Check for clauses  
            for pattern in self.clause_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    clause_id = self._extract_numbering(line, "clause")
                    boundaries.append((line_start, line_end, "clause", clause_id))
                    break
            
            # Check for sections
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    section_id = self._extract_numbering(line, "section")
                    boundaries.append((line_start, line_end, "section", section_id))
                    break
            
            # Check for paragraphs
            for pattern in self.paragraph_patterns:
                if re.match(pattern, line):
                    para_id = self._extract_numbering(line, "paragraph")
                    boundaries.append((line_start, line_end, "paragraph", para_id))
                    break
            
            current_pos = line_end + 1  # +1 for newline
        
        return boundaries
    
    def _extract_numbering(self, line: str, element_type: str) -> str:
        """Extract numbering/identification from legal element."""
        # Remove common prefixes and clean up
        line_clean = re.sub(r'^(?:ARTÍ?CULO|CLÁUSULA|Art\.?|Clause)\s*', '', line, flags=re.IGNORECASE)
        line_clean = re.sub(r'^(?:CAPÍ?TULO|TÍTULO|SECCIÓN|CHAPTER|TITLE|SECTION)\s*', '', line_clean, flags=re.IGNORECASE)
        
        # Extract number or letter
        number_match = re.match(r'([IVXLC\d]+)[º°]?\.?\s*[-–—]?\s*', line_clean)
        if number_match:
            return f"{element_type}_{number_match.group(1)}"
        
        # For paragraphs, extract letter/number from parentheses  
        if element_type == "paragraph":
            para_match = re.match(r'\s*\(?([a-zA-Z\d]+)\)?\s*', line_clean)
            if para_match:
                return f"para_{para_match.group(1)}"
        
        return f"{element_type}_unknown"


class SemanticBoundaryDetector:
    """Detects semantic boundaries in legal text."""
    
    def __init__(self):
        """Initialize semantic patterns."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Semantic transition indicators
        self.transition_patterns = [
            r'\b(?:sin embargo|no obstante|por tanto|en consecuencia)\b',  # Spanish transitions
            r'\b(?:however|nevertheless|therefore|consequently)\b',         # English transitions
            r'\b(?:además|asimismo|por otra parte|por el contrario)\b',     # Additional Spanish
            r'\b(?:furthermore|moreover|on the other hand|conversely)\b',   # Additional English
        ]
        
        # Legal conclusion indicators
        self.conclusion_patterns = [
            r'\b(?:en conclusión|por último|finalmente|en resumen)\b',
            r'\b(?:in conclusion|finally|lastly|in summary)\b',
        ]
        
    def detect_semantic_breaks(
        self, 
        text: str, 
        max_chunk_size: int = 1000
    ) -> List[int]:
        """
        Detect semantic break points in text.
        
        Returns list of character positions where breaks should occur.
        """
        break_points = [0]  # Start with beginning
        
        # Split into sentences
        sentences = self._split_sentences(text)
        current_pos = 0
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Check if adding this sentence exceeds chunk size
            if current_chunk_size + sentence_tokens > max_chunk_size and current_chunk_size > 0:
                break_points.append(current_pos)
                current_chunk_size = sentence_tokens
            else:
                current_chunk_size += sentence_tokens
            
            # Check for semantic transition indicators
            if self._has_semantic_transition(sentence):
                # Add break after this sentence if chunk is substantial
                if current_chunk_size > max_chunk_size // 2:
                    sentence_end = current_pos + len(sentence)
                    break_points.append(sentence_end)
                    current_chunk_size = 0
            
            current_pos += len(sentence)
        
        # Add final position
        if break_points[-1] < len(text):
            break_points.append(len(text))
        
        return break_points
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with legal awareness."""
        # Handle legal citations that contain periods
        text = re.sub(r'(\b(?:Art|Inc|Cap)\.)(\s*\d)', r'\1<PERIOD>\2', text)
        
        # Split on sentence boundaries  
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore periods in legal citations
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _has_semantic_transition(self, sentence: str) -> bool:
        """Check if sentence contains semantic transition indicators."""
        sentence_lower = sentence.lower()
        
        for pattern in self.transition_patterns + self.conclusion_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False


class DocumentChunker:
    """
    Main document chunker with multiple strategies for legal documents.
    
    Implements chunking strategies optimized for legal document structure
    and semantic coherence.
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.LEGAL_SEMANTIC,
        max_chunk_size: int = 1000,
        overlap_size: int = 100,
        legal_context: Optional[Any] = None,
        config: Optional[LexCertaintyConfig] = None
    ):
        """Initialize document chunker."""
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.legal_context = legal_context
        self.config = config or LexCertaintyConfig()
        
        # Initialize detectors
        self.structure_detector = LegalStructureDetector()
        self.semantic_detector = SemanticBoundaryDetector()
        
        # Tokenizer for size calculations
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized DocumentChunker with {strategy.value} strategy")
    
    async def chunk_document(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_legal_structure: bool = True
    ) -> List[DocumentChunk]:
        """
        Chunk document using configured strategy.
        
        Args:
            document: Document text to chunk
            metadata: Document metadata
            preserve_legal_structure: Whether to preserve legal structure
            
        Returns:
            List of DocumentChunk objects
        """
        metadata = metadata or {}
        
        if self.strategy == ChunkingStrategy.SIMPLE:
            return await self._simple_chunking(document, metadata)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return await self._semantic_chunking(document, metadata)
        elif self.strategy == ChunkingStrategy.LEGAL_SEMANTIC:
            return await self._legal_semantic_chunking(document, metadata, preserve_legal_structure)
        elif self.strategy == ChunkingStrategy.HIERARCHICAL:
            return await self._hierarchical_chunking(document, metadata)
        elif self.strategy == ChunkingStrategy.CLAUSE_BASED:
            return await self._clause_based_chunking(document, metadata)
        elif self.strategy == ChunkingStrategy.ARTICLE_BASED:
            return await self._article_based_chunking(document, metadata)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, falling back to simple chunking")
            return await self._simple_chunking(document, metadata)
    
    async def _simple_chunking(
        self,
        document: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Simple fixed-size chunking."""
        chunks = []
        tokens = self.tokenizer.encode(document)
        
        start_pos = 0
        chunk_id = 0
        
        while start_pos < len(tokens):
            # Calculate chunk end with overlap
            end_pos = min(start_pos + self.max_chunk_size, len(tokens))
            
            # Decode chunk tokens back to text
            chunk_tokens = tokens[start_pos:end_pos]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find actual character positions in original text
            char_start = len(self.tokenizer.decode(tokens[:start_pos]))
            char_end = len(self.tokenizer.decode(tokens[:end_pos]))
            
            chunk = DocumentChunk(
                id=f"simple_{chunk_id}",
                content=chunk_text,
                start_position=char_start,
                end_position=char_end,
                chunk_type=ChunkType.BODY
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start_pos = max(start_pos + self.max_chunk_size - self.overlap_size, start_pos + 1)
        
        return chunks
    
    async def _semantic_chunking(
        self,
        document: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Semantic boundary-aware chunking."""
        chunks = []
        
        # Detect semantic boundaries
        break_points = self.semantic_detector.detect_semantic_breaks(
            document, 
            self.max_chunk_size
        )
        
        for i in range(len(break_points) - 1):
            start_pos = break_points[i]
            end_pos = break_points[i + 1]
            
            # Add overlap with previous chunk
            if i > 0 and self.overlap_size > 0:
                overlap_start = max(0, start_pos - self.overlap_size)
                chunk_content = document[overlap_start:end_pos]
                actual_start = overlap_start
            else:
                chunk_content = document[start_pos:end_pos]
                actual_start = start_pos
            
            # Calculate semantic score
            semantic_score = self._calculate_semantic_coherence(chunk_content)
            
            chunk = DocumentChunk(
                id=f"semantic_{i}",
                content=chunk_content.strip(),
                start_position=actual_start,
                end_position=end_pos,
                chunk_type=ChunkType.BODY,
                semantic_score=semantic_score
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _legal_semantic_chunking(
        self,
        document: str,
        metadata: Dict[str, Any],
        preserve_structure: bool = True
    ) -> List[DocumentChunk]:
        """Legal-aware semantic chunking."""
        chunks = []
        
        # First, detect legal structure boundaries
        legal_boundaries = []
        if preserve_structure:
            legal_boundaries = self.structure_detector.detect_legal_boundaries(document)
        
        # Then detect semantic boundaries
        semantic_breaks = self.semantic_detector.detect_semantic_breaks(
            document,
            self.max_chunk_size
        )
        
        # Combine and prioritize legal boundaries
        all_boundaries = set(semantic_breaks)
        
        # Add legal structure boundaries with higher priority
        for start, end, element_type, element_id in legal_boundaries:
            all_boundaries.add(start)
            # For legal elements, also add end positions
            if element_type in ["article", "clause"]:
                all_boundaries.add(end)
        
        # Sort boundaries
        sorted_boundaries = sorted(all_boundaries)
        
        # Create chunks respecting legal structure
        chunk_id = 0
        i = 0
        
        while i < len(sorted_boundaries) - 1:
            start_pos = sorted_boundaries[i]
            
            # Find appropriate end position
            end_pos = sorted_boundaries[i + 1]
            current_size = len(self.tokenizer.encode(document[start_pos:end_pos]))
            
            # Extend chunk if it's too small and doesn't break legal structure
            while (i + 2 < len(sorted_boundaries) and 
                   current_size < self.max_chunk_size // 2):
                
                next_end = sorted_boundaries[i + 2]
                
                # Check if next boundary is a legal structure boundary
                is_legal_boundary = any(
                    boundary[0] <= next_end <= boundary[1]
                    for boundary in legal_boundaries
                    if boundary[2] in ["article", "clause", "section"]
                )
                
                if is_legal_boundary:
                    break  # Don't cross major legal boundaries
                
                end_pos = next_end
                current_size = len(self.tokenizer.encode(document[start_pos:end_pos]))
                i += 1
            
            # Create chunk with legal metadata
            chunk_content = document[start_pos:end_pos].strip()
            
            if chunk_content:  # Only add non-empty chunks
                # Determine chunk type from legal boundaries
                chunk_type = self._determine_chunk_type(
                    start_pos, end_pos, legal_boundaries
                )
                
                # Extract legal elements in chunk
                legal_elements = self._extract_legal_elements(
                    chunk_content, start_pos, legal_boundaries
                )
                
                chunk = DocumentChunk(
                    id=f"legal_semantic_{chunk_id}",
                    content=chunk_content,
                    start_position=start_pos,
                    end_position=end_pos,
                    chunk_type=chunk_type,
                    legal_elements=legal_elements,
                    semantic_score=self._calculate_semantic_coherence(chunk_content)
                )
                
                chunks.append(chunk)
                chunk_id += 1
            
            i += 1
        
        return chunks
    
    async def _hierarchical_chunking(
        self,
        document: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Hierarchical chunking preserving document structure."""
        chunks = []
        legal_boundaries = self.structure_detector.detect_legal_boundaries(document)
        
        # Group boundaries by hierarchy level
        hierarchy = {
            "section": [],
            "article": [],
            "clause": [],
            "paragraph": []
        }
        
        for start, end, element_type, element_id in legal_boundaries:
            if element_type in hierarchy:
                hierarchy[element_type].append((start, end, element_id))
        
        # Create hierarchical chunks
        chunk_id = 0
        
        # Process sections first (highest level)
        if hierarchy["section"]:
            for i, (start, end, section_id) in enumerate(hierarchy["section"]):
                # Find next section or end of document
                next_start = (hierarchy["section"][i + 1][0] 
                            if i + 1 < len(hierarchy["section"]) 
                            else len(document))
                
                section_content = document[start:next_start]
                
                # Create section-level chunk
                section_chunk = DocumentChunk(
                    id=f"section_{chunk_id}",
                    content=section_content.strip(),
                    start_position=start,
                    end_position=next_start,
                    chunk_type=ChunkType.BODY,
                    legal_elements={"section_id": section_id}
                )
                
                chunks.append(section_chunk)
                
                # Create sub-chunks within section if needed
                if len(self.tokenizer.encode(section_content)) > self.max_chunk_size:
                    sub_chunks = await self._create_sub_chunks(
                        section_content, start, chunk_id, section_chunk.id
                    )
                    chunks.extend(sub_chunks)
                
                chunk_id += 1
        
        # Process standalone articles and clauses
        standalone_elements = []
        for element_type in ["article", "clause", "paragraph"]:
            standalone_elements.extend([
                (start, end, element_type, element_id)
                for start, end, element_id in hierarchy[element_type]
            ])
        
        # Sort by position
        standalone_elements.sort(key=lambda x: x[0])
        
        for start, end, element_type, element_id in standalone_elements:
            content = document[start:end].strip()
            
            if content and not any(chunk.start_position <= start < chunk.end_position 
                                 for chunk in chunks):
                chunk = DocumentChunk(
                    id=f"{element_type}_{chunk_id}",
                    content=content,
                    start_position=start,
                    end_position=end,
                    chunk_type=getattr(ChunkType, element_type.upper(), ChunkType.BODY),
                    legal_elements={f"{element_type}_id": element_id}
                )
                
                chunks.append(chunk)
                chunk_id += 1
        
        return sorted(chunks, key=lambda x: x.start_position)
    
    async def _clause_based_chunking(
        self,
        document: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk document by contract clauses."""
        chunks = []
        legal_boundaries = self.structure_detector.detect_legal_boundaries(document)
        
        # Extract clause boundaries
        clause_boundaries = [
            (start, end, element_id)
            for start, end, element_type, element_id in legal_boundaries
            if element_type == "clause"
        ]
        
        if not clause_boundaries:
            # Fallback to semantic chunking if no clauses detected
            return await self._semantic_chunking(document, metadata)
        
        for i, (start, end, clause_id) in enumerate(clause_boundaries):
            # Find clause content (until next clause or end)
            next_start = (clause_boundaries[i + 1][0] 
                         if i + 1 < len(clause_boundaries) 
                         else len(document))
            
            clause_content = document[start:next_start].strip()
            
            # Split large clauses into sub-chunks
            if len(self.tokenizer.encode(clause_content)) > self.max_chunk_size:
                sub_chunks = await self._create_clause_sub_chunks(
                    clause_content, start, i, clause_id
                )
                chunks.extend(sub_chunks)
            else:
                chunk = DocumentChunk(
                    id=f"clause_{i}",
                    content=clause_content,
                    start_position=start,
                    end_position=next_start,
                    chunk_type=ChunkType.CLAUSE,
                    legal_elements={"clause_id": clause_id}
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _article_based_chunking(
        self,
        document: str,
        metadata: Dict[str, Any]  
    ) -> List[DocumentChunk]:
        """Chunk document by legal articles."""
        chunks = []
        legal_boundaries = self.structure_detector.detect_legal_boundaries(document)
        
        # Extract article boundaries
        article_boundaries = [
            (start, end, element_id)
            for start, end, element_type, element_id in legal_boundaries
            if element_type == "article"
        ]
        
        if not article_boundaries:
            # Fallback to semantic chunking if no articles detected
            return await self._semantic_chunking(document, metadata)
        
        for i, (start, end, article_id) in enumerate(article_boundaries):
            # Find article content (until next article or end)
            next_start = (article_boundaries[i + 1][0]
                         if i + 1 < len(article_boundaries)
                         else len(document))
            
            article_content = document[start:next_start].strip()
            
            # Split large articles into sub-chunks
            if len(self.tokenizer.encode(article_content)) > self.max_chunk_size:
                sub_chunks = await self._create_article_sub_chunks(
                    article_content, start, i, article_id
                )
                chunks.extend(sub_chunks)
            else:
                chunk = DocumentChunk(
                    id=f"article_{i}",
                    content=article_content,
                    start_position=start,
                    end_position=next_start,
                    chunk_type=ChunkType.ARTICLE,
                    legal_elements={"article_id": article_id}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(
        self,
        start_pos: int,
        end_pos: int,
        legal_boundaries: List[Tuple[int, int, str, str]]
    ) -> ChunkType:
        """Determine chunk type based on legal boundaries."""
        # Check if chunk contains or starts with specific legal elements
        for bound_start, bound_end, element_type, element_id in legal_boundaries:
            if bound_start >= start_pos and bound_start < end_pos:
                if element_type == "article":
                    return ChunkType.ARTICLE
                elif element_type == "clause":
                    return ChunkType.CLAUSE
                elif element_type == "section":
                    return ChunkType.HEADER
        
        return ChunkType.BODY
    
    def _extract_legal_elements(
        self,
        content: str,
        start_pos: int,
        legal_boundaries: List[Tuple[int, int, str, str]]
    ) -> Dict[str, Any]:
        """Extract legal elements present in chunk."""
        elements = {
            "articles": [],
            "clauses": [],
            "sections": [],
            "paragraphs": [],
            "references": []
        }
        
        # Find overlapping legal boundaries
        for bound_start, bound_end, element_type, element_id in legal_boundaries:
            if bound_start >= start_pos:
                if element_type in elements:
                    elements[element_type].append({
                        "id": element_id,
                        "relative_start": bound_start - start_pos,
                        "relative_end": bound_end - start_pos
                    })
        
        # Extract cross-references
        ref_patterns = [
            r'\b(?:véase|ver|cfr\.?|cf\.?)\s+(?:artículo|art\.?)\s+\d+',
            r'\b(?:según|conforme)\s+(?:artículo|art\.?)\s+\d+',
            r'\bley\s+\d+(?:\.\d+)*\b',
            r'\bdecreto\s+\d+(?:/\d+)?\b'
        ]
        
        references = []
        for pattern in ref_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                references.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        elements["references"] = references
        
        return elements
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence score for text chunk."""
        # Simple coherence metrics
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words that indicate coherence
        transition_words = [
            "además", "asimismo", "por tanto", "sin embargo", "no obstante",
            "en consecuencia", "por ende", "furthermore", "however", "therefore"
        ]
        
        transition_count = 0
        for sentence in sentences:
            for word in transition_words:
                if word in sentence.lower():
                    transition_count += 1
                    break
        
        # Calculate coherence as ratio of sentences with transitions
        coherence = transition_count / len(sentences) if sentences else 0
        
        # Boost score for legal structure indicators
        legal_indicators = ["artículo", "cláusula", "inciso", "párrafo"]
        legal_count = sum(1 for word in legal_indicators if word in text.lower())
        legal_boost = min(legal_count * 0.1, 0.3)
        
        return min(coherence + legal_boost, 1.0)
    
    async def _create_sub_chunks(
        self,
        content: str,
        base_start: int,
        parent_id: int,
        parent_chunk_id: str
    ) -> List[DocumentChunk]:
        """Create sub-chunks for oversized content."""
        sub_chunks = []
        
        # Use semantic chunking for sub-division
        semantic_breaks = self.semantic_detector.detect_semantic_breaks(
            content, 
            self.max_chunk_size
        )
        
        for i in range(len(semantic_breaks) - 1):
            start_pos = semantic_breaks[i]
            end_pos = semantic_breaks[i + 1]
            
            sub_content = content[start_pos:end_pos].strip()
            
            if sub_content:
                sub_chunk = DocumentChunk(
                    id=f"sub_{parent_id}_{i}",
                    content=sub_content,
                    start_position=base_start + start_pos,
                    end_position=base_start + end_pos,
                    chunk_type=ChunkType.BODY,
                    parent_chunk=parent_chunk_id,
                    semantic_score=self._calculate_semantic_coherence(sub_content)
                )
                sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    async def _create_clause_sub_chunks(
        self,
        clause_content: str,
        base_start: int,
        clause_index: int,
        clause_id: str
    ) -> List[DocumentChunk]:
        """Create sub-chunks for large clauses."""
        sub_chunks = []
        
        # Split by paragraphs first
        paragraphs = clause_content.split('\n\n')
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                sub_chunk = DocumentChunk(
                    id=f"clause_{clause_index}_para_{i}",
                    content=paragraph.strip(),
                    start_position=base_start + current_pos,
                    end_position=base_start + current_pos + len(paragraph),
                    chunk_type=ChunkType.CLAUSE,
                    legal_elements={"clause_id": clause_id, "paragraph": i}
                )
                sub_chunks.append(sub_chunk)
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return sub_chunks
    
    async def _create_article_sub_chunks(
        self,
        article_content: str,
        base_start: int,
        article_index: int,
        article_id: str
    ) -> List[DocumentChunk]:
        """Create sub-chunks for large articles."""
        sub_chunks = []
        
        # Look for sub-paragraphs or incisors within article
        inciso_pattern = r'^(?:\s*(?:\([a-z]\)|\w+\)|\d+\.))'
        lines = article_content.split('\n')
        
        current_chunk = []
        current_pos = 0
        chunk_start = 0
        sub_index = 0
        
        for line in lines:
            if re.match(inciso_pattern, line.strip()) and current_chunk:
                # Create chunk from accumulated lines
                chunk_content = '\n'.join(current_chunk).strip()
                if chunk_content:
                    sub_chunk = DocumentChunk(
                        id=f"article_{article_index}_sub_{sub_index}",
                        content=chunk_content,
                        start_position=base_start + chunk_start,
                        end_position=base_start + current_pos,
                        chunk_type=ChunkType.ARTICLE,
                        legal_elements={"article_id": article_id, "sub_section": sub_index}
                    )
                    sub_chunks.append(sub_chunk)
                    sub_index += 1
                
                # Start new chunk
                current_chunk = [line]
                chunk_start = current_pos
            else:
                current_chunk.append(line)
            
            current_pos += len(line) + 1  # +1 for \n
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                sub_chunk = DocumentChunk(
                    id=f"article_{article_index}_sub_{sub_index}",
                    content=chunk_content,
                    start_position=base_start + chunk_start,
                    end_position=base_start + current_pos,
                    chunk_type=ChunkType.ARTICLE,
                    legal_elements={"article_id": article_id, "sub_section": sub_index}
                )
                sub_chunks.append(sub_chunk)
        
        return sub_chunks