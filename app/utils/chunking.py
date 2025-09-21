"""
Advanced text chunking system with multiple strategies for optimal document processing.
Supports fixed-size, semantic, and sliding window chunking with structure preservation.
"""

import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.exceptions import ProcessingError
from app.models.schemas import DocumentChunk
from app.utils.text_processing import text_preprocessor, TextQuality

logger = get_logger(__name__)


class ChunkType(Enum):
    """Types of text chunks."""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    HEADER = "header"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    TABLE = "table"


class ChunkStrategy(Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 200
    preserve_structure: bool = True
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    include_headers: bool = True
    split_on_whitespace: bool = True
    token_based: bool = False  # If True, use token count; if False, use character count
    
    def __post_init__(self):
        """Validate configuration."""
        if self.overlap_size >= self.max_chunk_size:
            raise ValueError("Overlap size must be less than max chunk size")
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("Min chunk size must be less than or equal to max chunk size")


@dataclass
class ProcessedChunk:
    """Processed text chunk with metadata."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    start_position: int = 0
    end_position: int = 0
    chunk_type: ChunkType = ChunkType.FIXED_SIZE
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_document_chunk(self) -> DocumentChunk:
        """Convert to DocumentChunk schema."""
        return DocumentChunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            chunk_index=self.chunk_index,
            start_char=self.start_position,
            end_char=self.end_position,
            metadata={
                "chunk_type": self.chunk_type.value,
                **self.metadata
            }
        )


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration."""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Chunk text into segments."""
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _get_text_size(self, text: str) -> int:
        """Get text size based on configuration."""
        if self.config.token_based:
            return self._estimate_tokens(text)
        return len(text)
    
    def _create_chunk(self, text: str, start_pos: int, end_pos: int, 
                     document_id: str, chunk_index: int, 
                     chunk_type: ChunkType = ChunkType.FIXED_SIZE,
                     metadata: Optional[Dict[str, Any]] = None) -> ProcessedChunk:
        """Create a processed chunk."""
        return ProcessedChunk(
            document_id=document_id,
            content=text.strip(),
            start_position=start_pos,
            end_position=end_pos,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            metadata=metadata or {}
        )


class FixedSizeChunker(BaseChunker):
    """Fixed-size text chunker."""
    
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        max_size = self.config.max_chunk_size
        overlap = self.config.overlap_size
        
        if not text.strip():
            return chunks
        
        # Split into words for better boundary handling
        if self.config.split_on_whitespace:
            words = text.split()
            current_chunk = []
            current_size = 0
            chunk_index = 0
            start_pos = 0
            
            for word in words:
                word_size = self._get_text_size(word + " ")
                
                if current_size + word_size > max_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    end_pos = start_pos + len(chunk_text)
                    
                    chunk = self._create_chunk(
                        chunk_text, start_pos, end_pos, document_id, chunk_index
                    )
                    chunks.append(chunk)
                    
                    # Handle overlap
                    if overlap > 0:
                        overlap_words = []
                        overlap_size = 0
                        for w in reversed(current_chunk):
                            w_size = self._get_text_size(w + " ")
                            if overlap_size + w_size <= overlap:
                                overlap_words.insert(0, w)
                                overlap_size += w_size
                            else:
                                break
                        current_chunk = overlap_words
                        current_size = overlap_size
                        start_pos = end_pos - len(" ".join(overlap_words))
                    else:
                        current_chunk = []
                        current_size = 0
                        start_pos = end_pos
                    
                    chunk_index += 1
                
                current_chunk.append(word)
                current_size += word_size
            
            # Handle remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunk = self._create_chunk(
                    chunk_text, start_pos, end_pos, document_id, chunk_index
                )
                chunks.append(chunk)
        
        else:
            # Character-based splitting
            step = max_size - overlap
            for i in range(0, len(text), step):
                end = min(i + max_size, len(text))
                chunk_text = text[i:end]
                
                if chunk_text.strip():
                    chunk = self._create_chunk(
                        chunk_text, i, end, document_id, len(chunks)
                    )
                    chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic text chunker that preserves meaning."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.sentence_endings = re.compile(r'[.!?]+(?:\s|$)')
        self.paragraph_separators = re.compile(r'\n\s*\n')
    
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Split text semantically by sentences and paragraphs."""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # First, split by paragraphs if configured
        if self.config.respect_paragraph_boundaries:
            paragraphs = self.paragraph_separators.split(text)
        else:
            paragraphs = [text]
        
        chunk_index = 0
        current_position = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_chunks = self._chunk_paragraph(
                paragraph, document_id, chunk_index, current_position
            )
            chunks.extend(para_chunks)
            chunk_index += len(para_chunks)
            current_position += len(paragraph) + 2  # Account for paragraph separator
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _chunk_paragraph(self, paragraph: str, document_id: str, 
                        start_index: int, start_position: int) -> List[ProcessedChunk]:
        """Chunk a single paragraph."""
        if not paragraph.strip():
            return []
        
        # Split by sentences if configured
        if self.config.respect_sentence_boundaries:
            sentences = self._split_sentences(paragraph)
        else:
            sentences = [paragraph]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start_pos = start_position
        
        for sentence in sentences:
            sentence_size = self._get_text_size(sentence)
            
            # Check if adding this sentence would exceed max size
            if (current_size + sentence_size > self.config.max_chunk_size and 
                current_chunk and current_size >= self.config.min_chunk_size):
                
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunk_end_pos = chunk_start_pos + len(chunk_text)
                
                chunk = self._create_chunk(
                    chunk_text, chunk_start_pos, chunk_end_pos, 
                    document_id, start_index + len(chunks),
                    ChunkType.SEMANTIC
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size
                chunk_start_pos = chunk_end_pos
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_end_pos = chunk_start_pos + len(chunk_text)
            
            chunk = self._create_chunk(
                chunk_text, chunk_start_pos, chunk_end_pos,
                document_id, start_index + len(chunks),
                ChunkType.SEMANTIC
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = []
        current_sentence = ""
        
        # Simple sentence splitting
        parts = self.sentence_endings.split(text)
        
        for i, part in enumerate(parts):
            if i < len(parts) - 1:  # Not the last part
                current_sentence += part + "."
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # Last part (no sentence ending)
                if part.strip():
                    sentences.append(part.strip())
        
        return [s for s in sentences if s.strip()]


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunker with configurable overlap."""
    
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Create overlapping chunks using sliding window."""
        chunks = []
        
        if not text.strip():
            return chunks
        
        max_size = self.config.max_chunk_size
        overlap = self.config.overlap_size
        step = max_size - overlap
        
        words = text.split()
        chunk_index = 0
        
        for i in range(0, len(words), step):
            # Calculate word boundaries for this chunk
            chunk_words = words[i:i + max_size]
            
            if not chunk_words:
                break
            
            chunk_text = " ".join(chunk_words)
            
            # Calculate character positions (approximate)
            start_pos = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_pos = start_pos + len(chunk_text)
            
            chunk = self._create_chunk(
                chunk_text, start_pos, end_pos, document_id, chunk_index,
                ChunkType.SLIDING_WINDOW,
                {"overlap_size": overlap, "step_size": step}
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Break if we've covered all the text
            if i + max_size >= len(words):
                break
        
        self.logger.info(f"Created {len(chunks)} sliding window chunks")
        return chunks


class StructurePreservingChunker(BaseChunker):
    """Chunker that preserves document structure like headers, lists, etc."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^(\s*[-*+•]|\s*\d+\.)\s+(.+)$', re.MULTILINE)
        self.code_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Chunk text while preserving structure."""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Extract structural elements
        structures = self._extract_structures(text)
        
        # Group structures into chunks
        chunk_index = 0
        current_chunk_elements = []
        current_size = 0
        
        for element in structures:
            element_size = self._get_text_size(element['content'])
            
            # Check if we should create a new chunk
            if (current_size + element_size > self.config.max_chunk_size and 
                current_chunk_elements and 
                current_size >= self.config.min_chunk_size):
                
                # Create chunk from current elements
                chunk = self._create_structure_chunk(
                    current_chunk_elements, document_id, chunk_index
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_elements = [element]
                current_size = element_size
                chunk_index += 1
            else:
                current_chunk_elements.append(element)
                current_size += element_size
        
        # Handle remaining elements
        if current_chunk_elements:
            chunk = self._create_structure_chunk(
                current_chunk_elements, document_id, chunk_index
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} structure-preserving chunks")
        return chunks
    
    def _extract_structures(self, text: str) -> List[Dict[str, Any]]:
        """Extract structural elements from text."""
        elements = []
        lines = text.split('\n')
        current_paragraph = []
        current_pos = 0
        
        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Check for headers
            header_match = self.header_pattern.match(line)
            if header_match:
                # Finish current paragraph if any
                if current_paragraph:
                    para_text = '\n'.join(current_paragraph)
                    elements.append({
                        'type': ChunkType.PARAGRAPH,
                        'content': para_text,
                        'start_pos': line_start - len(para_text),
                        'end_pos': line_start
                    })
                    current_paragraph = []
                
                # Add header
                elements.append({
                    'type': ChunkType.HEADER,
                    'content': line,
                    'start_pos': line_start,
                    'end_pos': line_end,
                    'level': len(header_match.group(1))
                })
            
            # Check for list items
            elif self.list_pattern.match(line):
                # Finish current paragraph if any
                if current_paragraph:
                    para_text = '\n'.join(current_paragraph)
                    elements.append({
                        'type': ChunkType.PARAGRAPH,
                        'content': para_text,
                        'start_pos': line_start - len(para_text),
                        'end_pos': line_start
                    })
                    current_paragraph = []
                
                # Add list item
                elements.append({
                    'type': ChunkType.LIST_ITEM,
                    'content': line,
                    'start_pos': line_start,
                    'end_pos': line_end
                })
            
            # Regular text line
            elif line.strip():
                current_paragraph.append(line)
            
            # Empty line - potential paragraph break
            else:
                if current_paragraph:
                    para_text = '\n'.join(current_paragraph)
                    elements.append({
                        'type': ChunkType.PARAGRAPH,
                        'content': para_text,
                        'start_pos': line_start - len(para_text),
                        'end_pos': line_start
                    })
                    current_paragraph = []
            
            current_pos = line_end + 1  # +1 for newline
        
        # Handle final paragraph
        if current_paragraph:
            para_text = '\n'.join(current_paragraph)
            elements.append({
                'type': ChunkType.PARAGRAPH,
                'content': para_text,
                'start_pos': current_pos - len(para_text),
                'end_pos': current_pos
            })
        
        return elements
    
    def _create_structure_chunk(self, elements: List[Dict[str, Any]], 
                              document_id: str, chunk_index: int) -> ProcessedChunk:
        """Create a chunk from structural elements."""
        if not elements:
            return None
        
        # Combine content
        content_parts = [elem['content'] for elem in elements]
        combined_content = '\n'.join(content_parts)
        
        # Calculate positions
        start_pos = min(elem.get('start_pos', 0) for elem in elements)
        end_pos = max(elem.get('end_pos', 0) for elem in elements)
        
        # Determine chunk type
        types = [elem['type'] for elem in elements]
        if ChunkType.HEADER in types:
            chunk_type = ChunkType.HEADER
        elif ChunkType.LIST_ITEM in types:
            chunk_type = ChunkType.LIST_ITEM
        else:
            chunk_type = ChunkType.PARAGRAPH
        
        # Create metadata
        metadata = {
            'element_count': len(elements),
            'element_types': [t.value for t in types],
            'has_header': ChunkType.HEADER in types,
            'has_list': ChunkType.LIST_ITEM in types
        }
        
        return self._create_chunk(
            combined_content, start_pos, end_pos, document_id, chunk_index,
            chunk_type, metadata
        )


class HybridChunker(BaseChunker):
    """Hybrid chunker that combines multiple strategies."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.semantic_chunker = SemanticChunker(config)
        self.structure_chunker = StructurePreservingChunker(config)
        self.fixed_chunker = FixedSizeChunker(config)
    
    def chunk_text(self, text: str, document_id: str) -> List[ProcessedChunk]:
        """Apply hybrid chunking strategy."""
        if not text.strip():
            return []
        
        # Use structure-preserving chunker if structure preservation is enabled
        if self.config.preserve_structure:
            chunks = self.structure_chunker.chunk_text(text, document_id)
        else:
            # Use semantic chunker if sentence/paragraph boundaries are respected
            if (self.config.respect_sentence_boundaries or 
                self.config.respect_paragraph_boundaries):
                chunks = self.semantic_chunker.chunk_text(text, document_id)
            else:
                # Fall back to fixed-size chunker
                chunks = self.fixed_chunker.chunk_text(text, document_id)
        
        # Post-process chunks that are too large
        final_chunks = []
        for chunk in chunks:
            if self._get_text_size(chunk.content) > self.config.max_chunk_size:
                # Split large chunks using fixed-size strategy
                sub_chunks = self.fixed_chunker.chunk_text(chunk.content, document_id)
                # Update indices
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.chunk_index = len(final_chunks) + i
                    sub_chunk.start_position += chunk.start_position
                    sub_chunk.end_position += chunk.start_position
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        self.logger.info(f"Created {len(final_chunks)} hybrid chunks")
        return final_chunks


class AdvancedTextChunker:
    """Main interface for advanced text chunking with multiple strategies."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize with configuration."""
        self.config = config or ChunkingConfig()
        self.logger = get_logger(__name__)
        
        # Initialize chunkers
        self._chunkers = {
            ChunkStrategy.FIXED_SIZE: FixedSizeChunker(self.config),
            ChunkStrategy.SEMANTIC: SemanticChunker(self.config),
            ChunkStrategy.SLIDING_WINDOW: SlidingWindowChunker(self.config),
            ChunkStrategy.HYBRID: HybridChunker(self.config)
        }
    
    def chunk_document(self, text: str, document_id: str, 
                      preprocess: bool = True) -> List[DocumentChunk]:
        """
        Chunk a document using the configured strategy.
        
        Args:
            text: Input text to chunk
            document_id: ID of the source document
            preprocess: Whether to preprocess text before chunking
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            if not text or not text.strip():
                self.logger.warning("Empty or whitespace-only text provided for chunking")
                return []
            
            # Preprocess text if requested
            if preprocess:
                preprocessing_result = text_preprocessor.preprocess_document(text)
                
                if preprocessing_result.get("skipped", False):
                    self.logger.info(f"Skipping chunking due to: {preprocessing_result.get('skip_reason')}")
                    return []
                
                processed_text = preprocessing_result.get("processed_text", text)
            else:
                processed_text = text
            
            # Get appropriate chunker
            chunker = self._chunkers[self.config.strategy]
            
            # Create chunks
            processed_chunks = chunker.chunk_text(processed_text, document_id)
            
            # Convert to DocumentChunk objects
            document_chunks = [chunk.to_document_chunk() for chunk in processed_chunks]
            
            # Add chunking metadata
            for i, chunk in enumerate(document_chunks):
                chunk.metadata.update({
                    "chunking_strategy": self.config.strategy.value,
                    "total_chunks": len(document_chunks),
                    "chunk_size_config": self.config.max_chunk_size,
                    "overlap_size_config": self.config.overlap_size,
                    "preprocessed": preprocess
                })
            
            self.logger.info(f"Successfully chunked document {document_id} into {len(document_chunks)} chunks")
            
            return document_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk document {document_id}: {str(e)}")
            raise ProcessingError(f"Chunking failed: {str(e)}")
    
    def update_config(self, new_config: ChunkingConfig):
        """Update chunking configuration."""
        self.config = new_config
        
        # Reinitialize chunkers with new config
        self._chunkers = {
            ChunkStrategy.FIXED_SIZE: FixedSizeChunker(self.config),
            ChunkStrategy.SEMANTIC: SemanticChunker(self.config),
            ChunkStrategy.SLIDING_WINDOW: SlidingWindowChunker(self.config),
            ChunkStrategy.HYBRID: HybridChunker(self.config)
        }
        
        self.logger.info(f"Updated chunking configuration: strategy={self.config.strategy.value}")


# Global instance with default configuration
default_chunker = AdvancedTextChunker()
