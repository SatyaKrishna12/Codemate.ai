"""
Enhanced text splitter with sophisticated chunking strategies.
Provides LangChain-compatible interface while using advanced chunking internally.
"""

import re
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from app.utils.chunking import (
    AdvancedTextChunker, ChunkingConfig, ChunkStrategy, ChunkType
)
from app.utils.text_processing import text_preprocessor
from app.core.logging import get_logger

logger = get_logger(__name__)


class SplitterType(Enum):
    """Types of text splitters."""
    RECURSIVE_CHARACTER = "recursive_character"
    SEMANTIC = "semantic"
    TOKEN = "token"
    MARKDOWN_HEADER = "markdown_header"
    SENTENCE = "sentence"


class RecursiveCharacterTextSplitter:
    """
    Enhanced text splitter that splits text recursively by different separators.
    Integrates with our sophisticated chunking system.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
            keep_separator: Whether to keep separators in chunks
            is_separator_regex: Whether separators are regex patterns
            **kwargs: Additional configuration options
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        
        # Configure advanced chunker with validation
        min_chunk_size = max(50, chunk_size // 20)
        # Ensure overlap is less than chunk size and reasonable
        validated_overlap = min(chunk_overlap, chunk_size - 1, chunk_size // 2)
        
        self.chunking_config = ChunkingConfig(
            strategy=ChunkStrategy.HYBRID,
            max_chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=validated_overlap,
            preserve_structure=True,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using sophisticated chunking.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        try:
            # Use advanced chunker for better results
            document_chunks = self.chunker.chunk_document(
                text, 
                document_id="temp_split", 
                preprocess=True
            )
            
            # Extract just the content strings
            chunks = [chunk.content for chunk in document_chunks if chunk.content.strip()]
            
            logger.debug(f"Split text into {len(chunks)} chunks using advanced chunker")
            return chunks
            
        except Exception as e:
            logger.warning(f"Advanced chunking failed, falling back to simple splitting: {str(e)}")
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """Fallback to simple recursive splitting if advanced chunking fails."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by separators in order of preference
        for separator in self.separators:
            if separator in text:
                if self.is_separator_regex:
                    parts = re.split(separator, text)
                else:
                    parts = text.split(separator)
                
                for i, part in enumerate(parts):
                    # Add separator back if keeping separators
                    if (i < len(parts) - 1 and separator and 
                        self.keep_separator and not self.is_separator_regex):
                        part += separator
                    
                    # Check if adding part would exceed chunk size
                    if len(current_chunk) + len(part) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        
                        # Handle overlap
                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            current_chunk = current_chunk[-self.chunk_overlap:] + part
                        else:
                            current_chunk = part
                    else:
                        current_chunk += part
                
                break
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle oversized chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Force split large chunks
                for i in range(0, len(chunk), self.chunk_size - self.chunk_overlap):
                    sub_chunk = chunk[i:i + self.chunk_size]
                    if sub_chunk.strip():
                        final_chunks.append(sub_chunk.strip())
        
        return [chunk for chunk in final_chunks if chunk.strip()]


class SemanticChunker:
    """
    Semantic chunker that preserves meaning and context.
    Uses our advanced semantic chunking capabilities.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        sentence_split_regex: str = r'[.!?]+(?:\s|$)',
        **kwargs
    ):
        """Initialize semantic chunker."""
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_split_regex = sentence_split_regex
        
        # Configure for semantic chunking
        self.chunking_config = ChunkingConfig(
            strategy=ChunkStrategy.SEMANTIC,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=0,  # Semantic chunks typically don't overlap
            preserve_structure=True,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
    
    def split_text(self, text: str) -> List[str]:
        """Split text semantically."""
        if not text or not text.strip():
            return []
        
        try:
            document_chunks = self.chunker.chunk_document(
                text, 
                document_id="semantic_split", 
                preprocess=True
            )
            
            chunks = [chunk.content for chunk in document_chunks if chunk.content.strip()]
            logger.debug(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}")
            return [text]  # Return original text as fallback


class TokenTextSplitter:
    """
    Token-based text splitter with approximate token counting.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "gpt2",
        **kwargs
    ):
        """Initialize token splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        
        # Configure for token-based chunking with validation
        validated_overlap = min(chunk_overlap, chunk_size - 1, chunk_size // 2)
        
        self.chunking_config = ChunkingConfig(
            strategy=ChunkStrategy.FIXED_SIZE,
            max_chunk_size=chunk_size,
            min_chunk_size=max(50, chunk_size // 20),
            overlap_size=validated_overlap,
            token_based=True,  # Use token counting
            preserve_structure=False,
            respect_sentence_boundaries=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
    
    def split_text(self, text: str) -> List[str]:
        """Split text by token count."""
        if not text or not text.strip():
            return []
        
        try:
            document_chunks = self.chunker.chunk_document(
                text, 
                document_id="token_split", 
                preprocess=False  # Don't preprocess for token splitting
            )
            
            chunks = [chunk.content for chunk in document_chunks if chunk.content.strip()]
            logger.debug(f"Created {len(chunks)} token-based chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Token-based chunking failed: {str(e)}")
            # Fallback to character-based approximation
            return self._approximate_token_split(text)
    
    def _approximate_token_split(self, text: str) -> List[str]:
        """Approximate token-based splitting using character count."""
        # Rough approximation: 1 token ≈ 4 characters
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap
        )
        return splitter.split_text(text)


class MarkdownHeaderTextSplitter:
    """
    Markdown header-based text splitter that preserves document structure.
    """
    
    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple]] = None,
        return_each_line: bool = False,
        **kwargs
    ):
        """
        Initialize markdown header splitter.
        
        Args:
            headers_to_split_on: List of tuples (header_level, header_name)
            return_each_line: Whether to return each line separately
        """
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6")
        ]
        self.return_each_line = return_each_line
        
        # Configure for structure-preserving chunking
        self.chunking_config = ChunkingConfig(
            strategy=ChunkStrategy.HYBRID,
            max_chunk_size=2000,  # Larger chunks for structured content
            min_chunk_size=100,
            overlap_size=0,  # No overlap for header-based splitting
            preserve_structure=True,
            include_headers=True,
            respect_paragraph_boundaries=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
    
    def split_text(self, text: str) -> List[str]:
        """Split markdown text by headers."""
        if not text or not text.strip():
            return []
        
        try:
            document_chunks = self.chunker.chunk_document(
                text, 
                document_id="markdown_split", 
                preprocess=False  # Preserve original markdown structure
            )
            
            # Filter for chunks that contain headers or significant content
            chunks = []
            for chunk in document_chunks:
                if (chunk.content.strip() and 
                    (chunk.metadata.get('has_header', False) or 
                     len(chunk.content) > 50)):
                    chunks.append(chunk.content)
            
            logger.debug(f"Created {len(chunks)} markdown header-based chunks")
            return chunks or [text]  # Return original if no good chunks
            
        except Exception as e:
            logger.error(f"Markdown header splitting failed: {str(e)}")
            return self._fallback_markdown_split(text)
    
    def _fallback_markdown_split(self, text: str) -> List[str]:
        """Fallback markdown splitting by headers."""
        chunks = []
        current_chunk = ""
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a header
            is_header = False
            for header_prefix, _ in self.headers_to_split_on:
                if line.strip().startswith(header_prefix + ' '):
                    is_header = True
                    break
            
            if is_header and current_chunk.strip():
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks or [text]


class TextSplitterFactory:
    """Factory for creating different types of text splitters."""
    
    @staticmethod
    def create_splitter(
        splitter_type: Union[str, SplitterType],
        **kwargs
    ) -> Union[RecursiveCharacterTextSplitter, SemanticChunker, 
               TokenTextSplitter, MarkdownHeaderTextSplitter]:
        """
        Create a text splitter of the specified type.
        
        Args:
            splitter_type: Type of splitter to create
            **kwargs: Configuration arguments for the splitter
            
        Returns:
            Configured text splitter instance
        """
        if isinstance(splitter_type, str):
            splitter_type = SplitterType(splitter_type)
        
        if splitter_type == SplitterType.RECURSIVE_CHARACTER:
            return RecursiveCharacterTextSplitter(**kwargs)
        elif splitter_type == SplitterType.SEMANTIC:
            return SemanticChunker(**kwargs)
        elif splitter_type == SplitterType.TOKEN:
            return TokenTextSplitter(**kwargs)
        elif splitter_type == SplitterType.MARKDOWN_HEADER:
            return MarkdownHeaderTextSplitter(**kwargs)
        elif splitter_type == SplitterType.SENTENCE:
            # Sentence splitter is essentially semantic with small chunks
            kwargs.setdefault('max_chunk_size', 500)
            kwargs.setdefault('min_chunk_size', 50)
            return SemanticChunker(**kwargs)
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")


# Convenience functions for compatibility
def create_recursive_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Create a recursive character text splitter."""
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_semantic_splitter(max_chunk_size: int = 1000) -> SemanticChunker:
    """Create a semantic chunker."""
    return SemanticChunker(max_chunk_size=max_chunk_size)


def create_token_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> TokenTextSplitter:
    """Create a token-based text splitter."""
    return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """Create a markdown header text splitter."""
    return MarkdownHeaderTextSplitter()
