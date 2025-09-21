"""Enhanced document processor with sophisticated text processing and chunking."""
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import uuid
from datetime import datetime

from PyPDF2 import PdfReader
from docx import Document as DocxDocument

from app.core.logging import get_logger
from app.core.exceptions import DocumentError, ProcessingError
from app.models.schemas import DocumentChunk, DocumentType
from app.utils.text_processing import text_preprocessor, TextQuality
from app.utils.chunking import AdvancedTextChunker, ChunkingConfig, ChunkStrategy

logger = get_logger(__name__)


class DocumentProcessor:
    """Enhanced document processor with sophisticated text processing and chunking."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        
        # Initialize advanced chunker with default configuration
        self.chunking_config = ChunkingConfig(
            strategy=ChunkStrategy.HYBRID,
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=200,
            preserve_structure=True,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True,
            include_headers=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
    
    async def process_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a file and extract text content and metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (text_content, metadata)
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise DocumentError(f"File not found: {file_path}")
            
            extension = file_path_obj.suffix.lower()
            
            if extension == '.pdf':
                return await self._process_pdf(file_path)
            elif extension == '.docx':
                return await self._process_docx(file_path)
            elif extension in ['.txt', '.md']:
                return await self._process_text(file_path)
            else:
                raise DocumentError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to process file: {str(e)}")
    
    async def _process_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Extract text
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # Extract metadata
                metadata = {
                    'page_count': len(reader.pages),
                    'title': reader.metadata.get('/Title') if reader.metadata else None,
                    'author': reader.metadata.get('/Author') if reader.metadata else None,
                    'subject': reader.metadata.get('/Subject') if reader.metadata else None,
                    'creation_date': reader.metadata.get('/CreationDate') if reader.metadata else None,
                    'modification_date': reader.metadata.get('/ModDate') if reader.metadata else None
                }
                
                return text_content.strip(), metadata
                
        except Exception as e:
            raise ProcessingError(f"Error processing PDF: {str(e)}")
    
    async def _process_docx(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process DOCX file."""
        try:
            doc = DocxDocument(file_path)
            
            # Extract text
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            # Extract metadata
            metadata = {
                'title': doc.core_properties.title,
                'author': doc.core_properties.author,
                'subject': doc.core_properties.subject,
                'creation_date': doc.core_properties.created,
                'modification_date': doc.core_properties.modified
            }
            
            return text_content.strip(), metadata
            
        except Exception as e:
            raise ProcessingError(f"Error processing DOCX: {str(e)}")
    
    async def _process_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process text file with encoding detection."""
        try:
            file_path_obj = Path(file_path)
            
            # Read raw bytes first for encoding detection
            with open(file_path, 'rb') as file:
                raw_data = file.read()
            
            # Detect encoding
            encoding = text_preprocessor.detect_encoding(raw_data)
            
            # Read with detected encoding
            try:
                text_content = raw_data.decode(encoding)
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode with {encoding}, falling back to utf-8 with error handling")
                text_content = raw_data.decode('utf-8', errors='replace')
            
            # Basic metadata
            file_stat = os.stat(file_path)
            metadata = {
                'creation_date': datetime.fromtimestamp(file_stat.st_ctime),
                'modification_date': datetime.fromtimestamp(file_stat.st_mtime),
                'encoding': encoding,
                'file_extension': file_path_obj.suffix.lower()
            }
            
            return text_content.strip(), metadata
            
        except Exception as e:
            raise ProcessingError(f"Error processing text file: {str(e)}")
    
    async def create_chunks(self, text: str, document_id: str, 
                          custom_config: ChunkingConfig = None) -> List[DocumentChunk]:
        """
        Create sophisticated text chunks using advanced chunking strategies.
        
        Args:
            text: Text to chunk
            document_id: ID of the source document
            custom_config: Optional custom chunking configuration
            
        Returns:
            List of document chunks
        """
        try:
            if not text:
                return []
            
            # Use custom config if provided
            if custom_config:
                chunker = AdvancedTextChunker(custom_config)
            else:
                chunker = self.chunker
            
            # Create chunks with preprocessing
            chunks = chunker.chunk_document(text, document_id, preprocess=True)
            
            logger.info(f"Created {len(chunks)} sophisticated chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise ProcessingError(f"Failed to create chunks: {str(e)}")
    
    def configure_chunking(self, strategy: ChunkStrategy = ChunkStrategy.HYBRID,
                          max_chunk_size: int = 1000, overlap_size: int = 200,
                          preserve_structure: bool = True) -> None:
        """
        Configure chunking strategy and parameters.
        
        Args:
            strategy: Chunking strategy to use
            max_chunk_size: Maximum size of chunks
            overlap_size: Overlap between chunks
            preserve_structure: Whether to preserve document structure
        """
        self.chunking_config = ChunkingConfig(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(100, max_chunk_size // 10),
            overlap_size=overlap_size,
            preserve_structure=preserve_structure,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True,
            include_headers=True
        )
        self.chunker = AdvancedTextChunker(self.chunking_config)
        
        logger.info(f"Updated chunking configuration: strategy={strategy.value}, "
                   f"max_size={max_chunk_size}, overlap={overlap_size}")
    
    async def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyze text quality and characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with quality metrics and analysis
        """
        try:
            # Preprocess and analyze
            preprocessing_result = text_preprocessor.preprocess_document(text)
            
            # Additional analysis
            lines = text.split('\n')
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            analysis = {
                **preprocessing_result,
                'line_count': len(lines),
                'paragraph_count': len(paragraphs),
                'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
                'has_structure': any(line.strip().startswith('#') for line in lines),
                'has_lists': any(line.strip().startswith(('-', '*', '+')) or 
                               bool(re.match(r'^\s*\d+\.', line.strip())) for line in lines)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text quality: {str(e)}")
            raise ProcessingError(f"Failed to analyze text quality: {str(e)}")
