"""
Embedding service for generating vector embeddings from text using sentence transformers.
Integrates with LangChain for consistent embedding operations.
"""

import asyncio
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import EmbeddingError
from app.models.schemas import DocumentChunk

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self._model: Optional[SentenceTransformer] = None
        self._model_name = settings.embedding_model_name
        self._cache_dir = settings.embedding_model_cache_dir
    
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info(
                "Initializing embedding model",
                model_name=self._model_name,
                cache_dir=self._cache_dir
            )
            
            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                self._load_model
            )
            
            logger.info(
                "Embedding model initialized successfully",
                model_name=self._model_name,
                embedding_dim=self.embedding_dimension
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize embedding model",
                model_name=self._model_name,
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(
                f"Failed to initialize embedding model: {str(e)}",
                details={"model_name": self._model_name}
            )
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            return SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_dir
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load model: {str(e)}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self._model is not None
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        if self._model is None:
            return None
        return self._model.get_sentence_embedding_dimension()
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if not text.strip():
                raise EmbeddingError("Cannot embed empty text")
            
            # Generate embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._model.encode,
                text,
                {"convert_to_numpy": True, "normalize_embeddings": True}
            )
            
            return embedding
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Error generating embedding",
                text_length=len(text),
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}",
                details={"text_length": len(text)}
            )
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                raise EmbeddingError("No valid texts to embed")
            
            logger.info(
                "Generating embeddings for multiple texts",
                text_count=len(valid_texts)
            )
            
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._model.encode,
                valid_texts,
                {"convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": True}
            )
            
            logger.info(
                "Embeddings generated successfully",
                text_count=len(valid_texts),
                embedding_shape=embeddings.shape
            )
            
            return [embeddings[i] for i in range(len(embeddings))]
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Error generating embeddings",
                text_count=len(texts),
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(
                f"Failed to generate embeddings: {str(e)}",
                details={"text_count": len(texts)}
            )
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            logger.info(
                "Generating embeddings for document chunks",
                chunk_count=len(chunks)
            )
            
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.embed_texts(texts)
            
            logger.info(
                "Chunk embeddings generated successfully",
                chunk_count=len(chunks),
                embedding_count=len(embeddings)
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "Error generating chunk embeddings",
                chunk_count=len(chunks),
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(
                f"Failed to generate chunk embeddings: {str(e)}",
                details={"chunk_count": len(chunks)}
            )
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            logger.debug(
                "Generating query embedding",
                query_length=len(query)
            )
            
            embedding = await self.embed_text(query)
            
            logger.debug(
                "Query embedding generated successfully",
                embedding_dim=len(embedding)
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Error generating query embedding",
                query=query[:100],  # Log first 100 chars only
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(
                f"Failed to generate query embedding: {str(e)}",
                details={"query_length": len(query)}
            )
    
    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize embeddings if not already normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(
                "Error computing similarity",
                error=str(e),
                exc_info=True
            )
            raise EmbeddingError(f"Failed to compute similarity: {str(e)}")
    
    async def health_check(self) -> dict:
        """
        Perform health check for the embedding service.
        
        Returns:
            Health check status
        """
        try:
            status = {
                "service": "embedding",
                "status": "healthy",
                "model_name": self._model_name,
                "initialized": self.is_initialized,
                "embedding_dimension": self.embedding_dimension
            }
            
            if self.is_initialized:
                # Test embedding generation
                test_embedding = await self.embed_text("Health check test")
                status["test_embedding_shape"] = test_embedding.shape
            
            return status
            
        except Exception as e:
            logger.error(
                "Embedding service health check failed",
                error=str(e),
                exc_info=True
            )
            return {
                "service": "embedding",
                "status": "unhealthy",
                "error": str(e),
                "model_name": self._model_name,
                "initialized": self.is_initialized
            }


# Global embedding service instance
embedding_service = EmbeddingService()
