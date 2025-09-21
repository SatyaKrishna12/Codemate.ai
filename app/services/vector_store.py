"""
Vector database service using FAISS for efficient similarity search.
Integrates with LangChain for vector store operations.
"""

import os
import pickle
import asyncio
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import faiss

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import VectorStoreError
from app.models.schemas import DocumentChunk, SearchResult
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)


class VectorStore:
    """FAISS-based vector store for document chunks."""
    
    def __init__(self):
        """Initialize the vector store."""
        self._index: Optional[faiss.Index] = None
        self._chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index = 0
        self._dimension: Optional[int] = None
        
        # Ensure vectors directory exists
        os.makedirs(settings.vectors_dir, exist_ok=True)
        
        self._index_file = os.path.join(settings.vectors_dir, f"{settings.vector_index_name}.index")
        self._metadata_file = os.path.join(settings.vectors_dir, f"{settings.vector_index_name}_metadata.pkl")
    
    async def initialize(self, dimension: Optional[int] = None) -> None:
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (will be inferred if not provided)
        """
        try:
            logger.info("Initializing vector store")
            
            # Ensure embedding service is initialized
            if not embedding_service.is_initialized:
                await embedding_service.initialize()
            
            # Get dimension from embedding service if not provided
            if dimension is None:
                dimension = embedding_service.embedding_dimension
            
            if dimension is None:
                raise VectorStoreError("Could not determine embedding dimension")
            
            self._dimension = dimension
            
            # Try to load existing index
            if await self._load_index():
                logger.info(
                    "Loaded existing vector index",
                    dimension=self._dimension,
                    num_vectors=self._index.ntotal
                )
            else:
                # Create new index
                await self._create_index()
                logger.info(
                    "Created new vector index",
                    dimension=self._dimension
                )
            
        except Exception as e:
            logger.error(
                "Failed to initialize vector store",
                error=str(e),
                exc_info=True
            )
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
    
    async def _create_index(self) -> None:
        """Create a new FAISS index."""
        try:
            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self._index = faiss.IndexFlatIP(self._dimension)
            
            # Initialize metadata structures
            self._chunk_metadata = {}
            self._id_to_index = {}
            self._index_to_id = {}
            self._next_index = 0
            
            logger.info(
                "Created new FAISS index",
                dimension=self._dimension,
                index_type="IndexFlatIP"
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create FAISS index: {str(e)}")
    
    async def _load_index(self) -> bool:
        """
        Load existing index from disk.
        
        Returns:
            True if loaded successfully, False if no index exists
        """
        try:
            if not os.path.exists(self._index_file) or not os.path.exists(self._metadata_file):
                return False
            
            # Load FAISS index
            loop = asyncio.get_event_loop()
            self._index = await loop.run_in_executor(
                None,
                faiss.read_index,
                self._index_file
            )
            
            # Load metadata
            with open(self._metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self._chunk_metadata = metadata.get('chunk_metadata', {})
                self._id_to_index = metadata.get('id_to_index', {})
                self._index_to_id = metadata.get('index_to_id', {})
                self._next_index = metadata.get('next_index', 0)
            
            return True
            
        except Exception as e:
            logger.warning(
                "Failed to load existing index",
                error=str(e)
            )
            return False
    
    async def save_index(self) -> None:
        """Save index and metadata to disk."""
        try:
            if self._index is None:
                return
            
            logger.info("Saving vector index to disk")
            
            # Save FAISS index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                faiss.write_index,
                self._index,
                self._index_file
            )
            
            # Save metadata
            metadata = {
                'chunk_metadata': self._chunk_metadata,
                'id_to_index': self._id_to_index,
                'index_to_id': self._index_to_id,
                'next_index': self._next_index
            }
            
            with open(self._metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(
                "Vector index saved successfully",
                num_vectors=self._index.ntotal
            )
            
        except Exception as e:
            logger.error(
                "Failed to save vector index",
                error=str(e),
                exc_info=True
            )
            raise VectorStoreError(f"Failed to save vector index: {str(e)}")
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
        """
        try:
            if not chunks:
                return
            
            if self._index is None:
                await self.initialize()
            
            logger.info(
                "Adding chunks to vector store",
                chunk_count=len(chunks)
            )
            
            # Generate embeddings for chunks
            embeddings = await embedding_service.embed_chunks(chunks)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._index.add,
                embeddings_array
            )
            
            # Update metadata
            for i, chunk in enumerate(chunks):
                index_id = self._next_index + i
                
                self._chunk_metadata[index_id] = {
                    'chunk_id': chunk.chunk_id,
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata or {}
                }
                
                self._id_to_index[chunk.chunk_id] = index_id
                self._index_to_id[index_id] = chunk.chunk_id
            
            self._next_index += len(chunks)
            
            # Save index
            await self.save_index()
            
            logger.info(
                "Chunks added successfully",
                chunk_count=len(chunks),
                total_vectors=self._index.ntotal
            )
            
        except Exception as e:
            logger.error(
                "Failed to add chunks to vector store",
                chunk_count=len(chunks),
                error=str(e),
                exc_info=True
            )
            raise VectorStoreError(f"Failed to add chunks: {str(e)}")
    
    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        similarity_threshold: float = 0.7,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            document_ids: Filter by document IDs (optional)
            
        Returns:
            List of search results
        """
        try:
            if self._index is None or self._index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            loop = asyncio.get_event_loop()
            scores, indices = await loop.run_in_executor(
                None,
                self._index.search,
                query_embedding,
                min(k * 2, self._index.ntotal)  # Get more results for filtering
            )
            
            # Process results
            results = []
            scores = scores[0]  # Get first (and only) query results
            indices = indices[0]
            
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Convert FAISS inner product score to cosine similarity
                similarity_score = float(score)
                
                if similarity_score < similarity_threshold:
                    continue
                
                # Get chunk metadata
                chunk_metadata = self._chunk_metadata.get(idx)
                if not chunk_metadata:
                    continue
                
                # Filter by document IDs if specified
                if document_ids and chunk_metadata['document_id'] not in document_ids:
                    continue
                
                # Create search result
                result = SearchResult(
                    document_id=chunk_metadata['document_id'],
                    chunk_id=chunk_metadata['chunk_id'],
                    content=chunk_metadata['content'],
                    similarity_score=similarity_score,
                    document_filename=chunk_metadata['metadata'].get('filename', ''),
                    chunk_index=chunk_metadata['chunk_index'],
                    metadata=chunk_metadata['metadata']
                )
                
                results.append(result)
                
                if len(results) >= k:
                    break
            
            logger.info(
                "Vector search completed",
                query_results=len(results),
                similarity_threshold=similarity_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to perform vector search",
                error=str(e),
                exc_info=True
            )
            raise VectorStoreError(f"Failed to perform search: {str(e)}")
    
    async def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            if self._index is None:
                return 0
            
            # Find chunks to delete
            chunks_to_delete = []
            for idx, metadata in self._chunk_metadata.items():
                if metadata['document_id'] == document_id:
                    chunks_to_delete.append(idx)
            
            if not chunks_to_delete:
                return 0
            
            # FAISS doesn't support deletion, so we need to rebuild the index
            logger.info(
                "Rebuilding index after document deletion",
                document_id=document_id,
                chunks_to_delete=len(chunks_to_delete)
            )
            
            # Collect remaining chunks
            remaining_embeddings = []
            remaining_metadata = {}
            new_id_to_index = {}
            new_index_to_id = {}
            new_index = 0
            
            for idx in range(self._index.ntotal):
                if idx not in chunks_to_delete:
                    # Get embedding from index
                    embedding = self._index.reconstruct(idx)
                    remaining_embeddings.append(embedding)
                    
                    # Update metadata
                    chunk_metadata = self._chunk_metadata[idx]
                    remaining_metadata[new_index] = chunk_metadata
                    
                    chunk_id = chunk_metadata['chunk_id']
                    new_id_to_index[chunk_id] = new_index
                    new_index_to_id[new_index] = chunk_id
                    
                    new_index += 1
            
            # Rebuild index
            if remaining_embeddings:
                embeddings_array = np.array(remaining_embeddings, dtype=np.float32)
                self._index = faiss.IndexFlatIP(self._dimension)
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._index.add,
                    embeddings_array
                )
            else:
                # Create empty index
                self._index = faiss.IndexFlatIP(self._dimension)
            
            # Update metadata
            self._chunk_metadata = remaining_metadata
            self._id_to_index = new_id_to_index
            self._index_to_id = new_index_to_id
            self._next_index = new_index
            
            # Save updated index
            await self.save_index()
            
            logger.info(
                "Document chunks deleted successfully",
                document_id=document_id,
                chunks_deleted=len(chunks_to_delete),
                remaining_chunks=self._index.ntotal
            )
            
            return len(chunks_to_delete)
            
        except Exception as e:
            logger.error(
                "Failed to delete document chunks",
                document_id=document_id,
                error=str(e),
                exc_info=True
            )
            raise VectorStoreError(f"Failed to delete document chunks: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary
        """
        try:
            if self._index is None:
                return {
                    "initialized": False,
                    "total_vectors": 0,
                    "dimension": self._dimension
                }
            
            # Count documents
            document_ids = set()
            for metadata in self._chunk_metadata.values():
                document_ids.add(metadata['document_id'])
            
            return {
                "initialized": True,
                "total_vectors": self._index.ntotal,
                "total_documents": len(document_ids),
                "dimension": self._dimension,
                "index_type": type(self._index).__name__
            }
            
        except Exception as e:
            logger.error(
                "Failed to get vector store stats",
                error=str(e),
                exc_info=True
            )
            return {
                "initialized": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the vector store.
        
        Returns:
            Health check status
        """
        try:
            stats = await self.get_stats()
            
            status = {
                "service": "vector_store",
                "status": "healthy" if stats.get("initialized", False) else "unhealthy",
                "stats": stats
            }
            
            return status
            
        except Exception as e:
            logger.error(
                "Vector store health check failed",
                error=str(e),
                exc_info=True
            )
            return {
                "service": "vector_store",
                "status": "unhealthy",
                "error": str(e)
            }


# Global vector store instance
vector_store = VectorStore()
