"""
LangChain adapter for the custom FAISS vector store service.
Provides compatibility with LangChain's vector store interface.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.services.vector_store_service import (
    VectorStoreService, SearchService, VectorMetadata, 
    SearchFilters, IndexConfig, EmbeddingProvider
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class LangChainEmbeddingAdapter(EmbeddingProvider):
    """Adapter to make LangChain embeddings work with our vector store."""
    
    def __init__(self, langchain_embeddings: Embeddings):
        """Initialize adapter with LangChain embeddings."""
        self.embeddings = langchain_embeddings
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # LangChain embeddings are typically synchronous
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding, dtype=np.float32)
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = self.embeddings.embed_documents(texts)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]


class CustomFAISS(VectorStore):
    """
    LangChain-compatible wrapper for our custom FAISS vector store.
    
    This allows you to use our advanced vector store service with 
    LangChain's standard interface.
    """
    
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        search_service: SearchService,
        embedding_function: Embeddings,
        **kwargs
    ):
        """Initialize the LangChain adapter."""
        self.vector_store_service = vector_store_service
        self.search_service = search_service
        self.embedding_function = embedding_function
        
        # Create adapter for embedding provider
        self.embedding_adapter = LangChainEmbeddingAdapter(embedding_function)
        
        # Update search service embedding provider
        self.search_service.embedding_provider = self.embedding_adapter
    
    @classmethod
    async def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        storage_path: str = "./langchain_vector_store",
        config: Optional[IndexConfig] = None,
        **kwargs: Any,
    ) -> "CustomFAISS":
        """Create vector store from texts."""
        
        # Create vector store service
        vector_store_service = VectorStoreService(
            storage_path=storage_path,
            config=config
        )
        
        await vector_store_service.init_index()
        
        # Create search service
        embedding_adapter = LangChainEmbeddingAdapter(embedding)
        search_service = SearchService(vector_store_service, embedding_adapter)
        
        # Create instance
        instance = cls(
            vector_store_service=vector_store_service,
            search_service=search_service,
            embedding_function=embedding,
            **kwargs
        )
        
        # Add texts
        await instance.aadd_texts(texts, metadatas)
        
        return instance
    
    @classmethod
    async def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        storage_path: str = "./langchain_vector_store",
        config: Optional[IndexConfig] = None,
        **kwargs: Any,
    ) -> "CustomFAISS":
        """Create vector store from documents."""
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return await cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            storage_path=storage_path,
            config=config,
            **kwargs
        )
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to vector store (synchronous wrapper)."""
        import asyncio
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))
    
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to vector store (async)."""
        
        texts_list = list(texts)
        if not texts_list:
            return []
        
        # Generate embeddings
        embeddings = await self.embedding_adapter.embed_texts(texts_list)
        
        # Prepare data for vector store
        embeddings_data = []
        chunk_ids = []
        
        for i, (text, embedding) in enumerate(zip(texts_list, embeddings)):
            chunk_id = f"chunk_{i}_{hash(text) % 1000000}"
            chunk_ids.append(chunk_id)
            
            # Get metadata for this text
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Create vector metadata
            vector_metadata = VectorMetadata(
                chunk_id=chunk_id,
                document_id=metadata.get('source', 'unknown'),
                content=text,
                metadata=metadata
            )
            
            embeddings_data.append((chunk_id, embedding, vector_metadata))
        
        # Add to vector store
        await self.vector_store_service.add_embeddings(embeddings_data)
        
        return chunk_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents (synchronous wrapper)."""
        import asyncio
        return asyncio.run(self.asimilarity_search(query, k, filter, **kwargs))
    
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents (async)."""
        
        # Convert LangChain filter to our SearchFilters
        search_filters = self._convert_langchain_filter(filter) if filter else None
        
        # Perform search
        results = await self.search_service.search(
            query_text=query,
            k=k,
            filters=search_filters,
            **kwargs
        )
        
        # Convert results to LangChain Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.content,
                metadata={
                    **result.metadata,
                    'chunk_id': result.chunk_id,
                    'document_id': result.document_id,
                    'score': result.final_score,
                    'semantic_score': result.semantic_score,
                    'keyword_score': result.keyword_score
                }
            )
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search with scores (synchronous wrapper)."""
        import asyncio
        return asyncio.run(self.asimilarity_search_with_score(query, k, filter, **kwargs))
    
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search with scores (async)."""
        
        # Convert LangChain filter to our SearchFilters
        search_filters = self._convert_langchain_filter(filter) if filter else None
        
        # Perform search
        results = await self.search_service.search(
            query_text=query,
            k=k,
            filters=search_filters,
            **kwargs
        )
        
        # Convert results to LangChain Documents with scores
        documents_with_scores = []
        for result in results:
            doc = Document(
                page_content=result.content,
                metadata={
                    **result.metadata,
                    'chunk_id': result.chunk_id,
                    'document_id': result.document_id,
                    'semantic_score': result.semantic_score,
                    'keyword_score': result.keyword_score
                }
            )
            documents_with_scores.append((doc, result.final_score))
        
        return documents_with_scores
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search by vector (not fully implemented - would need vector search)."""
        raise NotImplementedError(
            "Direct vector search not implemented. Use similarity_search with text."
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """MMR search (simplified implementation)."""
        # For now, just return regular search results
        # A full MMR implementation would require additional logic
        return self.similarity_search(query, k, filter, **kwargs)
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs."""
        if not ids:
            return False
        
        import asyncio
        removed_count = asyncio.run(self.vector_store_service.remove_by_id(ids))
        return removed_count > 0
    
    async def asave(self, path: Optional[str] = None) -> None:
        """Save vector store."""
        await self.vector_store_service.save(path)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save vector store (sync wrapper)."""
        import asyncio
        asyncio.run(self.asave(path))
    
    @classmethod
    async def aload(
        cls,
        path: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "CustomFAISS":
        """Load vector store from disk."""
        
        # Create vector store service
        vector_store_service = VectorStoreService(storage_path=path)
        await vector_store_service.init_index()
        
        # Load existing data
        loaded = await vector_store_service.load(path)
        if not loaded:
            raise ValueError(f"Could not load vector store from {path}")
        
        # Create search service
        embedding_adapter = LangChainEmbeddingAdapter(embedding)
        search_service = SearchService(vector_store_service, embedding_adapter)
        
        return cls(
            vector_store_service=vector_store_service,
            search_service=search_service,
            embedding_function=embedding,
            **kwargs
        )
    
    @classmethod
    def load(
        cls,
        path: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "CustomFAISS":
        """Load vector store from disk (sync wrapper)."""
        import asyncio
        return asyncio.run(cls.aload(path, embedding, **kwargs))
    
    def _convert_langchain_filter(self, filter_dict: dict) -> SearchFilters:
        """Convert LangChain filter format to our SearchFilters."""
        
        # This is a simple conversion - you might need to adjust
        # based on your specific filter requirements
        
        return SearchFilters(
            document_types=filter_dict.get('document_types'),
            tags=filter_dict.get('tags'),
            date_from=filter_dict.get('date_from'),
            date_to=filter_dict.get('date_to'),
            metadata_filters={k: v for k, v in filter_dict.items() 
                             if k not in ['document_types', 'tags', 'date_from', 'date_to']}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get vector store metrics."""
        return self.vector_store_service.get_metrics()


# Convenience function for LangChain users
def create_custom_faiss(
    texts: List[str],
    embeddings: Embeddings,
    metadatas: Optional[List[dict]] = None,
    storage_path: str = "./custom_faiss_store",
    config: Optional[IndexConfig] = None
) -> CustomFAISS:
    """
    Convenience function to create CustomFAISS vector store.
    
    Args:
        texts: List of text documents
        embeddings: LangChain embeddings instance
        metadatas: Optional metadata for each text
        storage_path: Path to store the vector database
        config: Optional index configuration
        
    Returns:
        CustomFAISS instance
    """
    import asyncio
    return asyncio.run(
        CustomFAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            storage_path=storage_path,
            config=config
        )
    )
