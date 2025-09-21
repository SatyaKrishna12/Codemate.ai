"""
Query processing service for handling search requests and retrieving relevant documents.
Supports semantic search, keyword search, and hybrid search modes.
"""

import time
from typing import List, Optional, Dict, Any
import asyncio
from enum import Enum

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import SearchError, ValidationError
from app.models.schemas import SearchRequest, SearchResponse, SearchResult, SearchType
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)


class QueryProcessor:
    """Service for processing search queries and retrieving relevant documents."""
    
    def __init__(self):
        """Initialize the query processor."""
        self._search_stats = {
            "total_searches": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "total_processing_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the query processor and its dependencies."""
        try:
            logger.info("Initializing query processor")
            
            # Initialize embedding service
            if not embedding_service.is_initialized:
                await embedding_service.initialize()
            
            # Initialize vector store
            await vector_store.initialize()
            
            logger.info("Query processor initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize query processor",
                error=str(e),
                exc_info=True
            )
            raise SearchError(f"Failed to initialize query processor: {str(e)}")
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Process a search request and return results.
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            Search response with results and metadata
            
        Raises:
            SearchError: If search processing fails
            ValidationError: If request validation fails
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Processing search request",
                query=request.query[:100],  # Log first 100 chars
                search_type=request.search_type,
                max_results=request.max_results,
                similarity_threshold=request.similarity_threshold
            )
            
            # Validate request
            await self._validate_search_request(request)
            
            # Process search based on type
            if request.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(request)
            elif request.search_type == SearchType.KEYWORD:
                results = await self._keyword_search(request)
            elif request.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(request)
            else:
                raise ValidationError(f"Unsupported search type: {request.search_type}")
            
            processing_time = time.time() - start_time
            
            # Update statistics
            await self._update_search_stats(request.search_type, processing_time)
            
            # Create response
            response = SearchResponse(
                query=request.query,
                search_type=request.search_type,
                results=results,
                total_results=len(results),
                processing_time=processing_time,
                similarity_threshold=request.similarity_threshold
            )
            
            logger.info(
                "Search completed successfully",
                query=request.query[:100],
                search_type=request.search_type,
                results_count=len(results),
                processing_time=processing_time
            )
            
            return response
            
        except (SearchError, ValidationError):
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Unexpected error during search",
                query=request.query[:100],
                search_type=request.search_type,
                processing_time=processing_time,
                error=str(e),
                exc_info=True
            )
            raise SearchError(
                f"Search failed: {str(e)}",
                details={"query": request.query, "search_type": request.search_type}
            )
    
    async def _validate_search_request(self, request: SearchRequest) -> None:
        """
        Validate search request parameters.
        
        Args:
            request: Search request to validate
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check query length
            if len(request.query.strip()) == 0:
                raise ValidationError("Query cannot be empty")
            
            if len(request.query) > 1000:
                raise ValidationError("Query is too long (maximum 1000 characters)")
            
            # Check similarity threshold
            if not 0.0 <= request.similarity_threshold <= 1.0:
                raise ValidationError("Similarity threshold must be between 0.0 and 1.0")
            
            # Check max results
            if not 1 <= request.max_results <= 50:
                raise ValidationError("Max results must be between 1 and 50")
            
            logger.debug(
                "Search request validation passed",
                query_length=len(request.query),
                max_results=request.max_results,
                similarity_threshold=request.similarity_threshold
            )
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Error validating search request",
                error=str(e),
                exc_info=True
            )
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    async def _semantic_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            request: Search request
            
        Returns:
            List of search results
        """
        try:
            logger.debug("Performing semantic search", query=request.query[:100])
            
            # Generate query embedding
            query_embedding = await embedding_service.embed_query(request.query)
            
            # Search vector store
            results = await vector_store.search(
                query_embedding=query_embedding,
                k=request.max_results,
                similarity_threshold=request.similarity_threshold,
                document_ids=request.document_ids
            )
            
            logger.debug(
                "Semantic search completed",
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Error in semantic search",
                query=request.query[:100],
                error=str(e),
                exc_info=True
            )
            raise SearchError(f"Semantic search failed: {str(e)}")
    
    async def _keyword_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Perform keyword-based search using text matching.
        
        Args:
            request: Search request
            
        Returns:
            List of search results
        """
        try:
            logger.debug("Performing keyword search", query=request.query[:100])
            
            # For now, we'll use semantic search as fallback
            # In a production system, you might implement BM25 or Elasticsearch
            logger.warning("Keyword search not fully implemented, falling back to semantic search")
            
            # Use semantic search with lower threshold for keyword-like behavior
            modified_request = SearchRequest(
                query=request.query,
                search_type=SearchType.SEMANTIC,
                max_results=request.max_results,
                similarity_threshold=max(0.3, request.similarity_threshold - 0.2),
                document_ids=request.document_ids
            )
            
            results = await self._semantic_search(modified_request)
            
            # Filter results based on keyword presence (simple implementation)
            query_keywords = set(request.query.lower().split())
            filtered_results = []
            
            for result in results:
                content_words = set(result.content.lower().split())
                if query_keywords.intersection(content_words):
                    filtered_results.append(result)
                
                if len(filtered_results) >= request.max_results:
                    break
            
            logger.debug(
                "Keyword search completed",
                original_results=len(results),
                filtered_results=len(filtered_results)
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(
                "Error in keyword search",
                query=request.query[:100],
                error=str(e),
                exc_info=True
            )
            raise SearchError(f"Keyword search failed: {str(e)}")
    
    async def _hybrid_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            request: Search request
            
        Returns:
            List of search results
        """
        try:
            logger.debug("Performing hybrid search", query=request.query[:100])
            
            # Run semantic and keyword searches in parallel
            semantic_request = SearchRequest(
                query=request.query,
                search_type=SearchType.SEMANTIC,
                max_results=request.max_results * 2,  # Get more results for merging
                similarity_threshold=request.similarity_threshold,
                document_ids=request.document_ids
            )
            
            keyword_request = SearchRequest(
                query=request.query,
                search_type=SearchType.KEYWORD,
                max_results=request.max_results * 2,
                similarity_threshold=request.similarity_threshold,
                document_ids=request.document_ids
            )
            
            # Run searches concurrently
            semantic_results, keyword_results = await asyncio.gather(
                self._semantic_search(semantic_request),
                self._keyword_search(keyword_request),
                return_exceptions=True
            )
            
            # Handle potential exceptions
            if isinstance(semantic_results, Exception):
                logger.warning(f"Semantic search failed in hybrid mode: {semantic_results}")
                semantic_results = []
            
            if isinstance(keyword_results, Exception):
                logger.warning(f"Keyword search failed in hybrid mode: {keyword_results}")
                keyword_results = []
            
            # Merge and rank results
            merged_results = await self._merge_search_results(
                semantic_results,
                keyword_results,
                request.max_results
            )
            
            logger.debug(
                "Hybrid search completed",
                semantic_results=len(semantic_results) if isinstance(semantic_results, list) else 0,
                keyword_results=len(keyword_results) if isinstance(keyword_results, list) else 0,
                merged_results=len(merged_results)
            )
            
            return merged_results
            
        except Exception as e:
            logger.error(
                "Error in hybrid search",
                query=request.query[:100],
                error=str(e),
                exc_info=True
            )
            raise SearchError(f"Hybrid search failed: {str(e)}")
    
    async def _merge_search_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        max_results: int
    ) -> List[SearchResult]:
        """
        Merge and rank results from semantic and keyword searches.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            max_results: Maximum number of results to return
            
        Returns:
            Merged and ranked results
        """
        try:
            # Create a map to track unique chunks
            chunk_scores = {}
            
            # Process semantic results (weight: 0.7)
            for result in semantic_results:
                chunk_id = result.chunk_id
                score = result.similarity_score * 0.7
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]['score'] += score
                else:
                    chunk_scores[chunk_id] = {
                        'result': result,
                        'score': score,
                        'sources': ['semantic']
                    }
            
            # Process keyword results (weight: 0.3)
            for result in keyword_results:
                chunk_id = result.chunk_id
                score = result.similarity_score * 0.3
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]['score'] += score
                    chunk_scores[chunk_id]['sources'].append('keyword')
                else:
                    chunk_scores[chunk_id] = {
                        'result': result,
                        'score': score,
                        'sources': ['keyword']
                    }
            
            # Sort by combined score
            sorted_chunks = sorted(
                chunk_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            # Create final results with updated scores
            merged_results = []
            for chunk_id, data in sorted_chunks[:max_results]:
                result = data['result']
                # Update similarity score with combined score
                result.similarity_score = data['score']
                # Add source information to metadata
                if result.metadata is None:
                    result.metadata = {}
                result.metadata['search_sources'] = data['sources']
                
                merged_results.append(result)
            
            return merged_results
            
        except Exception as e:
            logger.error(
                "Error merging search results",
                error=str(e),
                exc_info=True
            )
            # Return semantic results as fallback
            return semantic_results[:max_results]
    
    async def _update_search_stats(self, search_type: SearchType, processing_time: float) -> None:
        """Update search statistics."""
        try:
            self._search_stats["total_searches"] += 1
            self._search_stats["total_processing_time"] += processing_time
            
            if search_type == SearchType.SEMANTIC:
                self._search_stats["semantic_searches"] += 1
            elif search_type == SearchType.KEYWORD:
                self._search_stats["keyword_searches"] += 1
            elif search_type == SearchType.HYBRID:
                self._search_stats["hybrid_searches"] += 1
            
        except Exception as e:
            logger.warning(
                "Error updating search statistics",
                error=str(e)
            )
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        try:
            stats = self._search_stats.copy()
            
            # Calculate average processing time
            if stats["total_searches"] > 0:
                stats["average_processing_time"] = (
                    stats["total_processing_time"] / stats["total_searches"]
                )
            else:
                stats["average_processing_time"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(
                "Error getting search statistics",
                error=str(e),
                exc_info=True
            )
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the query processor.
        
        Returns:
            Health check status
        """
        try:
            # Check dependencies
            embedding_health = await embedding_service.health_check()
            vector_store_health = await vector_store.health_check()
            
            # Determine overall status
            dependencies_healthy = (
                embedding_health.get("status") == "healthy" and
                vector_store_health.get("status") == "healthy"
            )
            
            status = {
                "service": "query_processor",
                "status": "healthy" if dependencies_healthy else "unhealthy",
                "dependencies": {
                    "embedding_service": embedding_health,
                    "vector_store": vector_store_health
                },
                "search_stats": await self.get_search_stats()
            }
            
            return status
            
        except Exception as e:
            logger.error(
                "Query processor health check failed",
                error=str(e),
                exc_info=True
            )
            return {
                "service": "query_processor",
                "status": "unhealthy",
                "error": str(e)
            }


# Global query processor instance
query_processor = QueryProcessor()
