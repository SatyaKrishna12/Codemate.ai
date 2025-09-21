"""
Enhanced Retrieval Service integrating LangChain components with existing vector store.

This service wraps your existing query processor and vector store with LangChain's
MultiQueryRetriever, EnsembleRetriever, and ContextualCompressionRetriever for
advanced information retrieval capabilities.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

"""
Enhanced Retrieval Service integrating advanced retrieval strategies with existing vector store.

This service extends your existing query processor and vector store with multi-query generation,
ensemble retrieval, and advanced ranking for comprehensive information retrieval.
"""

import asyncio
import re
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    IRSearchRequest, RankedResult, RetrievalMode, SearchResult,
    SearchRequest, SearchType, SearchResponse
)
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.services.query_processor import QueryProcessor

logger = get_logger(__name__)


@dataclass
class RetrievalStrategy:
    """Configuration for a retrieval strategy."""
    name: str
    weight: float
    search_type: SearchType
    max_results: int = 20


class QueryExpansionService:
    """Service for expanding queries into multiple variations."""
    
    def __init__(self):
        """Initialize query expansion service."""
        self.expansion_patterns = [
            # Synonym patterns
            ("AI", ["artificial intelligence", "machine learning", "ML"]),
            ("ML", ["machine learning", "artificial intelligence", "AI"]),
            ("NLP", ["natural language processing", "text processing"]),
            ("API", ["application programming interface", "web service"]),
            # Add more patterns based on your domain
        ]
    
    def expand_query(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate multiple query variations for enhanced retrieval.
        
        Args:
            query: Original query
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations including the original
        """
        variations = [query]  # Start with original
        
        try:
            # Generate synonym-based variations
            for term, synonyms in self.expansion_patterns:
                if term.lower() in query.lower():
                    for synonym in synonyms[:2]:  # Limit synonyms
                        variation = re.sub(
                            rf'\b{re.escape(term)}\b', 
                            synonym, 
                            query, 
                            flags=re.IGNORECASE
                        )
                        if variation != query and variation not in variations:
                            variations.append(variation)
                            if len(variations) >= max_variations + 1:
                                break
            
            # Generate keyword extraction variations
            words = query.split()
            if len(words) > 3:
                # Create focused variations with key terms
                key_terms = [word for word in words if len(word) > 3][:3]
                if key_terms:
                    focused_query = " ".join(key_terms)
                    if focused_query not in variations:
                        variations.append(focused_query)
            
            return variations[:max_variations + 1]
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return [query]


class EnsembleRetrieval:
    """Ensemble retrieval combining multiple search strategies."""
    
    def __init__(self, query_processor: QueryProcessor):
        """Initialize ensemble retrieval."""
        self.query_processor = query_processor
        self.strategies = [
            RetrievalStrategy("semantic", 0.6, SearchType.SEMANTIC, 15),
            RetrievalStrategy("keyword", 0.3, SearchType.KEYWORD, 10),
            RetrievalStrategy("hybrid", 0.1, SearchType.HYBRID, 10)
        ]
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform ensemble retrieval using multiple strategies.
        
        Args:
            query: Search query
            k: Total number of results desired
            document_ids: Optional filter for specific documents
            
        Returns:
            Combined and ranked results
        """
        all_results = []
        seen_chunks = set()
        
        try:
            # Execute all strategies in parallel
            tasks = []
            for strategy in self.strategies:
                task = self._execute_strategy(strategy, query, document_ids)
                tasks.append(task)
            
            strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results with weighted scoring
            for i, results in enumerate(strategy_results):
                if isinstance(results, Exception):
                    logger.error(f"Strategy {self.strategies[i].name} failed: {results}")
                    continue
                
                strategy = self.strategies[i]
                for result in results.results[:strategy.max_results]:
                    # Avoid duplicates
                    if result.chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(result.chunk_id)
                    
                    # Apply strategy weight to score
                    weighted_score = result.similarity_score * strategy.weight
                    
                    # Create enhanced result
                    enhanced_result = SearchResult(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        content=result.content,
                        similarity_score=weighted_score,
                        metadata={
                            **result.metadata,
                            'retrieval_strategy': strategy.name,
                            'original_score': result.similarity_score,
                            'strategy_weight': strategy.weight
                        }
                    )
                    all_results.append(enhanced_result)
            
            # Sort by weighted score and return top k
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return all_results[:k]
            
        except Exception as e:
            logger.error(f"Error in ensemble retrieval: {e}")
            return []
    
    async def _execute_strategy(
        self, 
        strategy: RetrievalStrategy, 
        query: str,
        document_ids: Optional[List[str]]
    ) -> SearchResponse:
        """Execute a single retrieval strategy."""
        request = SearchRequest(
            query=query,
            search_type=strategy.search_type,
            max_results=strategy.max_results,
            similarity_threshold=settings.similarity_threshold,
            document_ids=document_ids
        )
        
        return await self.query_processor.search(request)


class EnhancedRetrievalService:
    """
    Enhanced retrieval service that combines multiple advanced retrieval strategies
    with your existing vector store and query processing capabilities.
    """
    
    def __init__(self):
        """Initialize the enhanced retrieval service."""
        self.query_processor = QueryProcessor()
        self.query_expansion = QueryExpansionService()
        self.ensemble_retrieval = None
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize all retrieval components."""
        try:
            logger.info("Initializing enhanced retrieval service")
            
            # Initialize existing query processor
            await self.query_processor.initialize()
            
            # Initialize ensemble retrieval
            self.ensemble_retrieval = EnsembleRetrieval(self.query_processor)
            
            self.initialized = True
            logger.info("Enhanced retrieval service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced retrieval service: {e}")
            raise
    
    async def search(
        self, 
        request: IRSearchRequest
    ) -> List[RankedResult]:
        """
        Perform enhanced search using multiple retrieval strategies.
        
        Args:
            request: IR search request with query and parameters
            
        Returns:
            List of ranked results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Performing enhanced search: {request.query[:100]}...")
            
            # Choose retrieval approach based on mode
            if request.retrieval_mode == RetrievalMode.SEMANTIC_FIRST:
                results = await self._semantic_first_search(request)
            elif request.retrieval_mode == RetrievalMode.HYBRID:
                results = await self._hybrid_search(request)
            else:  # FILTER_FIRST
                results = await self._filter_first_search(request)
            
            # Convert to RankedResult format
            ranked_results = self._convert_to_ranked_results(results)
            
            logger.info(f"Enhanced search completed: {len(ranked_results)} results")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    async def _semantic_first_search(self, request: IRSearchRequest) -> List[SearchResult]:
        """Perform semantic-first search with query expansion."""
        # Generate query variations
        query_variations = self.query_expansion.expand_query(
            request.query, 
            settings.multi_query_retriever_queries - 1
        )
        
        all_results = []
        seen_chunks = set()
        
        # Search with each query variation
        for i, query_variant in enumerate(query_variations):
            search_request = SearchRequest(
                query=query_variant,
                search_type=SearchType.SEMANTIC,
                max_results=request.max_results // len(query_variations) + 5,
                similarity_threshold=request.similarity_threshold,
                document_ids=request.document_ids
            )
            
            response = await self.query_processor.search(search_request)
            
            # Add results with variation context
            for result in response.results:
                if result.chunk_id not in seen_chunks:
                    seen_chunks.add(result.chunk_id)
                    result.metadata['query_variation'] = query_variant
                    result.metadata['variation_index'] = i
                    all_results.append(result)
        
        # Sort and return top results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:request.max_results]
    
    async def _hybrid_search(self, request: IRSearchRequest) -> List[SearchResult]:
        """Perform hybrid search using ensemble retrieval."""
        return await self.ensemble_retrieval.retrieve(
            query=request.query,
            k=request.max_results,
            document_ids=request.document_ids
        )
    
    async def _filter_first_search(self, request: IRSearchRequest) -> List[SearchResult]:
        """Perform filter-first search with contextual filtering."""
        # Apply filters first, then search within filtered set
        filtered_doc_ids = request.document_ids
        
        # If no specific filters, use semantic search
        if not filtered_doc_ids and request.filters:
            # Apply metadata-based filtering logic here
            # For now, use standard hybrid approach
            return await self._hybrid_search(request)
        
        # Perform search within filtered documents
        search_request = SearchRequest(
            query=request.query,
            search_type=SearchType.HYBRID,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            document_ids=filtered_doc_ids
        )
        
        response = await self.query_processor.search(search_request)
        return response.results
    
    def _convert_to_ranked_results(self, results: List[SearchResult]) -> List[RankedResult]:
        """Convert SearchResult objects to RankedResult objects."""
        ranked_results = []
        
        for i, result in enumerate(results):
            ranked_result = RankedResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content_snippet=result.content[:500],  # Truncate for snippet
                semantic_score=result.similarity_score,
                metadata_score=self._calculate_metadata_score(result.metadata),
                final_score=result.similarity_score,  # Will be enhanced by RankingService
                relevance_explanation={
                    'retrieval_method': result.metadata.get('retrieval_strategy', 'semantic'),
                    'position': i,
                    'original_score': result.metadata.get('original_score', result.similarity_score),
                    'query_variation': result.metadata.get('query_variation', ''),
                    'strategy_weight': result.metadata.get('strategy_weight', 1.0)
                },
                source_metadata=result.metadata,
                matched_terms=[],  # Will be populated by PostProcessingService
                position_in_document=result.metadata.get('position')
            )
            ranked_results.append(ranked_result)
        
        return ranked_results
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate a basic metadata-based score."""
        score = 0.5  # Base score
        
        # Boost recent documents
        if 'created_at' in metadata:
            # Simple recency boost logic
            score += 0.1
        
        # Boost based on source credibility (if available)
        if metadata.get('source_type') == 'official':
            score += 0.2
        
        # Boost based on document type
        if metadata.get('document_type') in ['research_paper', 'documentation']:
            score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "query_expansion_enabled": True,
            "ensemble_strategies": [s.name for s in self.ensemble_retrieval.strategies] if self.ensemble_retrieval else [],
            "strategy_weights": {s.name: s.weight for s in self.ensemble_retrieval.strategies} if self.ensemble_retrieval else {},
            "max_results": settings.max_search_results,
            "multi_query_variations": settings.multi_query_retriever_queries
        }


# Global instance
enhanced_retrieval_service = EnhancedRetrievalService()
