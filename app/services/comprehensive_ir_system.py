"""
Comprehensive Information Retrieval (IR) System integrating all advanced components.

This module combines enhanced retrieval, multi-factor ranking, result aggregation,
validation, and post-processing services to provide a complete IR solution that
integrates with your existing FAISS-based vector store and query processing system.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    IRSearchRequest, IRSearchResponse, RankedResult, ValidationReport,
    RetrievalMode, SearchType
)

# Import all the IR services
from app.services.enhanced_retrieval_service import enhanced_retrieval_service
from app.services.ranking_service import ranking_service
from app.services.aggregation_service import aggregation_service
from app.services.validation_service import validation_service
from app.services.post_processing_service import post_processing_service

logger = get_logger(__name__)


class ComprehensiveIRSystem:
    """
    Main IR system that orchestrates all components for comprehensive
    information retrieval with enhanced ranking, validation, and processing.
    """
    
    def __init__(self):
        """Initialize the comprehensive IR system."""
        self.retrieval_service = enhanced_retrieval_service
        self.ranking_service = ranking_service
        self.aggregation_service = aggregation_service
        self.validation_service = validation_service
        self.post_processing_service = post_processing_service
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize all IR system components."""
        try:
            logger.info("Initializing Comprehensive IR System")
            
            # Initialize enhanced retrieval service
            await self.retrieval_service.initialize()
            
            self.initialized = True
            logger.info("Comprehensive IR System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IR system: {e}")
            raise
    
    async def search(
        self,
        request: IRSearchRequest,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> IRSearchResponse:
        """
        Perform comprehensive IR search with all advanced features.
        
        Args:
            request: IR search request with query and parameters
            user_preferences: Optional user preferences for personalization
            
        Returns:
            Complete IR search response with enhanced results
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting comprehensive IR search: '{request.query}'")
            
            # Step 1: Enhanced Retrieval
            logger.info("Step 1: Enhanced retrieval with multi-query and ensemble strategies")
            initial_results = await self.retrieval_service.search(request)
            logger.info(f"Retrieved {len(initial_results)} initial results")
            
            if not initial_results:
                return self._create_empty_response(request, start_time)
            
            # Step 2: Multi-factor Ranking
            logger.info("Step 2: Multi-factor ranking with metadata signals")
            ranked_results = self.ranking_service.rank_results(
                initial_results,
                user_preferences=user_preferences,
                query_context={'query': request.query}
            )
            logger.info(f"Ranking completed for {len(ranked_results)} results")
            
            # Step 3: Result Aggregation (if multiple strategies were used)
            logger.info("Step 3: Result aggregation and deduplication")
            # For now, we have one result list, but the aggregation service
            # can still handle deduplication and normalization
            strategy_name = f"{request.retrieval_mode.value}_retrieval"
            result_lists = [(strategy_name, ranked_results)]
            
            aggregated_results = self.aggregation_service.aggregate_results(
                result_lists=result_lists,
                max_results=request.max_results,
                fusion_strategy='reciprocal_rank',
                deduplicate=True
            )
            logger.info(f"Aggregation completed: {len(aggregated_results)} final results")
            
            # Step 4: Validation and Cross-referencing
            logger.info("Step 4: Information validation and cross-referencing")
            validation_report = await self.validation_service.validate_results(
                aggregated_results,
                query=request.query
            )
            logger.info(f"Validation completed with confidence: {validation_report.overall_confidence:.3f}")
            
            # Step 5: Post-processing (snippets, quotes, highlighting)
            logger.info("Step 5: Post-processing with snippets and quote extraction")
            processed_results = self.post_processing_service.process_results(
                aggregated_results,
                query=request.query,
                generate_snippets=True,
                extract_quotes=True,
                highlight_terms=True
            )
            logger.info(f"Post-processing completed for {len(processed_results)} results")
            
            # Step 6: Generate processing summary
            processing_summary = self._generate_processing_summary(
                request, processed_results, validation_report, start_time
            )
            
            # Create comprehensive response
            response = IRSearchResponse(
                query=request.query,
                results=processed_results,
                total_results=len(processed_results),
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                retrieval_mode=request.retrieval_mode,
                validation_report=validation_report,
                processing_summary=processing_summary
            )
            
            logger.info(f"Comprehensive IR search completed in {response.processing_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error in comprehensive IR search: {e}")
            return self._create_error_response(request, str(e), start_time)
    
    def _create_empty_response(
        self, 
        request: IRSearchRequest, 
        start_time: datetime
    ) -> IRSearchResponse:
        """Create response for empty results."""
        return IRSearchResponse(
            query=request.query,
            results=[],
            total_results=0,
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            retrieval_mode=request.retrieval_mode,
            validation_report=ValidationReport(
                overall_confidence=0.0,
                validation_results=[],
                consistency_checks=0,
                source_agreement_score=0.0,
                conflicting_information=[],
                supporting_evidence=[]
            ),
            processing_summary={
                "status": "no_results",
                "message": "No relevant results found for the query"
            }
        )
    
    def _create_error_response(
        self, 
        request: IRSearchRequest, 
        error_message: str,
        start_time: datetime
    ) -> IRSearchResponse:
        """Create response for error cases."""
        return IRSearchResponse(
            query=request.query,
            results=[],
            total_results=0,
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            retrieval_mode=request.retrieval_mode,
            validation_report=ValidationReport(
                overall_confidence=0.0,
                validation_results=[],
                consistency_checks=0,
                source_agreement_score=0.0,
                conflicting_information=[],
                supporting_evidence=[]
            ),
            processing_summary={
                "status": "error",
                "error_message": error_message
            }
        )
    
    def _generate_processing_summary(
        self,
        request: IRSearchRequest,
        results: List[RankedResult],
        validation_report: ValidationReport,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Generate a summary of the processing pipeline."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Collect statistics
        avg_score = sum(r.final_score for r in results) / len(results) if results else 0
        unique_sources = len(set(r.document_id for r in results))
        
        # Count results with various enhancements
        results_with_snippets = sum(
            1 for r in results 
            if 'snippet' in r.relevance_explanation
        )
        results_with_quotes = sum(
            1 for r in results 
            if r.relevance_explanation.get('key_quotes')
        )
        
        return {
            "status": "success",
            "processing_stages": [
                "enhanced_retrieval",
                "multi_factor_ranking", 
                "result_aggregation",
                "information_validation",
                "post_processing"
            ],
            "processing_time_seconds": round(processing_time, 3),
            "retrieval_mode": request.retrieval_mode.value,
            "statistics": {
                "total_results": len(results),
                "unique_sources": unique_sources,
                "average_relevance_score": round(avg_score, 3),
                "validation_confidence": round(validation_report.overall_confidence, 3),
                "source_agreement": round(validation_report.source_agreement_score, 3),
                "results_with_snippets": results_with_snippets,
                "results_with_quotes": results_with_quotes,
                "consistency_checks": validation_report.consistency_checks,
                "conflicting_info_found": len(validation_report.conflicting_information) > 0,
                "supporting_evidence_found": len(validation_report.supporting_evidence) > 0
            },
            "service_stats": {
                "retrieval": self.retrieval_service.get_retrieval_stats(),
                "ranking": self.ranking_service.get_ranking_stats(),
                "aggregation": self.aggregation_service.get_aggregation_stats(),
                "validation": self.validation_service.get_validation_stats(),
                "post_processing": self.post_processing_service.get_processing_stats()
            }
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health check."""
        return {
            "system_initialized": self.initialized,
            "services_status": {
                "enhanced_retrieval": hasattr(self.retrieval_service, 'initialized') and self.retrieval_service.initialized,
                "ranking": True,  # No async initialization required
                "aggregation": True,  # No async initialization required
                "validation": hasattr(self.validation_service, 'enabled') and self.validation_service.enabled,
                "post_processing": True  # No async initialization required
            },
            "configuration": {
                "max_results": settings.max_search_results,
                "similarity_threshold": settings.similarity_threshold,
                "deduplication_threshold": settings.deduplication_threshold,
                "cross_reference_enabled": settings.cross_reference_enabled,
                "snippet_max_chars": settings.snippet_max_chars
            },
            "capabilities": [
                "multi_query_retrieval",
                "ensemble_search_strategies",
                "multi_factor_ranking",
                "result_deduplication",
                "information_validation",
                "snippet_generation",
                "quote_extraction",
                "term_highlighting",
                "cross_referencing"
            ]
        }


# Global instance
comprehensive_ir_system = ComprehensiveIRSystem()
