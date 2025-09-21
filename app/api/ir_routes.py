"""
FastAPI Router for Comprehensive Information Retrieval (IR) System.

This module provides REST API endpoints for the comprehensive IR system,
integrating all advanced retrieval, ranking, validation, and processing features
with your existing FastAPI application.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
import time

from app.core.logging import get_logger
from app.models.schemas import IRSearchRequest, IRSearchResponse, RetrievalMode
from app.services.comprehensive_ir_system import comprehensive_ir_system

logger = get_logger(__name__)
router = APIRouter(prefix="/ir", tags=["Information Retrieval"])


@router.post("/search", response_model=IRSearchResponse)
async def comprehensive_search(
    request: IRSearchRequest,
    user_id: Optional[str] = Query(None, description="Optional user ID for personalization")
) -> IRSearchResponse:
    """
    Perform comprehensive information retrieval search.
    
    This endpoint combines:
    - Enhanced multi-query retrieval
    - Multi-factor ranking with metadata signals
    - Result aggregation and deduplication
    - Information validation and cross-referencing
    - Post-processing with snippets and quotes
    
    Args:
        request: IR search request with query and parameters
        user_id: Optional user ID for personalized results
    
    Returns:
        Comprehensive search response with enhanced results
    """
    try:
        logger.info(f"IR search request: {request.query[:100]}... (mode: {request.retrieval_mode})")
        
        # Get user preferences if user_id provided
        user_preferences = None
        if user_id:
            user_preferences = await _get_user_preferences(user_id)
        
        # Perform comprehensive search
        response = await comprehensive_ir_system.search(request, user_preferences)
        
        logger.info(f"IR search completed: {response.total_results} results in {response.processing_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error in comprehensive IR search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during IR search: {str(e)}"
        )


@router.get("/search", response_model=IRSearchResponse)
async def comprehensive_search_get(
    query: str = Query(..., description="Search query"),
    retrieval_mode: RetrievalMode = Query(RetrievalMode.HYBRID, description="Retrieval mode"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs to filter"),
    user_id: Optional[str] = Query(None, description="Optional user ID for personalization")
) -> IRSearchResponse:
    """
    Perform comprehensive IR search via GET request.
    
    This is a convenience endpoint that accepts search parameters as query parameters
    instead of a request body, useful for simple integrations and testing.
    """
    try:
        # Parse document IDs if provided
        document_id_list = None
        if document_ids:
            document_id_list = [doc_id.strip() for doc_id in document_ids.split(",") if doc_id.strip()]
        
        # Create search request
        request = IRSearchRequest(
            query=query,
            retrieval_mode=retrieval_mode,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            document_ids=document_id_list,
            filters={}
        )
        
        # Use the POST endpoint logic
        return await comprehensive_search(request, user_id)
        
    except Exception as e:
        logger.error(f"Error in GET IR search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during IR search: {str(e)}"
        )


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive IR system status and health check.
    
    Returns information about:
    - System initialization status
    - Individual service status
    - Configuration settings
    - Available capabilities
    """
    try:
        status = await comprehensive_ir_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system status: {str(e)}"
        )


@router.post("/initialize")
async def initialize_system() -> Dict[str, str]:
    """
    Initialize the comprehensive IR system.
    
    This endpoint can be called to explicitly initialize all IR components
    if they haven't been initialized automatically.
    """
    try:
        await comprehensive_ir_system.initialize()
        return {"status": "success", "message": "IR system initialized successfully"}
        
    except Exception as e:
        logger.error(f"Error initializing IR system: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize IR system: {str(e)}"
        )


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """
    Get detailed information about IR system capabilities.
    
    Returns:
        Detailed breakdown of available features and configurations
    """
    try:
        status = await comprehensive_ir_system.get_system_status()
        
        return {
            "retrieval_capabilities": {
                "modes": [mode.value for mode in RetrievalMode],
                "features": [
                    "multi_query_generation",
                    "ensemble_retrieval",
                    "query_expansion",
                    "semantic_search",
                    "keyword_search",
                    "hybrid_search"
                ]
            },
            "ranking_capabilities": {
                "factors": [
                    "semantic_similarity",
                    "recency_score",
                    "credibility_score",
                    "document_quality",
                    "user_preferences"
                ],
                "normalization_methods": ["minmax", "zscore", "log"]
            },
            "aggregation_capabilities": {
                "fusion_strategies": [
                    "reciprocal_rank",
                    "weighted_average", 
                    "max_score",
                    "borda_count"
                ],
                "deduplication": True
            },
            "validation_capabilities": {
                "cross_referencing": True,
                "consistency_checking": True,
                "confidence_scoring": True,
                "conflict_detection": True
            },
            "post_processing_capabilities": {
                "snippet_generation": True,
                "quote_extraction": True,
                "term_highlighting": True,
                "attribution": True
            },
            "configuration": status.get("configuration", {}),
            "system_status": status.get("services_status", {})
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving capabilities: {str(e)}"
        )


async def _get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user preferences for personalized search.
    
    This is a placeholder function that you can implement to retrieve
    user preferences from your user management system.
    """
    # Placeholder implementation - replace with your user preference logic
    default_preferences = {
        "preferred_sources": [],
        "preferred_document_types": ["research_paper", "documentation"],
        "preferred_language": "english",
        "recency_preference": "balanced",  # "recent", "balanced", "classic"
        "interested_topics": []
    }
    
    # In a real implementation, you would:
    # 1. Query your user database
    # 2. Retrieve stored preferences
    # 3. Apply any learned preferences from user behavior
    # 4. Return the preferences dictionary
    
    logger.info(f"Retrieved preferences for user {user_id}")
    return default_preferences


# Optional: Add middleware for request logging
@router.middleware("http")
async def log_requests(request, call_next):
    """Log IR system requests for analytics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"IR Request: {request.method} {request.url.path} "
        f"completed in {process_time:.3f}s with status {response.status_code}"
    )
    
    return response


# Export the router for inclusion in main app
__all__ = ["router"]
