"""
Search API routes for querying documents and retrieving relevant information.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.exceptions import SearchError, ValidationError
from app.models.schemas import SearchRequest, SearchResponse
from app.services.query_processor import query_processor

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search for relevant documents based on a query.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search response with relevant document chunks
        
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(
            "Search request received",
            query=request.query[:100],  # Log first 100 chars
            search_type=request.search_type,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        
        # Process search request
        response = await query_processor.search(request)
        
        logger.info(
            "Search completed successfully",
            query=request.query[:100],
            results_count=len(response.results),
            processing_time=response.processing_time
        )
        
        return response
        
    except (SearchError, ValidationError) as e:
        logger.error(
            "Search request failed",
            query=request.query[:100],
            error=str(e),
            error_type=type(e).__name__
        )
        
        if isinstance(e, ValidationError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
            
    except Exception as e:
        logger.error(
            "Unexpected error during search",
            query=request.query[:100],
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during search"
        )


@router.get("/stats")
async def get_search_stats() -> JSONResponse:
    """
    Get search statistics.
    
    Returns:
        JSON response with search statistics
    """
    try:
        stats = await query_processor.get_search_stats()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "search_statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(
            "Error getting search statistics",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search statistics"
        )
