"""
Research API routes for generating comprehensive answers from document knowledge base.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.exceptions import LLMError, SearchError, ValidationError
from app.models.schemas import ResearchRequest, ResearchResponse
from app.services.response_synthesizer import response_synthesizer

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/", response_model=ResearchResponse)
async def research_question(request: ResearchRequest) -> ResearchResponse:
    """
    Research a question using the document knowledge base.
    
    Args:
        request: Research request with question and parameters
        
    Returns:
        Research response with comprehensive answer and sources
        
    Raises:
        HTTPException: If research fails
    """
    try:
        logger.info(
            "Research request received",
            question=request.question[:100],  # Log first 100 chars
            context_limit=request.context_limit,
            include_sources=request.include_sources
        )
        
        # Process research request
        response = await response_synthesizer.research(request)
        
        logger.info(
            "Research completed successfully",
            question=request.question[:100],
            answer_length=len(response.answer),
            sources_count=len(response.sources),
            processing_time=response.processing_time,
            confidence_score=response.confidence_score
        )
        
        return response
        
    except (LLMError, SearchError, ValidationError) as e:
        logger.error(
            "Research request failed",
            question=request.question[:100],
            error=str(e),
            error_type=type(e).__name__
        )
        
        if isinstance(e, ValidationError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        elif isinstance(e, SearchError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve relevant documents: {str(e)}"
            )
        else:  # LLMError
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to generate response: {str(e)}"
            )
            
    except Exception as e:
        logger.error(
            "Unexpected error during research",
            question=request.question[:100],
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during research"
        )


@router.get("/stats")
async def get_research_stats() -> JSONResponse:
    """
    Get research statistics.
    
    Returns:
        JSON response with research statistics
    """
    try:
        stats = await response_synthesizer.get_synthesis_stats()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "research_statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(
            "Error getting research statistics",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve research statistics"
        )
