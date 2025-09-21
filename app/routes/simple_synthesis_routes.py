"""
Simplified synthesis routes using direct Groq API.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any

from app.core.logging import get_logger
from app.models.synthesis_schemas import (
    SynthesisRequest, SynthesisResponse, SynthesisConfig, 
    OutputFormat, OutputStyle, CitationStyle
)

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/synthesis", tags=["synthesis"])

# Import service with error handling
try:
    from app.services.simple_synthesis_service import simple_synthesis_service
    SYNTHESIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Synthesis service not available: {e}")
    SYNTHESIS_AVAILABLE = False


@router.get("/health")
async def synthesis_health():
    """Check synthesis service health."""
    if not SYNTHESIS_AVAILABLE:
        return {"status": "unavailable", "message": "Synthesis service not available"}
    
    try:
        health = simple_synthesis_service.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/quick")
async def quick_synthesis(
    query: str,
    sources: List[Dict[str, Any]] = None
):
    """Quick synthesis endpoint for immediate responses."""
    if not SYNTHESIS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthesis service not available"
        )
    
    try:
        if not sources:
            sources = []
        
        result = await simple_synthesis_service.quick_synthesis(query, sources)
        
        return {
            "query": query,
            "synthesis": result,
            "sources_count": len(sources),
            "service": "SimpleSynthesisService"
        }
        
    except Exception as e:
        logger.error(f"Quick synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}"
        )


@router.post("/generate")
async def generate_synthesis(request: SynthesisRequest):
    """Generate comprehensive synthesis response."""
    if not SYNTHESIS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthesis service not available"
        )
    
    try:
        result = await simple_synthesis_service.generate_synthesis(request)
        return result
        
    except Exception as e:
        logger.error(f"Synthesis generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}"
        )


@router.post("/demo")
async def demo_synthesis():
    """Demo synthesis with sample data."""
    if not SYNTHESIS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthesis service not available"
        )
    
    # Sample data for demo
    sample_sources = [
        {
            "title": "Python Benefits",
            "content": "Python is known for its simplicity and readability. It has extensive libraries and strong community support."
        },
        {
            "title": "Python Applications",
            "content": "Python is used in web development, data science, AI/ML, automation, and scientific computing."
        }
    ]
    
    sample_query = "What are the main advantages of using Python programming language?"
    
    try:
        result = await simple_synthesis_service.quick_synthesis(sample_query, sample_sources)
        
        return {
            "demo": True,
            "query": sample_query,
            "synthesis": result,
            "sources": sample_sources,
            "message": "This is a demo of the synthesis service using sample data"
        }
        
    except Exception as e:
        logger.error(f"Demo synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo failed: {str(e)}"
        )


@router.get("/")
async def synthesis_info():
    """Get synthesis service information."""
    return {
        "service": "Synthesis API",
        "version": "1.0.0-simplified",
        "available": SYNTHESIS_AVAILABLE,
        "endpoints": [
            "/health - Service health check",
            "/quick - Quick synthesis",
            "/generate - Full synthesis generation",
            "/demo - Demo with sample data"
        ],
        "model": "llama-3.1-8b-instant" if SYNTHESIS_AVAILABLE else "N/A"
    }
