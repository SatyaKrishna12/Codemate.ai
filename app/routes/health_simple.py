"""Simple health check API routes."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Deep Researcher Agent",
        "version": "1.0.0"
    })


@router.get("/status")
async def detailed_status():
    """Detailed status endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Deep Researcher Agent",
        "version": "1.0.0",
        "components": {
            "document_processor": "available",
            "api": "running"
        }
    })
