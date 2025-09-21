"""
Main FastAPI application for the Deep Researcher Agent system.
Provides comprehensive document processing, search, and research capabilities.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.exceptions import (
    ResearcherAgentException,
    researcher_agent_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.routes import documents, search, research, health
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.query_processor import query_processor
from app.services.response_synthesizer import response_synthesizer

# Initialize logging
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting Deep Researcher Agent application",
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug
    )
    
    try:
        # Initialize services in order
        logger.info("Initializing application services...")
        
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        await embedding_service.initialize()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        await vector_store.initialize()
        
        # Initialize query processor
        logger.info("Initializing query processor...")
        await query_processor.initialize()
        
        # Initialize response synthesizer
        logger.info("Initializing response synthesizer...")
        await response_synthesizer.initialize()
        
        logger.info("All services initialized successfully")
        
        # Yield control to the application
        yield
        
    except Exception as e:
        logger.error(
            "Failed to initialize application services",
            error=str(e),
            exc_info=True
        )
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Deep Researcher Agent application")
        
        try:
            # Save vector store index
            await vector_store.save_index()
            logger.info("Vector store index saved")
        except Exception as e:
            logger.error(
                "Error saving vector store during shutdown",
                error=str(e),
                exc_info=True
            )
        
        logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="A sophisticated document processing and research system with RAG capabilities",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with your actual allowed hosts
    )


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.middleware("http")
async def add_timestamp_middleware(request: Request, call_next):
    """Add timestamp to response."""
    response = await call_next(request)
    
    # Add timestamp to JSON responses
    if hasattr(response, 'body') and response.headers.get("content-type", "").startswith("application/json"):
        try:
            # This is a simple approach - in production you might want a more sophisticated method
            response.headers["X-Timestamp"] = datetime.utcnow().isoformat()
        except:
            pass
    
    return response


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log incoming requests and responses."""
    start_time = datetime.utcnow()
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers),
        request_id=request_id
    )
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        request_id=request_id
    )
    
    return response


# Add exception handlers
app.add_exception_handler(ResearcherAgentException, researcher_agent_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Include routers
app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(documents.router, prefix=settings.api_v1_prefix)
app.include_router(search.router, prefix=settings.api_v1_prefix)
app.include_router(research.router, prefix=settings.api_v1_prefix)

# Include synthesis router (new)
try:
    from app.routes.synthesis_routes import synthesis_router
    app.include_router(synthesis_router, prefix=settings.api_v1_prefix)
    logger.info("Synthesis router included successfully")
except ImportError as e:
    logger.warning(f"Synthesis router not available: {e}")
except Exception as e:
    logger.error(f"Error including synthesis router: {e}")


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """
    Root endpoint providing basic application information.
    """
    return JSONResponse(
        content={
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Deep Researcher Agent - Advanced document processing and research system with LLM-powered synthesis",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "health": f"{settings.api_v1_prefix}/health",
                "documents": f"{settings.api_v1_prefix}/documents",
                "search": f"{settings.api_v1_prefix}/search",
                "research": f"{settings.api_v1_prefix}/research",
                "synthesis": f"{settings.api_v1_prefix}/synthesis",
                "docs": "/docs" if settings.debug else "disabled"
            },
            "new_features": {
                "synthesis": "LLM-powered content synthesis with Groq integration",
                "citations": "APA, MLA, Chicago citation formats supported",
                "formats": "Executive summary, detailed report, FAQ, comparative analysis, bullet points",
                "quality_checks": "Hallucination detection, coherence analysis, factual accuracy"
            }
        }
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon endpoint."""
    return JSONResponse(content={"message": "No favicon available"}, status_code=404)


# Custom startup message
@app.on_event("startup")
async def startup_message():
    """Log startup message."""
    logger.info(
        f"""
        🚀 Deep Researcher Agent Started Successfully!
        
        ✅ Version: {settings.app_version}
        ✅ Environment: {settings.environment}
        ✅ Debug Mode: {settings.debug}
        ✅ API Prefix: {settings.api_v1_prefix}
        
        📚 Available Endpoints:
        • Health Check: {settings.api_v1_prefix}/health
        • Document Upload: {settings.api_v1_prefix}/documents/upload
        • Search: {settings.api_v1_prefix}/search
        • Research: {settings.api_v1_prefix}/research
        • Synthesis (NEW): {settings.api_v1_prefix}/synthesis
        
        🤖 New LLM-Powered Features:
        • Content Synthesis with Groq LLM
        • Multiple Output Formats (Executive Summary, Detailed Report, FAQ, etc.)
        • Citation Management (APA, MLA, Chicago)
        • Quality Checks & Hallucination Detection
        
        🔗 Documentation: {"http://localhost:" + str(settings.port) + "/docs" if settings.debug else "Disabled in production"}
        
        Ready to process documents and generate intelligent responses! 🎉
        """
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
