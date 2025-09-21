"""Simplified main.py without ML dependencies for now."""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import setup_exception_handlers
from app.routes import documents
from app.routes import health_simple as health

# Configure logging
setup_logging()
logger = get_logger(__name__)

# Import the intelligent query processing router
try:
    from intelligent_query_integration import query_processing_router
    QUERY_PROCESSING_AVAILABLE = True
    logger.info("Intelligent Query Processing system available")
except ImportError as e:
    try:
        from demo_query_routes import demo_router as query_processing_router
        QUERY_PROCESSING_AVAILABLE = True
        logger.info("Using Query Processing Demo system")
    except ImportError as e2:
        QUERY_PROCESSING_AVAILABLE = False
        logger.warning(f"No query processing system available: {e}, {e2}")

# Import the synthesis router (new)
try:
    from app.routes.simple_synthesis_routes import router as synthesis_router
    SYNTHESIS_AVAILABLE = True
    logger.info("Simple synthesis system available with Groq LLM integration")
except ImportError as e:
    SYNTHESIS_AVAILABLE = False
    logger.warning(f"Synthesis system not available: {e}")

# Import the unified Deep Researcher Agent router
try:
    from app.routes.deep_researcher_routes import router as researcher_router
    RESEARCHER_AVAILABLE = True
    logger.info("Deep Researcher Agent available")
except ImportError as e:
    RESEARCHER_AVAILABLE = False
    logger.warning(f"Deep Researcher Agent not available: {e}")

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Deep Researcher Agent",
        description="A comprehensive document processing and research system",
        version="1.0.0",
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="."), name="static")
    
    # Include routers
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    
    # Include intelligent query processing router if available
    if QUERY_PROCESSING_AVAILABLE:
        app.include_router(query_processing_router)
        logger.info("Intelligent Query Processing routes added to application")
    else:
        logger.warning("Intelligent Query Processing routes not available")
    
    # Include synthesis router if available (new)
    if SYNTHESIS_AVAILABLE:
        app.include_router(synthesis_router, prefix="/api/v1")
        logger.info("Synthesis routes with Groq LLM added to application")
    else:
        logger.warning("Synthesis routes not available")
    
    # Include Deep Researcher Agent router if available
    if RESEARCHER_AVAILABLE:
        app.include_router(researcher_router)
        logger.info("Deep Researcher Agent routes added to application")
    else:
        logger.warning("Deep Researcher Agent routes not available")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        endpoints = {
            "message": "Deep Researcher Agent API with Groq LLM Synthesis",
            "version": "1.0.0",
            "status": "running",
            "web_interface": {
                "dashboard": "/dashboard - Main Dashboard",
                "chat": "/chat - Chat Interface",
                "api_docs": "/docs - API Documentation"
            },
            "available_endpoints": {
                "health": "/api/v1/health",
                "documents": "/api/v1/documents/",
            }
        }
        
        # Add query processing endpoints if available
        if QUERY_PROCESSING_AVAILABLE:
            endpoints["available_endpoints"]["query_processing"] = {
                "process_single_query": "POST /api/v1/query-processing/process",
                "batch_process": "POST /api/v1/query-processing/batch-process", 
                "session_analysis": "GET /api/v1/query-processing/session/{session_id}/analysis",
                "statistics": "GET /api/v1/query-processing/statistics",
                "health_check": "GET /api/v1/query-processing/health"
            }
        
        # Add synthesis endpoints if available (new)
        if SYNTHESIS_AVAILABLE:
            endpoints["available_endpoints"]["synthesis"] = {
                "generate": "POST /api/v1/synthesis/generate",
                "quick": "POST /api/v1/synthesis/quick",
                "markdown": "POST /api/v1/synthesis/markdown",
                "dry_run": "POST /api/v1/synthesis/dry-run",
                "formats": "GET /api/v1/synthesis/formats",
                "health": "GET /api/v1/synthesis/health"
            }
            endpoints["features"] = {
                "llm_powered_synthesis": "Generate intelligent content with Groq LLM",
                "multiple_formats": "Executive summary, detailed report, FAQ, bullet points",
                "citation_styles": "APA, MLA, Chicago citation formats",
                "quality_checks": "Hallucination detection and content validation"
            }
        
        return JSONResponse(content=endpoints)
    
    @app.get("/deep_researcher_chat.html")
    async def deep_researcher_chat():
        """Serve the Deep Researcher chat interface."""
        try:
            with open("deep_researcher_chat.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return JSONResponse(content={"error": "Chat interface not found"}, status_code=404)
        
    @app.get("/researcher-chat")
    async def researcher_chat():
        """Serve the unified Deep Researcher Agent chat interface."""
        try:
            html_path = Path(__file__).parent / "chat_interface.html"
            
            if html_path.exists():
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content, status_code=200)
            else:
                return HTMLResponse(
                    content="<h1>Deep Researcher Chat Interface not found</h1>", 
                    status_code=404
                )
        except Exception as e:
            logger.error(f"Error serving researcher chat: {str(e)}")
            return HTMLResponse(
                content=f"<h1>Error loading chat interface: {str(e)}</h1>", 
                status_code=500
            )
    
    @app.get("/dashboard")
    async def comprehensive_dashboard():
        """Serve the comprehensive dashboard page."""
        try:
            with open("dashboard.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return JSONResponse({
                "error": "Dashboard page not found",
                "message": "Please ensure dashboard.html exists in the project root"
            }, status_code=404)

    @app.get("/chat")
    async def chat_interface():
        """Serve the chat interface page."""
        try:
            with open("chat_interface.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return JSONResponse({
                "error": "Chat interface not found", 
                "message": "Please ensure chat_interface.html exists in the project root"
            }, status_code=404)

    @app.get("/query-demo")
    async def query_demo():
        """Serve the query processing demo page."""
        try:
            with open("query_demo.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return JSONResponse({
                "error": "Demo page not found",
                "message": "Please ensure query_demo.html exists in the project root"
            }, status_code=404)
    
    return app

app = create_application()

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Deep Researcher Agent application", 
                debug=settings.debug, 
                environment=settings.environment,
                version="1.0.0")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
