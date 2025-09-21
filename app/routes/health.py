"""
Health check and system status API routes.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import HealthCheck, SystemStatus, ServiceStatus
from app.services.document_processor import document_processor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.query_processor import query_processor
from app.services.response_synthesizer import response_synthesizer

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Basic health check endpoint.
    
    Returns:
        Health check response with basic system information
    """
    try:
        return HealthCheck(
            status="healthy",
            version=settings.app_version,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(
            "Error in basic health check",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get("/detailed", response_model=SystemStatus)
async def detailed_health_check() -> SystemStatus:
    """
    Detailed health check that tests all system components.
    
    Returns:
        Detailed system status with component health information
    """
    try:
        logger.info("Performing detailed health check")
        
        # Check all services
        services = []
        overall_healthy = True
        
        # Check embedding service
        try:
            embedding_health = await embedding_service.health_check()
            services.append(ServiceStatus(
                service_name="embedding_service",
                status=embedding_health.get("status", "unknown"),
                details=embedding_health
            ))
            if embedding_health.get("status") != "healthy":
                overall_healthy = False
        except Exception as e:
            services.append(ServiceStatus(
                service_name="embedding_service",
                status="unhealthy",
                details={"error": str(e)}
            ))
            overall_healthy = False
        
        # Check vector store
        try:
            vector_health = await vector_store.health_check()
            services.append(ServiceStatus(
                service_name="vector_store",
                status=vector_health.get("status", "unknown"),
                details=vector_health
            ))
            if vector_health.get("status") != "healthy":
                overall_healthy = False
        except Exception as e:
            services.append(ServiceStatus(
                service_name="vector_store",
                status="unhealthy",
                details={"error": str(e)}
            ))
            overall_healthy = False
        
        # Check query processor
        try:
            query_health = await query_processor.health_check()
            services.append(ServiceStatus(
                service_name="query_processor",
                status=query_health.get("status", "unknown"),
                details=query_health
            ))
            if query_health.get("status") != "healthy":
                overall_healthy = False
        except Exception as e:
            services.append(ServiceStatus(
                service_name="query_processor",
                status="unhealthy",
                details={"error": str(e)}
            ))
            overall_healthy = False
        
        # Check response synthesizer
        try:
            synthesis_health = await response_synthesizer.health_check()
            services.append(ServiceStatus(
                service_name="response_synthesizer",
                status=synthesis_health.get("status", "unknown"),
                details=synthesis_health
            ))
            if synthesis_health.get("status") != "healthy":
                overall_healthy = False
        except Exception as e:
            services.append(ServiceStatus(
                service_name="response_synthesizer",
                status="unhealthy",
                details={"error": str(e)}
            ))
            overall_healthy = False
        
        # Collect system statistics
        statistics = {}
        try:
            # Get vector store stats
            vector_stats = await vector_store.get_stats()
            statistics["vector_store"] = vector_stats
            
            # Get search stats
            search_stats = await query_processor.get_search_stats()
            statistics["search"] = search_stats
            
            # Get synthesis stats
            synthesis_stats = await response_synthesizer.get_synthesis_stats()
            statistics["research"] = synthesis_stats
            
        except Exception as e:
            logger.warning(
                "Error collecting system statistics",
                error=str(e)
            )
            statistics["error"] = "Failed to collect statistics"
        
        system_status = SystemStatus(
            overall_status="healthy" if overall_healthy else "unhealthy",
            services=services,
            statistics=statistics
        )
        
        logger.info(
            "Detailed health check completed",
            overall_status=system_status.overall_status,
            healthy_services=sum(1 for s in services if s.status == "healthy"),
            total_services=len(services)
        )
        
        return system_status
        
    except Exception as e:
        logger.error(
            "Error in detailed health check",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detailed health check failed"
        )


@router.get("/readiness")
async def readiness_check() -> JSONResponse:
    """
    Readiness check to determine if the service is ready to handle requests.
    
    Returns:
        JSON response indicating readiness status
    """
    try:
        # Check if critical services are initialized
        ready = True
        checks = {}
        
        # Check embedding service
        checks["embedding_service"] = embedding_service.is_initialized
        if not embedding_service.is_initialized:
            ready = False
        
        # Check vector store (basic check)
        try:
            vector_stats = await vector_store.get_stats()
            checks["vector_store"] = vector_stats.get("initialized", False)
            if not vector_stats.get("initialized", False):
                ready = False
        except:
            checks["vector_store"] = False
            ready = False
        
        status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "ready": ready,
                "checks": checks,
                "message": "Service is ready" if ready else "Service is not ready"
            }
        )
        
    except Exception as e:
        logger.error(
            "Error in readiness check",
            error=str(e),
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "error": str(e),
                "message": "Readiness check failed"
            }
        )


@router.get("/liveness")
async def liveness_check() -> JSONResponse:
    """
    Liveness check to determine if the service is alive.
    
    Returns:
        JSON response indicating liveness status
    """
    try:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "alive": True,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Service is alive"
            }
        )
        
    except Exception as e:
        logger.error(
            "Error in liveness check",
            error=str(e),
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "alive": False,
                "error": str(e),
                "message": "Liveness check failed"
            }
        )
