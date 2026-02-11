"""
Health check endpoints.

Provides endpoints for monitoring application health and status.
"""

from typing import Dict, Any
from fastapi import APIRouter

from config.settings import get_settings
from config.database import get_supabase_client_optional


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status with basic info
    """
    return {
        "status": "healthy",
        "service": "recommendation-api",
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with dependency status.
    
    Checks:
    - Configuration loaded
    - Supabase connection
    
    Returns:
        Detailed health status
    """
    settings = get_settings()
    
    # Check Supabase
    supabase_status = "unknown"
    supabase_error = None
    try:
        client = get_supabase_client_optional()
        if client:
            # Try a simple query
            result = client.table("products").select("id").limit(1).execute()
            supabase_status = "connected" if result.data else "empty"
        else:
            supabase_status = "not_configured"
    except Exception as e:
        supabase_status = "error"
        supabase_error = str(e)
    
    return {
        "status": "healthy" if supabase_status == "connected" else "degraded",
        "service": "recommendation-api",
        "environment": settings.environment,
        "checks": {
            "config": "ok",
            "supabase": {
                "status": supabase_status,
                "error": supabase_error,
            },
        },
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    """
    # Check if we can connect to Supabase
    client = get_supabase_client_optional()
    if client is None:
        return {"status": "not_ready", "reason": "database_not_configured"}
    
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive.
    """
    return {"status": "alive"}
