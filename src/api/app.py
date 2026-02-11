"""
FastAPI Application Factory.

This module provides a clean, configurable FastAPI application setup.

Usage:
    # Development
    uvicorn api.app:create_app --factory --reload
    
    # Production
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Or use the convenience function
    from api.app import create_app
    app = create_app()
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from core.logging import configure_logging, get_logger
from core.middleware import RequestTracingMiddleware


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    
    Runs on startup:
    - Initialize logging
    - Load engines (lazy)
    - Warm up caches
    
    Runs on shutdown:
    - Clean up resources
    """
    settings = get_settings()
    
    # Configure logging based on environment
    configure_logging(
        json_logs=settings.is_production,
        log_level="DEBUG" if settings.debug else "INFO",
    )
    
    logger.info(
        "Starting recommendation API",
        environment=settings.environment,
        port=settings.port,
    )
    
    # Startup: Initialize resources here if needed
    # Engines are lazy-loaded on first request
    
    # Load brand names for search query classification
    try:
        from config.database import get_supabase_client
        from search.query_classifier import load_brands
        supabase = get_supabase_client()
        brands = load_brands(supabase)
        logger.info(f"Loaded {len(brands)} brand names for search classifier")
    except Exception as e:
        logger.warning(f"Could not load brands for search classifier: {e}")
    
    yield  # Application is running
    
    # Shutdown: Clean up resources
    logger.info("Shutting down recommendation API")


def create_app(
    include_static_files: bool = True,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        include_static_files: If True, mount static file directories
        
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Fashion Recommendation API",
        description="""
        Production fashion recommendation system with style preference learning.
        
        ## Features
        
        - **Style Learning**: Tinder-style 4-choice interface to learn user preferences
        - **Personalized Feed**: Recommendations based on learned taste vectors
        - **Vector Search**: pgvector-powered similarity search in Supabase
        
        ## Main Endpoints
        
        - `/api/women/*` - Women's fashion style learning and feed
        - `/api/unified/*` - Gender-aware unified API
        - `/api/recs/v2/*` - Full recommendation pipeline
        
        ## Health Checks
        
        - `/health` - Basic health check
        - `/health/detailed` - Detailed health with dependency status
        - `/ready` - Kubernetes readiness probe
        - `/live` - Kubernetes liveness probe
        """,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )
    
    # =========================================================================
    # Middleware (order matters - first added = outermost)
    # =========================================================================
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request tracing (adds X-Request-ID, logs timing)
    app.add_middleware(RequestTracingMiddleware)
    
    # =========================================================================
    # Routes
    # =========================================================================
    
    # Health checks
    from api.routes.health import router as health_router
    app.include_router(health_router, tags=["Health"])
    
    # Women's fashion routes (new modular version)
    from api.routes.women import router as women_router
    app.include_router(women_router)
    
    # Unified gender-aware routes
    from api.routes.unified import router as unified_router
    app.include_router(unified_router)

    # Pinterest integration routes
    from api.routes.pinterest import router as pinterest_router
    app.include_router(pinterest_router)
    
    # Recommendation pipeline (legacy + v2)
    try:
        from recs.api_endpoints import router as recs_router, v2_router as recs_v2_router
        app.include_router(recs_router)
        app.include_router(recs_v2_router)
        logger.info("Mounted recommendation routes: /api/recs/* and /api/recs/v2/*")
    except ImportError as e:
        logger.warning(f"Could not load recs router: {e}")
    
    # Hybrid search (Algolia + FashionCLIP)
    try:
        from api.routes.search import router as search_router
        app.include_router(search_router)
        logger.info("Mounted search routes: /api/search/*")
    except ImportError as e:
        logger.warning(f"Could not load search router: {e}")
    
    # =========================================================================
    # Static Files
    # =========================================================================
    
    if include_static_files:
        _mount_static_files(app, settings)
    
    return app


def _mount_static_files(app: FastAPI, settings) -> None:
    """
    Mount static file directories for images.
    """
    # Women's fashion images
    women_images_path = settings.images_dir
    if women_images_path.exists():
        app.mount(
            "/women-images",
            StaticFiles(directory=str(women_images_path)),
            name="women-images",
        )
        logger.info(f"Mounted women's images at /women-images from {women_images_path}")
    else:
        logger.warning(f"Women's images directory not found: {women_images_path}")
    
    # HP Images (men's fashion)
    hp_images_path = settings.hp_images_dir
    if hp_images_path.exists():
        app.mount(
            "/images",
            StaticFiles(directory=str(hp_images_path)),
            name="images",
        )
        logger.info(f"Mounted HP images at /images from {hp_images_path}")
    else:
        logger.warning(f"HP images directory not found: {hp_images_path}")


# Create default app instance for uvicorn
# Usage: uvicorn api.app:app
app = create_app()


# Alternative: Factory function for gunicorn
# Usage: gunicorn -k uvicorn.workers.UvicornWorker api.app:create_app()
def get_app() -> FastAPI:
    """Get the application instance (for ASGI servers)."""
    return app
