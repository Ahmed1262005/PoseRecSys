"""
FastAPI Application Factory.

Usage:
    # Development
    PYTHONPATH=src uvicorn api.app:app --reload

    # Production (gunicorn)
    PYTHONPATH=src gunicorn api.app:app -k uvicorn.workers.UvicornWorker \
        --workers 2 --bind 0.0.0.0:8000 --timeout 120
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from core.logging import configure_logging, get_logger
from core.middleware import RequestTracingMiddleware


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup warmup and shutdown."""
    settings = get_settings()

    configure_logging(
        json_logs=settings.is_production,
        log_level="DEBUG" if settings.debug else "INFO",
    )

    logger.info(
        "Starting recommendation API",
        environment=settings.environment,
        port=settings.port,
    )

    # Pre-load FashionCLIP model so first requests are not slow
    if settings.is_production:
        try:
            from core.clip_service import get_clip_service
            get_clip_service().warmup()
        except Exception as e:
            logger.warning("FashionCLIP warmup failed (will lazy-load)", error=str(e))

    yield

    logger.info("Shutting down recommendation API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Fashion Recommendation API",
        description="""
        Production fashion recommendation system.

        ## Features

        - **Hybrid Search**: Algolia + FashionCLIP semantic search with RRF merge
        - **Personalized Feed**: Recommendations via SASRec + taste vectors
        - **Outfit Engine**: Complete-the-fit with TATTOO scoring
        - **Canvas & Pinterest**: Inspiration management and taste extraction

        ## Main Endpoints

        - `/api/search/*` - Hybrid search
        - `/api/recs/v2/*` - Recommendation pipeline
        - `/api/canvas/*` - Inspiration canvas
        - `/api/integrations/pinterest/*` - Pinterest integration

        ## Health Checks

        - `/health` - Basic health check
        - `/health/detailed` - Detailed health with dependency status
        - `/ready` - Kubernetes readiness probe
        - `/live` - Kubernetes liveness probe
        """,
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # =========================================================================
    # Middleware (order matters - first added = outermost)
    # =========================================================================

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestTracingMiddleware)

    # =========================================================================
    # Routes
    # =========================================================================

    # Health checks (always available)
    from api.routes.health import router as health_router
    app.include_router(health_router, tags=["Health"])

    # Pinterest integration
    from api.routes.pinterest import router as pinterest_router
    app.include_router(pinterest_router)

    # Recommendation pipeline
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

    # POSE Canvas (inspiration management + style extraction)
    try:
        from canvas.routes import router as canvas_router
        app.include_router(canvas_router)
        logger.info("Mounted canvas routes: /api/canvas/*")
    except ImportError as e:
        logger.warning(f"Could not load canvas router: {e}")

    return app


# Create default app instance for uvicorn / gunicorn
app = create_app()


def get_app() -> FastAPI:
    """Get the application instance (for ASGI servers)."""
    return app
