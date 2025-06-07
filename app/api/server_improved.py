"""
Improved FastAPI Server for DataMCPServerAgent.

This module creates a production-ready FastAPI application with:
- Comprehensive middleware stack
- Error handling and validation
- API versioning and documentation
- Health checks and monitoring
- Security features
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config_improved import Settings
from app.core.exceptions_improved import BaseError, handle_exception
from app.core.logging_improved import (
    get_logger,
    set_agent_id,
    set_correlation_id,
    set_request_id,
    set_user_id,
    setup_logging,
)
from app.infrastructure.cache.manager import CacheManager
from app.infrastructure.database.manager import DatabaseManager
from app.infrastructure.monitoring.metrics import MetricsManager

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting DataMCPServerAgent API Server")

    # Initialize infrastructure
    try:
        # Database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        app.state.db_manager = db_manager
        logger.info("✅ Database initialized")

        # Cache
        cache_manager = CacheManager()
        await cache_manager.initialize()
        app.state.cache_manager = cache_manager
        logger.info("✅ Cache initialized")

        # Metrics
        metrics_manager = MetricsManager()
        await metrics_manager.initialize()
        app.state.metrics_manager = metrics_manager
        logger.info("✅ Metrics initialized")

        logger.info("🎉 Application startup complete")

    except Exception as e:
        logger.error(f"💥 Failed to initialize application: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("🛑 Shutting down DataMCPServerAgent API Server")

    try:
        if hasattr(app.state, "db_manager"):
            await app.state.db_manager.close()
            logger.info("✅ Database closed")

        if hasattr(app.state, "cache_manager"):
            await app.state.cache_manager.close()
            logger.info("✅ Cache closed")

        if hasattr(app.state, "metrics_manager"):
            await app.state.metrics_manager.close()
            logger.info("✅ Metrics closed")

        logger.info("👋 Application shutdown complete")

    except Exception as e:
        logger.error(f"💥 Error during shutdown: {e}", exc_info=True)

def create_api_server(settings: Settings = None) -> FastAPI:
    """Create and configure FastAPI application."""

    if settings is None:
        from app.core.config_improved import settings as default_settings

        settings = default_settings

    # Setup logging
    setup_logging(settings)

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url=None,  # Custom docs
        redoc_url=None,  # Custom redoc
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )

    # Store settings
    app.state.settings = settings

    # Setup middleware
    _setup_middleware(app, settings)

    # Setup routes
    _setup_routes(app, settings)

    # Setup exception handlers
    _setup_exception_handlers(app, settings)

    # Setup custom docs
    _setup_custom_docs(app, settings)

    logger.info("🏗️ FastAPI application created and configured")

    return app

def _setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Setup application middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=settings.security.cors_methods,
        allow_headers=settings.security.cors_headers,
    )

    # Trusted host middleware (for production)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["*"]  # Configure this properly for production
        )

    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request correlation middleware
    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        """Add correlation ID to requests."""
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        set_correlation_id(correlation_id)

        # Extract user and agent IDs from headers if available
        user_id = request.headers.get("X-User-ID")
        if user_id:
            set_user_id(user_id)

        agent_id = request.headers.get("X-Agent-ID")
        if agent_id:
            set_agent_id(agent_id)

        # Generate request ID
        request_id = str(uuid.uuid4())
        set_request_id(request_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-ID"] = request_id

        return response

    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log requests and responses."""
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
            },
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )

        # Update metrics
        if hasattr(request.app.state, "metrics_manager"):
            await request.app.state.metrics_manager.record_request(
                method=request.method,
                endpoint=str(request.url.path),
                status_code=response.status_code,
                duration=duration,
            )

        return response

    # Security headers middleware
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """Add security headers."""
        response = await call_next(request)

        if settings.security.enable_security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

def _setup_routes(app: FastAPI, settings: Settings) -> None:
    """Setup application routes."""

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "timestamp": time.time(),
        }

        # Check database health
        if hasattr(app.state, "db_manager"):
            try:
                db_healthy = await app.state.db_manager.health_check()
                health_status["database"] = "healthy" if db_healthy else "unhealthy"
            except Exception:
                health_status["database"] = "unhealthy"

        # Check cache health
        if hasattr(app.state, "cache_manager"):
            try:
                cache_healthy = await app.state.cache_manager.health_check()
                health_status["cache"] = "healthy" if cache_healthy else "unhealthy"
            except Exception:
                health_status["cache"] = "unhealthy"

        return health_status

    # Metrics endpoint
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        if hasattr(app.state, "metrics_manager"):
            return await app.state.metrics_manager.get_metrics()
        return {"error": "Metrics not available"}

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs" if settings.debug else "Documentation not available in production",
            "health": "/health",
            "metrics": "/metrics",
        }

    # Include API v1 router
    app.include_router(api_v1_router, prefix="/api/v1", tags=["API v1"])

def _setup_exception_handlers(app: FastAPI, settings: Settings) -> None:
    """Setup global exception handlers."""

    @app.exception_handler(BaseError)
    async def base_error_handler(request: Request, exc: BaseError):
        """Handle custom application errors."""
        logger.warning(
            f"Application error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "category": exc.category.value,
                "details": exc.details,
            },
        )

        status_code = 400
        if exc.category.value == "not_found":
            status_code = 404
        elif exc.category.value == "authorization":
            status_code = 403
        elif exc.category.value == "authentication":
            status_code = 401
        elif exc.category.value == "conflict":
            status_code = 409
        elif exc.category.value == "rate_limit":
            status_code = 429
        elif exc.category.value in ["external_service", "infrastructure", "system"]:
            status_code = 500

        return JSONResponse(status_code=status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        error = handle_exception(exc)

        logger.error(
            f"Unhandled exception: {exc}", extra={"error_id": error.error_id}, exc_info=True
        )

        if settings.debug:
            return JSONResponse(status_code=500, content=error.to_dict())
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error_id": error.error_id,
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal error occurred",
                    "suggestions": ["Try again later", "Contact support if the problem persists"],
                },
            )

def _setup_custom_docs(app: FastAPI, settings: Settings) -> None:
    """Setup custom API documentation."""

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )

    def custom_openapi():
        """Custom OpenAPI schema."""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add custom info
        openapi_schema["info"]["x-logo"] = {"url": "https://datamcp.dev/logo.png"}

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": settings.security.api_key_header,
            },
            "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
