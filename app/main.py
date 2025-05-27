"""
Main application factory and configuration.
Creates and configures the FastAPI application with all dependencies.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings
from app.core.logging import get_logger, set_agent_id, set_correlation_id, set_user_id
from app.domain.services import AgentService, StateService, TaskService
from app.infrastructure.database import DatabaseManager
from app.infrastructure.dependencies import setup_dependencies
from app.infrastructure.monitoring import setup_monitoring

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting DataMCPServerAgent...")

    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    app.state.db_manager = db_manager

    # Setup dependencies
    await setup_dependencies(app)

    # Setup monitoring
    if settings.monitoring.enable_metrics:
        setup_monitoring(app)

    # Initialize services
    await _initialize_services(app)

    logger.info("✅ DataMCPServerAgent started successfully!")

    yield

    # Cleanup
    logger.info("🛑 Shutting down DataMCPServerAgent...")

    # Close database connections
    if hasattr(app.state, "db_manager"):
        await app.state.db_manager.close()

    logger.info("✅ DataMCPServerAgent shutdown complete!")


async def _initialize_services(app: FastAPI) -> None:
    """Initialize domain services."""
    try:
        # Initialize core services
        agent_service = AgentService()
        task_service = TaskService()
        state_service = StateService()

        # Store services in app state
        app.state.agent_service = agent_service
        app.state.task_service = task_service
        app.state.state_service = state_service

        logger.info("✅ Domain services initialized")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Add middleware
    _setup_middleware(app)

    # Setup monitoring
    setup_monitoring(app)

    # Add exception handlers
    _setup_exception_handlers(app)

    # Include routers
    app.include_router(api_router, prefix="/api/v1")

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment.value,
            "features": {
                "cloudflare": settings.enable_cloudflare,
                "email": settings.enable_email,
                "webrtc": settings.enable_webrtc,
                "self_hosting": settings.enable_self_hosting,
            },
        }

    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs" if settings.debug else "Documentation not available in production",
        }

    return app


def _setup_middleware(app: FastAPI) -> None:
    """Setup application middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Trusted host middleware (for production)
    if settings.environment.value == "production":
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["*"]  # Configure this properly for production
        )

    # Request correlation middleware
    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        """Add correlation ID to requests."""
        import uuid

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

        # Process request
        response = await call_next(request)

        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id

        return response

    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log requests and responses."""
        import time

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

        return response


def _setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""

    from app.domain.models.base import (
        BusinessRuleError,
        ConcurrencyError,
        EntityNotFoundError,
        ValidationError,
    )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle domain validation errors."""
        logger.warning(f"Validation error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(BusinessRuleError)
    async def business_rule_error_handler(request: Request, exc: BusinessRuleError):
        """Handle business rule violations."""
        logger.warning(f"Business rule error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=422,
            content={
                "error": "business_rule_error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(EntityNotFoundError)
    async def entity_not_found_error_handler(request: Request, exc: EntityNotFoundError):
        """Handle entity not found errors."""
        logger.warning(f"Entity not found: {exc.message}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "entity_not_found",
                "message": exc.message,
            },
        )

    @app.exception_handler(ConcurrencyError)
    async def concurrency_error_handler(request: Request, exc: ConcurrencyError):
        """Handle concurrency/optimistic locking errors."""
        logger.warning(f"Concurrency error: {exc.message}")
        return JSONResponse(
            status_code=409,
            content={
                "error": "concurrency_error",
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        if settings.debug:
            import traceback

            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal error occurred",
                },
            )


# Create app instance for direct import
app = create_app()


def run_server():
    """Run the application server."""
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.workers if settings.environment.value == "production" else 1,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run_server()
