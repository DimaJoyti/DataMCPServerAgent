"""
Main entry point for the API following Clean Architecture patterns.
Unified FastAPI application for /src directory components.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.config import get_settings
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limiting import RateLimitingMiddleware
from src.api.routers import agents, chat, health, memory, tools
from src.utils.env_config import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler following Clean Architecture patterns."""
    logger.info("Starting DataMCPServerAgent /src API")
    
    # Initialize services and dependencies
    settings = get_settings()
    logger.info(f"Loaded settings: {settings.app_name}")
    
    # Initialize Redis if distributed mode is enabled
    if settings.enable_distributed:
        from src.api.services.redis_service import RedisService
        redis_service = RedisService()
        await redis_service.connect()
        app.state.redis = redis_service
    
    yield
    
    # Cleanup
    if hasattr(app.state, "redis"):
        await app.state.redis.disconnect()
    
    logger.info("Shutting down DataMCPServerAgent /src API")


# Create the FastAPI application
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="DataMCPServerAgent - Unified API for /src components",
        version="1.0.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    return app


app = create_app()
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add rate limiting middleware if enabled
if settings.enable_rate_limiting:
    app.add_middleware(RateLimitingMiddleware)


# Add exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.

    Args:
        request (Request): Request object
        exc (StarletteHTTPException): Exception

    Returns:
        JSONResponse: JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "error_message": str(exc.detail),
            "error_details": None,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions.

    Args:
        request (Request): Request object
        exc (Exception): Exception

    Returns:
        JSONResponse: JSON response with error details
    """
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_message": "An internal server error occurred",
            "error_details": {"type": type(exc).__name__, "message": str(exc)},
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# Include routers
app.include_router(health.router)
app.include_router(agents.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(tools.router)


@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint.

    Returns:
        Dict[str, Any]: API information
    """
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "DataMCPServerAgent - Unified API for /src components",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


def start_api():
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start_api()
