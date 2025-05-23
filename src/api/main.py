"""
Main entry point for the API.
"""

import os
import sys
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.config import config
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limiting import RateLimitingMiddleware
from src.api.routers import agents, chat, health, memory, tools
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()

# Create the FastAPI application
app = FastAPI(
    title=config.title,
    description=config.description,
    version=config.version,
    openapi_url=config.openapi_url,
    docs_url=config.docs_url,
    redoc_url=config.redoc_url,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allow_origins,
    allow_credentials=True,
    allow_methods=config.allow_methods,
    allow_headers=config.allow_headers,
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add rate limiting middleware if enabled
if config.enable_rate_limiting:
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
        "name": config.title,
        "version": config.version,
        "description": config.description,
        "docs_url": config.docs_url,
        "redoc_url": config.redoc_url,
    }


def start_api():
    """Start the API server."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )


if __name__ == "__main__":
    start_api()
