"""
Web server for document processing pipeline.
"""

import logging
import os
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from .api import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_server():
    """Create and configure the web server."""
    # Create FastAPI app
    app, api_service = create_app()
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Serve index.html at root
        @app.get("/ui")
        async def serve_ui():
            """Serve the web UI."""
            from fastapi.responses import FileResponse
            return FileResponse(str(static_dir / "index.html"))
    
    return app, api_service


def main():
    """Main entry point for the web server."""
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Document Processing Pipeline API server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Log Level: {log_level}")
    
    # Create app
    app, _ = create_server()
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
