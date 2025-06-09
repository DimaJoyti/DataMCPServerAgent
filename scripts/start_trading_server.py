#!/usr/bin/env python3
"""
Trading Server Startup Script
Starts the FastAPI trading server with proper configuration
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from src.web_interface.trading_api_server import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the trading server"""
    logger.info("Starting Institutional Trading System API Server...")
    
    # Server configuration
    config = {
        "app": "src.web_interface.trading_api_server:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True,
        "workers": 1,  # Use 1 worker for development
    }
    
    # Production configuration
    if os.getenv("ENVIRONMENT") == "production":
        config.update({
            "reload": False,
            "workers": 4,
            "log_level": "warning"
        })
    
    logger.info(f"Server will start on http://{config['host']}:{config['port']}")
    logger.info("API Documentation available at http://localhost:8000/docs")
    logger.info("WebSocket endpoints:")
    logger.info("  - Trading: ws://localhost:8000/ws/trading")
    logger.info("  - Market Data: ws://localhost:8000/ws/market-data")
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
