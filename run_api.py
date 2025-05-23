#!/usr/bin/env python
"""
Script to run the API server.
"""

import argparse
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.main import start_api
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="DataMCPServerAgent API Server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind the server to",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind the server to",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("API_DEBUG", "false").lower() == "true",
        help="Enable debug mode",
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["API_RELOAD"] = str(args.reload).lower()
    os.environ["API_DEBUG"] = str(args.debug).lower()
    
    print(f"Starting DataMCPServerAgent API Server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Auto-reload: {args.reload}")
    
    # Start the API server
    start_api()


if __name__ == "__main__":
    main()
