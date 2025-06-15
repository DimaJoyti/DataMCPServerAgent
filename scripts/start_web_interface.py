#!/usr/bin/env python3
"""
Script to start the Document Processing Pipeline web interface.
"""

import os
import sys
from pathlib import Path


def main():
    """Start the web interface."""
    print("🚀 Starting Document Processing Pipeline Web Interface")
    print("=" * 60)

    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Set environment variables
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", "8000")
    os.environ.setdefault("LOG_LEVEL", "info")

    print(f"🌐 Host: {os.environ['HOST']}")
    print(f"🔌 Port: {os.environ['PORT']}")
    print(f"📝 Log Level: {os.environ['LOG_LEVEL']}")

    # Import and run the server
    try:
        from src.web_interface.server import main as server_main
        server_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("  python install_pipeline_deps.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
