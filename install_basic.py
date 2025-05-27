#!/usr/bin/env python3
"""
Install basic dependencies for DataMCPServerAgent.
"""

import subprocess
import sys

def install_basic_deps():
    """Install basic dependencies."""
    print("📦 Installing basic dependencies...")
    
    basic_deps = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.3",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "aiofiles>=23.2.1"
    ]
    
    for dep in basic_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("✅ Basic dependencies installed successfully!")
    return True

if __name__ == "__main__":
    success = install_basic_deps()
    sys.exit(0 if success else 1)
