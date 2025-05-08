#!/usr/bin/env python3
"""
Installation script for DataMCPServerAgent dependencies.
This script installs all required dependencies for the advanced enhanced agent.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies for DataMCPServerAgent...")
    
    # Get the requirements file path
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Error: Requirements file not found at {requirements_path}")
        return False
    
    try:
        # Install dependencies using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    if success:
        print("\nYou can now run the advanced enhanced agent with:")
        print("python advanced_enhanced_main.py")
    else:
        print("\nFailed to install dependencies. Please try installing them manually:")
        print("pip install -r requirements.txt")
    
    sys.exit(0 if success else 1)
