#!/usr/bin/env python3
"""
Quick installation script using uv package manager.
Installs essential dependencies for DataMCPServerAgent v2.0.
"""

import subprocess
import sys
import shutil
import platform


def print_header():
    """Print installation header."""
    print("🚀 DataMCPServerAgent Quick Install (uv edition)")
    print("=" * 60)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def install_uv():
    """Install uv package manager."""
    print("\n📦 Installing uv package manager...")
    
    if shutil.which("uv"):
        print("✅ uv already installed")
        return True
    
    try:
        if platform.system() == "Windows":
            # Windows installation
            subprocess.run([
                "powershell", "-c", 
                "irm https://astral.sh/uv/install.ps1 | iex"
            ], check=True)
        else:
            # Try pip install first (simpler)
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "uv"
                ], check=True, capture_output=True)
                print("✅ uv installed via pip")
                return True
            except subprocess.CalledProcessError:
                # Fallback to curl install
                subprocess.run([
                    "curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"
                ], shell=True, check=True)
        
        print("✅ uv installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install uv: {e}")
        print("💡 Try installing manually: pip install uv")
        return False


def install_core_deps():
    """Install core dependencies with uv."""
    print("\n🔧 Installing core dependencies...")
    
    deps = [
        "pydantic>=2.5.0",
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "rich>=13.7.0",
        "typer[all]>=0.9.0",
        "structlog>=23.2.0",
        "python-dotenv>=1.0.0"
    ]
    
    for dep in deps:
        try:
            print(f"  Installing {dep}...")
            subprocess.run([
                "uv", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed: {dep}")
            return False
    
    print("✅ Core dependencies installed")
    return True


def install_test_deps():
    """Install testing dependencies."""
    print("\n🧪 Installing test dependencies...")
    
    deps = [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0"
    ]
    
    for dep in deps:
        try:
            print(f"  Installing {dep}...")
            subprocess.run([
                "uv", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed: {dep}")
            return False
    
    print("✅ Test dependencies installed")
    return True


def verify_installation():
    """Verify installation."""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        "pydantic",
        "fastapi", 
        "rich",
        "typer"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            return False
    
    print("✅ Installation verified")
    return True


def run_quick_test():
    """Run a quick test."""
    print("\n🧪 Running quick test...")
    
    try:
        # Simple test
        import json
        test_data = {"test": True, "version": "2.0"}
        json_str = json.dumps(test_data)
        parsed = json.loads(json_str)
        assert parsed["test"] is True
        
        print("✅ Quick test passed")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main installation function."""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Install uv
    if not install_uv():
        return 1
    
    # Install dependencies
    if not install_core_deps():
        return 1
    
    if not install_test_deps():
        print("⚠️ Test dependencies failed, but continuing...")
    
    # Verify installation
    if not verify_installation():
        return 1
    
    # Run quick test
    if not run_quick_test():
        return 1
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("  1. Run tests: python scripts/test_runner_uv.py")
    print("  2. Start API: python scripts/main.py api")
    print("  3. Check status: python scripts/main.py status")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
