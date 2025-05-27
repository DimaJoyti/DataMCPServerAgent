#!/usr/bin/env python3
"""
Script to install dependencies for the Document Processing Pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str = ""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description or command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command.split(),
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success: {description or command}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description or command}")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def install_core_dependencies():
    """Install core dependencies."""
    core_deps = [
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.0.0",
        "httpx>=0.24.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0"
    ]
    
    for dep in core_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_document_processing():
    """Install document processing dependencies."""
    doc_deps = [
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.12.0",
        "markdown>=3.4.0",
        "html2text>=2020.1.16",
        "python-magic>=0.4.27",
        "chardet>=5.0.0",
        "langdetect>=1.0.9",
        "textstat>=0.7.3"
    ]
    
    for dep in doc_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_excel_powerpoint():
    """Install Excel and PowerPoint support."""
    office_deps = [
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "python-pptx>=0.6.21"
    ]
    
    for dep in office_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_vectorization():
    """Install vectorization dependencies."""
    vector_deps = [
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "openai>=1.0.0"
    ]
    
    for dep in vector_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_vector_stores():
    """Install vector store dependencies."""
    store_deps = [
        "chromadb>=0.4.0",
        "faiss-cpu>=1.7.4"
    ]
    
    for dep in store_deps:
        success = run_command(f"uv pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"⚠️  Warning: Failed to install {dep}. This is optional.")


def install_web_interface():
    """Install web interface dependencies."""
    web_deps = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.0"
    ]
    
    for dep in web_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_development():
    """Install development dependencies."""
    dev_deps = [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "ruff>=0.0.280",
        "mypy>=1.5.0"
    ]
    
    for dep in dev_deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")


def install_optional_dependencies():
    """Install optional dependencies."""
    optional_deps = [
        ("pinecone-client>=2.2.0", "Pinecone vector store"),
        ("weaviate-client>=3.20.0", "Weaviate vector store"),
        ("qdrant-client>=1.3.0", "Qdrant vector store"),
        ("gunicorn>=21.0.0", "Production WSGI server"),
        ("redis>=4.6.0", "Redis caching"),
        ("structlog>=23.0.0", "Structured logging")
    ]
    
    print(f"\n{'='*60}")
    print("🔧 Installing optional dependencies")
    print(f"{'='*60}")
    
    for dep, description in optional_deps:
        success = run_command(f"uv pip install {dep}", f"Installing {description}")
        if not success:
            print(f"⚠️  Warning: Failed to install {dep}. This is optional.")


def main():
    """Main installation function."""
    print("🚀 Document Processing Pipeline - Dependency Installation")
    print("=" * 80)
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV package manager not found. Please install UV first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Install dependencies in order
    install_core_dependencies()
    install_document_processing()
    install_excel_powerpoint()
    install_vectorization()
    install_vector_stores()
    install_web_interface()
    install_development()
    
    # Ask about optional dependencies
    response = input("\n🤔 Install optional dependencies? (y/N): ").lower().strip()
    if response in ['y', 'yes']:
        install_optional_dependencies()
    
    print(f"\n{'='*80}")
    print("🎉 Installation completed!")
    print("=" * 80)
    
    print("\n📋 Next steps:")
    print("1. Run examples: python examples/advanced_features_example.py")
    print("2. Start web interface: python src/web_interface/server.py")
    print("3. Run tests: python -m pytest tests/")
    
    print("\n📚 Documentation:")
    print("- Setup guide: ADVANCED_FEATURES_SETUP.md")
    print("- Pipeline guide: PIPELINE_SETUP.md")
    print("- API docs: http://localhost:8000/docs (after starting web interface)")


if __name__ == "__main__":
    main()
