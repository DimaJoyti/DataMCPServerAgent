#!/usr/bin/env python3
"""
Enhanced installation script for DataMCPServerAgent dependencies.
Uses uv for fast and reliable package management.
"""

import subprocess
import sys
import shutil
from typing import List
import platform

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import typer

console = Console()


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        console.print(
            f"[red]❌ Python 3.9+ required, found {version.major}.{version.minor}[/red]"
        )
        return False

    console.print(f"[green]✅ Python {version.major}.{version.minor} detected[/green]")
    return True


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    return shutil.which("uv") is not None


def install_uv() -> bool:
    """Install uv package manager."""
    console.print("📦 Installing uv package manager...")

    try:
        if platform.system() == "Windows":
            # Windows installation
            subprocess.run([
                "powershell", "-c",
                "irm https://astral.sh/uv/install.ps1 | iex"
            ], check=True)
        else:
            # Unix-like systems - try multiple methods
            try:
                # Method 1: Direct curl install
                subprocess.run([
                    "curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"
                ], shell=True, check=True)
            except subprocess.CalledProcessError:
                # Method 2: Fallback to pip install
                console.print("🔄 Trying pip install as fallback...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "uv"
                ], check=True)

        console.print("[green]✅ uv installed successfully[/green]")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to install uv: {e}[/red]")
        console.print("[yellow]💡 Try installing manually: pip install uv[/yellow]")
        return False


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command with progress indication."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            progress.update(task, completed=True)
            console.print(f"[green]✅ {description}[/green]")
            return True

        except subprocess.CalledProcessError as e:
            progress.update(task, completed=True)
            console.print(f"[red]❌ {description} failed[/red]")
            console.print(f"[red]Error: {e.stderr}[/red]")
            return False


def install_core_dependencies() -> bool:
    """Install core dependencies using uv."""
    console.print("\n🔧 Installing core dependencies...")

    # Core dependencies
    core_deps = [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "typer[all]>=0.9.0",
        "rich>=13.7.0",
        "structlog>=23.2.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.2.1",
        "httpx>=0.25.2",
        "tenacity>=8.2.3",
        "psutil>=5.9.0"
    ]

    success = True
    for dep in core_deps:
        if not run_command(["uv", "pip", "install", dep], f"Installing {dep}"):
            success = False

    return success


def install_security_dependencies() -> bool:
    """Install security-related dependencies."""
    console.print("\n🔒 Installing security dependencies...")

    security_deps = [
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "cryptography>=41.0.0",
        "pyjwt>=2.8.0"
    ]

    success = True
    for dep in security_deps:
        if not run_command(["uv", "pip", "install", dep], f"Installing {dep}"):
            success = False

    return success


def install_ai_dependencies() -> bool:
    """Install AI/ML dependencies."""
    console.print("\n🤖 Installing AI/ML dependencies...")

    ai_deps = [
        "langchain-anthropic>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-mcp-adapters>=0.1.0",
        "mcp>=1.0.0"
    ]

    success = True
    for dep in ai_deps:
        if not run_command(["uv", "pip", "install", dep], f"Installing {dep}"):
            success = False

    return success


def install_development_dependencies() -> bool:
    """Install development dependencies."""
    console.print("\n🛠️ Installing development dependencies...")

    dev_deps = [
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "ruff>=0.1.6",
        "mypy>=1.7.1",
        "pre-commit>=3.6.0"
    ]

    success = True
    for dep in dev_deps:
        if not run_command(["uv", "pip", "install", dep], f"Installing {dep}"):
            success = False

    return success


def verify_installation() -> bool:
    """Verify that key packages can be imported."""
    console.print("\n🔍 Verifying installation...")

    test_imports = [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("structlog", "Structured logging"),
        ("rich", "Rich console"),
        ("typer", "Typer CLI"),
        ("passlib", "Password hashing"),
        ("jwt", "JWT tokens")
    ]

    success = True
    for module, description in test_imports:
        try:
            __import__(module)
            console.print(f"[green]✅ {description}[/green]")
        except ImportError:
            console.print(f"[red]❌ {description} - import failed[/red]")
            success = False

    return success


def display_summary(success: bool) -> None:
    """Display installation summary."""
    if success:
        panel = Panel(
            "[green]🎉 Installation completed successfully![/green]\n\n"
            "You can now run the DataMCPServerAgent with:\n"
            "[cyan]python -m app.main_improved[/cyan]\n\n"
            "Or use the CLI:\n"
            "[cyan]datamcp --help[/cyan]",
            title="Installation Complete",
            border_style="green"
        )
    else:
        panel = Panel(
            "[red]❌ Installation failed![/red]\n\n"
            "Please check the errors above and try again.\n"
            "You may need to install dependencies manually.",
            title="Installation Failed",
            border_style="red"
        )

    console.print(panel)


def main(
    skip_dev: bool = typer.Option(False, "--skip-dev", help="Skip development dependencies"),
    skip_ai: bool = typer.Option(False, "--skip-ai", help="Skip AI/ML dependencies"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify installation")
) -> None:
    """Install DataMCPServerAgent dependencies using uv."""

    console.print(Panel(
        "[bold blue]DataMCPServerAgent Dependency Installer[/bold blue]\n"
        "Using uv for fast and reliable package management",
        title="🚀 Installation Starting"
    ))

    # Check prerequisites
    if not check_python_version():
        raise typer.Exit(1)

    if not check_uv_installed():
        console.print("📦 uv not found, installing...")
        if not install_uv():
            console.print("[red]❌ Failed to install uv[/red]")
            raise typer.Exit(1)
    else:
        console.print("[green]✅ uv is already installed[/green]")

    # Install dependencies
    success = True

    if not install_core_dependencies():
        success = False

    if not install_security_dependencies():
        success = False

    if not skip_ai and not install_ai_dependencies():
        success = False

    if not skip_dev and not install_development_dependencies():
        success = False

    # Verify installation
    if verify and success:
        success = verify_installation()

    # Display summary
    display_summary(success)

    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
