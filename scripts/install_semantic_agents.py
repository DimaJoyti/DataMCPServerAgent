#!/usr/bin/env python3
"""
Installation script for Semantic Agents dependencies.

This script installs all required dependencies for the semantic agents system,
including AI/ML libraries, performance monitoring tools, and development dependencies.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

console = Console()

def run_command(
    command: List[str],
    description: str,
    check: bool = True,
    capture_output: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run a command with progress indication."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=True,
            )

            progress.update(task, description=f"✅ {description}")
            return result

        except subprocess.CalledProcessError as e:
            progress.update(task, description=f"❌ {description}")
            console.print(f"[red]Error running command: {' '.join(command)}[/red]")
            console.print(f"[red]Error: {e}[/red]")
            if capture_output and e.stdout:
                console.print(f"[yellow]stdout: {e.stdout}[/yellow]")
            if capture_output and e.stderr:
                console.print(f"[red]stderr: {e.stderr}[/red]")
            raise

def check_python_version() -> None:
    """Check if Python version is compatible."""
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        console.print(
            "[red]❌ Python 3.9 or higher is required for semantic agents.[/red]"
        )
        console.print(f"[yellow]Current version: {version.major}.{version.minor}.{version.micro}[/yellow]")
        raise typer.Exit(1)

    console.print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(f"✅ uv version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_uv() -> None:
    """Install uv package manager."""
    console.print("📦 Installing uv package manager...")

    try:
        # Try installing via pip first
        run_command(
            [sys.executable, "-m", "pip", "install", "uv"],
            "Installing uv via pip"
        )
    except subprocess.CalledProcessError:
        # Fallback to curl installation
        console.print("[yellow]pip installation failed, trying curl...[/yellow]")
        run_command(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"],
            "Installing uv via curl",
            check=False,
        )

def install_core_dependencies() -> None:
    """Install core dependencies for semantic agents."""
    console.print("🔧 Installing core dependencies...")

    # Core AI/ML dependencies
    core_deps = [
        "langchain-anthropic>=0.1.0",
        "langchain-core>=0.1.0",
        "psutil>=5.9.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
        "shortuuid>=1.0.11",
        "tenacity>=8.2.3",
    ]

    for dep in core_deps:
        run_command(
            ["uv", "pip", "install", dep],
            f"Installing {dep.split('>=')[0]}"
        )

def install_development_dependencies() -> None:
    """Install development dependencies."""
    console.print("🛠️ Installing development dependencies...")

    dev_deps = [
        "pylint>=3.0.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "ruff>=0.1.6",
        "mypy>=1.7.1",
        "pre-commit>=3.6.0",
        "bandit>=1.7.5",
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
    ]

    for dep in dev_deps:
        run_command(
            ["uv", "pip", "install", dep],
            f"Installing {dep.split('>=')[0]}"
        )

def install_optional_dependencies() -> None:
    """Install optional dependencies for enhanced functionality."""
    console.print("🎯 Installing optional dependencies...")

    optional_deps = [
        "httpx>=0.25.2",
        "aiohttp>=3.9.1",
        "aiofiles>=23.2.1",
        "structlog>=23.2.0",
        "rich>=13.7.0",
    ]

    for dep in optional_deps:
        try:
            run_command(
                ["uv", "pip", "install", dep],
                f"Installing {dep.split('>=')[0]}",
                check=False,
            )
        except subprocess.CalledProcessError:
            console.print(f"[yellow]⚠️ Failed to install {dep}, skipping...[/yellow]")

def setup_pre_commit_hooks() -> None:
    """Setup pre-commit hooks."""
    console.print("🪝 Setting up pre-commit hooks...")

    project_root = Path(__file__).parent.parent

    try:
        run_command(
            ["pre-commit", "install"],
            "Installing pre-commit hooks",
            check=False,
        )

        run_command(
            ["pre-commit", "install", "--hook-type", "commit-msg"],
            "Installing commit-msg hooks",
            check=False,
        )

    except subprocess.CalledProcessError:
        console.print("[yellow]⚠️ Failed to setup pre-commit hooks[/yellow]")

def verify_installation() -> None:
    """Verify that semantic agents can be imported."""
    console.print("🔍 Verifying installation...")

    try:
        # Test core imports
        import_tests = [
            "import langchain_anthropic",
            "import langchain_core",
            "import psutil",
            "from src.agents.semantic.base_semantic_agent import BaseSemanticAgent",
            "from src.agents.semantic.coordinator import SemanticCoordinator",
        ]

        for test in import_tests:
            try:
                exec(test)
                console.print(f"✅ {test}")
            except ImportError as e:
                console.print(f"❌ {test} - {e}")

    except Exception as e:
        console.print(f"[red]❌ Verification failed: {e}[/red]")
        raise

def display_banner() -> None:
    """Display installation banner."""
    banner = Text()
    banner.append("Semantic Agents Installation", style="bold blue")
    banner.append("\n")
    banner.append("Installing dependencies for advanced AI agent system", style="italic")

    panel = Panel(banner, title="🧠 Setup", border_style="blue", padding=(1, 2))
    console.print(panel)

def display_completion_message() -> None:
    """Display completion message with next steps."""
    completion_text = Text()
    completion_text.append("✅ Installation completed successfully!\n\n", style="bold green")
    completion_text.append("Next steps:\n", style="bold")
    completion_text.append("1. Set your ANTHROPIC_API_KEY environment variable\n")
    completion_text.append("2. Run: python app/main_improved.py semantic-agents\n")
    completion_text.append("3. Visit: http://localhost:8003/semantic-agents/docs\n")
    completion_text.append("4. Check the documentation: docs/SEMANTIC_AGENTS_GUIDE.md\n")

    panel = Panel(completion_text, title="🎉 Success", border_style="green", padding=(1, 2))
    console.print(panel)

def main(
    skip_dev: bool = typer.Option(False, help="Skip development dependencies"),
    skip_optional: bool = typer.Option(False, help="Skip optional dependencies"),
    skip_hooks: bool = typer.Option(False, help="Skip pre-commit hooks setup"),
    verify: bool = typer.Option(True, help="Verify installation"),
) -> None:
    """Install semantic agents dependencies."""

    display_banner()

    try:
        # Check Python version
        check_python_version()

        # Check/install uv
        if not check_uv_installed():
            install_uv()

        # Install dependencies
        install_core_dependencies()

        if not skip_dev:
            install_development_dependencies()

        if not skip_optional:
            install_optional_dependencies()

        if not skip_hooks:
            setup_pre_commit_hooks()

        # Verify installation
        if verify:
            verify_installation()

        display_completion_message()

    except Exception as e:
        console.print(f"[red]💥 Installation failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
