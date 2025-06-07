#!/usr/bin/env python3
"""
Phase 1 Setup Script for DataMCPServerAgent

This script sets up the development environment for Phase 1:
- Installs dependencies with uv
- Sets up pre-commit hooks
- Runs code quality checks
- Validates configuration
- Creates necessary directories
"""

import subprocess
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command with progress indication."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=None)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            progress.update(task, completed=True)

        console.print(f"✅ {description}", style="green")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"❌ {description}", style="red")
        console.print(f"Error: {e.stderr}", style="red")
        return False

def check_prerequisites() -> bool:
    """Check if required tools are installed."""
    console.print("🔍 Checking Prerequisites", style="bold blue")

    tools = [
        ("python", ["python", "--version"]),
        ("uv", ["uv", "--version"]),
        ("node", ["node", "--version"]),
        ("npm", ["npm", "--version"]),
    ]

    all_good = True

    for tool_name, cmd in tools:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            console.print(f"✅ {tool_name}: {version}", style="green")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print(f"❌ {tool_name}: Not found", style="red")
            all_good = False

    return all_good

def install_dependencies() -> bool:
    """Install Python dependencies with uv."""
    console.print("\n📦 Installing Dependencies", style="bold blue")

    commands = [
        (["uv", "pip", "install", "-e", ".[dev]"], "Installing development dependencies"),
        (["uv", "pip", "install", "pre-commit"], "Installing pre-commit"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False

    return True

def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    console.print("\n🪝 Setting up Pre-commit Hooks", style="bold blue")

    commands = [
        (["pre-commit", "install"], "Installing pre-commit hooks"),
        (["pre-commit", "install", "--hook-type", "commit-msg"], "Installing commit-msg hooks"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False

    return True

def run_code_quality_checks() -> bool:
    """Run initial code quality checks."""
    console.print("\n🔍 Running Code Quality Checks", style="bold blue")

    commands = [
        (["black", "--check", "app/", "src/"], "Checking code formatting with Black"),
        (["isort", "--check-only", "app/", "src/"], "Checking import sorting with isort"),
        (["ruff", "check", "app/", "src/"], "Running Ruff linter"),
        (["mypy", "app/"], "Running MyPy type checking"),
        (["pylint", "app/core/"], "Running Pylint on core modules"),
    ]

    results = []
    for cmd, desc in commands:
        success = run_command(cmd, desc)
        results.append((desc, success))

    # Show summary
    table = Table(title="Code Quality Check Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")

    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        style = "green" if success else "red"
        table.add_row(desc, status)

    console.print(table)

    return all(success for _, success in results)

def create_directories() -> bool:
    """Create necessary directories."""
    console.print("\n📁 Creating Directories", style="bold blue")

    directories = [
        "data",
        "temp",
        "logs",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs/api",
        "docs/architecture",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        console.print(f"✅ Created {dir_path}", style="green")

    return True

def validate_configuration() -> bool:
    """Validate project configuration."""
    console.print("\n⚙️ Validating Configuration", style="bold blue")

    config_files = [
        "pyproject.toml",
        ".pre-commit-config.yaml",
        "app/core/config.py",
        "app/core/logging.py",
    ]

    all_valid = True

    for config_file in config_files:
        if Path(config_file).exists():
            console.print(f"✅ {config_file} exists", style="green")
        else:
            console.print(f"❌ {config_file} missing", style="red")
            all_valid = False

    return all_valid

def show_summary() -> None:
    """Show setup summary."""
    summary_text = """
🎉 Phase 1 Setup Complete!

✅ Dependencies installed with uv
✅ Pre-commit hooks configured
✅ Code quality tools ready
✅ Project structure validated
✅ Configuration files in place

Next Steps:
1. Run: python app/main_improved.py --help
2. Start API: python app/main_improved.py api
3. Start CLI: python app/main_improved.py cli
4. Run tests: python app/main_improved.py test

For semantic agents:
python app/main_improved.py semantic-agents
    """

    panel = Panel(
        summary_text.strip(),
        title="🚀 Setup Complete",
        border_style="green",
        padding=(1, 2)
    )

    console.print(panel)

def main():
    """Main setup function."""
    console.print("🚀 DataMCPServerAgent Phase 1 Setup", style="bold magenta")
    console.print("=" * 50)

    # Check prerequisites
    if not check_prerequisites():
        console.print("\n❌ Prerequisites check failed. Please install missing tools.", style="red")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        console.print("\n❌ Dependency installation failed.", style="red")
        sys.exit(1)

    # Setup pre-commit
    if not setup_pre_commit():
        console.print("\n❌ Pre-commit setup failed.", style="red")
        sys.exit(1)

    # Create directories
    if not create_directories():
        console.print("\n❌ Directory creation failed.", style="red")
        sys.exit(1)

    # Validate configuration
    if not validate_configuration():
        console.print("\n❌ Configuration validation failed.", style="red")
        sys.exit(1)

    # Run code quality checks (optional - don't fail on this)
    console.print("\n🔍 Running optional code quality checks...")
    run_code_quality_checks()

    # Show summary
    show_summary()

if __name__ == "__main__":
    main()
