#!/usr/bin/env python3
"""
Simple Consolidated Main Entry Point for DataMCPServerAgent.

Simplified version without complex imports for initial testing.
"""

import asyncio
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.simple_config import SimpleSettings

# Initialize console
console = Console()

# Create Typer app
app = typer.Typer(
    name="datamcp-consolidated",
    help="DataMCPServerAgent - Simple Consolidated System",
    add_completion=False,
    rich_markup_mode="rich",
)

def display_banner():
    """Display application banner."""
    banner = Text()
    banner.append("DataMCPServerAgent", style="bold blue")
    banner.append(" - Consolidated v2.0.0", style="dim")
    banner.append("\n")
    banner.append("Single app/ Structure with Clean Architecture", style="italic")

    panel = Panel(banner, title="🤖 Consolidated System", border_style="blue", padding=(1, 2))
    console.print(panel)

@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8003, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the consolidated API server."""
    display_banner()

    settings = SimpleSettings()

    console.print("🚀 Starting Consolidated DataMCPServerAgent API", style="bold green")
    console.print(f"📍 Host: {host}:{port}")
    console.print(f"🔄 Reload: {reload}")
    console.print("🏗️ Structure: Single app/ directory")
    console.print("📐 Architecture: Clean Architecture + DDD")

    try:
        uvicorn.run(
            "app.api.simple_consolidated_server:create_simple_consolidated_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("👋 API server stopped by user", style="yellow")
    except Exception as e:
        console.print(f"💥 API server failed: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def cli():
    """Start the consolidated CLI interface."""
    display_banner()

    console.print("🖥️ Starting Consolidated CLI Interface", style="bold green")

    try:
        from app.cli.simple_consolidated_interface import SimpleConsolidatedCLI

        settings = SimpleSettings()
        cli_interface = SimpleConsolidatedCLI(settings)
        asyncio.run(cli_interface.run_interactive())

    except KeyboardInterrupt:
        console.print("👋 CLI interface stopped by user", style="yellow")
    except Exception as e:
        console.print(f"💥 CLI interface failed: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def status():
    """Show consolidated system status."""
    display_banner()

    console.print("📊 Consolidated System Status", style="bold green")
    console.print("=" * 50)

    settings = SimpleSettings()

    # System info
    console.print(f"📱 App: {settings.app_name}")
    console.print(f"📦 Version: {settings.app_version}")
    console.print(f"🌍 Environment: {settings.environment}")
    console.print(f"🔧 Debug: {settings.debug}")
    console.print("🏗️ Structure: Single app/ directory")
    console.print("📐 Architecture: Clean Architecture + DDD")

    # Check API server
    try:
        import httpx

        response = httpx.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            console.print("🌐 API Server: ✅ RUNNING (port 8003)")
        else:
            console.print("🌐 API Server: ⚠️ UNHEALTHY")
    except:
        console.print("🌐 API Server: ❌ NOT RUNNING")
        console.print("💡 Start with: python app/main_simple_consolidated.py api")

@app.command()
def info():
    """Show consolidated system information."""
    display_banner()

    settings = SimpleSettings()

    info_text = f"""
[bold]Consolidated DataMCPServerAgent v2.0[/bold]

[bold blue]Structure:[/bold blue]
• Single app/ directory (consolidated)
• Clean separation of concerns
• Domain-driven design principles

[bold blue]Architecture Layers:[/bold blue]
• [bold]Domain[/bold]: Business logic and models
• [bold]Application[/bold]: Use cases and orchestration
• [bold]Infrastructure[/bold]: External dependencies
• [bold]API[/bold]: REST endpoints and schemas
• [bold]CLI[/bold]: Command-line interface

[bold blue]Benefits:[/bold blue]
• Maintainable codebase
• Clear import paths
• Reduced complexity
• Better organization

[bold blue]Configuration:[/bold blue]
• App: {settings.app_name}
• Version: {settings.app_version}
• Environment: {settings.environment}
• Debug: {settings.debug}

[bold blue]Commands:[/bold blue]
• api    - Start API server
• cli    - Start CLI interface
• status - Show system status
• info   - Show this information
    """

    panel = Panel(info_text.strip(), title="📋 Consolidated System Info", border_style="green")
    console.print(panel)

@app.command()
def structure():
    """Show consolidated directory structure."""
    display_banner()

    console.print("📁 Consolidated Directory Structure", style="bold blue")
    console.print("=" * 50)

    structure_text = """
app/                           # Single source of truth
├── __init__.py
├── main_simple_consolidated.py # This entry point
│
├── core/                      # Core functionality
│   ├── simple_config.py       # Configuration
│   ├── logging_improved.py    # Logging
│   └── exceptions_improved.py # Exceptions
│
├── domain/                    # Business logic
│   ├── models/                # Domain models
│   ├── services/              # Domain services
│   └── events/                # Domain events
│
├── application/               # Use cases
│   ├── commands/              # Command handlers
│   ├── queries/               # Query handlers
│   └── use_cases/             # Business use cases
│
├── infrastructure/            # External concerns
│   ├── database/              # Data persistence
│   ├── cache/                 # Caching
│   └── messaging/             # Message queues
│
├── api/                       # REST API
│   ├── consolidated_server.py # Main API server
│   └── v1/                    # API endpoints
│
└── cli/                       # Command line
    └── consolidated_interface.py # CLI interface

Benefits:
✅ Single app/ directory
✅ Clear separation of concerns
✅ Clean import paths
✅ Domain-driven design
✅ Maintainable structure
    """

    console.print(structure_text)

@app.command()
def test():
    """Test the consolidated system."""
    display_banner()

    console.print("🧪 Testing Consolidated System", style="bold blue")

    # Test configuration
    try:
        settings = SimpleSettings()
        console.print(f"✅ Configuration: {settings.app_name} v{settings.app_version}")
    except Exception as e:
        console.print(f"❌ Configuration failed: {e}")
        return

    # Test API server creation
    try:
        from app.api.simple_consolidated_server import create_simple_consolidated_app

        app_instance = create_simple_consolidated_app()
        console.print(f"✅ API Server: {app_instance.title}")
    except Exception as e:
        console.print(f"❌ API Server failed: {e}")
        return

    # Test CLI interface
    try:
        from app.cli.simple_consolidated_interface import SimpleConsolidatedCLI

        cli = SimpleConsolidatedCLI(settings)
        console.print("✅ CLI Interface: Available")
    except Exception as e:
        console.print(f"❌ CLI Interface failed: {e}")
        return

    console.print("\n🎉 All tests passed! Consolidated system is ready!", style="bold green")
    console.print("\n📋 Next steps:")
    console.print("1. Start API: python app/main_simple_consolidated.py api")
    console.print("2. Start CLI: python app/main_simple_consolidated.py cli")
    console.print("3. Check docs: http://localhost:8003/docs")

if __name__ == "__main__":
    app()
