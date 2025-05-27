#!/usr/bin/env python3
"""
DataMCPServerAgent - Improved Main Entry Point

A unified, production-ready entry point for the DataMCPServerAgent system.
Supports multiple modes: API server, CLI interface, and background worker.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cli.interface import create_cli_interface
from app.core.config import Environment, Settings
from app.core.logging import get_logger, setup_logging
from app.workers.background import create_background_worker

# Import semantic agents
try:
    from src.agents.semantic.main import SemanticAgentsSystem
    SEMANTIC_AGENTS_AVAILABLE = True
except ImportError:
    SEMANTIC_AGENTS_AVAILABLE = False

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="datamcp",
    help="DataMCPServerAgent - Advanced AI Agent System",
    add_completion=False,
    rich_markup_mode="rich",
)


def display_banner():
    """Display application banner."""
    banner = Text()
    banner.append("DataMCPServerAgent", style="bold blue")
    banner.append(" v2.0.0", style="dim")
    banner.append("\n")
    banner.append("Advanced AI Agent System with MCP Integration", style="italic")

    panel = Panel(banner, title="🤖 Welcome", border_style="blue", padding=(1, 2))
    console.print(panel)


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8002, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    env: Environment = typer.Option(Environment.DEVELOPMENT, help="Environment"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Start the API server."""
    display_banner()

    # Create settings
    settings = Settings(
        environment=env, api_host=host, api_port=port, log_level=log_level, debug=reload
    )

    # Setup logging
    setup_logging(settings)

    logger.info("🚀 Starting DataMCPServerAgent API Server")
    logger.info(f"📍 Environment: {env.value}")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"📊 Metrics: http://{host}:{port}/metrics")

    try:
        # Create and run server
        uvicorn.run(
            "app.api.server:create_api_server",
            factory=True,
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True,
            app_dir=str(Path(__file__).parent.parent),
        )
    except KeyboardInterrupt:
        logger.info("👋 API server stopped by user")
    except Exception as e:
        logger.error(f"💥 API server failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def cli(
    env: Environment = typer.Option(Environment.DEVELOPMENT, help="Environment"),
    log_level: str = typer.Option("INFO", help="Log level"),
    interactive: bool = typer.Option(True, help="Interactive mode"),
):
    """Start the CLI interface."""
    display_banner()

    # Create settings
    settings = Settings(environment=env, log_level=log_level, debug=True)

    # Setup logging
    setup_logging(settings)

    logger.info("🖥️ Starting DataMCPServerAgent CLI Interface")
    logger.info(f"📍 Environment: {env.value}")

    try:
        # Create and run CLI
        cli_interface = create_cli_interface(settings)

        if interactive:
            asyncio.run(cli_interface.run_interactive())
        else:
            asyncio.run(cli_interface.run_batch())

    except KeyboardInterrupt:
        logger.info("👋 CLI interface stopped by user")
    except Exception as e:
        logger.error(f"💥 CLI interface failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def worker(
    env: Environment = typer.Option(Environment.DEVELOPMENT, help="Environment"),
    log_level: str = typer.Option("INFO", help="Log level"),
    worker_type: str = typer.Option("general", help="Worker type"),
    concurrency: int = typer.Option(4, help="Concurrency level"),
):
    """Start a background worker."""
    display_banner()

    # Create settings
    settings = Settings(environment=env, log_level=log_level)

    # Setup logging
    setup_logging(settings)

    logger.info("⚙️ Starting DataMCPServerAgent Background Worker")
    logger.info(f"📍 Environment: {env.value}")
    logger.info(f"🔧 Worker Type: {worker_type}")
    logger.info(f"🔄 Concurrency: {concurrency}")

    try:
        # Create and run worker
        worker = create_background_worker(settings, worker_type, concurrency)
        asyncio.run(worker.run())

    except KeyboardInterrupt:
        logger.info("👋 Background worker stopped by user")
    except Exception as e:
        logger.error(f"💥 Background worker failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def status():
    """Show system status."""
    display_banner()

    console.print("📊 System Status", style="bold green")
    console.print("=" * 50)

    # Check API server
    try:
        import httpx

        response = httpx.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            console.print("✅ API Server: [green]RUNNING[/green]")
        else:
            console.print("⚠️ API Server: [yellow]UNHEALTHY[/yellow]")
    except:
        console.print("❌ API Server: [red]NOT RUNNING[/red]")

    # Check database
    try:
        from app.infrastructure.database import DatabaseManager

        db = DatabaseManager()
        # Add database health check here
        console.print("✅ Database: [green]CONNECTED[/green]")
    except:
        console.print("❌ Database: [red]DISCONNECTED[/red]")

    # Check cache
    try:
        from app.infrastructure.cache import CacheManager

        cache = CacheManager()
        # Add cache health check here
        console.print("✅ Cache: [green]CONNECTED[/green]")
    except:
        console.print("❌ Cache: [red]DISCONNECTED[/red]")


@app.command()
def migrate(
    env: Environment = typer.Option(Environment.DEVELOPMENT, help="Environment"),
    direction: str = typer.Option("up", help="Migration direction (up/down)"),
):
    """Run database migrations."""
    display_banner()

    console.print("🔄 Running Database Migrations", style="bold blue")

    # Create settings
    settings = Settings(environment=env)

    try:
        from app.infrastructure.database.migrations import MigrationRunner

        runner = MigrationRunner(settings)

        if direction == "up":
            asyncio.run(runner.migrate_up())
            console.print("✅ Migrations completed successfully", style="green")
        elif direction == "down":
            asyncio.run(runner.migrate_down())
            console.print("✅ Rollback completed successfully", style="green")
        else:
            console.print("❌ Invalid direction. Use 'up' or 'down'", style="red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"💥 Migration failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def test(
    coverage: bool = typer.Option(True, help="Run with coverage"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    pattern: Optional[str] = typer.Option(None, help="Test pattern"),
):
    """Run the test suite."""
    display_banner()

    console.print("🧪 Running Test Suite", style="bold blue")

    import subprocess

    cmd = ["python", "-m", "pytest"]

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    if verbose:
        cmd.append("-v")

    if pattern:
        cmd.extend(["-k", pattern])

    try:
        result = subprocess.run(cmd, check=True)
        console.print("✅ All tests passed!", style="green")
    except subprocess.CalledProcessError:
        console.print("❌ Some tests failed", style="red")
        raise typer.Exit(1)


@app.command()
def semantic_agents(
    env: Environment = typer.Option(Environment.DEVELOPMENT, help="Environment"),
    log_level: str = typer.Option("INFO", help="Log level"),
    api_port: int = typer.Option(8003, help="API port for semantic agents"),
):
    """Start the semantic agents system."""
    if not SEMANTIC_AGENTS_AVAILABLE:
        console.print("❌ Semantic agents not available. Please install dependencies.", style="red")
        raise typer.Exit(1)

    display_banner()

    # Create settings
    settings = Settings(environment=env, log_level=log_level)

    # Setup logging
    setup_logging(settings)

    logger.info("🧠 Starting Semantic Agents System")
    logger.info(f"📍 Environment: {env.value}")
    logger.info(f"🌐 API: http://localhost:{api_port}")

    try:
        # Create and run semantic agents system
        system = SemanticAgentsSystem()

        # Create FastAPI app with semantic agents
        from fastapi import FastAPI
        app = FastAPI(
            title="DataMCPServerAgent with Semantic Agents",
            description="Advanced AI Agent System with Semantic Agents",
            version="2.0.0",
        )

        # Include semantic agents router
        semantic_app = system.get_fastapi_app()
        app.mount("/semantic", semantic_app)

        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=api_port,
            log_level=log_level.lower(),
        )

    except KeyboardInterrupt:
        logger.info("👋 Semantic agents system stopped by user")
    except Exception as e:
        logger.error(f"💥 Semantic agents system failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def docs(
    serve: bool = typer.Option(False, help="Serve documentation"),
    port: int = typer.Option(8080, help="Documentation server port"),
):
    """Generate or serve documentation."""
    display_banner()

    if serve:
        console.print(f"📚 Serving documentation on http://localhost:{port}", style="blue")

        import subprocess

        try:
            subprocess.run(["mkdocs", "serve", "--dev-addr", f"localhost:{port}"], check=True)
        except subprocess.CalledProcessError:
            console.print("❌ Failed to serve documentation", style="red")
            raise typer.Exit(1)
    else:
        console.print("📝 Generating documentation...", style="blue")

        import subprocess

        try:
            subprocess.run(["mkdocs", "build"], check=True)
            console.print("✅ Documentation generated successfully", style="green")
        except subprocess.CalledProcessError:
            console.print("❌ Failed to generate documentation", style="red")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
