"""
Enhanced main entry point for DataMCPServerAgent v2.0.
Provides a unified interface to launch different agent modes and services.
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "src"))

console = Console()

# Import configuration and logging
try:
    from app.core.config import Settings
    from app.core.logging import get_logger, setup_logging
    from app.main_improved import create_app

    # Load settings
    settings = Settings()
    setup_logging(settings)
    logger = get_logger(__name__)

except ImportError as e:
    console.print(f"[red]❌ Failed to import core modules: {e}[/red]")
    console.print("[yellow]⚠️ Falling back to basic mode[/yellow]")

    # Fallback imports
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    settings = None


def display_banner():
    """Display application banner."""
    banner = Panel(
        "[bold blue]🤖 DataMCPServerAgent v2.0[/bold blue]\n"
        "[italic]Enhanced AI Agent System with MCP Integration[/italic]\n\n"
        "[dim]Advanced architecture with Clean Code principles[/dim]",
        title="Welcome",
        border_style="blue"
    )
    console.print(banner)


def start_api_server(host: str, port: int, reload: bool, debug: bool):
    """Start the API server."""
    try:
        import uvicorn

        from app.main_improved import app

        console.print(f"[green]🚀 Starting API server on {host}:{port}[/green]")

        uvicorn.run(
            "app.main_improved:app",
            host=host,
            port=port,
            reload=reload,
            log_level="debug" if debug else "info"
        )

    except ImportError as e:
        console.print(f"[red]❌ Failed to start API server: {e}[/red]")
        console.print("[yellow]💡 Try: pip install uvicorn[/yellow]")
        sys.exit(1)


def start_cli_interface():
    """Start the CLI interface."""
    try:
        from app.cli.interface_improved import main as cli_main
        cli_main()
    except ImportError:
        console.print("[yellow]⚠️ CLI interface not available[/yellow]")
        console.print("[dim]Using basic mode...[/dim]")
        basic_chat_mode()


def basic_chat_mode():
    """Basic chat mode fallback."""
    console.print("[cyan]💬 Basic Chat Mode[/cyan]")
    console.print("[dim]Type 'exit' to quit[/dim]\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                console.print("[green]👋 Goodbye![/green]")
                break

            # Simple echo response
            console.print(f"Agent: I received your message: {user_input}")

        except KeyboardInterrupt:
            console.print("\n[green]👋 Goodbye![/green]")
            break


def show_status():
    """Show system status."""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # Check core components
    components = [
        ("Configuration", "app.core.config", "Settings management"),
        ("Logging", "app.core.logging", "Structured logging"),
        ("API Server", "app.main_improved", "FastAPI application"),
        ("CLI Interface", "app.cli.interface_improved", "Command line interface"),
    ]

    for name, module, description in components:
        try:
            __import__(module)
            table.add_row(name, "[green]✅ Available[/green]", description)
        except ImportError:
            table.add_row(name, "[red]❌ Missing[/red]", description)

    console.print(table)


def main():
    """Enhanced main entry point."""
    parser = argparse.ArgumentParser(
        description="DataMCPServerAgent v2.0 - Enhanced AI Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s api --port 8000 --reload    # Start API server with auto-reload
  %(prog)s cli                         # Start CLI interface
  %(prog)s status                      # Show system status
  %(prog)s chat                        # Start basic chat mode
        """
    )

    parser.add_argument(
        "mode",
        choices=["api", "cli", "chat", "status"],
        default="status",
        nargs="?",
        help="Operation mode to run"
    )

    # API server options
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind the API server (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind the API server (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("API_DEBUG", "false").lower() == "true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Display banner
    display_banner()

    # Route to appropriate mode
    if args.mode == "api":
        start_api_server(args.host, args.port, args.reload, args.debug)
    elif args.mode == "cli":
        start_cli_interface()
    elif args.mode == "chat":
        basic_chat_mode()
    elif args.mode == "status":
        show_status()
    else:
        console.print(f"[red]❌ Unknown mode: {args.mode}[/red]")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
