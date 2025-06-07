"""
Simple Consolidated CLI Interface for DataMCPServerAgent.

Simplified version without complex logging to avoid recursion issues.
"""

import asyncio
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from app.core.simple_config import SimpleSettings

console = Console()

class SimpleConsolidatedCLI:
    """Simple consolidated CLI interface."""

    def __init__(self, settings: SimpleSettings):
        self.settings = settings
        self.session_history: List[Dict[str, Any]] = []

    async def run_interactive(self) -> None:
        """Run interactive CLI session."""
        console.print(self._create_welcome_panel())

        while True:
            try:
                command = Prompt.ask("[bold blue]datamcp-consolidated[/bold blue]", default="help")

                if command.lower() in ["exit", "quit", "q"]:
                    console.print("👋 Goodbye from Consolidated System!", style="bold green")
                    break

                await self._process_command(command)

            except KeyboardInterrupt:
                console.print("\n👋 Goodbye from Consolidated System!", style="bold green")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="bold red")

    def _create_welcome_panel(self) -> Panel:
        """Create welcome panel."""
        welcome_text = f"""
[bold blue]{self.settings.app_name} - Consolidated[/bold blue] [dim]v{self.settings.app_version}[/dim]

Single app/ Structure with Clean Architecture

[bold]Structure:[/bold] Unified app/ directory
[bold]Architecture:[/bold] Clean Architecture + DDD
[bold]Environment:[/bold] {self.settings.environment.value}

[dim]Type 'help' for available commands[/dim]
        """

        return Panel(
            welcome_text.strip(),
            title="🤖 Consolidated System",
            border_style="blue",
            padding=(1, 2),
        )

    async def _process_command(self, command: str) -> None:
        """Process CLI command."""
        parts = command.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        # Record command
        self.session_history.append(
            {"command": command, "timestamp": asyncio.get_event_loop().time()}
        )

        # Route commands
        if cmd == "help":
            self._show_help()
        elif cmd == "status":
            self._show_status()
        elif cmd == "structure":
            self._show_structure()
        elif cmd == "architecture":
            self._show_architecture()
        elif cmd == "agents":
            self._show_agents_info()
        elif cmd == "tasks":
            self._show_tasks_info()
        elif cmd == "api":
            self._show_api_info()
        elif cmd == "history":
            self._show_history()
        elif cmd == "clear":
            console.clear()
        else:
            console.print(f"❓ Unknown command: {cmd}", style="yellow")
            console.print("💡 Type 'help' for available commands", style="dim")

    def _show_help(self) -> None:
        """Show help."""
        help_table = Table(title="Consolidated System Commands", show_header=True)
        help_table.add_column("Command", style="bold blue")
        help_table.add_column("Description", style="dim")

        commands = [
            ("help", "Show this help message"),
            ("status", "Show system status"),
            ("structure", "Show directory structure"),
            ("architecture", "Show architecture info"),
            ("agents", "Show agents information"),
            ("tasks", "Show tasks information"),
            ("api", "Show API information"),
            ("history", "Show command history"),
            ("clear", "Clear screen"),
            ("exit", "Exit CLI (or quit, q)"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)

    def _show_status(self) -> None:
        """Show system status."""
        status_tree = Tree("🖥️ Consolidated System Status")

        # System info
        system_node = status_tree.add("📊 System")
        system_node.add(f"Name: [bold]{self.settings.app_name} - Consolidated[/bold]")
        system_node.add(f"Version: [bold]{self.settings.app_version}[/bold]")
        system_node.add(f"Environment: [bold]{self.settings.environment}[/bold]")

        # Structure info
        structure_node = status_tree.add("🏗️ Structure")
        structure_node.add("Pattern: [bold]Single app/ directory[/bold]")
        structure_node.add("Architecture: [bold]Clean Architecture + DDD[/bold]")
        structure_node.add("Organization: [bold]Domain-driven[/bold]")

        # Components
        components_node = status_tree.add("🔧 Components")
        components_node.add("Domain Layer: ✅ Available")
        components_node.add("Application Layer: ✅ Available")
        components_node.add("Infrastructure Layer: ✅ Available")
        components_node.add("API Layer: ✅ Available")
        components_node.add("CLI Layer: ✅ Active")

        console.print(status_tree)

    def _show_structure(self) -> None:
        """Show directory structure."""
        structure_tree = Tree("📁 Consolidated Directory Structure")

        app_node = structure_tree.add("[bold blue]app/[/bold blue] (Single source of truth)")

        # Core
        core_node = app_node.add("core/ - Core functionality")
        core_node.add("simple_config.py - Configuration")
        core_node.add("logging_improved.py - Logging")
        core_node.add("exceptions_improved.py - Exceptions")

        # Domain
        domain_node = app_node.add("domain/ - Business logic")
        domain_node.add("models/ - Domain models")
        domain_node.add("services/ - Domain services")
        domain_node.add("events/ - Domain events")

        # Application
        app_layer_node = app_node.add("application/ - Use cases")
        app_layer_node.add("commands/ - Command handlers")
        app_layer_node.add("queries/ - Query handlers")
        app_layer_node.add("use_cases/ - Business use cases")

        # Infrastructure
        infra_node = app_node.add("infrastructure/ - External concerns")
        infra_node.add("database/ - Data persistence")
        infra_node.add("cache/ - Caching layer")
        infra_node.add("messaging/ - Message queues")

        # API
        api_node = app_node.add("api/ - REST API")
        api_node.add("simple_consolidated_server.py - Main server")
        api_node.add("v1/ - API endpoints")

        # CLI
        cli_node = app_node.add("cli/ - Command line")
        cli_node.add("simple_consolidated_interface.py - This interface")

        console.print(structure_tree)

    def _show_architecture(self) -> None:
        """Show architecture information."""
        arch_panel = """
[bold]Clean Architecture + Domain-Driven Design[/bold]

[bold blue]Consolidation Benefits:[/bold blue]
• Single app/ directory structure
• Clear import paths (app.*)
• Reduced complexity
• Better organization
• Unified codebase

[bold blue]Architecture Layers:[/bold blue]
• [bold]Domain[/bold]: Business logic, models, services
• [bold]Application[/bold]: Use cases, commands, queries
• [bold]Infrastructure[/bold]: External dependencies
• [bold]API[/bold]: REST endpoints, schemas
• [bold]CLI[/bold]: Command-line interface

[bold blue]Design Principles:[/bold blue]
• Dependency Inversion
• Separation of Concerns
• Single Responsibility
• Domain-Driven Design
• Clean Code practices
        """

        panel = Panel(arch_panel.strip(), title="🏗️ Consolidated Architecture", border_style="blue")
        console.print(panel)

    def _show_agents_info(self) -> None:
        """Show agents information."""
        agents_panel = """
[bold]Agent Management in Consolidated System[/bold]

[bold blue]Available Operations:[/bold blue]
• List agents: GET /api/v1/agents
• Create agent: POST /api/v1/agents
• Get agent: GET /api/v1/agents/{id}
• Delete agent: DELETE /api/v1/agents/{id}

[bold blue]Agent Types:[/bold blue]
• Worker - General purpose agents
• Coordinator - Task coordination
• Specialist - Domain-specific tasks

[bold blue]Web Interface:[/bold blue]
• API Docs: http://localhost:8003/docs
• Interactive testing available
• Real-time agent management
        """

        panel = Panel(agents_panel.strip(), title="🤖 Agents Information", border_style="green")
        console.print(panel)

    def _show_tasks_info(self) -> None:
        """Show tasks information."""
        tasks_panel = """
[bold]Task Management in Consolidated System[/bold]

[bold blue]Available Operations:[/bold blue]
• List tasks: GET /api/v1/tasks
• Create task: POST /api/v1/tasks
• Get task: GET /api/v1/tasks/{id}
• Update status: PUT /api/v1/tasks/{id}/status

[bold blue]Task Statuses:[/bold blue]
• Pending - Waiting to start
• Running - Currently executing
• Completed - Successfully finished
• Failed - Execution failed

[bold blue]Web Interface:[/bold blue]
• API Docs: http://localhost:8003/docs
• Interactive task management
• Real-time status updates
        """

        panel = Panel(tasks_panel.strip(), title="📋 Tasks Information", border_style="yellow")
        console.print(panel)

    def _show_api_info(self) -> None:
        """Show API information."""
        api_panel = """
[bold]API Information for Consolidated System[/bold]

[bold blue]Base URL:[/bold blue]
• http://localhost:8003

[bold blue]Documentation:[/bold blue]
• Swagger UI: /docs
• ReDoc: /redoc
• OpenAPI spec: /openapi.json

[bold blue]Main Endpoints:[/bold blue]
• GET / - System information
• GET /health - Health check
• GET /api/v1/status - System status
• GET /api/v1/architecture - Architecture info

[bold blue]Features:[/bold blue]
• OpenAPI documentation
• CORS enabled
• Request/response validation
• Error handling
        """

        panel = Panel(api_panel.strip(), title="🌐 API Information", border_style="blue")
        console.print(panel)

    def _show_history(self) -> None:
        """Show command history."""
        if not self.session_history:
            console.print("📭 No command history", style="dim")
            return

        history_table = Table(title="📜 Command History", show_header=True)
        history_table.add_column("#", style="dim")
        history_table.add_column("Command", style="bold")
        history_table.add_column("Time", style="dim")

        for i, entry in enumerate(self.session_history[-10:], 1):
            history_table.add_row(str(i), entry["command"], f"{entry['timestamp']:.2f}s")

        console.print(history_table)
