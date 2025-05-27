"""
Consolidated CLI Interface for DataMCPServerAgent.

Unified command-line interface that provides access to all
system functionality through a single, clean interface.
"""

import asyncio
import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from app.core.logging_improved import get_logger
from app.core.simple_config import SimpleSettings

logger = get_logger(__name__)
console = Console()


class ConsolidatedCLI:
    """Consolidated CLI interface for DataMCPServerAgent."""

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
                logger.error(f"CLI error: {e}", exc_info=True)

    async def run_batch(self) -> None:
        """Run batch processing from stdin."""
        console.print("📥 Reading from stdin (Consolidated Mode)...", style="dim")

        try:
            for line in sys.stdin:
                command = line.strip()
                if command:
                    console.print(f"▶️ Executing: {command}", style="dim")
                    await self._process_command(command)
        except KeyboardInterrupt:
            console.print("\n⏹️ Batch processing stopped", style="yellow")

    def _create_welcome_panel(self) -> Panel:
        """Create welcome panel for consolidated system."""
        welcome_text = f"""
[bold blue]{self.settings.app_name} - Consolidated[/bold blue] [dim]v{self.settings.app_version}[/dim]

Unified AI Agent System with Clean Architecture

[bold]Structure:[/bold] Single app/ directory
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
        """Process CLI command in consolidated system."""
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
            await self._show_status()
        elif cmd == "architecture":
            self._show_architecture()
        elif cmd == "structure":
            self._show_structure()
        elif cmd == "agents":
            await self._handle_agents_command(args)
        elif cmd == "tasks":
            await self._handle_tasks_command(args)
        elif cmd == "domain":
            self._show_domain_info()
        elif cmd == "application":
            self._show_application_info()
        elif cmd == "infrastructure":
            self._show_infrastructure_info()
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
        """Show help for consolidated system."""
        help_table = Table(title="Consolidated System Commands", show_header=True)
        help_table.add_column("Command", style="bold blue")
        help_table.add_column("Description", style="dim")
        help_table.add_column("Examples", style="green")

        commands = [
            ("help", "Show this help message", "help"),
            ("status", "Show consolidated system status", "status"),
            ("architecture", "Show architecture information", "architecture"),
            ("structure", "Show directory structure", "structure"),
            ("agents", "Manage agents", "agents list, agents create"),
            ("tasks", "Manage tasks", "tasks list, tasks create"),
            ("domain", "Show domain layer info", "domain"),
            ("application", "Show application layer info", "application"),
            ("infrastructure", "Show infrastructure layer info", "infrastructure"),
            ("api", "Show API layer info", "api"),
            ("history", "Show command history", "history"),
            ("clear", "Clear screen", "clear"),
            ("exit", "Exit CLI", "exit, quit, q"),
        ]

        for cmd, desc, examples in commands:
            help_table.add_row(cmd, desc, examples)

        console.print(help_table)

    async def _show_status(self) -> None:
        """Show consolidated system status."""
        status_tree = Tree("🖥️ Consolidated System Status")

        # System info
        system_node = status_tree.add("📊 System")
        system_node.add(f"Name: [bold]{self.settings.app_name} - Consolidated[/bold]")
        system_node.add(f"Version: [bold]{self.settings.app_version}[/bold]")
        system_node.add(f"Environment: [bold]{self.settings.environment}[/bold]")
        system_node.add(f"Debug: [bold]{self.settings.debug}[/bold]")

        # Architecture info
        arch_node = status_tree.add("🏗️ Architecture")
        arch_node.add("Pattern: [bold]Clean Architecture + DDD[/bold]")
        arch_node.add("Structure: [bold]Single app/ directory[/bold]")
        arch_node.add("Layers: [bold]Domain, Application, Infrastructure, API, CLI[/bold]")

        # Components
        components_node = status_tree.add("🔧 Components")
        components_node.add("Domain Layer: ✅ Available")
        components_node.add("Application Layer: ✅ Available")
        components_node.add("Infrastructure Layer: ✅ Available")
        components_node.add("API Layer: ✅ Available")
        components_node.add("CLI Layer: ✅ Active")

        console.print(status_tree)

    def _show_architecture(self) -> None:
        """Show architecture information."""
        arch_panel = """
[bold]Clean Architecture + Domain-Driven Design[/bold]

[bold blue]Layers:[/bold blue]
• [bold]Domain[/bold]: Business logic, models, services
• [bold]Application[/bold]: Use cases, commands, queries
• [bold]Infrastructure[/bold]: External dependencies
• [bold]API[/bold]: REST endpoints, schemas
• [bold]CLI[/bold]: Command-line interface

[bold blue]Principles:[/bold blue]
• Dependency Inversion
• Separation of Concerns
• Single Responsibility
• Domain-Driven Design

[bold blue]Benefits:[/bold blue]
• Maintainable codebase
• Testable components
• Clear boundaries
• Scalable architecture
        """

        panel = Panel(arch_panel.strip(), title="🏗️ Architecture Overview", border_style="blue")
        console.print(panel)

    def _show_structure(self) -> None:
        """Show consolidated directory structure."""
        structure_tree = Tree("📁 Consolidated Structure")

        app_node = structure_tree.add("[bold blue]app/[/bold blue] (Single source of truth)")

        # Core
        core_node = app_node.add("core/ - Core functionality")
        core_node.add("config.py - Unified configuration")
        core_node.add("logging.py - Structured logging")
        core_node.add("exceptions.py - Exception handling")

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
        api_node.add("v1/ - API version 1")
        api_node.add("schemas/ - API schemas")
        api_node.add("middleware/ - API middleware")

        # CLI
        cli_node = app_node.add("cli/ - Command line")
        cli_node.add("commands/ - CLI commands")
        cli_node.add("interface.py - Main interface")

        # Other directories
        structure_tree.add("tests/ - Test suite")
        structure_tree.add("docs/ - Documentation")
        structure_tree.add("examples/ - Usage examples")

        console.print(structure_tree)

    def _show_domain_info(self) -> None:
        """Show domain layer information."""
        domain_panel = """
[bold]Domain Layer - Business Logic[/bold]

[bold blue]Models:[/bold blue]
• Agent - AI agent entities
• Task - Work units with lifecycle
• User - System users
• Memory - Knowledge storage

[bold blue]Services:[/bold blue]
• AgentService - Agent operations
• TaskService - Task management
• MemoryService - Knowledge management

[bold blue]Events:[/bold blue]
• AgentCreated, AgentStatusChanged
• TaskStarted, TaskCompleted
• MemoryUpdated, KnowledgeAdded

[bold blue]Principles:[/bold blue]
• Pure business logic
• Framework independent
• Rich domain models
• Event-driven communication
        """

        panel = Panel(domain_panel.strip(), title="🧠 Domain Layer", border_style="green")
        console.print(panel)

    def _show_application_info(self) -> None:
        """Show application layer information."""
        app_panel = """
[bold]Application Layer - Use Cases[/bold]

[bold blue]Commands (CQRS):[/bold blue]
• CreateAgent, UpdateAgent, DeleteAgent
• CreateTask, UpdateTaskStatus
• AddMemory, UpdateKnowledge

[bold blue]Queries (CQRS):[/bold blue]
• GetAgent, ListAgents
• GetTask, ListTasks
• SearchMemory, GetKnowledge

[bold blue]Use Cases:[/bold blue]
• Agent lifecycle management
• Task orchestration
• Memory operations
• System coordination

[bold blue]Handlers:[/bold blue]
• Event handlers for domain events
• Command validation and execution
• Query optimization and caching
        """

        panel = Panel(app_panel.strip(), title="⚙️ Application Layer", border_style="yellow")
        console.print(panel)

    def _show_infrastructure_info(self) -> None:
        """Show infrastructure layer information."""
        infra_panel = """
[bold]Infrastructure Layer - External Concerns[/bold]

[bold blue]Database:[/bold blue]
• PostgreSQL for production
• SQLite for development
• Async ORM with SQLAlchemy

[bold blue]Cache:[/bold blue]
• Redis for high-performance caching
• In-memory cache for development
• Distributed caching support

[bold blue]Messaging:[/bold blue]
• RabbitMQ for reliable messaging
• Redis pub/sub for real-time events
• Event sourcing capabilities

[bold blue]External APIs:[/bold blue]
• Cloudflare integrations
• Email service providers
• Third-party AI services
        """

        panel = Panel(infra_panel.strip(), title="🔧 Infrastructure Layer", border_style="red")
        console.print(panel)

    def _show_api_info(self) -> None:
        """Show API layer information."""
        api_panel = """
[bold]API Layer - REST Endpoints[/bold]

[bold blue]Endpoints:[/bold blue]
• GET /api/v1/agents - List agents
• POST /api/v1/agents - Create agent
• GET /api/v1/tasks - List tasks
• POST /api/v1/tasks - Create task

[bold blue]Features:[/bold blue]
• OpenAPI documentation
• Request/response validation
• Error handling
• CORS support
• Rate limiting

[bold blue]Documentation:[/bold blue]
• Swagger UI at /docs
• ReDoc at /redoc
• OpenAPI spec at /openapi.json

[bold blue]Testing:[/bold blue]
• Interactive API testing
• Automated test suite
• Performance benchmarks
        """

        panel = Panel(api_panel.strip(), title="🌐 API Layer", border_style="blue")
        console.print(panel)

    async def _handle_agents_command(self, args: List[str]) -> None:
        """Handle agents command in consolidated system."""
        if not args:
            args = ["list"]

        subcommand = args[0].lower()

        if subcommand == "list":
            console.print("🤖 Agents in Consolidated System:")
            console.print("  • Use API: GET /api/v1/agents")
            console.print("  • Or web interface: http://localhost:8002/docs")
        elif subcommand == "create":
            console.print("🔧 Create Agent in Consolidated System:")
            console.print("  • Use API: POST /api/v1/agents")
            console.print("  • Or web interface: http://localhost:8002/docs")
        else:
            console.print("❓ Usage: agents [list|create]", style="yellow")

    async def _handle_tasks_command(self, args: List[str]) -> None:
        """Handle tasks command in consolidated system."""
        if not args:
            args = ["list"]

        subcommand = args[0].lower()

        if subcommand == "list":
            console.print("📋 Tasks in Consolidated System:")
            console.print("  • Use API: GET /api/v1/tasks")
            console.print("  • Or web interface: http://localhost:8002/docs")
        elif subcommand == "create":
            console.print("🔧 Create Task in Consolidated System:")
            console.print("  • Use API: POST /api/v1/tasks")
            console.print("  • Or web interface: http://localhost:8002/docs")
        else:
            console.print("❓ Usage: tasks [list|create]", style="yellow")

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
