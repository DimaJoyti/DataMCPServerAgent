"""
Improved CLI Interface for DataMCPServerAgent.

This module provides a rich, interactive command-line interface with:
- Beautiful output with Rich
- Interactive prompts
- Command history
- Auto-completion
- Progress indicators
- Error handling
"""

import asyncio
import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from app.core.config import Settings
from app.core.logging import get_logger


# Temporary mock managers until they are implemented
class MockAgentManager:
    def __init__(self, settings): pass
    async def list_agents(self): return []
    async def create_agent(self, **kwargs): return type('Agent', (), {'name': kwargs['name'], 'id': 'mock-id'})()
    async def delete_agent(self, agent_id): pass
    async def get_agent(self, agent_id): return None

class MockTaskManager:
    def __init__(self, settings): pass
    async def list_tasks(self): return []

class MockToolManager:
    def __init__(self, settings): pass
    async def list_tools(self): return []

logger = get_logger(__name__)
console = Console()


class CLIInterface:
    """Interactive CLI interface for DataMCPServerAgent."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent_manager = MockAgentManager(settings)
        self.task_manager = MockTaskManager(settings)
        self.tool_manager = MockToolManager(settings)
        self.session_history: List[Dict[str, Any]] = []

    async def run_interactive(self) -> None:
        """Run interactive CLI session."""
        console.print(self._create_welcome_panel())

        while True:
            try:
                # Get user input
                command = Prompt.ask("[bold blue]datamcp[/bold blue]", default="help")

                if command.lower() in ["exit", "quit", "q"]:
                    console.print("👋 Goodbye!", style="bold green")
                    break

                # Process command
                await self._process_command(command)

            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="bold green")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="bold red")
                logger.error(f"CLI error: {e}", exc_info=True)

    async def run_batch(self) -> None:
        """Run batch processing from stdin."""
        console.print("📥 Reading from stdin...", style="dim")

        try:
            for line in sys.stdin:
                command = line.strip()
                if command:
                    console.print(f"▶️ Executing: {command}", style="dim")
                    await self._process_command(command)
        except KeyboardInterrupt:
            console.print("\n⏹️ Batch processing stopped", style="yellow")

    async def _process_command(self, command: str) -> None:
        """Process a CLI command."""
        parts = command.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        # Record command in history
        self.session_history.append(
            {"command": command, "timestamp": asyncio.get_event_loop().time()}
        )

        # Route command
        if cmd == "help":
            self._show_help()
        elif cmd == "status":
            await self._show_status()
        elif cmd == "agents":
            await self._handle_agents_command(args)
        elif cmd == "tasks":
            await self._handle_tasks_command(args)
        elif cmd == "tools":
            await self._handle_tools_command(args)
        elif cmd == "chat":
            await self._handle_chat_command(args)
        elif cmd == "history":
            self._show_history()
        elif cmd == "clear":
            console.clear()
        else:
            console.print(f"❓ Unknown command: {cmd}", style="yellow")
            console.print("💡 Type 'help' for available commands", style="dim")

    def _create_welcome_panel(self) -> Panel:
        """Create welcome panel."""
        welcome_text = f"""
[bold blue]{self.settings.app_name}[/bold blue] [dim]v{self.settings.app_version}[/dim]

{self.settings.app_description}

[dim]Environment: {self.settings.environment.value}[/dim]
[dim]Type 'help' for available commands[/dim]
        """

        return Panel(welcome_text.strip(), title="🤖 Welcome", border_style="blue", padding=(1, 2))

    def _show_help(self) -> None:
        """Show help information."""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="bold blue")
        help_table.add_column("Description", style="dim")
        help_table.add_column("Examples", style="green")

        commands = [
            ("help", "Show this help message", "help"),
            ("status", "Show system status", "status"),
            ("agents", "Manage agents", "agents list, agents create, agents delete <id>"),
            ("tasks", "Manage tasks", "tasks list, tasks create, tasks status <id>"),
            ("tools", "Manage tools", "tools list, tools info <name>"),
            ("chat", "Start chat session", "chat, chat with <agent-id>"),
            ("history", "Show command history", "history"),
            ("clear", "Clear screen", "clear"),
            ("exit", "Exit the CLI", "exit, quit, q"),
        ]

        for cmd, desc, examples in commands:
            help_table.add_row(cmd, desc, examples)

        console.print(help_table)

    async def _show_status(self) -> None:
        """Show system status."""
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)

            # Gather status information
            status_info = {
                "Application": {
                    "Name": self.settings.app_name,
                    "Version": self.settings.app_version,
                    "Environment": self.settings.environment.value,
                    "Debug": "Enabled" if self.settings.debug else "Disabled",
                },
                "Agents": {},
                "Tasks": {},
                "Tools": {},
            }

            try:
                # Get agent status
                agents = await self.agent_manager.list_agents()
                status_info["Agents"] = {
                    "Total": len(agents),
                    "Active": len([a for a in agents if a.status == "active"]),
                    "Inactive": len([a for a in agents if a.status == "inactive"]),
                }

                # Get task status
                tasks = await self.task_manager.list_tasks()
                status_info["Tasks"] = {
                    "Total": len(tasks),
                    "Running": len([t for t in tasks if t.status == "running"]),
                    "Completed": len([t for t in tasks if t.status == "completed"]),
                    "Failed": len([t for t in tasks if t.status == "failed"]),
                }

                # Get tool status
                tools = await self.tool_manager.list_tools()
                status_info["Tools"] = {
                    "Available": len(tools),
                    "Enabled": len([t for t in tools if t.enabled]),
                }

            except Exception as e:
                console.print(f"⚠️ Error gathering status: {e}", style="yellow")

            progress.remove_task(task)

        # Display status
        status_tree = Tree("🖥️ System Status")

        for category, info in status_info.items():
            category_node = status_tree.add(f"📊 {category}")

            if isinstance(info, dict):
                for key, value in info.items():
                    category_node.add(f"{key}: [bold]{value}[/bold]")
            else:
                category_node.add(str(info))

        console.print(status_tree)

    async def _handle_agents_command(self, args: List[str]) -> None:
        """Handle agents command."""
        if not args:
            args = ["list"]

        subcommand = args[0].lower()

        if subcommand == "list":
            await self._list_agents()
        elif subcommand == "create":
            await self._create_agent()
        elif subcommand == "delete" and len(args) > 1:
            await self._delete_agent(args[1])
        elif subcommand == "info" and len(args) > 1:
            await self._show_agent_info(args[1])
        else:
            console.print("❓ Usage: agents [list|create|delete <id>|info <id>]", style="yellow")

    async def _list_agents(self) -> None:
        """List all agents."""
        try:
            agents = await self.agent_manager.list_agents()

            if not agents:
                console.print("📭 No agents found", style="dim")
                return

            agents_table = Table(title="🤖 Agents", show_header=True)
            agents_table.add_column("ID", style="dim")
            agents_table.add_column("Name", style="bold")
            agents_table.add_column("Type", style="blue")
            agents_table.add_column("Status", style="green")
            agents_table.add_column("Created", style="dim")

            for agent in agents:
                status_style = "green" if agent.status == "active" else "red"
                agents_table.add_row(
                    agent.id[:8],
                    agent.name,
                    agent.agent_type,
                    f"[{status_style}]{agent.status}[/{status_style}]",
                    agent.created_at.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(agents_table)

        except Exception as e:
            console.print(f"❌ Error listing agents: {e}", style="red")

    async def _create_agent(self) -> None:
        """Create a new agent."""
        try:
            console.print("🔧 Creating new agent...", style="bold blue")

            name = Prompt.ask("Agent name")
            agent_type = Prompt.ask(
                "Agent type",
                choices=["worker", "analytics", "communication", "coordinator"],
                default="worker",
            )
            description = Prompt.ask("Description", default="")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating agent...", total=None)

                agent = await self.agent_manager.create_agent(
                    name=name, agent_type=agent_type, description=description
                )

                progress.remove_task(task)

            console.print(f"✅ Agent created: {agent.name} ({agent.id[:8]})", style="green")

        except Exception as e:
            console.print(f"❌ Error creating agent: {e}", style="red")

    async def _delete_agent(self, agent_id: str) -> None:
        """Delete an agent."""
        try:
            # Confirm deletion
            if not Confirm.ask(f"Are you sure you want to delete agent {agent_id}?"):
                console.print("❌ Deletion cancelled", style="yellow")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Deleting agent...", total=None)

                await self.agent_manager.delete_agent(agent_id)

                progress.remove_task(task)

            console.print(f"✅ Agent {agent_id} deleted", style="green")

        except Exception as e:
            console.print(f"❌ Error deleting agent: {e}", style="red")

    async def _show_agent_info(self, agent_id: str) -> None:
        """Show detailed agent information."""
        try:
            agent = await self.agent_manager.get_agent(agent_id)

            if not agent:
                console.print(f"❌ Agent {agent_id} not found", style="red")
                return

            # Create info panel
            info_text = f"""
[bold]Name:[/bold] {agent.name}
[bold]Type:[/bold] {agent.agent_type}
[bold]Status:[/bold] {agent.status}
[bold]Description:[/bold] {agent.description or 'N/A'}
[bold]Created:[/bold] {agent.created_at}
[bold]Updated:[/bold] {agent.updated_at}
[bold]Capabilities:[/bold] {len(agent.capabilities)}
            """

            panel = Panel(info_text.strip(), title=f"🤖 Agent {agent.id[:8]}", border_style="blue")

            console.print(panel)

        except Exception as e:
            console.print(f"❌ Error getting agent info: {e}", style="red")

    async def _handle_tasks_command(self, args: List[str]) -> None:
        """Handle tasks command."""
        if not args:
            args = ["list"]

        subcommand = args[0].lower()

        if subcommand == "list":
            await self._list_tasks()
        elif subcommand == "create":
            await self._create_task()
        elif subcommand == "status" and len(args) > 1:
            await self._show_task_status(args[1])
        else:
            console.print("❓ Usage: tasks [list|create|status <id>]", style="yellow")

    async def _list_tasks(self) -> None:
        """List all tasks."""
        try:
            tasks = await self.task_manager.list_tasks()

            if not tasks:
                console.print("📭 No tasks found", style="dim")
                return

            tasks_table = Table(title="📋 Tasks", show_header=True)
            tasks_table.add_column("ID", style="dim")
            tasks_table.add_column("Name", style="bold")
            tasks_table.add_column("Agent", style="blue")
            tasks_table.add_column("Status", style="green")
            tasks_table.add_column("Progress", style="yellow")
            tasks_table.add_column("Created", style="dim")

            for task in tasks:
                status_style = {
                    "pending": "yellow",
                    "running": "blue",
                    "completed": "green",
                    "failed": "red",
                }.get(task.status, "dim")

                tasks_table.add_row(
                    task.id[:8],
                    task.name,
                    task.agent_id[:8] if task.agent_id else "N/A",
                    f"[{status_style}]{task.status}[/{status_style}]",
                    f"{task.progress.percentage:.1f}%" if task.progress else "0%",
                    task.created_at.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(tasks_table)

        except Exception as e:
            console.print(f"❌ Error listing tasks: {e}", style="red")

    async def _handle_tools_command(self, args: List[str]) -> None:
        """Handle tools command."""
        if not args:
            args = ["list"]

        subcommand = args[0].lower()

        if subcommand == "list":
            await self._list_tools()
        elif subcommand == "info" and len(args) > 1:
            await self._show_tool_info(args[1])
        else:
            console.print("❓ Usage: tools [list|info <name>]", style="yellow")

    async def _list_tools(self) -> None:
        """List available tools."""
        try:
            tools = await self.tool_manager.list_tools()

            if not tools:
                console.print("📭 No tools found", style="dim")
                return

            tools_table = Table(title="🔧 Tools", show_header=True)
            tools_table.add_column("Name", style="bold")
            tools_table.add_column("Category", style="blue")
            tools_table.add_column("Status", style="green")
            tools_table.add_column("Description", style="dim")

            for tool in tools:
                status_style = "green" if tool.enabled else "red"
                status_text = "Enabled" if tool.enabled else "Disabled"

                tools_table.add_row(
                    tool.name,
                    tool.category,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    (
                        tool.description[:50] + "..."
                        if len(tool.description) > 50
                        else tool.description
                    ),
                )

            console.print(tools_table)

        except Exception as e:
            console.print(f"❌ Error listing tools: {e}", style="red")

    async def _handle_chat_command(self, args: List[str]) -> None:
        """Handle chat command."""
        console.print("💬 Starting chat session...", style="bold blue")
        console.print("Type 'exit' to end the chat session", style="dim")

        agent_id = args[0] if args else None

        while True:
            try:
                message = Prompt.ask("[bold green]You[/bold green]")

                if message.lower() in ["exit", "quit"]:
                    console.print("👋 Chat session ended", style="dim")
                    break

                # Process message with agent
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Processing...", total=None)

                    # Simulate agent response
                    await asyncio.sleep(1)
                    response = f"Echo: {message}"

                    progress.remove_task(task)

                console.print(f"[bold blue]Agent[/bold blue]: {response}")

            except KeyboardInterrupt:
                console.print("\n👋 Chat session ended", style="dim")
                break

    def _show_history(self) -> None:
        """Show command history."""
        if not self.session_history:
            console.print("📭 No command history", style="dim")
            return

        history_table = Table(title="📜 Command History", show_header=True)
        history_table.add_column("#", style="dim")
        history_table.add_column("Command", style="bold")
        history_table.add_column("Time", style="dim")

        for i, entry in enumerate(self.session_history[-20:], 1):  # Show last 20 commands
            history_table.add_row(str(i), entry["command"], f"{entry['timestamp']:.2f}s")

        console.print(history_table)


def create_cli_interface(settings: Settings) -> CLIInterface:
    """Create CLI interface instance."""
    return CLIInterface(settings)
