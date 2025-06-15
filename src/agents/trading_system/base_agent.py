"""
Base agent class for the Fetch.ai Advanced Crypto Trading System.

This module provides the foundation for all specialized agents in the system.
"""

import logging
from typing import Optional

from uagents import Agent, Context, Model


class BaseAgentState(Model):
    """Base state model for all agents."""

    is_active: bool = True
    last_update: Optional[str] = None
    message_count: int = 0
    error_count: int = 0


class BaseAgent:
    """Base class for all specialized agents in the trading system."""

    def __init__(
        self,
        name: str,
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base agent.

        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
        """
        self.name = name
        self.seed = seed
        self.port = port
        self.endpoint = endpoint

        # Set up logging
        self.logger = logger or logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Create the agent
        self.agent = Agent(name=name, seed=seed, port=port, endpoint=endpoint)

        # Set up the agent state
        self.state = BaseAgentState()

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default handlers for the agent."""

        @self.agent.on_interval(period=60.0)
        async def heartbeat(ctx: Context):
            """Send a heartbeat to indicate the agent is alive."""
            ctx.logger.debug(f"Heartbeat from {self.name}")

        @self.agent.on_event("startup")
        async def on_startup(ctx: Context):
            """Handle agent startup."""
            ctx.logger.info(f"Agent {self.name} started")

        @self.agent.on_event("shutdown")
        async def on_shutdown(ctx: Context):
            """Handle agent shutdown."""
            ctx.logger.info(f"Agent {self.name} shutting down")

    def run(self):
        """Run the agent."""
        self.logger.info(f"Starting agent {self.name}")
        self.agent.run()

    async def run_async(self):
        """Run the agent asynchronously."""
        self.logger.info(f"Starting agent {self.name} asynchronously")
        await self.agent.async_run()

    def stop(self):
        """Stop the agent."""
        self.logger.info(f"Stopping agent {self.name}")
        # Implementation depends on uagents library capabilities

    def get_address(self) -> str:
        """Get the agent's address."""
        return str(self.agent.address)
