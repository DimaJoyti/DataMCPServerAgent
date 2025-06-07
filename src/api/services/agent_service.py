"""
Agent service for the API.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..models.response_models import AgentResponse
from ..config import config

class AgentService:
    """Service for interacting with agents."""

    async def list_agents(self) -> List[AgentResponse]:
        """
        List all available agent modes.

        Returns:
            List[AgentResponse]: List of agent responses
        """
        agents = []

        for agent_mode in config.available_agent_modes:
            agent = await self.get_agent(agent_mode)
            agents.append(agent)

        return agents

    async def get_agent(self, agent_mode: str) -> AgentResponse:
        """
        Get information about a specific agent mode.

        Args:
            agent_mode (str): Agent mode

        Returns:
            AgentResponse: Agent response
        """
        # Define capabilities for each agent mode
        capabilities = {
            "basic": ["chat", "web_search", "web_browsing"],
            "advanced": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents"],
            "enhanced": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning"],
            "advanced_enhanced": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "context_aware_memory", "adaptive_learning"],
            "multi_agent": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "context_aware_memory", "adaptive_learning", "collaborative_learning", "knowledge_sharing"],
            "reinforcement_learning": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "reinforcement_learning"],
            "distributed_memory": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "distributed_memory"],
            "knowledge_graph": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "knowledge_graph"],
            "error_recovery": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "error_recovery"],
            "research_reports": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "research", "report_generation"],
            "seo": ["chat", "web_search", "web_browsing", "tool_selection", "specialized_sub_agents", "memory_persistence", "learning", "seo_analysis", "seo_optimization"],
        }

        # Get capabilities for the agent mode
        agent_capabilities = capabilities.get(agent_mode, [])

        return AgentResponse(
            agent_id=str(uuid.uuid4()),
            agent_mode=agent_mode,
            status="available",
            capabilities=agent_capabilities,
            created_at=datetime.now(),
            metadata={
                "description": f"{agent_mode.replace('_', ' ').title()} Agent",
            },
        )

    async def create_agent(self, agent_mode: str, agent_config: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Create a new agent instance.

        Args:
            agent_mode (str): Agent mode
            agent_config (Optional[Dict[str, Any]]): Agent configuration

        Returns:
            AgentResponse: Agent response
        """
        # Get agent information
        agent = await self.get_agent(agent_mode)

        # Update agent with configuration
        if agent_config:
            agent.metadata["config"] = agent_config

        return agent
