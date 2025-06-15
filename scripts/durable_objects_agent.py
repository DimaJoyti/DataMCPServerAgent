"""
Durable Objects simulation for Agent state management.
In production, this would be implemented as Cloudflare Durable Objects.
"""

import asyncio
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_INPUT = "waiting_for_input"
    ERROR = "error"
    TERMINATED = "terminated"

@dataclass
class AgentMemory:
    conversation_history: List[Dict[str, Any]]
    context_variables: Dict[str, Any]
    tool_usage_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    session_metadata: Dict[str, Any]

@dataclass
class AgentInstance:
    agent_id: str
    user_id: str
    session_id: str
    agent_type: str
    state: AgentState
    memory: AgentMemory
    created_at: datetime
    last_activity: datetime
    configuration: Dict[str, Any]
    is_persistent: bool = True

class DurableAgentObject:
    """
    Simulates Cloudflare Durable Objects for agent state management.
    In production, this would be a real Durable Object.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent_instance: Optional[AgentInstance] = None
        self.storage_path = f"agent_storage/{agent_id}"
        self.lock = asyncio.Lock()

        # Ensure storage directory exists
        os.makedirs("agent_storage", exist_ok=True)

        # Load existing state if available
        self._load_state()

    def _load_state(self):
        """Load agent state from persistent storage."""
        try:
            if os.path.exists(f"{self.storage_path}.pkl"):
                with open(f"{self.storage_path}.pkl", 'rb') as f:
                    self.agent_instance = pickle.load(f)
        except Exception as e:
            print(f"Error loading agent state: {e}")

    def _save_state(self):
        """Save agent state to persistent storage."""
        try:
            if self.agent_instance and self.agent_instance.is_persistent:
                with open(f"{self.storage_path}.pkl", 'wb') as f:
                    pickle.dump(self.agent_instance, f)
        except Exception as e:
            print(f"Error saving agent state: {e}")

    async def initialize_agent(
        self,
        user_id: str,
        session_id: str,
        agent_type: str,
        configuration: Dict[str, Any] = None
    ) -> AgentInstance:
        """Initialize a new agent instance."""
        async with self.lock:
            if self.agent_instance:
                # Update existing agent
                self.agent_instance.user_id = user_id
                self.agent_instance.session_id = session_id
                self.agent_instance.last_activity = datetime.utcnow()
                self.agent_instance.state = AgentState.IDLE
            else:
                # Create new agent
                memory = AgentMemory(
                    conversation_history=[],
                    context_variables={},
                    tool_usage_history=[],
                    user_preferences={},
                    session_metadata={}
                )

                self.agent_instance = AgentInstance(
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=session_id,
                    agent_type=agent_type,
                    state=AgentState.IDLE,
                    memory=memory,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    configuration=configuration or {}
                )

            self._save_state()
            return self.agent_instance

    async def update_state(self, new_state: AgentState) -> bool:
        """Update agent state."""
        async with self.lock:
            if self.agent_instance:
                self.agent_instance.state = new_state
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

    async def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to conversation history."""
        async with self.lock:
            if self.agent_instance:
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata or {}
                }
                self.agent_instance.memory.conversation_history.append(message)
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

    async def add_tool_usage(self, tool_name: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Record tool usage."""
        async with self.lock:
            if self.agent_instance:
                tool_record = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.agent_instance.memory.tool_usage_history.append(tool_record)
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

    async def update_context(self, key: str, value: Any) -> bool:
        """Update context variable."""
        async with self.lock:
            if self.agent_instance:
                self.agent_instance.memory.context_variables[key] = value
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

    async def get_context(self, key: str) -> Any:
        """Get context variable."""
        if self.agent_instance:
            return self.agent_instance.memory.context_variables.get(key)
        return None

    async def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if self.agent_instance:
            return self.agent_instance.memory.conversation_history[-limit:]
        return []

    async def get_tool_usage_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get tool usage history."""
        if self.agent_instance:
            return self.agent_instance.memory.tool_usage_history[-limit:]
        return []

    async def update_user_preference(self, key: str, value: Any) -> bool:
        """Update user preference."""
        async with self.lock:
            if self.agent_instance:
                self.agent_instance.memory.user_preferences[key] = value
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

    async def get_agent_info(self) -> Optional[Dict[str, Any]]:
        """Get agent information."""
        if self.agent_instance:
            return {
                "agent_id": self.agent_instance.agent_id,
                "user_id": self.agent_instance.user_id,
                "session_id": self.agent_instance.session_id,
                "agent_type": self.agent_instance.agent_type,
                "state": self.agent_instance.state.value,
                "created_at": self.agent_instance.created_at.isoformat(),
                "last_activity": self.agent_instance.last_activity.isoformat(),
                "configuration": self.agent_instance.configuration,
                "memory_stats": {
                    "conversation_messages": len(self.agent_instance.memory.conversation_history),
                    "context_variables": len(self.agent_instance.memory.context_variables),
                    "tools_used": len(self.agent_instance.memory.tool_usage_history),
                    "user_preferences": len(self.agent_instance.memory.user_preferences)
                }
            }
        return None

    async def cleanup_old_data(self, days: int = 30):
        """Clean up old conversation data."""
        async with self.lock:
            if self.agent_instance:
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                # Filter conversation history
                self.agent_instance.memory.conversation_history = [
                    msg for msg in self.agent_instance.memory.conversation_history
                    if datetime.fromisoformat(msg["timestamp"]) > cutoff_date
                ]

                # Filter tool usage history
                self.agent_instance.memory.tool_usage_history = [
                    tool for tool in self.agent_instance.memory.tool_usage_history
                    if datetime.fromisoformat(tool["timestamp"]) > cutoff_date
                ]

                self._save_state()

    async def terminate_agent(self) -> bool:
        """Terminate the agent."""
        async with self.lock:
            if self.agent_instance:
                self.agent_instance.state = AgentState.TERMINATED
                self.agent_instance.last_activity = datetime.utcnow()
                self._save_state()
                return True
            return False

class DurableObjectManager:
    """Manager for Durable Object instances."""

    def __init__(self):
        self.agents: Dict[str, DurableAgentObject] = {}
        self.cleanup_task = None
        # Don't start cleanup task immediately - will be started when needed

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_inactive_agents()

        self.cleanup_task = asyncio.create_task(cleanup_loop())

    async def get_agent(self, agent_id: str) -> DurableAgentObject:
        """Get or create agent Durable Object."""
        if agent_id not in self.agents:
            self.agents[agent_id] = DurableAgentObject(agent_id)
        return self.agents[agent_id]

    async def create_agent(
        self,
        user_id: str,
        session_id: str,
        agent_type: str,
        configuration: Dict[str, Any] = None
    ) -> str:
        """Create a new agent and return its ID."""
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        agent_obj = await self.get_agent(agent_id)
        await agent_obj.initialize_agent(user_id, session_id, agent_type, configuration)
        return agent_id

    async def list_user_agents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all agents for a user."""
        user_agents = []
        for agent_id, agent_obj in self.agents.items():
            if agent_obj.agent_instance and agent_obj.agent_instance.user_id == user_id:
                info = await agent_obj.get_agent_info()
                if info:
                    user_agents.append(info)
        return user_agents

    async def cleanup_inactive_agents(self, hours: int = 24):
        """Clean up inactive agents."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        inactive_agents = []

        for agent_id, agent_obj in self.agents.items():
            if (agent_obj.agent_instance and
                agent_obj.agent_instance.last_activity < cutoff_time and
                agent_obj.agent_instance.state != AgentState.PROCESSING):
                inactive_agents.append(agent_id)

        for agent_id in inactive_agents:
            await self.agents[agent_id].terminate_agent()
            del self.agents[agent_id]

        return len(inactive_agents)

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        total_agents = len(self.agents)
        active_agents = 0
        states_count = {}

        for agent_obj in self.agents.values():
            if agent_obj.agent_instance:
                if agent_obj.agent_instance.state != AgentState.TERMINATED:
                    active_agents += 1

                state = agent_obj.agent_instance.state.value
                states_count[state] = states_count.get(state, 0) + 1

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "states_distribution": states_count,
            "storage_path": "agent_storage/"
        }

# Global manager instance
durable_manager = DurableObjectManager()

# Decorator for Durable Object operations
def with_durable_state(agent_id_param: str = "agent_id"):
    """Decorator to automatically manage Durable Object state."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            agent_id = kwargs.get(agent_id_param)
            if not agent_id:
                raise ValueError(f"Missing {agent_id_param} parameter")

            # Get agent Durable Object
            agent_obj = await durable_manager.get_agent(agent_id)
            kwargs['agent_obj'] = agent_obj

            # Update state to processing
            await agent_obj.update_state(AgentState.PROCESSING)

            try:
                result = await func(*args, **kwargs)
                # Update state back to idle
                await agent_obj.update_state(AgentState.IDLE)
                return result
            except Exception:
                # Update state to error
                await agent_obj.update_state(AgentState.ERROR)
                raise

        return wrapper
    return decorator
