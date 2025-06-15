"""
Consolidated Agent Model for DataMCPServerAgent.

Unified agent model that consolidates all agent-related functionality
into a single, clean domain model following DDD principles.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Types of agents in the consolidated system."""

    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    COMMUNICATOR = "communicator"


class AgentStatus(str, Enum):
    """Status of agents in the consolidated system."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""

    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    COMMUNICATION = "communication"
    TASK_COORDINATION = "task_coordination"
    MEMORY_MANAGEMENT = "memory_management"
    LEARNING = "learning"
    VISUALIZATION = "visualization"
    API_INTEGRATION = "api_integration"


class AgentConfiguration(BaseModel):
    """Configuration for agent behavior."""

    max_concurrent_tasks: int = Field(default=5, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    memory_limit_mb: int = Field(default=512, ge=64, le=4096)
    learning_enabled: bool = Field(default=True)
    auto_retry: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    priority_level: int = Field(default=5, ge=1, le=10)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Metrics for agent performance."""

    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    average_task_duration: float = Field(default=0.0, ge=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    uptime_seconds: int = Field(default=0, ge=0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    last_activity: Optional[datetime] = None


class ConsolidatedAgent(BaseModel):
    """
    Consolidated Agent model representing an AI agent in the system.

    This is the main aggregate root for agent-related operations,
    containing all necessary information and behavior for agent management.
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)

    # Classification
    agent_type: AgentType = Field(default=AgentType.WORKER)
    capabilities: List[AgentCapability] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # State
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    configuration: AgentConfiguration = Field(default_factory=AgentConfiguration)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)

    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None
    version: int = Field(default=1, ge=1)

    # Relationships
    owner_id: Optional[str] = None
    parent_agent_id: Optional[str] = None
    child_agent_ids: List[str] = Field(default_factory=list)

    # Extended properties
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent({self.name}, {self.agent_type}, {self.status})"

    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (
            f"ConsolidatedAgent(id='{self.id[:8]}', name='{self.name}', "
            f"type='{self.agent_type}', status='{self.status}')"
        )

    # Domain methods

    def activate(self) -> None:
        """Activate the agent."""
        if self.status == AgentStatus.MAINTENANCE:
            raise ValueError("Cannot activate agent in maintenance mode")

        self.status = AgentStatus.ACTIVE
        self.last_seen = datetime.now()
        self._update_version()

    def deactivate(self) -> None:
        """Deactivate the agent."""
        self.status = AgentStatus.INACTIVE
        self._update_version()

    def set_busy(self) -> None:
        """Mark agent as busy."""
        if self.status != AgentStatus.ACTIVE:
            raise ValueError("Only active agents can be set to busy")

        self.status = AgentStatus.BUSY
        self.last_seen = datetime.now()
        self._update_version()

    def set_error(self, error_message: str = "") -> None:
        """Mark agent as in error state."""
        self.status = AgentStatus.ERROR
        if error_message:
            self.metadata["last_error"] = error_message
            self.metadata["error_timestamp"] = datetime.now().isoformat()
        self._update_version()

    def enter_maintenance(self) -> None:
        """Put agent in maintenance mode."""
        self.status = AgentStatus.MAINTENANCE
        self._update_version()

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self._update_version()

    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove a capability from the agent."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self._update_version()

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def update_configuration(self, **kwargs) -> None:
        """Update agent configuration."""
        for key, value in kwargs.items():
            if hasattr(self.configuration, key):
                setattr(self.configuration, key, value)
        self._update_version()

    def record_task_completion(self, duration_seconds: float, success: bool = True) -> None:
        """Record completion of a task."""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1

        # Update average duration
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            current_total = self.metrics.average_task_duration * (total_tasks - 1)
            self.metrics.average_task_duration = (current_total + duration_seconds) / total_tasks

        # Update success rate
        if total_tasks > 0:
            self.metrics.success_rate = self.metrics.tasks_completed / total_tasks

        self.metrics.last_activity = datetime.now()
        self.last_seen = datetime.now()
        self._update_version()

    def update_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Update resource usage metrics."""
        self.metrics.memory_usage_mb = memory_mb
        self.metrics.cpu_usage_percent = cpu_percent
        self.last_seen = datetime.now()

    def add_child_agent(self, child_id: str) -> None:
        """Add a child agent."""
        if child_id not in self.child_agent_ids:
            self.child_agent_ids.append(child_id)
            self._update_version()

    def remove_child_agent(self, child_id: str) -> None:
        """Remove a child agent."""
        if child_id in self.child_agent_ids:
            self.child_agent_ids.remove(child_id)
            self._update_version()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the agent."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._update_version()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the agent."""
        if tag in self.tags:
            self.tags.remove(tag)
            self._update_version()

    def has_tag(self, tag: str) -> bool:
        """Check if agent has a specific tag."""
        return tag in self.tags

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value
        self._update_version()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def is_healthy(self) -> bool:
        """Check if agent is in a healthy state."""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.BUSY]

    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.status == AgentStatus.ACTIVE

    def get_uptime_seconds(self) -> int:
        """Calculate uptime in seconds."""
        if self.last_seen:
            return int((datetime.now() - self.created_at).total_seconds())
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "status": self.status,
            "configuration": self.configuration.dict(),
            "metrics": self.metrics.dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "version": self.version,
            "owner_id": self.owner_id,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": self.child_agent_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidatedAgent":
        """Create agent from dictionary."""
        # Convert datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if "last_seen" in data and data["last_seen"] and isinstance(data["last_seen"], str):
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])

        return cls(**data)

    def _update_version(self) -> None:
        """Update version and timestamp."""
        self.version += 1
        self.updated_at = datetime.now()


# Type alias for backward compatibility
Agent = ConsolidatedAgent
