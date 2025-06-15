"""
Agent domain models.
Defines the core Agent entity and related value objects.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator

from .base import AggregateRoot, BaseValueObject, DomainEvent, ValidationError


class AgentType(str, Enum):
    """Types of agents in the system."""

    WORKER = "worker"
    ANALYTICS = "analytics"
    MARKETPLACE = "marketplace"
    OBSERVABILITY = "observability"
    EMAIL = "email"
    WEBRTC = "webrtc"
    ORCHESTRATOR = "orchestrator"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Agent operational status."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    SCALING = "scaling"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(BaseValueObject):
    """Represents a capability that an agent possesses."""

    name: str = Field(description="Capability name")
    version: str = Field(description="Capability version")
    description: str = Field(description="Capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")
    enabled: bool = Field(default=True, description="Whether capability is enabled")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Capability name cannot be empty")
        return v.strip().lower()


class AgentConfiguration(BaseValueObject):
    """Agent configuration settings."""

    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent tasks")
    timeout_seconds: int = Field(default=300, description="Task timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    memory_limit_mb: int = Field(default=512, description="Memory limit in MB")
    cpu_limit_cores: float = Field(default=1.0, description="CPU limit in cores")
    environment_variables: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    secrets: Set[str] = Field(default_factory=set, description="Required secrets")

    @field_validator("max_concurrent_tasks")
    @classmethod
    def validate_max_concurrent_tasks(cls, v):
        if v <= 0:
            raise ValidationError("Max concurrent tasks must be positive")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValidationError("Timeout must be positive")
        return v


class AgentMetrics(BaseValueObject):
    """Agent performance metrics."""

    tasks_completed: int = Field(default=0, description="Total tasks completed")
    tasks_failed: int = Field(default=0, description="Total tasks failed")
    average_response_time_ms: float = Field(
        default=0.0, description="Average response time in milliseconds"
    )
    cpu_usage_percent: float = Field(default=0.0, description="Current CPU usage percentage")
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage in MB")
    uptime_seconds: int = Field(default=0, description="Agent uptime in seconds")
    last_heartbeat: Optional[datetime] = Field(default=None, description="Last heartbeat timestamp")

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 0.0
        return (self.tasks_completed / total_tasks) * 100.0

    @property
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on metrics."""
        if not self.last_heartbeat:
            return False

        # Consider unhealthy if no heartbeat in last 5 minutes
        time_since_heartbeat = datetime.now(timezone.utc) - self.last_heartbeat
        if time_since_heartbeat.total_seconds() > 300:
            return False

        # Consider unhealthy if CPU or memory usage is too high
        if self.cpu_usage_percent > 90 or self.memory_usage_mb > 450:  # 90% of 512MB limit
            return False

        return True


class AgentCreatedEvent(DomainEvent):
    """Event raised when an agent is created."""

    def __init__(self, agent_id: str, agent_type: AgentType, name: str):
        super().__init__(
            event_type="AgentCreated",
            aggregate_id=agent_id,
            aggregate_type="Agent",
            version=1,
            data={
                "agent_type": agent_type.value,
                "name": name,
            },
        )


class AgentStatusChangedEvent(DomainEvent):
    """Event raised when agent status changes."""

    def __init__(
        self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus, version: int
    ):
        super().__init__(
            event_type="AgentStatusChanged",
            aggregate_id=agent_id,
            aggregate_type="Agent",
            version=version,
            data={
                "old_status": old_status.value,
                "new_status": new_status.value,
            },
        )


class AgentScaledEvent(DomainEvent):
    """Event raised when agent is scaled."""

    def __init__(self, agent_id: str, old_instances: int, new_instances: int, version: int):
        super().__init__(
            event_type="AgentScaled",
            aggregate_id=agent_id,
            aggregate_type="Agent",
            version=version,
            data={
                "old_instances": old_instances,
                "new_instances": new_instances,
            },
        )


class Agent(AggregateRoot):
    """Agent aggregate root."""

    name: str = Field(description="Agent name")
    agent_type: AgentType = Field(description="Type of agent")
    status: AgentStatus = Field(default=AgentStatus.INITIALIZING, description="Current status")
    description: Optional[str] = Field(default=None, description="Agent description")

    # Configuration and capabilities
    configuration: AgentConfiguration = Field(
        default_factory=AgentConfiguration, description="Agent configuration"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=list, description="Agent capabilities"
    )

    # Scaling and deployment
    desired_instances: int = Field(default=1, description="Desired number of instances")
    current_instances: int = Field(default=0, description="Current number of instances")

    # Metrics and monitoring
    metrics: AgentMetrics = Field(default_factory=AgentMetrics, description="Agent metrics")

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict, description="Agent tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __init__(self, **data):
        super().__init__(**data)
        if self.version == 1:  # New agent
            self.add_domain_event(AgentCreatedEvent(self.id, self.agent_type, self.name))

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Agent name cannot be empty")
        return v.strip()

    @field_validator("desired_instances")
    @classmethod
    def validate_desired_instances(cls, v):
        if v < 0:
            raise ValidationError("Desired instances cannot be negative")
        return v

    def change_status(self, new_status: AgentStatus) -> None:
        """Change agent status."""
        if self.status != new_status:
            old_status = self.status
            self.status = new_status
            self.apply_event(AgentStatusChangedEvent(self.id, old_status, new_status, self.version))

    def scale_to(self, target_instances: int) -> None:
        """Scale agent to target number of instances."""
        if target_instances < 0:
            raise ValidationError("Target instances cannot be negative")

        if self.desired_instances != target_instances:
            old_instances = self.desired_instances
            self.desired_instances = target_instances
            self.change_status(AgentStatus.SCALING)
            self.apply_event(
                AgentScaledEvent(self.id, old_instances, target_instances, self.version)
            )

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        # Check if capability already exists
        existing = next((c for c in self.capabilities if c.name == capability.name), None)
        if existing:
            # Update existing capability
            self.capabilities = [
                c if c.name != capability.name else capability for c in self.capabilities
            ]
        else:
            # Add new capability
            self.capabilities.append(capability)

    def remove_capability(self, capability_name: str) -> bool:
        """Remove a capability from the agent."""
        original_count = len(self.capabilities)
        self.capabilities = [c for c in self.capabilities if c.name != capability_name]
        return len(self.capabilities) < original_count

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(c.name == capability_name and c.enabled for c in self.capabilities)

    def update_metrics(self, metrics: AgentMetrics) -> None:
        """Update agent metrics."""
        self.metrics = metrics

        # Auto-update status based on metrics
        if not metrics.is_healthy and self.status == AgentStatus.ACTIVE:
            self.change_status(AgentStatus.ERROR)
        elif metrics.is_healthy and self.status == AgentStatus.ERROR:
            self.change_status(AgentStatus.ACTIVE)

    def set_instances_count(self, count: int) -> None:
        """Set current instances count."""
        if count < 0:
            raise ValidationError("Instances count cannot be negative")

        self.current_instances = count

        # Update status based on scaling progress
        if self.status == AgentStatus.SCALING:
            if count == self.desired_instances:
                self.change_status(AgentStatus.ACTIVE)

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the agent."""
        self.tags[key] = value

    def remove_tag(self, key: str) -> bool:
        """Remove a tag from the agent."""
        return self.tags.pop(key, None) is not None

    def is_scalable(self) -> bool:
        """Check if agent can be scaled."""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]

    def can_accept_tasks(self) -> bool:
        """Check if agent can accept new tasks."""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE] and self.current_instances > 0
