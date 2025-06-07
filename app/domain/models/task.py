"""
Task domain models.
Defines task entities, value objects, and related domain logic.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .base import (
    AggregateRoot,
    BaseValueObject,
    BusinessRuleError,
    DomainEvent,
    ValidationError,
)

class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    """Types of tasks."""

    DATA_ANALYSIS = "data_analysis"
    EMAIL_PROCESSING = "email_processing"
    WEBRTC_CALL = "webrtc_call"
    STATE_SYNC = "state_sync"
    SCALING_OPERATION = "scaling_operation"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"

class TaskProgress(BaseValueObject):
    """Task progress information."""

    percentage: float = Field(default=0.0, description="Progress percentage (0-100)")
    current_step: str = Field(default="", description="Current step description")
    total_steps: int = Field(default=1, description="Total number of steps")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    estimated_completion: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )

    @validator("percentage")
    def validate_percentage(cls, v):
        if v < 0 or v > 100:
            raise ValidationError("Progress percentage must be between 0 and 100")
        return v

    @validator("completed_steps")
    def validate_completed_steps(cls, v, values):
        total_steps = values.get("total_steps", 1)
        if v < 0 or v > total_steps:
            raise ValidationError("Completed steps must be between 0 and total steps")
        return v

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.percentage >= 100.0 or self.completed_steps >= self.total_steps

class TaskResult(BaseValueObject):
    """Task execution result."""

    success: bool = Field(description="Whether task succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    execution_time_ms: int = Field(default=0, description="Execution time in milliseconds")
    resource_usage: Dict[str, Any] = Field(
        default_factory=dict, description="Resource usage metrics"
    )

    @validator("execution_time_ms")
    def validate_execution_time(cls, v):
        if v < 0:
            raise ValidationError("Execution time cannot be negative")
        return v

class TaskDependency(BaseValueObject):
    """Task dependency specification."""

    task_id: str = Field(description="ID of the dependent task")
    dependency_type: str = Field(default="completion", description="Type of dependency")
    required: bool = Field(default=True, description="Whether dependency is required")

    @validator("task_id")
    def validate_task_id(cls, v):
        if not v or not v.strip():
            raise ValidationError("Task ID cannot be empty")
        return v.strip()

class TaskCreatedEvent(DomainEvent):
    """Event raised when a task is created."""

    def __init__(self, task_id: str, task_type: TaskType, agent_id: str, priority: TaskPriority):
        super().__init__(
            event_type="TaskCreated",
            aggregate_id=task_id,
            aggregate_type="Task",
            version=1,
            data={
                "task_type": task_type.value,
                "agent_id": agent_id,
                "priority": priority.value,
            },
        )

class TaskStatusChangedEvent(DomainEvent):
    """Event raised when task status changes."""

    def __init__(self, task_id: str, old_status: TaskStatus, new_status: TaskStatus, version: int):
        super().__init__(
            event_type="TaskStatusChanged",
            aggregate_id=task_id,
            aggregate_type="Task",
            version=version,
            data={
                "old_status": old_status.value,
                "new_status": new_status.value,
            },
        )

class TaskProgressUpdatedEvent(DomainEvent):
    """Event raised when task progress is updated."""

    def __init__(self, task_id: str, progress: TaskProgress, version: int):
        super().__init__(
            event_type="TaskProgressUpdated",
            aggregate_id=task_id,
            aggregate_type="Task",
            version=version,
            data={
                "percentage": progress.percentage,
                "current_step": progress.current_step,
                "completed_steps": progress.completed_steps,
                "total_steps": progress.total_steps,
            },
        )

class TaskCompletedEvent(DomainEvent):
    """Event raised when a task is completed."""

    def __init__(self, task_id: str, result: TaskResult, version: int):
        super().__init__(
            event_type="TaskCompleted",
            aggregate_id=task_id,
            aggregate_type="Task",
            version=version,
            data={
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "error_message": result.error_message,
                "error_code": result.error_code,
            },
        )

class Task(AggregateRoot):
    """Task aggregate root."""

    # Basic task information
    name: str = Field(description="Task name")
    task_type: TaskType = Field(description="Type of task")
    description: Optional[str] = Field(default=None, description="Task description")

    # Assignment and execution
    agent_id: str = Field(description="ID of the agent assigned to this task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")

    # Task data and configuration
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input data")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Task configuration")

    # Progress and results
    progress: TaskProgress = Field(default_factory=TaskProgress, description="Task progress")
    result: Optional[TaskResult] = Field(default=None, description="Task result")

    # Timing
    scheduled_at: Optional[datetime] = Field(
        default=None, description="When task is scheduled to run"
    )
    started_at: Optional[datetime] = Field(default=None, description="When task execution started")
    completed_at: Optional[datetime] = Field(default=None, description="When task was completed")
    timeout_at: Optional[datetime] = Field(default=None, description="When task times out")

    # Dependencies and relationships
    dependencies: List[TaskDependency] = Field(
        default_factory=list, description="Task dependencies"
    )
    parent_task_id: Optional[str] = Field(
        default=None, description="Parent task ID if this is a subtask"
    )

    # Retry and error handling
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_count: int = Field(default=0, description="Current retry count")
    retry_delay_seconds: int = Field(default=60, description="Delay between retries in seconds")

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict, description="Task tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __init__(self, **data):
        super().__init__(**data)
        if self.version == 1:  # New task
            self.add_domain_event(
                TaskCreatedEvent(self.id, self.task_type, self.agent_id, self.priority)
            )

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Task name cannot be empty")
        return v.strip()

    @validator("agent_id")
    def validate_agent_id(cls, v):
        if not v or not v.strip():
            raise ValidationError("Agent ID cannot be empty")
        return v.strip()

    @validator("max_retries")
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValidationError("Max retries cannot be negative")
        return v

    @validator("retry_delay_seconds")
    def validate_retry_delay(cls, v):
        if v < 0:
            raise ValidationError("Retry delay cannot be negative")
        return v

    def change_status(self, new_status: TaskStatus) -> None:
        """Change task status with validation."""
        if not self._can_transition_to(new_status):
            raise BusinessRuleError(f"Cannot transition from {self.status} to {new_status}")

        old_status = self.status
        self.status = new_status

        # Update timestamps based on status
        current_time = datetime.now(timezone.utc)
        if new_status == TaskStatus.RUNNING and not self.started_at:
            self.started_at = current_time
        elif new_status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]:
            self.completed_at = current_time

        self.apply_event(TaskStatusChangedEvent(self.id, old_status, new_status, self.version))

    def _can_transition_to(self, new_status: TaskStatus) -> bool:
        """Check if status transition is valid."""
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.QUEUED, TaskStatus.CANCELLED],
            TaskStatus.QUEUED: [TaskStatus.RUNNING, TaskStatus.CANCELLED],
            TaskStatus.RUNNING: [
                TaskStatus.PAUSED,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.CANCELLED,
            ],
            TaskStatus.PAUSED: [TaskStatus.RUNNING, TaskStatus.CANCELLED],
            TaskStatus.COMPLETED: [],  # Terminal state
            TaskStatus.FAILED: [TaskStatus.QUEUED],  # Can retry
            TaskStatus.CANCELLED: [],  # Terminal state
            TaskStatus.TIMEOUT: [TaskStatus.QUEUED],  # Can retry
        }

        return new_status in valid_transitions.get(self.status, [])

    def update_progress(self, progress: TaskProgress) -> None:
        """Update task progress."""
        if self.status != TaskStatus.RUNNING:
            raise BusinessRuleError("Can only update progress for running tasks")

        self.progress = progress
        self.apply_event(TaskProgressUpdatedEvent(self.id, progress, self.version))

        # Auto-complete if progress indicates completion
        if progress.is_complete and self.status == TaskStatus.RUNNING:
            self.complete_successfully()

    def complete_successfully(self, result_data: Dict[str, Any] = None) -> None:
        """Mark task as completed successfully."""
        execution_time = 0
        if self.started_at:
            execution_time = int(
                (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000
            )

        self.result = TaskResult(
            success=True, data=result_data or {}, execution_time_ms=execution_time
        )

        self.progress = TaskProgress(percentage=100.0, completed_steps=self.progress.total_steps)
        self.change_status(TaskStatus.COMPLETED)
        self.apply_event(TaskCompletedEvent(self.id, self.result, self.version))

    def fail(self, error_message: str, error_code: str = None) -> None:
        """Mark task as failed."""
        execution_time = 0
        if self.started_at:
            execution_time = int(
                (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000
            )

        self.result = TaskResult(
            success=False,
            error_message=error_message,
            error_code=error_code,
            execution_time_ms=execution_time,
        )

        self.change_status(TaskStatus.FAILED)
        self.apply_event(TaskCompletedEvent(self.id, self.result, self.version))

    def cancel(self, reason: str = None) -> None:
        """Cancel the task."""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            raise BusinessRuleError("Cannot cancel completed or already cancelled task")

        self.change_status(TaskStatus.CANCELLED)
        if reason:
            self.metadata["cancellation_reason"] = reason

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]
            and self.retry_count < self.max_retries
        )

    def retry(self) -> None:
        """Retry the task."""
        if not self.can_retry():
            raise BusinessRuleError("Task cannot be retried")

        self.retry_count += 1
        self.result = None
        self.progress = TaskProgress()
        self.started_at = None
        self.completed_at = None
        self.change_status(TaskStatus.QUEUED)

    def set_timeout(self, timeout_seconds: int) -> None:
        """Set task timeout."""
        if timeout_seconds <= 0:
            raise ValidationError("Timeout must be positive")

        self.timeout_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.timeout_at:
            return False
        return datetime.now(timezone.utc) > self.timeout_at

    def add_dependency(self, dependency: TaskDependency) -> None:
        """Add a task dependency."""
        # Check if dependency already exists
        existing = next((d for d in self.dependencies if d.task_id == dependency.task_id), None)
        if not existing:
            self.dependencies.append(dependency)

    def remove_dependency(self, task_id: str) -> bool:
        """Remove a task dependency."""
        original_count = len(self.dependencies)
        self.dependencies = [d for d in self.dependencies if d.task_id != task_id]
        return len(self.dependencies) < original_count

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the task."""
        self.tags[key] = value

    def remove_tag(self, key: str) -> bool:
        """Remove a tag from the task."""
        return self.tags.pop(key, None) is not None

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]

    @property
    def is_active(self) -> bool:
        """Check if task is actively running."""
        return self.status == TaskStatus.RUNNING

    @property
    def duration_seconds(self) -> Optional[int]:
        """Get task duration in seconds."""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now(timezone.utc)
        return int((end_time - self.started_at).total_seconds())
