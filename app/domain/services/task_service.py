"""
Task domain services.
Contains business logic for task management, scheduling, and dependencies.
"""

from datetime import datetime
from typing import Any, Dict, List

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService, ValidationError
from app.domain.models.task import Task, TaskPriority, TaskStatus, TaskType

logger = get_logger(__name__)


class TaskService(DomainService, LoggerMixin):
    """Core task management service."""

    async def create_task(
        self,
        name: str,
        task_type: TaskType,
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        description: str = None,
        input_data: Dict[str, Any] = None,
        configuration: Dict[str, Any] = None,
    ) -> Task:
        """Create a new task."""
        self.logger.info(f"Creating task: {name} for agent {agent_id}")

        # Create task
        task = Task(
            name=name,
            task_type=task_type,
            agent_id=agent_id,
            priority=priority,
            description=description,
            input_data=input_data or {},
            configuration=configuration or {},
        )

        # Save task
        task_repo = self.get_repository("task")
        saved_task = await task_repo.save(task)

        self.logger.info(f"Task created successfully: {saved_task.id}")
        return saved_task

    async def get_tasks_by_agent(self, agent_id: str) -> List[Task]:
        """Get all tasks for a specific agent."""
        task_repo = self.get_repository("task")
        return await task_repo.list(agent_id=agent_id)

    async def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specific status."""
        task_repo = self.get_repository("task")
        return await task_repo.list(status=status)


class TaskSchedulingService(DomainService, LoggerMixin):
    """Service for task scheduling operations."""

    async def schedule_task(self, task_id: str, scheduled_at: datetime) -> Task:
        """Schedule a task for execution."""
        task_repo = self.get_repository("task")
        task = await task_repo.get_by_id(task_id)

        if not task:
            raise ValidationError(f"Task not found: {task_id}")

        task.scheduled_at = scheduled_at
        return await task_repo.save(task)


class TaskDependencyService(DomainService, LoggerMixin):
    """Service for managing task dependencies."""

    async def add_dependency(self, task_id: str, dependency_task_id: str) -> Task:
        """Add a dependency to a task."""
        task_repo = self.get_repository("task")
        task = await task_repo.get_by_id(task_id)

        if not task:
            raise ValidationError(f"Task not found: {task_id}")

        from app.domain.models.task import TaskDependency

        dependency = TaskDependency(task_id=dependency_task_id)
        task.add_dependency(dependency)

        return await task_repo.save(task)
