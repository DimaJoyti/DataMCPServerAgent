"""
Task management API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_task_service
from app.domain.models.task import TaskPriority, TaskStatus, TaskType
from app.domain.services.task_service import TaskService

router = APIRouter()

class CreateTaskRequest(BaseModel):
    """Request model for creating a task."""

    name: str = Field(description="Task name")
    task_type: TaskType = Field(description="Type of task")
    agent_id: str = Field(description="Agent ID to assign task to")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    description: Optional[str] = Field(default=None, description="Task description")

class TaskResponse(BaseModel):
    """Response model for task data."""

    id: str
    name: str
    task_type: TaskType
    agent_id: str
    status: TaskStatus
    priority: TaskPriority
    description: Optional[str]
    created_at: str

    class Config:
        from_attributes = True

@router.post("/", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    task_service: TaskService = Depends(get_task_service),
    current_user=Depends(get_current_user),
):
    """Create a new task."""
    task = await task_service.create_task(
        name=request.name,
        task_type=request.task_type,
        agent_id=request.agent_id,
        priority=request.priority,
        description=request.description,
    )

    return TaskResponse.from_orm(task)

@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    task_service: TaskService = Depends(get_task_service), current_user=Depends(get_current_user)
):
    """List all tasks."""
    # Mock implementation
    return []
