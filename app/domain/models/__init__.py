"""
Domain models for DataMCPServerAgent - Consolidated Version.
Contains simplified business entities for the consolidated system.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Agent models
class AgentType(str, Enum):
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"


class AgentCapability(str, Enum):
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    COMMUNICATION = "communication"


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_type: AgentType = AgentType.WORKER
    status: AgentStatus = AgentStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)


# Task models
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TaskType(str, Enum):
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_id: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = Field(default_factory=datetime.now)


__all__ = [
    "Agent",
    "AgentType",
    "AgentStatus",
    "AgentCapability",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "TaskType",
]
