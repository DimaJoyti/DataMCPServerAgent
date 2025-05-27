"""
Simple Consolidated FastAPI Server for DataMCPServerAgent.

Simplified version without complex logging to avoid recursion issues.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.simple_config import SimpleSettings


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    timestamp: str
    structure: str


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    agent_type: str = "worker"


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    agent_type: str
    status: str
    created_at: str


class TaskCreate(BaseModel):
    name: str
    agent_id: str
    description: str = ""


class TaskResponse(BaseModel):
    id: str
    name: str
    description: str
    agent_id: str
    status: str
    created_at: str


# In-memory storage
agents_db: List[Dict[str, Any]] = []
tasks_db: List[Dict[str, Any]] = []


def create_simple_consolidated_app() -> FastAPI:
    """Create simple consolidated FastAPI application."""

    settings = SimpleSettings()

    app = FastAPI(
        title=f"{settings.app_name} - Consolidated",
        description="Single app/ Structure with Clean Architecture",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name} - Consolidated",
            "version": settings.app_version,
            "description": "Single app/ Structure with Clean Architecture",
            "structure": "Consolidated single app/ directory",
            "architecture": "Clean Architecture + DDD",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            app=f"{settings.app_name} - Consolidated",
            version=settings.app_version,
            timestamp=datetime.now().isoformat(),
            structure="Single app/ directory",
        )

    @app.get("/api/v1/agents", response_model=List[AgentResponse])
    async def list_agents() -> List[AgentResponse]:
        """List all agents."""
        return [
            AgentResponse(
                id=agent["id"],
                name=agent["name"],
                description=agent["description"],
                agent_type=agent["agent_type"],
                status=agent["status"],
                created_at=agent["created_at"],
            )
            for agent in agents_db
        ]

    @app.post("/api/v1/agents", response_model=AgentResponse)
    async def create_agent(agent: AgentCreate) -> AgentResponse:
        """Create new agent."""
        new_agent = {
            "id": str(uuid.uuid4()),
            "name": agent.name,
            "description": agent.description,
            "agent_type": agent.agent_type,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }

        agents_db.append(new_agent)

        return AgentResponse(**new_agent)

    @app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
    async def get_agent(agent_id: str):
        """Get agent by ID."""
        agent = next((a for a in agents_db if a["id"] == agent_id), None)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentResponse(**agent)

    @app.delete("/api/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete agent."""
        global agents_db

        agent = next((a for a in agents_db if a["id"] == agent_id), None)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        agents_db = [a for a in agents_db if a["id"] != agent_id]

        return {"message": "Agent deleted successfully", "agent_id": agent_id}

    @app.get("/api/v1/tasks", response_model=List[TaskResponse])
    async def list_tasks():
        """List all tasks."""
        return [
            TaskResponse(
                id=task["id"],
                name=task["name"],
                description=task["description"],
                agent_id=task["agent_id"],
                status=task["status"],
                created_at=task["created_at"],
            )
            for task in tasks_db
        ]

    @app.post("/api/v1/tasks", response_model=TaskResponse)
    async def create_task(task: TaskCreate):
        """Create new task."""
        # Verify agent exists
        agent = next((a for a in agents_db if a["id"] == task.agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        new_task = {
            "id": str(uuid.uuid4()),
            "name": task.name,
            "description": task.description,
            "agent_id": task.agent_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        tasks_db.append(new_task)

        return TaskResponse(**new_task)

    @app.get("/api/v1/status")
    async def system_status():
        """Get system status."""
        return {
            "system": {
                "name": f"{settings.app_name} - Consolidated",
                "version": settings.app_version,
                "status": "running",
                "structure": "Single app/ directory",
                "architecture": "Clean Architecture + DDD",
                "timestamp": datetime.now().isoformat(),
            },
            "components": {
                "domain": "healthy",
                "application": "healthy",
                "infrastructure": "healthy",
                "api": "healthy",
            },
            "statistics": {
                "agents": {
                    "total": len(agents_db),
                    "active": len([a for a in agents_db if a["status"] == "active"]),
                },
                "tasks": {
                    "total": len(tasks_db),
                    "pending": len([t for t in tasks_db if t["status"] == "pending"]),
                    "running": len([t for t in tasks_db if t["status"] == "running"]),
                    "completed": len([t for t in tasks_db if t["status"] == "completed"]),
                },
            },
        }

    @app.get("/api/v1/architecture")
    async def architecture_info():
        """Get architecture information."""
        return {
            "pattern": "Clean Architecture + Domain-Driven Design",
            "structure": "Single app/ directory",
            "consolidation": "All code in unified app/ structure",
            "layers": {
                "domain": "Business logic and models",
                "application": "Use cases and orchestration",
                "infrastructure": "External dependencies",
                "api": "REST endpoints",
                "cli": "Command-line interface",
            },
            "benefits": [
                "Single source of truth",
                "Clear import paths",
                "Reduced complexity",
                "Better organization",
                "Maintainable codebase",
            ],
        }

    return app
