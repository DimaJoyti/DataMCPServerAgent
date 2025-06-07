"""
Consolidated FastAPI Server for DataMCPServerAgent.

Single, unified API server that consolidates all functionality
into a clean, maintainable structure.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.logging_improved import get_logger, setup_logging
from app.core.simple_config import SimpleSettings

logger = get_logger(__name__)

# Pydantic models for consolidated API
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    timestamp: str
    components: Dict[str, str]

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
    priority: str = "normal"

class TaskResponse(BaseModel):
    id: str
    name: str
    description: str
    agent_id: str
    status: str
    priority: str
    created_at: str

class SystemStatus(BaseModel):
    system: Dict[str, Any]
    components: Dict[str, str]
    statistics: Dict[str, Any]

# In-memory storage for demonstration
agents_db: List[Dict[str, Any]] = []
tasks_db: List[Dict[str, Any]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting Consolidated DataMCPServerAgent")

    # Initialize components
    logger.info("✅ Consolidated system initialized")

    yield

    # Cleanup
    logger.info("🛑 Shutting down Consolidated DataMCPServerAgent")
    logger.info("👋 Consolidated system shutdown complete")

def create_consolidated_app() -> FastAPI:
    """Create consolidated FastAPI application."""

    settings = SimpleSettings()
    setup_logging(settings)

    app = FastAPI(
        title=f"{settings.app_name} - Consolidated",
        description="Unified AI Agent System with Clean Architecture",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root endpoints
    @app.get("/")
    async def root():
        """Root endpoint for consolidated system."""
        return {
            "message": f"Welcome to {settings.app_name} - Consolidated",
            "version": settings.app_version,
            "description": "Unified AI Agent System with Clean Architecture",
            "structure": "Single app/ directory with Clean Architecture",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Consolidated health check."""
        return HealthResponse(
            status="healthy",
            app=f"{settings.app_name} - Consolidated",
            version=settings.app_version,
            timestamp=datetime.now().isoformat(),
            components={
                "api": "healthy",
                "domain": "healthy",
                "application": "healthy",
                "infrastructure": "healthy",
            },
        )

    # API v1 routes
    @app.get("/api/v1/agents", response_model=List[AgentResponse])
    async def list_agents():
        """List all agents in consolidated system."""
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
    async def create_agent(agent: AgentCreate):
        """Create new agent in consolidated system."""
        new_agent = {
            "id": str(uuid.uuid4()),
            "name": agent.name,
            "description": agent.description,
            "agent_type": agent.agent_type,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }

        agents_db.append(new_agent)

        logger.info(f"Created agent: {agent.name} ({new_agent['id'][:8]})")

        return AgentResponse(**new_agent)

    @app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
    async def get_agent(agent_id: str):
        """Get agent by ID from consolidated system."""
        agent = next((a for a in agents_db if a["id"] == agent_id), None)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentResponse(**agent)

    @app.delete("/api/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete agent from consolidated system."""
        global agents_db

        agent = next((a for a in agents_db if a["id"] == agent_id), None)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        agents_db = [a for a in agents_db if a["id"] != agent_id]

        logger.info(f"Deleted agent: {agent_id[:8]}")

        return {"message": "Agent deleted successfully", "agent_id": agent_id}

    @app.get("/api/v1/tasks", response_model=List[TaskResponse])
    async def list_tasks():
        """List all tasks in consolidated system."""
        return [
            TaskResponse(
                id=task["id"],
                name=task["name"],
                description=task["description"],
                agent_id=task["agent_id"],
                status=task["status"],
                priority=task["priority"],
                created_at=task["created_at"],
            )
            for task in tasks_db
        ]

    @app.post("/api/v1/tasks", response_model=TaskResponse)
    async def create_task(task: TaskCreate):
        """Create new task in consolidated system."""
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
            "priority": task.priority,
            "created_at": datetime.now().isoformat(),
        }

        tasks_db.append(new_task)

        logger.info(f"Created task: {task.name} ({new_task['id'][:8]})")

        return TaskResponse(**new_task)

    @app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str):
        """Get task by ID from consolidated system."""
        task = next((t for t in tasks_db if t["id"] == task_id), None)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskResponse(**task)

    @app.put("/api/v1/tasks/{task_id}/status")
    async def update_task_status(task_id: str, status: str):
        """Update task status in consolidated system."""
        task = next((t for t in tasks_db if t["id"] == task_id), None)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        valid_statuses = ["pending", "running", "completed", "failed"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}"
            )

        task["status"] = status
        task["updated_at"] = datetime.now().isoformat()

        logger.info(f"Updated task {task_id[:8]} status to {status}")

        return {"message": "Task status updated", "task": TaskResponse(**task)}

    @app.get("/api/v1/status", response_model=SystemStatus)
    async def system_status():
        """Get consolidated system status."""
        return SystemStatus(
            system={
                "name": f"{settings.app_name} - Consolidated",
                "version": settings.app_version,
                "status": "running",
                "architecture": "Clean Architecture + DDD",
                "structure": "Single app/ directory",
                "timestamp": datetime.now().isoformat(),
            },
            components={
                "domain": "healthy",
                "application": "healthy",
                "infrastructure": "healthy",
                "api": "healthy",
                "cli": "available",
            },
            statistics={
                "agents": {
                    "total": len(agents_db),
                    "active": len([a for a in agents_db if a["status"] == "active"]),
                },
                "tasks": {
                    "total": len(tasks_db),
                    "pending": len([t for t in tasks_db if t["status"] == "pending"]),
                    "running": len([t for t in tasks_db if t["status"] == "running"]),
                    "completed": len([t for t in tasks_db if t["status"] == "completed"]),
                    "failed": len([t for t in tasks_db if t["status"] == "failed"]),
                },
            },
        )

    # Architecture information endpoints
    @app.get("/api/v1/architecture")
    async def architecture_info():
        """Get consolidated architecture information."""
        return {
            "pattern": "Clean Architecture + Domain-Driven Design",
            "structure": "Single app/ directory",
            "layers": {
                "domain": "Business logic and models",
                "application": "Use cases and orchestration",
                "infrastructure": "External dependencies",
                "api": "REST endpoints and schemas",
                "cli": "Command-line interface",
            },
            "principles": [
                "Dependency Inversion",
                "Separation of Concerns",
                "Single Responsibility",
                "Domain-Driven Design",
            ],
            "benefits": [
                "Maintainable codebase",
                "Testable components",
                "Clear boundaries",
                "Scalable architecture",
            ],
        }

    # Error handlers
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions in consolidated system."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc) if settings.debug else "An error occurred",
                "system": "Consolidated DataMCPServerAgent",
                "timestamp": datetime.now().isoformat(),
            },
        )

    logger.info("🏗️ Consolidated FastAPI application created")

    return app
