"""
Agent management API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_agent_scaling_service, get_agent_service, get_current_user
from app.api.models.requests import PaginationParams
from app.api.models.responses import PaginatedResponse, SuccessResponse
from app.core.logging import get_logger
from app.domain.models.agent import (
    AgentCapability,
    AgentConfiguration,
    AgentStatus,
    AgentType,
)
from app.domain.services.agent_service import AgentScalingService, AgentService

logger = get_logger(__name__)
router = APIRouter()

# Request models
class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    name: str = Field(description="Agent name")
    agent_type: AgentType = Field(description="Type of agent")
    description: Optional[str] = Field(default=None, description="Agent description")
    configuration: Optional[AgentConfiguration] = Field(
        default=None, description="Agent configuration"
    )

class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = Field(default=None, description="Agent name")
    description: Optional[str] = Field(default=None, description="Agent description")
    configuration: Optional[AgentConfiguration] = Field(
        default=None, description="Agent configuration"
    )

class ScaleAgentRequest(BaseModel):
    """Request model for scaling an agent."""

    target_instances: int = Field(description="Target number of instances", ge=0, le=10)

class AddCapabilityRequest(BaseModel):
    """Request model for adding a capability to an agent."""

    capability: AgentCapability = Field(description="Capability to add")

# Response models
class AgentResponse(BaseModel):
    """Response model for agent data."""

    id: str
    name: str
    agent_type: AgentType
    status: AgentStatus
    description: Optional[str]
    configuration: AgentConfiguration
    capabilities: List[AgentCapability]
    desired_instances: int
    current_instances: int
    created_at: str
    updated_at: str
    version: int

    class Config:
        from_attributes = True

class AgentMetricsResponse(BaseModel):
    """Response model for agent metrics."""

    agent_id: str
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    average_response_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    uptime_seconds: int
    is_healthy: bool
    last_heartbeat: Optional[str]

class ScalingRecommendationResponse(BaseModel):
    """Response model for scaling recommendations."""

    agent_id: str
    agent_name: str
    current_instances: int
    recommended_action: str
    recommended_instances: int
    reason: str
    priority: str

# Endpoints
@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: CreateAgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Create a new agent."""
    logger.info(f"Creating agent: {request.name}")

    agent = await agent_service.create_agent(
        name=request.name,
        agent_type=request.agent_type,
        configuration=request.configuration,
        description=request.description,
    )

    return AgentResponse.from_orm(agent)

@router.get("/", response_model=PaginatedResponse[AgentResponse])
async def list_agents(
    pagination: PaginationParams = Depends(),
    agent_type: Optional[AgentType] = Query(default=None, description="Filter by agent type"),
    status: Optional[AgentStatus] = Query(default=None, description="Filter by agent status"),
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """List agents with pagination and filters."""
    filters = {}
    if agent_type:
        filters["agent_type"] = agent_type
    if status:
        filters["status"] = status

    agents = await agent_service.get_repository("agent").list(
        limit=pagination.limit, offset=pagination.offset, **filters
    )

    total = await agent_service.get_repository("agent").count(**filters)

    return PaginatedResponse(
        items=[AgentResponse.from_orm(agent) for agent in agents],
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=(total + pagination.size - 1) // pagination.size,
    )

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Get agent by ID."""
    agent = await agent_service.get_repository("agent").get_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return AgentResponse.from_orm(agent)

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Update agent."""
    agent_repo = agent_service.get_repository("agent")
    agent = await agent_repo.get_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update fields
    if request.name is not None:
        agent.name = request.name
    if request.description is not None:
        agent.description = request.description
    if request.configuration is not None:
        agent.configuration = request.configuration

    updated_agent = await agent_repo.save(agent)
    return AgentResponse.from_orm(updated_agent)

@router.delete("/{agent_id}", response_model=SuccessResponse)
async def delete_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Delete agent."""
    success = await agent_service.get_repository("agent").delete(agent_id)

    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return SuccessResponse(message="Agent deleted successfully")

@router.post("/{agent_id}/scale", response_model=AgentResponse)
async def scale_agent(
    agent_id: str,
    request: ScaleAgentRequest,
    scaling_service: AgentScalingService = Depends(get_agent_scaling_service),
    current_user=Depends(get_current_user),
):
    """Scale agent to target number of instances."""
    logger.info(f"Scaling agent {agent_id} to {request.target_instances} instances")

    agent = await scaling_service.scale_agent(agent_id, request.target_instances)
    return AgentResponse.from_orm(agent)

@router.get("/{agent_id}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Get agent metrics."""
    agent = await agent_service.get_repository("agent").get_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return AgentMetricsResponse(
        agent_id=agent.id,
        tasks_completed=agent.metrics.tasks_completed,
        tasks_failed=agent.metrics.tasks_failed,
        success_rate=agent.metrics.success_rate,
        average_response_time_ms=agent.metrics.average_response_time_ms,
        cpu_usage_percent=agent.metrics.cpu_usage_percent,
        memory_usage_mb=agent.metrics.memory_usage_mb,
        uptime_seconds=agent.metrics.uptime_seconds,
        is_healthy=agent.metrics.is_healthy,
        last_heartbeat=(
            agent.metrics.last_heartbeat.isoformat() if agent.metrics.last_heartbeat else None
        ),
    )

@router.post("/{agent_id}/capabilities", response_model=AgentResponse)
async def add_capability(
    agent_id: str,
    request: AddCapabilityRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Add capability to agent."""
    agent_repo = agent_service.get_repository("agent")
    agent = await agent_repo.get_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent.add_capability(request.capability)
    updated_agent = await agent_repo.save(agent)

    return AgentResponse.from_orm(updated_agent)

@router.delete("/{agent_id}/capabilities/{capability_name}", response_model=AgentResponse)
async def remove_capability(
    agent_id: str,
    capability_name: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user=Depends(get_current_user),
):
    """Remove capability from agent."""
    agent_repo = agent_service.get_repository("agent")
    agent = await agent_repo.get_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    removed = agent.remove_capability(capability_name)
    if not removed:
        raise HTTPException(status_code=404, detail="Capability not found")

    updated_agent = await agent_repo.save(agent)
    return AgentResponse.from_orm(updated_agent)

@router.post("/auto-scale", response_model=List[AgentResponse])
async def auto_scale_agents(
    background_tasks: BackgroundTasks,
    scaling_service: AgentScalingService = Depends(get_agent_scaling_service),
    current_user=Depends(get_current_user),
):
    """Trigger auto-scaling for all agents."""
    logger.info("Triggering auto-scaling for all agents")

    # Run auto-scaling in background
    background_tasks.add_task(scaling_service.auto_scale_agents)

    return SuccessResponse(message="Auto-scaling triggered")

@router.get("/scaling/recommendations", response_model=List[ScalingRecommendationResponse])
async def get_scaling_recommendations(
    scaling_service: AgentScalingService = Depends(get_agent_scaling_service),
    current_user=Depends(get_current_user),
):
    """Get scaling recommendations for all agents."""
    recommendations = await scaling_service.get_scaling_recommendations()

    return [ScalingRecommendationResponse(**rec) for rec in recommendations]
