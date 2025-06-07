"""
Semantic Agents API

Provides REST API endpoints for interacting with semantic agents,
managing agent coordination, and monitoring performance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from .base_semantic_agent import SemanticAgentConfig, SemanticContext
from .coordinator import SemanticCoordinator
from .performance import PerformanceTracker, CacheManager
from .scaling import AutoScaler, LoadBalancer
from .specialized_agents import (
    DataAnalysisAgent,
    DocumentProcessingAgent,
    KnowledgeExtractionAgent,
    ReasoningAgent,
    SearchAgent,
)

# Request/Response Models
class TaskRequest(BaseModel):
    """Request model for task execution."""

    task_description: str = Field(..., description="Description of the task to execute")
    agent_type: Optional[str] = Field(None, description="Preferred agent type")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    required_capabilities: Optional[List[str]] = Field(None, description="Required capabilities")
    priority: int = Field(1, description="Task priority (1-10)")
    collaborative: bool = Field(False, description="Use multiple agents")
    session_id: Optional[str] = Field(None, description="Session ID for sticky routing")

class TaskResponse(BaseModel):
    """Response model for task execution."""

    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    collaborative: bool = False

class AgentStatusResponse(BaseModel):
    """Response model for agent status."""

    agent_id: str
    name: str
    specialization: Optional[str]
    is_active: bool
    current_tasks: int
    capabilities: List[str]
    performance_metrics: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    """Response model for system status."""

    coordinator_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    scaling_status: Dict[str, Any]
    cache_stats: Dict[str, Any]
    registered_agents: int
    active_tasks: int

class AgentCreateRequest(BaseModel):
    """Request model for creating new agents."""

    agent_type: str = Field(..., description="Type of agent to create")
    name: Optional[str] = Field(None, description="Agent name")
    capabilities: Optional[List[str]] = Field(None, description="Agent capabilities")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")

# API Router
router = APIRouter(prefix="/semantic-agents", tags=["Semantic Agents"])

# Global instances (would be properly injected in production)
_coordinator: Optional[SemanticCoordinator] = None
_performance_tracker: Optional[PerformanceTracker] = None
_auto_scaler: Optional[AutoScaler] = None
_load_balancer: Optional[LoadBalancer] = None
_cache_manager: Optional[CacheManager] = None

logger = logging.getLogger("semantic_agents_api")

async def get_coordinator() -> SemanticCoordinator:
    """Dependency to get the semantic coordinator."""
    global _coordinator
    if _coordinator is None:
        raise HTTPException(status_code=503, detail="Semantic coordinator not initialized")
    return _coordinator

async def get_performance_tracker() -> PerformanceTracker:
    """Dependency to get the performance tracker."""
    global _performance_tracker
    if _performance_tracker is None:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    return _performance_tracker

async def get_cache_manager() -> CacheManager:
    """Dependency to get the cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

# API Endpoints

@router.post("/tasks/execute", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    coordinator: SemanticCoordinator = Depends(get_coordinator),
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> TaskResponse:
    """Execute a task using semantic agents."""

    # Check cache first
    cache_key = f"task:{hash(request.task_description)}"
    cached_result = await cache_manager.get(cache_key)

    if cached_result and not request.collaborative:
        logger.info(f"Returning cached result for task: {request.task_description[:50]}...")
        return TaskResponse(
            task_id=cached_result["task_id"],
            success=cached_result["success"],
            result=cached_result["result"],
            agent_id=cached_result.get("agent_id"),
            execution_time_ms=0,  # Cached result
        )

    # Start performance tracking
    operation_id = performance_tracker.start_operation(
        agent_id="coordinator",
        operation_type="task_execution",
        metadata={"task_description": request.task_description},
    )

    start_time = datetime.now()

    try:
        # Create semantic context
        context = None
        if request.context:
            context = SemanticContext(
                user_intent=request.task_description,
                context_data=request.context,
            )

        # Execute task
        result = await coordinator.execute_task(
            task_description=request.task_description,
            context=context,
            required_capabilities=request.required_capabilities,
            priority=request.priority,
            collaborative=request.collaborative,
        )

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Cache successful results
        if result.get("success") and not request.collaborative:
            await cache_manager.set(
                cache_key,
                {
                    "task_id": result.get("task_id"),
                    "success": result.get("success"),
                    "result": result.get("result"),
                    "agent_id": result.get("agent_id"),
                },
                ttl=3600,  # Cache for 1 hour
            )

        # End performance tracking
        performance_tracker.end_operation(
            operation_id,
            success=result.get("success", False),
            result_metadata={"execution_time_ms": execution_time},
        )

        return TaskResponse(
            task_id=result.get("task_id", "unknown"),
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            agent_id=result.get("agent_id"),
            execution_time_ms=execution_time,
            collaborative=result.get("collaborative", False),
        )

    except Exception as e:
        # End performance tracking with error
        performance_tracker.end_operation(
            operation_id,
            success=False,
            error_message=str(e),
        )

        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents", response_model=List[AgentStatusResponse])
async def list_agents(
    coordinator: SemanticCoordinator = Depends(get_coordinator),
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
) -> List[AgentStatusResponse]:
    """List all registered semantic agents."""

    agents = []

    for agent_id, agent in coordinator.registered_agents.items():
        # Get performance metrics
        perf_metrics = performance_tracker.get_agent_performance(agent_id)

        agents.append(AgentStatusResponse(
            agent_id=agent_id,
            name=agent.config.name,
            specialization=agent.config.specialization,
            is_active=agent.is_active,
            current_tasks=len(agent.current_tasks),
            capabilities=agent.config.capabilities,
            performance_metrics=perf_metrics,
        ))

    return agents

@router.get("/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    coordinator: SemanticCoordinator = Depends(get_coordinator),
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
) -> AgentStatusResponse:
    """Get status of a specific agent."""

    if agent_id not in coordinator.registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = coordinator.registered_agents[agent_id]
    perf_metrics = performance_tracker.get_agent_performance(agent_id)

    return AgentStatusResponse(
        agent_id=agent_id,
        name=agent.config.name,
        specialization=agent.config.specialization,
        is_active=agent.is_active,
        current_tasks=len(agent.current_tasks),
        capabilities=agent.config.capabilities,
        performance_metrics=perf_metrics,
    )

@router.post("/agents", response_model=AgentStatusResponse)
async def create_agent(
    request: AgentCreateRequest,
    coordinator: SemanticCoordinator = Depends(get_coordinator),
) -> AgentStatusResponse:
    """Create a new semantic agent."""

    # Create agent configuration
    config = SemanticAgentConfig(
        name=request.name or f"{request.agent_type}_agent",
        specialization=request.agent_type,
        capabilities=request.capabilities or [],
    )

    # Apply configuration overrides
    if request.config_overrides:
        for key, value in request.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create agent based on type
    agent = None
    if request.agent_type == "data_analysis":
        agent = DataAnalysisAgent(config)
    elif request.agent_type == "document_processing":
        agent = DocumentProcessingAgent(config)
    elif request.agent_type == "knowledge_extraction":
        agent = KnowledgeExtractionAgent(config)
    elif request.agent_type == "reasoning":
        agent = ReasoningAgent(config)
    elif request.agent_type == "search":
        agent = SearchAgent(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")

    # Initialize and register agent
    await agent.initialize()
    await coordinator.register_agent(agent)

    return AgentStatusResponse(
        agent_id=agent.config.agent_id,
        name=agent.config.name,
        specialization=agent.config.specialization,
        is_active=agent.is_active,
        current_tasks=len(agent.current_tasks),
        capabilities=agent.config.capabilities,
        performance_metrics={},
    )

@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    coordinator: SemanticCoordinator = Depends(get_coordinator),
) -> Dict[str, str]:
    """Delete a semantic agent."""

    if agent_id not in coordinator.registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Shutdown and unregister agent
    agent = coordinator.registered_agents[agent_id]
    await agent.shutdown()
    await coordinator.unregister_agent(agent_id)

    return {"message": f"Agent {agent_id} deleted successfully"}

@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    coordinator: SemanticCoordinator = Depends(get_coordinator),
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> SystemStatusResponse:
    """Get overall system status."""

    coordinator_status = coordinator.get_coordinator_status()
    performance_metrics = performance_tracker.get_system_performance()
    cache_stats = cache_manager.get_stats()

    # Get scaling status if available
    scaling_status = {}
    global _auto_scaler
    if _auto_scaler:
        scaling_status = _auto_scaler.get_scaling_status()

    return SystemStatusResponse(
        coordinator_status=coordinator_status,
        performance_metrics=performance_metrics,
        scaling_status=scaling_status,
        cache_stats=cache_stats,
        registered_agents=len(coordinator.registered_agents),
        active_tasks=len(coordinator.active_tasks),
    )

@router.get("/performance/bottlenecks")
async def get_performance_bottlenecks(
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
) -> Dict[str, Any]:
    """Get performance bottlenecks and optimization recommendations."""

    bottlenecks = performance_tracker.identify_bottlenecks()
    recommendations = performance_tracker.get_optimization_recommendations()

    return {
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "timestamp": datetime.now(),
    }

@router.post("/cache/clear")
async def clear_cache(
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> Dict[str, str]:
    """Clear the system cache."""

    await cache_manager.clear()
    return {"message": "Cache cleared successfully"}

@router.get("/cache/stats")
async def get_cache_stats(
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> Dict[str, Any]:
    """Get cache statistics."""

    return cache_manager.get_stats()

# Initialization function
async def initialize_semantic_agents_api(
    coordinator: SemanticCoordinator,
    performance_tracker: PerformanceTracker,
    auto_scaler: Optional[AutoScaler] = None,
    load_balancer: Optional[LoadBalancer] = None,
    cache_manager: Optional[CacheManager] = None,
) -> None:
    """Initialize the semantic agents API with required components."""

    global _coordinator, _performance_tracker, _auto_scaler, _load_balancer, _cache_manager

    _coordinator = coordinator
    _performance_tracker = performance_tracker
    _auto_scaler = auto_scaler
    _load_balancer = load_balancer
    _cache_manager = cache_manager or CacheManager()

    logger.info("Semantic agents API initialized")
