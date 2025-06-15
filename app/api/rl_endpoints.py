"""
API endpoints for Reinforcement Learning system.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.logging_improved import get_logger
from app.core.rl_integration import RLMode, get_rl_manager

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/rl", tags=["Reinforcement Learning"])


# Request/Response models
class RLRequest(BaseModel):
    """Request for RL processing."""
    request: str = Field(..., description="User request to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    mode: Optional[RLMode] = Field(None, description="RL mode to use")


class RLResponse(BaseModel):
    """Response from RL processing."""
    success: bool = Field(..., description="Whether processing was successful")
    response: str = Field(..., description="Generated response")
    response_time: float = Field(..., description="Response time in seconds")
    rl_mode: str = Field(..., description="RL mode used")
    action: Optional[int] = Field(None, description="Selected action")
    reward: Optional[float] = Field(None, description="Reward received")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explanation data")
    safety_info: Optional[Dict[str, Any]] = Field(None, description="Safety information")
    error: Optional[str] = Field(None, description="Error message if failed")


class TrainingRequest(BaseModel):
    """Request for RL training."""
    episodes: int = Field(1, ge=1, le=100, description="Number of episodes to train")
    mode: Optional[RLMode] = Field(None, description="RL mode to use")


class TrainingResponse(BaseModel):
    """Response from RL training."""
    success: bool = Field(..., description="Whether training was successful")
    episodes_completed: int = Field(..., description="Number of episodes completed")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    error: Optional[str] = Field(None, description="Error message if failed")


class SystemStatus(BaseModel):
    """RL system status."""
    initialized: bool = Field(..., description="Whether system is initialized")
    training: bool = Field(..., description="Whether training is active")
    mode: str = Field(..., description="Current RL mode")
    algorithm: str = Field(..., description="Current algorithm")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    config: Dict[str, Any] = Field(..., description="System configuration")


class PerformanceReport(BaseModel):
    """Performance report."""
    summary: Dict[str, Any] = Field(..., description="Performance summary")
    rl_config: Dict[str, Any] = Field(..., description="RL configuration")
    system_status: Dict[str, Any] = Field(..., description="System status")


# Dependency to get RL manager
def get_rl_manager_dep() -> Any:
    """Get RL manager dependency."""
    return get_rl_manager()


@router.get("/status", response_model=SystemStatus)
async def get_rl_status(
    manager = Depends(get_rl_manager_dep)
) -> SystemStatus:
    """Get RL system status."""
    try:
        status = manager.get_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"Error getting RL status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_rl_system(
    background_tasks: BackgroundTasks,
    manager = Depends(get_rl_manager_dep)
) -> Dict[str, str]:
    """Initialize the RL system."""
    try:
        if manager.is_initialized:
            return {"message": "RL system already initialized", "status": "success"}

        # Initialize in background
        background_tasks.add_task(manager.initialize)

        return {"message": "RL system initialization started", "status": "initializing"}
    except Exception as e:
        logger.error(f"Error initializing RL system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=RLResponse)
async def process_request(
    request: RLRequest,
    manager = Depends(get_rl_manager_dep)
) -> RLResponse:
    """Process a request using the RL system."""
    try:
        # Initialize if needed
        if not manager.is_initialized:
            await manager.initialize()

        # Process request
        result = await manager.process_request(request.request, request.context)

        return RLResponse(**result)
    except Exception as e:
        logger.error(f"Error processing RL request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainingResponse)
async def train_rl_system(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    manager = Depends(get_rl_manager_dep)
) -> TrainingResponse:
    """Train the RL system."""
    try:
        # Initialize if needed
        if not manager.is_initialized:
            await manager.initialize()

        if not manager.config.training_enabled:
            raise HTTPException(status_code=400, detail="Training is disabled")

        # Train episodes
        episodes_completed = 0
        all_metrics = []

        for episode in range(request.episodes):
            try:
                metrics = await manager.train_episode()

                if "error" in metrics:
                    logger.warning(f"Training episode {episode + 1} failed: {metrics['error']}")
                    break

                all_metrics.append(metrics)
                episodes_completed += 1

            except Exception as e:
                logger.error(f"Error in training episode {episode + 1}: {e}")
                break

        # Aggregate metrics
        aggregated_metrics = {}
        if all_metrics:
            # Simple aggregation - can be enhanced
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)):
                    values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                    if values:
                        aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
                        aggregated_metrics[f"total_{key}"] = sum(values)

        aggregated_metrics["episodes"] = episodes_completed
        aggregated_metrics["individual_metrics"] = all_metrics

        return TrainingResponse(
            success=episodes_completed > 0,
            episodes_completed=episodes_completed,
            metrics=aggregated_metrics,
            error=None if episodes_completed > 0 else "No episodes completed successfully"
        )

    except Exception as e:
        logger.error(f"Error training RL system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceReport)
async def get_performance_report(
    manager = Depends(get_rl_manager_dep)
) -> PerformanceReport:
    """Get detailed performance report."""
    try:
        report = manager.get_performance_report()
        return PerformanceReport(**report)
    except Exception as e:
        logger.error(f"Error getting performance report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-model")
async def save_model(
    background_tasks: BackgroundTasks,
    manager = Depends(get_rl_manager_dep)
) -> Dict[str, str]:
    """Save the current RL model."""
    try:
        if not manager.is_initialized:
            raise HTTPException(status_code=400, detail="RL system not initialized")

        # Save model in background
        background_tasks.add_task(manager.save_model)

        return {"message": "Model save initiated", "status": "saving"}
    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modes")
async def get_available_modes() -> Dict[str, List[str]]:
    """Get available RL modes."""
    return {
        "modes": [mode.value for mode in RLMode],
        "descriptions": {
            "basic": "Basic Q-learning and policy gradient methods",
            "advanced": "Advanced RL with experience replay and target networks",
            "multi_objective": "Multi-objective optimization",
            "hierarchical": "Hierarchical RL with temporal abstraction",
            "modern_deep": "Modern deep RL algorithms (DQN, PPO, A2C)",
            "rainbow": "Rainbow DQN with all improvements",
            "multi_agent": "Multi-agent cooperative and competitive learning",
            "curriculum": "Curriculum learning with progressive difficulty",
            "meta_learning": "Meta-learning for fast adaptation (MAML)",
            "distributed": "Distributed training with multiple workers",
            "safe": "Safe RL with constraints and risk management",
            "explainable": "Explainable RL with interpretable decisions",
        }
    }


@router.get("/config")
async def get_rl_config(
    manager = Depends(get_rl_manager_dep)
) -> Dict[str, Any]:
    """Get current RL configuration."""
    try:
        config = manager.config
        return {
            "mode": config.mode.value,
            "algorithm": config.algorithm,
            "state_representation": config.state_representation,
            "training_enabled": config.training_enabled,
            "safety_enabled": config.safety_enabled,
            "explanation_enabled": config.explanation_enabled,
            "distributed_workers": config.distributed_workers,
            "num_agents": config.num_agents,
            "cooperation_mode": config.cooperation_mode,
            "max_resource_usage": config.max_resource_usage,
            "max_response_time": config.max_response_time,
            "safety_weight": config.safety_weight,
        }
    except Exception as e:
        logger.error(f"Error getting RL config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_rl_system(
    manager = Depends(get_rl_manager_dep)
) -> Dict[str, str]:
    """Reset the RL system."""
    try:
        # Reset performance metrics
        manager.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "average_reward": 0.0,
            "training_episodes": 0,
        }

        # Reset training state
        manager.is_training = False

        return {"message": "RL system reset successfully", "status": "reset"}
    except Exception as e:
        logger.error(f"Error resetting RL system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    manager = Depends(get_rl_manager_dep)
) -> Dict[str, Any]:
    """Health check for RL system."""
    try:
        status = manager.get_status()

        health_status = "healthy" if status["initialized"] else "unhealthy"

        return {
            "status": health_status,
            "initialized": status["initialized"],
            "mode": status["mode"],
            "total_requests": status["performance_metrics"]["total_requests"],
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error in RL health check: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# WebSocket endpoint for real-time RL interaction
@router.websocket("/ws")
async def websocket_rl_interaction(websocket):
    """WebSocket endpoint for real-time RL interaction."""
    await websocket.accept()

    try:
        manager = get_rl_manager()

        # Initialize if needed
        if not manager.is_initialized:
            await websocket.send_json({
                "type": "status",
                "message": "Initializing RL system..."
            })
            await manager.initialize()
            await websocket.send_json({
                "type": "status",
                "message": "RL system initialized"
            })

        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "request":
                # Process request
                request = data.get("request", "")
                context = data.get("context", {})

                result = await manager.process_request(request, context)

                await websocket.send_json({
                    "type": "response",
                    "data": result
                })

            elif data.get("type") == "train":
                # Train episode
                metrics = await manager.train_episode()

                await websocket.send_json({
                    "type": "training_result",
                    "data": metrics
                })

            elif data.get("type") == "status":
                # Get status
                status = manager.get_status()

                await websocket.send_json({
                    "type": "status_response",
                    "data": status
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {data.get('type')}"
                })

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()
