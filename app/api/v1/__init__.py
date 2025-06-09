"""
API v1 endpoints.
"""

from fastapi import APIRouter

from .agents import router as agents_router
from .brand_agents import router as brand_agents_router
from .communication import router as communication_router
from .deployment import router as deployment_router
from .state import router as state_router
from .tasks import router as tasks_router

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(brand_agents_router, prefix="/brand-agents", tags=["brand-agents"])
api_router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
api_router.include_router(state_router, prefix="/state", tags=["state"])
api_router.include_router(communication_router, prefix="/communication", tags=["communication"])
api_router.include_router(deployment_router, prefix="/deployment", tags=["deployment"])

__all__ = ["api_router"]
