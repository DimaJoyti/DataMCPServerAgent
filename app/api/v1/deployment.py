"""
Deployment API endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_deployment_service
from app.api.models.responses import SuccessResponse
from app.domain.models.deployment import Environment
from app.domain.services.deployment_service import DeploymentService

router = APIRouter()


class CreateDeploymentRequest(BaseModel):
    """Request model for creating deployment."""

    name: str = Field(description="Deployment name")
    environment: Environment = Field(description="Target environment")
    deployment_type: str = Field(description="Deployment type")


@router.post("/", response_model=SuccessResponse)
async def create_deployment(
    request: CreateDeploymentRequest,
    deployment_service: DeploymentService = Depends(get_deployment_service),
    current_user=Depends(get_current_user),
):
    """Create a deployment configuration."""
    await deployment_service.create_deployment_config(
        name=request.name, environment=request.environment, deployment_type=request.deployment_type
    )

    return SuccessResponse(message="Deployment configuration created successfully")
