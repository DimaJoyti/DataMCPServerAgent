"""
State management API endpoints.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_state_service
from app.api.models.responses import SuccessResponse
from app.domain.models.state import StateType
from app.domain.services.state_service import StateService

router = APIRouter()


class SaveStateRequest(BaseModel):
    """Request model for saving state."""

    entity_id: str = Field(description="Entity ID")
    entity_type: str = Field(description="Entity type")
    state_type: StateType = Field(description="State type")
    state_data: Dict[str, Any] = Field(description="State data")


@router.post("/save", response_model=SuccessResponse)
async def save_state(
    request: SaveStateRequest,
    state_service: StateService = Depends(get_state_service),
    current_user=Depends(get_current_user),
):
    """Save entity state."""
    await state_service.save_state(
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        state_type=request.state_type,
        state_data=request.state_data,
        created_by=current_user.username,
    )

    return SuccessResponse(message="State saved successfully")
