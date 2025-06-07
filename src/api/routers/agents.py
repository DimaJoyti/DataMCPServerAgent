"""
Agents router for the API.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

from ..models.request_models import AgentRequest
from ..models.response_models import AgentResponse, ErrorResponse
from ..services.agent_service import AgentService
from ..middleware.auth import get_api_key
from ..config import config

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    api_key: Optional[str] = Depends(get_api_key),
) -> List[AgentResponse]:
    """
    List all available agent modes.
    """
    try:
        # Create an agent service
        agent_service = AgentService()

        # Get all available agent modes
        agents = await agent_service.list_agents()

        return agents
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@router.get("/{agent_mode}", response_model=AgentResponse)
async def get_agent(
    agent_mode: str = Path(..., description="Agent mode"),
    api_key: Optional[str] = Depends(get_api_key),
) -> AgentResponse:
    """
    Get information about a specific agent mode.

    - **agent_mode**: Agent mode to get information about
    """
    try:
        # Validate agent mode
        if agent_mode not in config.available_agent_modes:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Agent mode {agent_mode} not found",
            )

        # Create an agent service
        agent_service = AgentService()

        # Get agent information
        agent = await agent_service.get_agent(agent_mode)

        if not agent:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Agent mode {agent_mode} not found",
            )

        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: AgentRequest,
    api_key: Optional[str] = Depends(get_api_key),
) -> AgentResponse:
    """
    Create a new agent instance.

    - **agent_mode**: Agent mode to use
    - **config**: (Optional) Agent configuration
    """
    try:
        # Validate agent mode
        if request.agent_mode not in config.available_agent_modes:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Agent mode {request.agent_mode} not found",
            )

        # Create an agent service
        agent_service = AgentService()

        # Create a new agent instance
        agent = await agent_service.create_agent(
            agent_mode=request.agent_mode,
            agent_config=request.config,
        )

        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
