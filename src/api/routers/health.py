"""
Health router for the API.
"""

import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from ..config import config
from ..models.response_models import HealthResponse
from ..services.health_service import HealthService

router = APIRouter(prefix="/health", tags=["health"])

# Store the start time for uptime calculation
start_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health of the API.
    """
    try:
        # Create a health service
        health_service = HealthService()

        # Check the health of components
        components = await health_service.check_components()

        # Calculate uptime
        uptime = time.time() - start_time

        return HealthResponse(
            status="ok",
            version=config.version,
            timestamp=datetime.now(),
            components=components,
            uptime=uptime,
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
