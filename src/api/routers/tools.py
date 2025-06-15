"""
Tools router for the API.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST

from ..middleware.auth import get_api_key
from ..models.request_models import ToolRequest
from ..models.response_models import ToolResponse
from ..services.tool_service import ToolService

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/", response_model=List[Dict[str, Any]])
async def list_tools(
    agent_mode: Optional[str] = Query(None, description="Agent mode to filter tools"),
    api_key: Optional[str] = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    """
    List all available tools.

    - **agent_mode**: (Optional) Agent mode to filter tools
    """
    try:
        # Create a tool service
        tool_service = ToolService()

        # List all available tools
        tools = await tool_service.list_tools(agent_mode=agent_mode)

        return tools
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/execute", response_model=ToolResponse)
async def execute_tool(
    request: ToolRequest,
    api_key: Optional[str] = Depends(get_api_key),
) -> ToolResponse:
    """
    Execute a tool.

    - **tool_name**: Name of the tool to use
    - **tool_input**: Input for the tool
    - **session_id**: (Optional) Session ID for the tool operation
    - **agent_mode**: (Optional) Agent mode to use for the tool operation
    """
    try:
        # Create a tool service
        tool_service = ToolService()

        # Execute the tool
        response = await tool_service.execute_tool(
            tool_name=request.tool_name,
            tool_input=request.tool_input,
            session_id=request.session_id,
            agent_mode=request.agent_mode,
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
