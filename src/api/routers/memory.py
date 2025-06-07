"""
Memory router for the API.
"""

from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

from ..models.request_models import MemoryRequest
from ..models.response_models import MemoryResponse, ErrorResponse
from ..services.memory_service import MemoryService
from ..middleware.auth import get_api_key

router = APIRouter(prefix="/memory", tags=["memory"])

@router.post("/", response_model=MemoryResponse)
async def store_memory(
    request: MemoryRequest,
    api_key: Optional[str] = Depends(get_api_key),
) -> MemoryResponse:
    """
    Store a memory item.

    - **session_id**: Session ID for the memory
    - **memory_item**: Memory item to store
    - **memory_backend**: (Optional) Memory backend to use
    """
    try:
        # Validate request
        if not request.memory_item:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Memory item is required",
            )

        # Create a memory service
        memory_service = MemoryService()

        # Store the memory item
        response = await memory_service.store_memory(
            session_id=request.session_id,
            memory_item=request.memory_item,
            memory_backend=request.memory_backend,
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@router.get("/{session_id}", response_model=MemoryResponse)
async def retrieve_memory(
    session_id: str = Path(..., description="Session ID"),
    query: Optional[str] = Query(None, description="Query for retrieving memory"),
    limit: int = Query(10, description="Maximum number of memory items to return"),
    offset: int = Query(0, description="Offset for pagination"),
    memory_backend: Optional[str] = Query(None, description="Memory backend to use"),
    api_key: Optional[str] = Depends(get_api_key),
) -> MemoryResponse:
    """
    Retrieve memory items for a session.

    - **session_id**: Session ID for the memory
    - **query**: (Optional) Query for retrieving memory
    - **limit**: (Optional) Maximum number of memory items to return
    - **offset**: (Optional) Offset for pagination
    - **memory_backend**: (Optional) Memory backend to use
    """
    try:
        # Create a memory service
        memory_service = MemoryService()

        # Retrieve memory items
        response = await memory_service.retrieve_memory(
            session_id=session_id,
            query=query,
            limit=limit,
            offset=offset,
            memory_backend=memory_backend,
        )

        if not response.memory_items:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No memory items found for session {session_id}",
            )

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@router.delete("/{session_id}", response_model=MemoryResponse)
async def clear_memory(
    session_id: str = Path(..., description="Session ID"),
    memory_backend: Optional[str] = Query(None, description="Memory backend to use"),
    api_key: Optional[str] = Depends(get_api_key),
) -> MemoryResponse:
    """
    Clear memory for a session.

    - **session_id**: Session ID for the memory
    - **memory_backend**: (Optional) Memory backend to use
    """
    try:
        # Create a memory service
        memory_service = MemoryService()

        # Clear memory
        response = await memory_service.clear_memory(
            session_id=session_id,
            memory_backend=memory_backend,
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
