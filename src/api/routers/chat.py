"""
Chat router for the API.
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ..middleware.auth import get_api_key
from ..models.request_models import ChatRequest
from ..models.response_models import ChatResponse
from ..services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key),
) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    - **message**: The message to send to the agent
    - **session_id**: (Optional) Session ID for continuing a conversation
    - **agent_mode**: (Optional) Agent mode to use
    - **user_id**: (Optional) User ID for personalized responses
    - **context**: (Optional) Additional context for the agent
    - **stream**: (Optional) Whether to stream the response
    """
    try:
        # If streaming is requested, use the streaming endpoint
        if request.stream:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="For streaming responses, use the /chat/stream endpoint",
            )

        # Create a chat service
        chat_service = ChatService()

        # Generate a session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process the chat request
        response = await chat_service.process_chat(
            message=request.message,
            session_id=session_id,
            agent_mode=request.agent_mode,
            user_id=request.user_id,
            context=request.context,
        )

        # Add any background tasks if needed
        background_tasks.add_task(
            chat_service.log_interaction, session_id, request.message, response
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/stream", response_model=None)
async def chat_stream(
    request: ChatRequest,
    api_key: Optional[str] = Depends(get_api_key),
) -> StreamingResponse:
    """
    Send a message to the agent and get a streaming response.

    - **message**: The message to send to the agent
    - **session_id**: (Optional) Session ID for continuing a conversation
    - **agent_mode**: (Optional) Agent mode to use
    - **user_id**: (Optional) User ID for personalized responses
    - **context**: (Optional) Additional context for the agent
    """
    try:
        # Ensure streaming is enabled
        if not request.stream:
            request.stream = True

        # Create a chat service
        chat_service = ChatService()

        # Generate a session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Create a streaming response
        return StreamingResponse(
            chat_service.stream_chat(
                message=request.message,
                session_id=session_id,
                agent_mode=request.agent_mode,
                user_id=request.user_id,
                context=request.context,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/sessions/{session_id}", response_model=List[ChatResponse])
async def get_chat_history(
    session_id: str = Path(..., description="Session ID"),
    limit: int = Query(10, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Offset for pagination"),
    api_key: Optional[str] = Depends(get_api_key),
) -> List[ChatResponse]:
    """
    Get chat history for a session.

    - **session_id**: Session ID
    - **limit**: (Optional) Maximum number of messages to return
    - **offset**: (Optional) Offset for pagination
    """
    try:
        # Create a chat service
        chat_service = ChatService()

        # Get chat history
        history = await chat_service.get_chat_history(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

        if not history:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No chat history found for session {session_id}",
            )

        return history
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
