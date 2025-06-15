"""
Playground router for agent-ui compatibility.
This router provides endpoints that match the agent-ui expectations.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ..middleware.auth import get_api_key
from ..services.agent_service import AgentService

router = APIRouter(prefix="/v1/playground", tags=["playground"])

# In-memory storage for sessions (in production, use Redis or database)
sessions_storage = {}


@router.post("/clear_sessions")
async def clear_all_sessions(api_key: Optional[str] = Depends(get_api_key)):
    """
    Clear all stored sessions.
    """
    sessions_storage.clear()
    return {"message": "All sessions cleared successfully"}


@router.get("/status")
async def get_playground_status():
    """
    Get playground status.
    """
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/agents")
async def get_playground_agents(
    api_key: Optional[str] = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    """
    Get all available agents in the format expected by agent-ui.
    """
    try:
        # Create an agent service
        agent_service = AgentService()

        # Get all available agent modes
        agents = await agent_service.list_agents()

        # Transform to agent-ui format
        playground_agents = []
        for agent in agents:
            playground_agents.append(
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "model": agent.model,
                    "storage": True,  # Enable storage for all agents
                    "description": agent.description,
                    "status": "active",
                }
            )

        return playground_agents
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/agents/{agent_id}/runs")
async def create_agent_run(
    agent_id: str,
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
):
    """
    Create a new agent run (chat with agent).
    This endpoint handles streaming responses for real-time chat.
    """
    try:
        # Parse the request body
        body = await request.json()
        messages = body.get("messages", [])
        session_id = body.get("session_id")

        if not messages:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Messages are required",
            )

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="No user message found",
            )

        # Create an agent service
        agent_service = AgentService()

        # Generate a session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Store session data
        if session_id not in sessions_storage:
            sessions_storage[session_id] = {
                "id": session_id,
                "agent_id": agent_id,
                "created_at": datetime.utcnow().isoformat(),
                "messages": [],
            }

        # Add user message to session
        sessions_storage[session_id]["messages"].extend(messages)

        async def generate_response():
            """Generate streaming response."""
            try:
                # Get agent response
                response = await agent_service.chat_with_agent(
                    agent_mode=agent_id, message=user_message, session_id=session_id
                )

                # Create response message
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("response", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Add to session
                sessions_storage[session_id]["messages"].append(assistant_message)

                # Stream the response in chunks
                content = response.get("response", "")
                chunk_size = 50  # Characters per chunk

                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]

                    # Format as server-sent event
                    event_data = {"type": "content", "content": chunk, "session_id": session_id}

                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

                # Send completion event
                completion_data = {
                    "type": "completion",
                    "session_id": session_id,
                    "message": assistant_message,
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as e:
                error_data = {"type": "error", "error": str(e), "session_id": session_id}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/agents/{agent_id}/sessions")
async def get_agent_sessions(
    agent_id: str,
    api_key: Optional[str] = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    """
    Get all sessions for an agent.
    """
    try:
        # Filter sessions by agent_id
        agent_sessions = []
        for session_id, session_data in sessions_storage.items():
            if session_data.get("agent_id") == agent_id:
                agent_sessions.append(
                    {
                        "id": session_id,
                        "agent_id": agent_id,
                        "created_at": session_data.get("created_at"),
                        "message_count": len(session_data.get("messages", [])),
                        "last_message": (
                            session_data.get("messages", [])[-1]
                            if session_data.get("messages")
                            else None
                        ),
                    }
                )

        return agent_sessions
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/agents/{agent_id}/sessions/{session_id}")
async def get_agent_session(
    agent_id: str,
    session_id: str,
    api_key: Optional[str] = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Get a specific session.
    """
    try:
        if session_id not in sessions_storage:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        session_data = sessions_storage[session_id]

        if session_data.get("agent_id") != agent_id:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found for agent {agent_id}",
            )

        return session_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/agents/{agent_id}/sessions/{session_id}")
async def delete_agent_session(
    agent_id: str,
    session_id: str,
    api_key: Optional[str] = Depends(get_api_key),
):
    """
    Delete a specific session.
    """
    try:
        if session_id not in sessions_storage:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        session_data = sessions_storage[session_id]

        if session_data.get("agent_id") != agent_id:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found for agent {agent_id}",
            )

        del sessions_storage[session_id]

        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
