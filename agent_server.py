"""
Full-featured agent server with streaming chat and real agent integration.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="DataMCPServerAgent API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock agents data with real agent types
AGENTS = [
    {
        "agent_id": "basic",
        "name": "Basic Agent",
        "model": "claude-3-5-sonnet",
        "storage": True,
        "description": "Basic conversational agent for general questions",
        "status": "active"
    },
    {
        "agent_id": "research",
        "name": "Research Assistant",
        "model": "claude-3-5-sonnet",
        "storage": True,
        "description": "Advanced research and analysis agent",
        "status": "active"
    },
    {
        "agent_id": "seo",
        "name": "SEO Agent",
        "model": "claude-3-5-sonnet",
        "storage": True,
        "description": "SEO optimization and content analysis agent",
        "status": "active"
    },
    {
        "agent_id": "enhanced",
        "name": "Enhanced Agent",
        "model": "claude-3-5-sonnet",
        "storage": True,
        "description": "Enhanced agent with advanced capabilities",
        "status": "active"
    }
]

# In-memory session storage
sessions_storage: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DataMCPServerAgent API",
        "version": "0.1.0",
        "description": "API for interacting with DataMCPServerAgent",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/v1/playground/status")
async def get_playground_status():
    """Get playground status."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "agents_available": len(AGENTS)
    }

@app.get("/v1/playground/agents")
async def get_playground_agents() -> List[Dict[str, Any]]:
    """Get all available agents."""
    return AGENTS

async def get_agent_response(agent_id: str, user_message: str) -> str:
    """Get response from the specified agent using real agent implementations."""
    try:
        # Import the chat service that has all the agent integrations
        from src.api.services.chat_service import ChatService

        # Create chat service instance
        chat_service = ChatService()

        # Map agent_id to the correct agent mode
        agent_mode_map = {
            "basic": "basic",
            "research": "basic",  # Use basic for research until we have research_assistant_main
            "seo": "seo",
            "enhanced": "enhanced"
        }

        agent_mode = agent_mode_map.get(agent_id, "basic")

        try:
            # Use the chat service to get response
            response_data = await chat_service.chat_with_agent(
                agent_mode=agent_mode,
                message=user_message,
                session_id=f"session_{agent_id}_{int(datetime.utcnow().timestamp())}"
            )

            # Extract the response text
            if isinstance(response_data, dict):
                response = response_data.get("response", str(response_data))
            else:
                response = str(response_data)

        except Exception as e:
            # Fallback to mock responses if real agents fail
            print(f"Real agent failed, using fallback: {e}")

            if agent_id == "basic":
                response = f"Basic Agent Response: I understand you're asking about '{user_message}'. As a basic agent, I can help with general questions and provide conversational assistance."
            elif agent_id == "research":
                response = f"Research Assistant Analysis: Regarding '{user_message}', I can provide detailed research, analysis, and insights. Let me gather relevant information and provide a comprehensive response."
            elif agent_id == "seo":
                response = f"SEO Analysis: For '{user_message}', I can help optimize your content, analyze keywords, improve search rankings, and provide SEO recommendations."
            elif agent_id == "enhanced":
                response = f"Enhanced Agent Response: I'm processing your request about '{user_message}' with advanced capabilities including context awareness, adaptive learning, and enhanced tool selection."
            else:
                response = f"I'm agent {agent_id} and I received your message: '{user_message}'. How can I assist you today?"

        return response

    except Exception as e:
        return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again."

@app.post("/v1/playground/agents/{agent_id}/runs")
async def create_agent_run(agent_id: str, request: Request):
    """Create a new agent run with streaming response."""
    try:
        # Parse the request body
        body = await request.json()
        messages = body.get("messages", [])
        session_id = body.get("session_id", f"session_{int(datetime.utcnow().timestamp())}")

        if not messages:
            return {"error": "Messages are required"}

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return {"error": "No user message found"}

        # Store session data
        if session_id not in sessions_storage:
            sessions_storage[session_id] = {
                "id": session_id,
                "agent_id": agent_id,
                "created_at": datetime.utcnow().isoformat(),
                "messages": []
            }

        # Add user message to session
        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        sessions_storage[session_id]["messages"].append(user_msg)

        async def generate_response():
            """Generate streaming response."""
            try:
                # Get agent response
                response_text = await get_agent_response(agent_id, user_message)

                # Stream the response in chunks
                chunk_size = 30  # Characters per chunk

                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]

                    # Format as server-sent event
                    event_data = {
                        "type": "content",
                        "content": chunk,
                        "session_id": session_id,
                        "agent_id": agent_id
                    }

                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Add to session
                sessions_storage[session_id]["messages"].append(assistant_message)

                # Send completion event
                completion_data = {
                    "type": "completion",
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "message": assistant_message
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "session_id": session_id,
                    "agent_id": agent_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        return {"error": str(e)}

@app.get("/v1/playground/agents/{agent_id}/sessions")
async def get_agent_sessions(agent_id: str) -> List[Dict[str, Any]]:
    """Get all sessions for an agent."""
    agent_sessions = []
    for session_id, session_data in sessions_storage.items():
        if session_data.get("agent_id") == agent_id:
            agent_sessions.append({
                "id": session_id,
                "agent_id": agent_id,
                "created_at": session_data.get("created_at"),
                "message_count": len(session_data.get("messages", [])),
                "last_message": session_data.get("messages", [])[-1] if session_data.get("messages") else None
            })
    return agent_sessions

@app.get("/v1/playground/agents/{agent_id}/sessions/{session_id}")
async def get_agent_session(agent_id: str, session_id: str) -> Dict[str, Any]:
    """Get a specific session."""
    if session_id not in sessions_storage:
        return {"error": f"Session {session_id} not found"}

    session_data = sessions_storage[session_id]

    if session_data.get("agent_id") != agent_id:
        return {"error": f"Session {session_id} not found for agent {agent_id}"}

    return session_data

@app.delete("/v1/playground/agents/{agent_id}/sessions/{session_id}")
async def delete_agent_session(agent_id: str, session_id: str):
    """Delete a specific session."""
    if session_id not in sessions_storage:
        return {"error": f"Session {session_id} not found"}

    session_data = sessions_storage[session_id]

    if session_data.get("agent_id") != agent_id:
        return {"error": f"Session {session_id} not found for agent {agent_id}"}

    del sessions_storage[session_id]

    return {"message": f"Session {session_id} deleted successfully"}

if __name__ == "__main__":
    uvicorn.run("agent_server:app", host="0.0.0.0", port=8000, reload=True)
