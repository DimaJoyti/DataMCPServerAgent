"""
Integrated Agent Server with MCP, Authentication, Authorization, and Durable Objects.
The complete Agent Puzzle solution.
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
from typing import Optional
import uuid

from auth_system import auth_system, User, Role
from mcp_inspector import mcp_inspector
from durable_objects_agent import durable_manager
from secure_mcp_client import secure_mcp_client, ToolCall

app = FastAPI(
    title="Integrated Cloudflare AI Agents",
    version="1.0.0",
    description="Complete Agent system with MCP, Auth, and Durable Objects"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Get current authenticated user from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract API key from "Bearer <api_key>" format
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    api_key = authorization[7:]  # Remove "Bearer " prefix
    user = auth_system.authenticate_api_key(api_key)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Integrated Cloudflare AI Agents",
        "version": "1.0.0",
        "description": "Complete Agent system with MCP, Auth, and Durable Objects",
        "features": [
            "MCP Inspector for debugging",
            "Authentication & Authorization",
            "Durable Objects for agent state",
            "Secure tool access control",
            "Real-time observability"
        ],
        "endpoints": {
            "auth": "/v1/auth/*",
            "agents": "/v1/agents/*",
            "tools": "/v1/tools/*",
            "inspector": "/v1/inspector/*",
            "playground": "/v1/playground/*"
        }
    }

# Authentication endpoints
@app.post("/v1/auth/users")
async def create_user(request: Request):
    """Create a new user."""
    body = await request.json()
    username = body.get("username")
    email = body.get("email")
    password = body.get("password")
    role = Role(body.get("role", "user"))

    if not all([username, email, password]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        user = auth_system.create_user(username, email, password, role)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "api_key": user.api_key,
            "created_at": user.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return auth_system.get_user_info(current_user.user_id)

@app.get("/v1/auth/users")
async def list_users(current_user: User = Depends(get_current_user)):
    """List all users (admin only)."""
    if current_user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return auth_system.get_users_summary()

# Agent management endpoints
@app.post("/v1/agents")
async def create_agent(request: Request, current_user: User = Depends(get_current_user)):
    """Create a new agent instance."""
    body = await request.json()
    agent_type = body.get("agent_type", "cloudflare_worker")
    configuration = body.get("configuration", {})

    session_id = f"session_{uuid.uuid4().hex[:12]}"

    # Create agent using secure MCP client
    tool_call = ToolCall(
        tool_name="create_agent",
        parameters={
            "agent_type": agent_type,
            "configuration": configuration
        },
        user=current_user,
        session_id=session_id
    )

    result = await secure_mcp_client.call_tool(tool_call)

    if result["success"]:
        return result["result"]
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/v1/agents")
async def list_user_agents(current_user: User = Depends(get_current_user)):
    """List all agents for the current user."""
    tool_call = ToolCall(
        tool_name="list_user_agents",
        parameters={},
        user=current_user,
        session_id=f"session_{uuid.uuid4().hex[:8]}"
    )

    result = await secure_mcp_client.call_tool(tool_call)

    if result["success"]:
        return result["result"]
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/v1/agents/{agent_id}")
async def get_agent_info(agent_id: str, current_user: User = Depends(get_current_user)):
    """Get agent information."""
    tool_call = ToolCall(
        tool_name="get_agent_info",
        parameters={},
        user=current_user,
        session_id=f"session_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id
    )

    result = await secure_mcp_client.call_tool(tool_call)

    if result["success"]:
        return result["result"]
    else:
        raise HTTPException(status_code=404, detail=result["error"])

@app.delete("/v1/agents/{agent_id}")
async def terminate_agent(agent_id: str, current_user: User = Depends(get_current_user)):
    """Terminate an agent."""
    tool_call = ToolCall(
        tool_name="terminate_agent",
        parameters={},
        user=current_user,
        session_id=f"session_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id
    )

    result = await secure_mcp_client.call_tool(tool_call)

    if result["success"]:
        return result["result"]
    else:
        raise HTTPException(status_code=400, detail=result["error"])

# Tool execution endpoints
@app.post("/v1/tools/{tool_name}")
async def execute_tool(
    tool_name: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    agent_id: Optional[str] = None
):
    """Execute a tool with authentication and authorization."""
    body = await request.json()
    parameters = body.get("parameters", {})
    session_id = body.get("session_id", f"session_{uuid.uuid4().hex[:8]}")

    tool_call = ToolCall(
        tool_name=tool_name,
        parameters=parameters,
        user=current_user,
        session_id=session_id,
        agent_id=agent_id
    )

    result = await secure_mcp_client.call_tool(tool_call)

    if result["success"]:
        return result["result"]
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/v1/tools")
async def list_available_tools(current_user: User = Depends(get_current_user)):
    """List all available tools for the current user."""
    available_tools = []

    for tool_name, tool_func in secure_mcp_client.tools.items():
        # Check if user has access to this tool
        if auth_system.check_tool_access(current_user, tool_name):
            required_permission = secure_mcp_client.tool_permissions.get(tool_name)
            available_tools.append({
                "name": tool_name,
                "description": tool_func.__doc__ or "No description available",
                "required_permission": required_permission.value if required_permission else None,
                "accessible": True
            })

    return {
        "tools": available_tools,
        "count": len(available_tools),
        "user_role": current_user.role.value,
        "user_permissions": [p.value for p in current_user.permissions]
    }

# MCP Inspector endpoints
@app.get("/v1/inspector/events")
async def get_recent_events(
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get recent MCP events."""
    if current_user.role not in [Role.ADMIN, Role.DEVELOPER]:
        raise HTTPException(status_code=403, detail="Developer or admin access required")

    return {
        "events": mcp_inspector.get_recent_events(limit),
        "total_events": len(mcp_inspector.events)
    }

@app.get("/v1/inspector/sessions")
async def get_active_sessions(current_user: User = Depends(get_current_user)):
    """Get active sessions summary."""
    if current_user.role not in [Role.ADMIN, Role.DEVELOPER]:
        raise HTTPException(status_code=403, detail="Developer or admin access required")

    return mcp_inspector.get_active_sessions_summary()

@app.get("/v1/inspector/stats")
async def get_system_stats(current_user: User = Depends(get_current_user)):
    """Get comprehensive system statistics."""
    if current_user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return {
        "mcp_inspector": {
            "total_events": len(mcp_inspector.events),
            "active_sessions": len(mcp_inspector.active_sessions),
            "tool_usage_stats": mcp_inspector.get_tool_usage_stats(),
            "auth_failures": len(mcp_inspector.auth_failures)
        },
        "auth_system": auth_system.get_users_summary(),
        "durable_objects": await durable_manager.get_system_stats(),
        "system_health": {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@app.post("/v1/inspector/export")
async def export_events(current_user: User = Depends(get_current_user)):
    """Export MCP events to file."""
    if current_user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    filename = mcp_inspector.export_events()
    return {"filename": filename, "message": "Events exported successfully"}

# Playground compatibility endpoints
@app.get("/v1/playground/status")
async def get_playground_status():
    """Get playground status."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "features": ["mcp", "auth", "durable_objects"],
        "version": "1.0.0"
    }

@app.get("/v1/playground/agents")
async def get_playground_agents():
    """Get available agent types."""
    return [
        {
            "agent_id": "cloudflare_worker",
            "name": "Cloudflare Worker Agent",
            "model": "claude-3-5-sonnet",
            "storage": True,
            "description": "AI Agent with Cloudflare Workers integration",
            "status": "active",
            "features": ["mcp", "auth", "durable_objects"]
        },
        {
            "agent_id": "data_analytics",
            "name": "Data Analytics Agent",
            "model": "claude-3-5-sonnet",
            "storage": True,
            "description": "Analytics agent with D1 and R2 integration",
            "status": "active",
            "features": ["mcp", "auth", "durable_objects"]
        }
    ]

if __name__ == "__main__":
    print("🚀 Starting Integrated Cloudflare AI Agents Server...")
    print("📋 Features: MCP Inspector + Auth + Durable Objects")
    print("🔐 Default API Keys:")
    print(f"   Admin: {auth_system.users['admin_001'].api_key}")
    print(f"   Developer: {auth_system.users['dev_001'].api_key}")
    print(f"   User: {auth_system.users['user_001'].api_key}")
    print("🌐 Server starting on http://localhost:8002")

    uvicorn.run("integrated_agent_server:app", host="0.0.0.0", port=8002, reload=True)
