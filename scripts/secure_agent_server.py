"""
Secure Integrated Agent Server with environment-based configuration.
"""

import uuid
from datetime import datetime
from typing import Optional

import uvicorn
from auth_system import Role, User, auth_system
from cloudflare_workflows import EventType, WorkflowEvent, workflow_engine
from durable_objects_agent import durable_manager
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp_inspector import mcp_inspector
from secure_config import config, logger

# Initialize FastAPI with secure configuration
app = FastAPI(
    title="Secure Cloudflare AI Agents",
    version="2.0.0",
    description="Production-ready Agent system with MCP, Auth, Workflows, and Durable Objects",
    debug=config.debug,
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None
)

# Security middleware would be added here in production
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=[...])

# CORS middleware with secure origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting (would use Redis in production)
request_counts = {}

async def rate_limit_check(request: Request):
    """Simple rate limiting check."""
    client_ip = request.client.host
    current_time = datetime.utcnow().timestamp()

    if client_ip not in request_counts:
        request_counts[client_ip] = []

    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < config.security.rate_limit_window
    ]

    # Check rate limit
    if len(request_counts[client_ip]) >= config.security.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    request_counts[client_ip].append(current_time)

# Authentication dependency with rate limiting
async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> User:
    """Get current authenticated user with rate limiting."""
    await rate_limit_check(request)

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    api_key = authorization[7:]
    user = auth_system.authenticate_api_key(api_key)

    if not user:
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info(f"User {user.username} authenticated from {request.client.host}")
    return user

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Secure Cloudflare AI Agents",
        "version": "2.0.0",
        "environment": config.environment,
        "features": [
            "Secure MCP Inspector",
            "Environment-based Configuration",
            "Rate Limiting",
            "Cloudflare Workflows",
            "Durable Execution",
            "Enhanced Security"
        ],
        "endpoints": {
            "auth": "/v1/auth/*",
            "agents": "/v1/agents/*",
            "tools": "/v1/tools/*",
            "workflows": "/v1/workflows/*",
            "inspector": "/v1/inspector/*",
            "playground": "/v1/playground/*"
        },
        "security": {
            "rate_limit": f"{config.security.rate_limit_requests} requests per {config.security.rate_limit_window}s",
            "encryption": "enabled",
            "cors_origins": len(config.security.allowed_origins)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": config.environment,
        "services": {
            "auth_system": "active",
            "mcp_inspector": "active",
            "durable_objects": "active",
            "workflow_engine": "active"
        }
    }

# Authentication endpoints
@app.post("/v1/auth/users")
async def create_user(request: Request):
    """Create a new user."""
    await rate_limit_check(request)

    body = await request.json()
    username = body.get("username")
    email = body.get("email")
    password = body.get("password")
    role = Role(body.get("role", "user"))

    if not all([username, email, password]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        user = auth_system.create_user(username, email, password, role)
        logger.info(f"User created: {username} ({role.value})")

        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "api_key": user.api_key,
            "created_at": user.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return auth_system.get_user_info(current_user.user_id)

# Workflow endpoints
@app.post("/v1/workflows")
async def create_workflow(request: Request, current_user: User = Depends(get_current_user)):
    """Create a new workflow."""
    body = await request.json()

    workflow_name = body.get("name", "Agent Workflow")
    steps = body.get("steps", [])
    context = body.get("context", {})
    timeout_seconds = body.get("timeout_seconds")

    if not steps:
        raise HTTPException(status_code=400, detail="Workflow steps are required")

    try:
        workflow_id = await workflow_engine.create_workflow(
            name=workflow_name,
            steps=steps,
            context=context,
            timeout_seconds=timeout_seconds,
            user_id=current_user.user_id
        )

        logger.info(f"Workflow created: {workflow_id} by {current_user.username}")

        return {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "status": "created",
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str, current_user: User = Depends(get_current_user)):
    """Start workflow execution."""
    workflow = workflow_engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.user_id != current_user.user_id and current_user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    success = await workflow_engine.start_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start workflow")

    logger.info(f"Workflow started: {workflow_id} by {current_user.username}")

    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Workflow execution started"
    }

@app.post("/v1/workflows/{workflow_id}/events")
async def send_workflow_event(
    workflow_id: str,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Send an event to a workflow (waitForEvent API)."""
    body = await request.json()

    event_type = body.get("event_type")
    payload = body.get("payload", {})
    source = body.get("source", "api")

    if not event_type:
        raise HTTPException(status_code=400, detail="Event type is required")

    try:
        event = WorkflowEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            event_type=EventType(event_type),
            payload=payload,
            timestamp=datetime.utcnow(),
            source=source,
            metadata={"user_id": current_user.user_id}
        )

        success = await workflow_engine.send_event(workflow_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")

        logger.info(f"Event sent to workflow {workflow_id}: {event_type}")

        return {
            "event_id": event.event_id,
            "workflow_id": workflow_id,
            "event_type": event_type,
            "status": "sent"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Failed to send event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str, current_user: User = Depends(get_current_user)):
    """Get workflow status and details."""
    workflow = workflow_engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.user_id != current_user.user_id and current_user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "status": workflow.status.value,
        "steps": [
            {
                "step_id": step.step_id,
                "name": step.name,
                "status": step.status.value,
                "started_at": step.started_at.isoformat() if step.started_at else None,
                "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                "error": step.error
            }
            for step in workflow.steps
        ],
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
        "events_count": len(workflow.events)
    }

@app.get("/v1/workflows")
async def list_workflows(current_user: User = Depends(get_current_user)):
    """List workflows for the current user."""
    if current_user.role == Role.ADMIN:
        workflows = workflow_engine.list_workflows()
    else:
        workflows = workflow_engine.list_workflows(user_id=current_user.user_id)

    return {
        "workflows": [
            {
                "workflow_id": w.workflow_id,
                "name": w.name,
                "status": w.status.value,
                "created_at": w.created_at.isoformat(),
                "user_id": w.user_id
            }
            for w in workflows
        ],
        "count": len(workflows)
    }

# Enhanced inspector endpoints with security
@app.get("/v1/inspector/stats")
async def get_system_stats(current_user: User = Depends(get_current_user)):
    """Get comprehensive system statistics (admin only)."""
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
        "workflow_engine": {
            "total_workflows": len(workflow_engine.workflows),
            "active_workflows": len([w for w in workflow_engine.workflows.values() if w.status.value in ["running", "waiting"]]),
            "completed_workflows": len([w for w in workflow_engine.workflows.values() if w.status.value == "completed"])
        },
        "system_health": {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": config.environment
        }
    }

# Playground compatibility
@app.get("/v1/playground/status")
async def get_playground_status():
    """Get playground status."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "features": ["secure_mcp", "workflows", "durable_execution"],
        "version": "2.0.0",
        "environment": config.environment
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Secure Cloudflare AI Agents Server...")
    logger.info(f"🔧 Environment: {config.environment}")
    logger.info(f"🔐 Security: Rate limiting enabled ({config.security.rate_limit_requests} req/{config.security.rate_limit_window}s)")
    logger.info(f"🌐 CORS Origins: {len(config.security.allowed_origins)}")
    logger.info(f"📊 Observability: Metrics={config.observability.metrics_enabled}, Tracing={config.observability.tracing_enabled}")

    uvicorn.run(
        "secure_agent_server:app",
        host="0.0.0.0",
        port=8003,
        reload=config.debug,
        log_level=config.observability.log_level.lower()
    )
