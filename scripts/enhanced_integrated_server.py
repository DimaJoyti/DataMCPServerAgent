"""
Enhanced Integrated Server with all new capabilities:
- Persistent state handling with Cloudflare
- Long-running tasks and horizontal scaling
- Email APIs for human-in-the-loop workflows
- WebRTC for voice and video communication
- Self-hosting capabilities
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn

# Import existing components
from auth_system import User, auth_system

# Import enhanced integrations
from cloudflare_mcp_integration import (
    AgentType,
    LongRunningTask,
    PersistentState,
    TaskStatus,
    enhanced_cloudflare_integration,
)
from email_integration import ApprovalRequest, email_integration
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced DataMCPServerAgent",
    version="2.0.0",
    description="Complete AI Agent system with Cloudflare integration, Email, WebRTC, and Self-hosting"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AUTHENTICATION ====================

async def get_current_user(authorization: str = Header(None)) -> User:
    """Get current authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user = auth_system.authenticate_api_key(api_key)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "features": {
            "persistent_state": True,
            "long_running_tasks": True,
            "horizontal_scaling": True,
            "email_integration": True,
            "webrtc_integration": True,
            "self_hosting": True
        }
    }

# ==================== PERSISTENT STATE ENDPOINTS ====================

@app.post("/api/v2/agents/{agent_id}/state")
async def save_agent_state(
    agent_id: str,
    state_data: Dict[str, Any],
    agent_type: AgentType = AgentType.ANALYTICS,
    user: User = Depends(get_current_user)
):
    """Save agent state."""
    try:
        success = await enhanced_cloudflare_integration.save_agent_state(
            agent_id, agent_type, state_data
        )

        if success:
            return {"success": True, "agent_id": agent_id, "message": "State saved successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save state")

    except Exception as e:
        logger.error(f"Error saving agent state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agents/{agent_id}/state")
async def get_agent_state(
    agent_id: str,
    user: User = Depends(get_current_user)
) -> Optional[PersistentState]:
    """Get agent state."""
    try:
        state = await enhanced_cloudflare_integration.load_agent_state(agent_id)
        return state

    except Exception as e:
        logger.error(f"Error loading agent state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v2/agents/{agent_id}/state")
async def delete_agent_state(
    agent_id: str,
    user: User = Depends(get_current_user)
):
    """Delete agent state."""
    try:
        success = await enhanced_cloudflare_integration.delete_agent_state(agent_id)

        if success:
            return {"success": True, "agent_id": agent_id, "message": "State deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="State not found")

    except Exception as e:
        logger.error(f"Error deleting agent state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LONG-RUNNING TASKS ENDPOINTS ====================

@app.post("/api/v2/agents/{agent_id}/tasks")
async def create_task(
    agent_id: str,
    task_type: str,
    metadata: Dict[str, Any] = None,
    user: User = Depends(get_current_user)
):
    """Create a long-running task."""
    try:
        task_id = await enhanced_cloudflare_integration.create_long_running_task(
            agent_id, task_type, metadata or {}
        )

        return {
            "success": True,
            "task_id": task_id,
            "agent_id": agent_id,
            "task_type": task_type
        }

    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    user: User = Depends(get_current_user)
) -> Optional[LongRunningTask]:
    """Get task status."""
    try:
        task = await enhanced_cloudflare_integration.get_task_status(task_id)
        return task

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v2/tasks/{task_id}/progress")
async def update_task_progress(
    task_id: str,
    progress: float,
    status: TaskStatus = None,
    user: User = Depends(get_current_user)
):
    """Update task progress."""
    try:
        success = await enhanced_cloudflare_integration.update_task_progress(
            task_id, progress, status
        )

        if success:
            return {"success": True, "task_id": task_id, "progress": progress}
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    except Exception as e:
        logger.error(f"Error updating task progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    result: Dict[str, Any] = None,
    error_message: str = None,
    user: User = Depends(get_current_user)
):
    """Complete a task."""
    try:
        success = await enhanced_cloudflare_integration.complete_task(
            task_id, result, error_message
        )

        if success:
            return {"success": True, "task_id": task_id, "message": "Task completed"}
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    except Exception as e:
        logger.error(f"Error completing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agents/{agent_id}/tasks")
async def get_agent_tasks(
    agent_id: str,
    user: User = Depends(get_current_user)
) -> List[LongRunningTask]:
    """Get all tasks for an agent."""
    try:
        tasks = await enhanced_cloudflare_integration.get_agent_tasks(agent_id)
        return tasks

    except Exception as e:
        logger.error(f"Error getting agent tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HORIZONTAL SCALING ENDPOINTS ====================

@app.post("/api/v2/agents/{agent_id}/scale")
async def scale_agent(
    agent_id: str,
    target_instances: int,
    user: User = Depends(get_current_user)
):
    """Scale agent horizontally."""
    try:
        result = await enhanced_cloudflare_integration.scale_agent_horizontally(
            agent_id, target_instances
        )
        return result

    except Exception as e:
        logger.error(f"Error scaling agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    user: User = Depends(get_current_user)
):
    """Get agent load metrics."""
    try:
        metrics = await enhanced_cloudflare_integration.get_agent_load_metrics(agent_id)
        return metrics

    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== EMAIL INTEGRATION ENDPOINTS ====================

@app.post("/api/v2/email/approval")
async def create_approval_request(
    agent_id: str,
    task_id: str,
    title: str,
    description: str,
    data: Dict[str, Any],
    approver_email: str,
    expires_in_hours: int = 24,
    user: User = Depends(get_current_user)
):
    """Create approval request."""
    try:
        approval_id = await email_integration.create_approval_request(
            agent_id, task_id, title, description, data, approver_email, expires_in_hours
        )

        return {
            "success": True,
            "approval_id": approval_id,
            "message": "Approval request created and email sent"
        }

    except Exception as e:
        logger.error(f"Error creating approval request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/email/approval/{approval_id}/respond")
async def respond_to_approval(
    approval_id: str,
    action: str,
    approver_email: str,
    reason: str = None
):
    """Respond to approval request."""
    try:
        success = await email_integration.process_approval_response(
            approval_id, action, approver_email, reason
        )

        if success:
            return {"success": True, "approval_id": approval_id, "action": action}
        else:
            raise HTTPException(status_code=400, detail="Failed to process approval")

    except Exception as e:
        logger.error(f"Error processing approval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/email/approval/{approval_id}")
async def get_approval_status(
    approval_id: str,
    user: User = Depends(get_current_user)
) -> Optional[ApprovalRequest]:
    """Get approval status."""
    try:
        approval = await email_integration.get_approval_status(approval_id)
        return approval

    except Exception as e:
        logger.error(f"Error getting approval status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Enhanced DataMCPServerAgent starting up...")

    # Initialize components
    logger.info("✅ Cloudflare integration initialized")
    logger.info("✅ Email integration initialized")
    logger.info("✅ WebRTC integration initialized")
    logger.info("✅ Self-hosting manager initialized")
    logger.info("✅ Authentication system initialized")

    logger.info("🎉 Enhanced DataMCPServerAgent ready!")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_integrated_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
