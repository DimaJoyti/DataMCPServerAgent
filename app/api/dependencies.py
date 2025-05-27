"""
FastAPI dependencies for dependency injection.
"""

from typing import Optional

from fastapi import Header

from app.domain.services.agent_service import AgentScalingService, AgentService
from app.domain.services.communication_service import EmailService, WebRTCService
from app.domain.services.deployment_service import DeploymentService
from app.domain.services.state_service import StateService
from app.domain.services.task_service import TaskService


# Mock user for demonstration
class MockUser:
    def __init__(self):
        self.id = "demo_user"
        self.username = "demo"
        self.email = "demo@example.com"


async def get_current_user(authorization: Optional[str] = Header(None)) -> MockUser:
    """Get current authenticated user (mock implementation)."""
    # In production, this would validate JWT token or API key
    return MockUser()


async def get_agent_service() -> AgentService:
    """Get agent service instance."""
    return AgentService()


async def get_agent_scaling_service() -> AgentScalingService:
    """Get agent scaling service instance."""
    return AgentScalingService()


async def get_task_service() -> TaskService:
    """Get task service instance."""
    return TaskService()


async def get_state_service() -> StateService:
    """Get state service instance."""
    return StateService()


async def get_email_service() -> EmailService:
    """Get email service instance."""
    return EmailService()


async def get_webrtc_service() -> WebRTCService:
    """Get WebRTC service instance."""
    return WebRTCService()


async def get_deployment_service() -> DeploymentService:
    """Get deployment service instance."""
    return DeploymentService()
