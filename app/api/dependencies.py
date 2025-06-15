"""
FastAPI dependencies for dependency injection.
"""

from typing import Optional

from fastapi import Header

from app.domain.services.ab_testing_service import ABTestingService
from app.domain.services.agent_service import AgentScalingService, AgentService
from app.domain.services.ai_response_service import AIResponseService
from app.domain.services.analytics_service import AnalyticsService
from app.domain.services.brand_agent_service import (
    BrandAgentService,
    ConversationService,
    KnowledgeService,
)
from app.domain.services.communication_service import EmailService, WebRTCService
from app.domain.services.conversation_engine import ConversationEngine
from app.domain.services.deployment_service import DeploymentService
from app.domain.services.knowledge_integration_service import KnowledgeIntegrationService
from app.domain.services.learning_service import LearningService
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


async def get_brand_agent_service() -> BrandAgentService:
    """Get brand agent service instance."""
    return BrandAgentService()


async def get_knowledge_service() -> KnowledgeService:
    """Get knowledge service instance."""
    return KnowledgeService()


async def get_conversation_service() -> ConversationService:
    """Get conversation service instance."""
    return ConversationService()


async def get_conversation_engine() -> ConversationEngine:
    """Get conversation engine instance."""
    return ConversationEngine()


async def get_ai_response_service() -> AIResponseService:
    """Get AI response service instance."""
    return AIResponseService()


async def get_knowledge_integration_service() -> KnowledgeIntegrationService:
    """Get knowledge integration service instance."""
    return KnowledgeIntegrationService()


async def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    return AnalyticsService()


async def get_learning_service() -> LearningService:
    """Get learning service instance."""
    return LearningService()


async def get_ab_testing_service() -> ABTestingService:
    """Get A/B testing service instance."""
    return ABTestingService()
