"""
Brand Agent API endpoints.
Provides REST API for managing brand agents, knowledge, and conversations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

from app.api.dependencies import (
    get_current_user,
    get_brand_agent_service,
    get_knowledge_service,
    get_conversation_service,
    get_conversation_engine,
    get_ai_response_service,
    get_knowledge_integration_service
)
from app.api.models.responses import PaginatedResponse, SuccessResponse
from app.domain.models.brand_agent import (
    BrandAgentType,
    BrandPersonality,
    BrandAgentConfiguration,
    ConversationChannel,
    KnowledgeType,
    PersonalityTrait,
)
from app.domain.models.conversation import MessageType, ConversationStatus
from app.domain.services.brand_agent_service import BrandAgentService, KnowledgeService, ConversationService
from app.domain.services.conversation_engine import ConversationEngine
from app.domain.services.ai_response_service import AIResponseService
from app.domain.services.knowledge_integration_service import KnowledgeIntegrationService
from app.domain.services.analytics_service import AnalyticsService
from app.domain.services.learning_service import LearningService
from app.domain.services.ab_testing_service import ABTestingService

router = APIRouter()

# Request Models
class CreateBrandAgentRequest(BaseModel):
    """Request model for creating a brand agent."""
    
    name: str = Field(description="Brand agent name")
    brand_id: str = Field(description="Brand/company ID")
    agent_type: BrandAgentType = Field(description="Type of brand agent")
    description: Optional[str] = Field(default=None, description="Agent description")
    personality: Optional[BrandPersonality] = Field(default=None, description="Agent personality")
    configuration: Optional[BrandAgentConfiguration] = Field(default=None, description="Agent configuration")


class UpdatePersonalityRequest(BaseModel):
    """Request model for updating agent personality."""
    
    traits: List[PersonalityTrait] = Field(description="Personality traits")
    tone: str = Field(description="Communication tone")
    communication_style: str = Field(description="Communication style")
    response_length: str = Field(description="Preferred response length")
    formality_level: str = Field(description="Formality level")
    emoji_usage: bool = Field(description="Whether to use emojis")
    custom_phrases: List[str] = Field(default_factory=list, description="Custom phrases")


class DeployAgentRequest(BaseModel):
    """Request model for deploying agent to channel."""
    
    channel: ConversationChannel = Field(description="Channel to deploy to")


class CreateKnowledgeRequest(BaseModel):
    """Request model for creating knowledge item."""
    
    title: str = Field(description="Knowledge title")
    content: str = Field(description="Knowledge content")
    knowledge_type: KnowledgeType = Field(description="Type of knowledge")
    brand_id: str = Field(description="Brand ID")
    tags: List[str] = Field(default_factory=list, description="Knowledge tags")
    priority: int = Field(default=1, ge=1, le=10, description="Priority (1-10)")
    source_url: Optional[str] = Field(default=None, description="Source URL")


class StartConversationRequest(BaseModel):
    """Request model for starting conversation."""
    
    agent_id: str = Field(description="Brand agent ID")
    channel: ConversationChannel = Field(description="Communication channel")
    user_id: Optional[str] = Field(default=None, description="User ID")


class AddMessageRequest(BaseModel):
    """Request model for adding message to conversation."""
    
    sender_type: str = Field(description="Sender type: 'user' or 'agent'")
    content: str = Field(description="Message content")
    message_type: str = Field(default="text", description="Message type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")


# Response Models
class BrandAgentResponse(BaseModel):
    """Response model for brand agent data."""
    
    id: str
    name: str
    brand_id: str
    agent_type: BrandAgentType
    description: Optional[str]
    is_active: bool
    is_deployed: bool
    deployment_channels: List[ConversationChannel]
    success_rate: float
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


class KnowledgeResponse(BaseModel):
    """Response model for knowledge data."""
    
    id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    tags: List[str]
    priority: int
    is_active: bool
    created_at: str
    last_updated: str
    
    class Config:
        from_attributes = True


class ConversationSessionResponse(BaseModel):
    """Response model for conversation session."""
    
    id: str
    brand_agent_id: str
    user_id: Optional[str]
    session_token: str
    channel: ConversationChannel
    status: str
    message_count: int
    started_at: str
    ended_at: Optional[str]
    user_satisfaction: Optional[int]
    
    class Config:
        from_attributes = True


class ConversationMessageResponse(BaseModel):
    """Response model for conversation message."""
    
    id: str
    session_id: str
    sender_type: str
    content: str
    message_type: str
    timestamp: str
    
    class Config:
        from_attributes = True


# Brand Agent Endpoints
@router.post("/", response_model=BrandAgentResponse)
async def create_brand_agent(
    request: CreateBrandAgentRequest,
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Create a new brand agent."""
    try:
        agent = await brand_agent_service.create_brand_agent(
            name=request.name,
            brand_id=request.brand_id,
            agent_type=request.agent_type,
            owner_id=current_user.id,
            personality=request.personality,
            configuration=request.configuration,
            description=request.description,
        )
        return BrandAgentResponse.from_orm(agent)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/", response_model=List[BrandAgentResponse])
async def list_brand_agents(
    brand_id: Optional[str] = Query(default=None, description="Filter by brand ID"),
    agent_type: Optional[BrandAgentType] = Query(default=None, description="Filter by agent type"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """List brand agents with filters."""
    try:
        # Build filters
        filters = {}
        if brand_id:
            filters["brand_id"] = brand_id
        if agent_type:
            filters["agent_type"] = agent_type
        if is_active is not None:
            filters["is_active"] = is_active
        
        # Get agents from repository
        agent_repo = brand_agent_service.get_repository("brand_agent")
        agents = await agent_repo.list(**filters)
        
        return [BrandAgentResponse.from_orm(agent) for agent in agents]
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{agent_id}", response_model=BrandAgentResponse)
async def get_brand_agent(
    agent_id: str = Path(..., description="Brand agent ID"),
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Get brand agent by ID."""
    try:
        agent_repo = brand_agent_service.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(agent_id)
        
        if not agent:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Brand agent not found")
        
        return BrandAgentResponse.from_orm(agent)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.put("/{agent_id}/personality", response_model=BrandAgentResponse)
async def update_agent_personality(
    agent_id: str = Path(..., description="Brand agent ID"),
    request: UpdatePersonalityRequest = ...,
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Update brand agent personality."""
    try:
        personality = BrandPersonality(
            traits=request.traits,
            tone=request.tone,
            communication_style=request.communication_style,
            response_length=request.response_length,
            formality_level=request.formality_level,
            emoji_usage=request.emoji_usage,
            custom_phrases=request.custom_phrases,
        )
        
        agent = await brand_agent_service.update_agent_personality(agent_id, personality)
        return BrandAgentResponse.from_orm(agent)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{agent_id}/deploy", response_model=BrandAgentResponse)
async def deploy_agent_to_channel(
    agent_id: str = Path(..., description="Brand agent ID"),
    request: DeployAgentRequest = ...,
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Deploy brand agent to a channel."""
    try:
        agent = await brand_agent_service.deploy_agent_to_channel(agent_id, request.channel)
        return BrandAgentResponse.from_orm(agent)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{agent_id}/performance")
async def get_agent_performance(
    agent_id: str = Path(..., description="Brand agent ID"),
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Get brand agent performance summary."""
    try:
        performance = await brand_agent_service.get_agent_performance_summary(agent_id)
        return performance
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/brands/{brand_id}/summary")
async def get_brand_summary(
    brand_id: str = Path(..., description="Brand ID"),
    brand_agent_service: BrandAgentService = Depends(get_brand_agent_service),
    current_user=Depends(get_current_user),
):
    """Get summary of all brand agents for a brand."""
    try:
        summary = await brand_agent_service.get_brand_agents_summary(brand_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Knowledge Management Endpoints
@router.post("/knowledge", response_model=KnowledgeResponse)
async def create_knowledge_item(
    request: CreateKnowledgeRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    current_user=Depends(get_current_user),
):
    """Create a new knowledge item."""
    try:
        knowledge = await knowledge_service.create_knowledge_item(
            title=request.title,
            content=request.content,
            knowledge_type=request.knowledge_type,
            brand_id=request.brand_id,
            tags=request.tags,
            priority=request.priority,
            source_url=request.source_url,
        )
        return KnowledgeResponse.from_orm(knowledge)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/knowledge/search")
async def search_knowledge(
    brand_id: str = Query(..., description="Brand ID"),
    query: str = Query(..., description="Search query"),
    knowledge_type: Optional[KnowledgeType] = Query(default=None, description="Filter by knowledge type"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    current_user=Depends(get_current_user),
):
    """Search knowledge items."""
    try:
        knowledge_items = await knowledge_service.search_knowledge(brand_id, query, knowledge_type)
        return [KnowledgeResponse.from_orm(item) for item in knowledge_items]
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Conversation Endpoints
@router.post("/conversations", response_model=ConversationSessionResponse)
async def start_conversation(
    request: StartConversationRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
    current_user=Depends(get_current_user),
):
    """Start a new conversation session."""
    try:
        session = await conversation_service.start_conversation(
            agent_id=request.agent_id,
            channel=request.channel,
            user_id=request.user_id,
        )
        return ConversationSessionResponse.from_orm(session)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/conversations/{session_id}/messages", response_model=ConversationMessageResponse)
async def add_message_to_conversation(
    session_id: str = Path(..., description="Conversation session ID"),
    request: AddMessageRequest = ...,
    conversation_service: ConversationService = Depends(get_conversation_service),
    current_user=Depends(get_current_user),
):
    """Add a message to a conversation."""
    try:
        message = await conversation_service.add_message_to_conversation(
            session_id=session_id,
            sender_type=request.sender_type,
            content=request.content,
            message_type=request.message_type,
            metadata=request.metadata,
        )
        return ConversationMessageResponse.from_orm(message)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/conversations/{session_id}/end", response_model=ConversationSessionResponse)
async def end_conversation(
    session_id: str = Path(..., description="Conversation session ID"),
    satisfaction_rating: Optional[int] = Query(default=None, ge=1, le=5, description="User satisfaction rating"),
    conversation_service: ConversationService = Depends(get_conversation_service),
    current_user=Depends(get_current_user),
):
    """End a conversation session."""
    try:
        session = await conversation_service.end_conversation(session_id, satisfaction_rating)
        return ConversationSessionResponse.from_orm(session)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Conversation Engine Endpoints (Phase 2)
class StartLiveConversationRequest(BaseModel):
    """Request model for starting live conversation."""

    brand_agent_id: str = Field(description="Brand agent ID")
    channel: ConversationChannel = Field(description="Communication channel")
    user_id: Optional[str] = Field(default=None, description="User ID")
    initial_context: Optional[Dict[str, Any]] = Field(default=None, description="Initial context")


class SendMessageRequest(BaseModel):
    """Request model for sending message."""

    content: str = Field(description="Message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Message type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Message metadata")


class LiveConversationResponse(BaseModel):
    """Response model for live conversation."""

    id: str
    brand_agent_id: str
    user_id: Optional[str]
    session_token: str
    channel: ConversationChannel
    status: ConversationStatus
    started_at: str
    last_activity_at: str
    message_count: int
    duration_seconds: int

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Response model for conversation message."""

    id: str
    conversation_id: str
    sender_type: str
    content: str
    message_type: MessageType
    status: str
    timestamp: str
    response_time_ms: Optional[int]
    knowledge_sources: List[str]

    class Config:
        from_attributes = True


@router.post("/live-conversations", response_model=LiveConversationResponse)
async def start_live_conversation(
    request: StartLiveConversationRequest,
    conversation_engine: ConversationEngine = Depends(get_conversation_engine),
    current_user=Depends(get_current_user),
):
    """Start a new live conversation with real-time capabilities."""
    try:
        conversation = await conversation_engine.start_conversation(
            brand_agent_id=request.brand_agent_id,
            channel=request.channel,
            user_id=request.user_id,
            initial_context=request.initial_context,
        )
        return LiveConversationResponse.from_orm(conversation)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/live-conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message_to_live_conversation(
    conversation_id: str = Path(..., description="Live conversation ID"),
    request: SendMessageRequest = ...,
    conversation_engine: ConversationEngine = Depends(get_conversation_engine),
    current_user=Depends(get_current_user),
):
    """Send a message to a live conversation and get AI response."""
    try:
        message = await conversation_engine.process_user_message(
            conversation_id=conversation_id,
            content=request.content,
            message_type=request.message_type,
            metadata=request.metadata,
        )
        return MessageResponse.from_orm(message)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/live-conversations/{conversation_id}/status")
async def get_live_conversation_status(
    conversation_id: str = Path(..., description="Live conversation ID"),
    conversation_engine: ConversationEngine = Depends(get_conversation_engine),
    current_user=Depends(get_current_user),
):
    """Get live conversation status and metrics."""
    try:
        status = await conversation_engine.get_conversation_status(conversation_id)
        if not status:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Conversation not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/live-conversations/{conversation_id}/end")
async def end_live_conversation(
    conversation_id: str = Path(..., description="Live conversation ID"),
    reason: str = Query(default="user_ended", description="End reason"),
    satisfaction_rating: Optional[int] = Query(default=None, ge=1, le=5, description="User satisfaction"),
    conversation_engine: ConversationEngine = Depends(get_conversation_engine),
    current_user=Depends(get_current_user),
):
    """End a live conversation."""
    try:
        conversation = await conversation_engine.end_conversation(
            conversation_id=conversation_id,
            reason=reason,
            user_satisfaction=satisfaction_rating,
        )
        return {"message": "Conversation ended successfully", "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Knowledge Integration Endpoints
@router.get("/knowledge/search")
async def search_brand_knowledge(
    brand_id: str = Query(..., description="Brand ID"),
    query: str = Query(..., description="Search query"),
    knowledge_types: Optional[List[KnowledgeType]] = Query(default=None, description="Knowledge types filter"),
    limit: int = Query(default=5, ge=1, le=20, description="Result limit"),
    min_relevance: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score"),
    knowledge_service: KnowledgeIntegrationService = Depends(get_knowledge_integration_service),
    current_user=Depends(get_current_user),
):
    """Search brand knowledge with relevance scoring."""
    try:
        results = await knowledge_service.search_relevant_knowledge(
            query=query,
            brand_id=brand_id,
            knowledge_types=knowledge_types,
            limit=limit,
            min_relevance=min_relevance,
        )
        return [result.to_dict() for result in results]
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/knowledge/analytics/{brand_id}")
async def get_knowledge_analytics(
    brand_id: str = Path(..., description="Brand ID"),
    knowledge_service: KnowledgeIntegrationService = Depends(get_knowledge_integration_service),
    current_user=Depends(get_current_user),
):
    """Get knowledge usage analytics for a brand."""
    try:
        analytics = await knowledge_service.get_knowledge_analytics(brand_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/knowledge/gaps/{brand_id}")
async def get_knowledge_gaps(
    brand_id: str = Path(..., description="Brand ID"),
    days: int = Query(default=7, ge=1, le=30, description="Days to analyze"),
    knowledge_service: KnowledgeIntegrationService = Depends(get_knowledge_integration_service),
    current_user=Depends(get_current_user),
):
    """Get suggested knowledge gaps based on recent conversations."""
    try:
        # This would get recent conversations from your repository
        recent_conversations = []  # Placeholder

        gaps = await knowledge_service.suggest_knowledge_gaps(brand_id, recent_conversations)
        return gaps
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Analytics Endpoints (Phase 3)
@router.get("/analytics/dashboard/{scope}/{scope_id}")
async def get_analytics_dashboard(
    scope: str = Path(..., description="Analytics scope (agent, brand, global)"),
    scope_id: str = Path(..., description="Scope ID"),
    time_range: str = Query(default="day", description="Time range (hour, day, week, month)"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user=Depends(get_current_user),
):
    """Get comprehensive analytics dashboard data."""
    try:
        from datetime import datetime, timedelta, timezone
        from app.domain.models.analytics import AnalyticsScope

        # Parse time range
        now = datetime.now(timezone.utc)
        if time_range == "hour":
            start_time = now - timedelta(hours=1)
        elif time_range == "day":
            start_time = now - timedelta(days=1)
        elif time_range == "week":
            start_time = now - timedelta(weeks=1)
        elif time_range == "month":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)

        # Convert scope string to enum
        analytics_scope = AnalyticsScope(scope.upper())

        dashboard_data = await analytics_service.get_analytics_dashboard_data(
            scope=analytics_scope,
            scope_id=scope_id,
            time_range=(start_time, now)
        )

        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/analytics/performance/{agent_id}")
async def get_agent_performance(
    agent_id: str = Path(..., description="Agent ID"),
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user=Depends(get_current_user),
):
    """Get detailed performance analytics for an agent."""
    try:
        from datetime import datetime, timedelta, timezone

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        performance = await analytics_service.collect_agent_performance(
            agent_id=agent_id,
            period_start=start_time,
            period_end=end_time
        )

        return {
            "agent_id": agent_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "metrics": {
                "total_conversations": performance.total_conversations,
                "completed_conversations": performance.completed_conversations,
                "avg_satisfaction": performance.avg_satisfaction,
                "resolution_rate": performance.resolution_rate,
                "escalation_rate": performance.escalation_rate,
                "avg_response_time_ms": performance.avg_response_time_ms,
                "avg_conversation_duration": performance.avg_conversation_duration,
                "messages_per_conversation": performance.messages_per_conversation,
                "utilization_rate": performance.utilization_rate,
                "knowledge_usage_rate": performance.knowledge_usage_rate,
                "performance_score": performance.calculate_performance_score(),
            },
            "trends": {
                "satisfaction": performance.satisfaction_trend,
                "response_time": performance.response_time_trend,
                "volume": performance.volume_trend,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/analytics/system")
async def get_system_metrics(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user=Depends(get_current_user),
):
    """Get system-wide performance metrics."""
    try:
        metrics = await analytics_service.collect_system_metrics()

        return {
            "timestamp": metrics.timestamp.isoformat(),
            "system": {
                "total_active_conversations": metrics.total_active_conversations,
                "total_agents": metrics.total_agents,
                "active_agents": metrics.active_agents,
                "avg_system_response_time_ms": metrics.avg_system_response_time_ms,
                "system_uptime_percentage": metrics.system_uptime_percentage,
                "error_rate": metrics.error_rate,
            },
            "resources": {
                "cpu_usage_percentage": metrics.cpu_usage_percentage,
                "memory_usage_percentage": metrics.memory_usage_percentage,
                "database_connections": metrics.database_connections,
                "websocket_connections": metrics.websocket_connections,
            },
            "throughput": {
                "messages_per_minute": metrics.messages_per_minute,
                "conversations_started_per_hour": metrics.conversations_started_per_hour,
                "ai_requests_per_minute": metrics.ai_requests_per_minute,
            },
            "quality": {
                "avg_ai_response_quality": metrics.avg_ai_response_quality,
                "knowledge_hit_rate": metrics.knowledge_hit_rate,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# Learning Endpoints
@router.get("/learning/insights/{agent_id}")
async def get_learning_insights(
    agent_id: str = Path(..., description="Agent ID"),
    days: int = Query(default=30, ge=7, le=90, description="Days to analyze"),
    learning_service: LearningService = Depends(get_learning_service),
    current_user=Depends(get_current_user),
):
    """Get AI-generated learning insights for an agent."""
    try:
        # This would get conversation analytics from the analytics service
        conversations = []  # Placeholder

        insights = await learning_service.analyze_conversation_patterns(
            agent_id=agent_id,
            conversations=conversations,
            time_window_days=days
        )

        return {
            "agent_id": agent_id,
            "analysis_period_days": days,
            "insights": [insight.to_dict() for insight in insights],
            "recommendations": await learning_service.get_learning_recommendations(agent_id)
        }
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/learning/feedback")
async def submit_learning_feedback(
    agent_id: str = Query(..., description="Agent ID"),
    conversation_id: str = Query(..., description="Conversation ID"),
    feedback_type: str = Query(default="satisfaction", description="Feedback type"),
    rating: Optional[int] = Query(default=None, ge=1, le=5, description="Rating"),
    comments: Optional[str] = Query(default=None, description="Feedback comments"),
    learning_service: LearningService = Depends(get_learning_service),
    current_user=Depends(get_current_user),
):
    """Submit feedback for learning improvement."""
    try:
        feedback = {
            "type": feedback_type,
            "rating": rating,
            "comments": comments,
        }

        await learning_service.learn_from_feedback(
            agent_id=agent_id,
            conversation_id=conversation_id,
            user_feedback=feedback
        )

        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


# A/B Testing Endpoints
class CreateExperimentRequest(BaseModel):
    """Request model for creating A/B test experiment."""

    name: str = Field(description="Experiment name")
    description: str = Field(description="Experiment description")
    experiment_type: str = Field(description="Experiment type")
    control_config: Dict[str, Any] = Field(description="Control variant configuration")
    test_configs: List[Dict[str, Any]] = Field(description="Test variant configurations")
    target_sample_size: int = Field(default=1000, description="Target sample size")
    confidence_level: float = Field(default=0.95, description="Statistical confidence level")


@router.post("/experiments")
async def create_experiment(
    agent_id: str = Query(..., description="Agent ID"),
    request: CreateExperimentRequest = ...,
    ab_testing_service: ABTestingService = Depends(get_ab_testing_service),
    current_user=Depends(get_current_user),
):
    """Create a new A/B test experiment."""
    try:
        from app.domain.services.ab_testing_service import ExperimentType

        experiment = await ab_testing_service.create_experiment(
            name=request.name,
            description=request.description,
            experiment_type=ExperimentType(request.experiment_type),
            agent_id=agent_id,
            control_config=request.control_config,
            test_configs=request.test_configs,
            target_sample_size=request.target_sample_size,
            confidence_level=request.confidence_level,
        )

        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "variants": [
                {
                    "id": variant.id,
                    "name": variant.name,
                    "is_control": variant.is_control,
                    "traffic_percentage": variant.traffic_percentage,
                }
                for variant in experiment.variants
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str = Path(..., description="Experiment ID"),
    ab_testing_service: ABTestingService = Depends(get_ab_testing_service),
    current_user=Depends(get_current_user),
):
    """Start an A/B test experiment."""
    try:
        success = await ab_testing_service.start_experiment(experiment_id)
        if success:
            return {"message": "Experiment started successfully"}
        else:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Experiment not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str = Path(..., description="Experiment ID"),
    ab_testing_service: ABTestingService = Depends(get_ab_testing_service),
    current_user=Depends(get_current_user),
):
    """Get results for an A/B test experiment."""
    try:
        results = await ab_testing_service.get_experiment_results(experiment_id)
        if results:
            return results
        else:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Experiment not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
