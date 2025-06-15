"""
Brand Agent domain models.
Defines models for brand agents, brand knowledge, personality, and conversations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import AggregateRoot, BaseEntity, BaseValueObject, DomainEvent, ValidationError


class BrandAgentType(str, Enum):
    """Types of brand agents."""

    CUSTOMER_SUPPORT = "customer_support"
    SALES_ASSISTANT = "sales_assistant"
    PRODUCT_EXPERT = "product_expert"
    BRAND_AMBASSADOR = "brand_ambassador"
    CONTENT_CREATOR = "content_creator"
    LEAD_QUALIFIER = "lead_qualifier"


class PersonalityTrait(str, Enum):
    """Personality traits for brand agents."""

    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    ENTHUSIASTIC = "enthusiastic"
    HELPFUL = "helpful"
    KNOWLEDGEABLE = "knowledgeable"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PERSUASIVE = "persuasive"


class ConversationChannel(str, Enum):
    """Channels where brand agents can operate."""

    WEBSITE_CHAT = "website_chat"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    MOBILE_APP = "mobile_app"
    VOICE_ASSISTANT = "voice_assistant"
    MESSAGING_PLATFORM = "messaging_platform"


class KnowledgeType(str, Enum):
    """Types of brand knowledge."""

    PRODUCT_INFO = "product_info"
    COMPANY_INFO = "company_info"
    FAQ = "faq"
    POLICIES = "policies"
    PROCEDURES = "procedures"
    BRAND_GUIDELINES = "brand_guidelines"
    COMPETITOR_INFO = "competitor_info"
    INDUSTRY_INSIGHTS = "industry_insights"


class BrandPersonality(BaseValueObject):
    """Brand agent personality configuration."""

    traits: List[PersonalityTrait] = Field(default_factory=list, description="Personality traits")
    tone: str = Field(default="professional", description="Communication tone")
    communication_style: str = Field(default="helpful", description="Communication style")
    response_length: str = Field(default="medium", description="Preferred response length")
    formality_level: str = Field(default="semi-formal", description="Formality level")
    emoji_usage: bool = Field(default=False, description="Whether to use emojis")
    custom_phrases: List[str] = Field(default_factory=list, description="Custom phrases to use")

    @field_validator("traits")
    @classmethod
    def validate_traits(cls, v):
        if len(v) > 5:
            raise ValueError("Maximum 5 personality traits allowed")
        return v


class BrandKnowledge(BaseEntity):
    """Brand knowledge item."""

    title: str = Field(description="Knowledge item title")
    content: str = Field(description="Knowledge content")
    knowledge_type: KnowledgeType = Field(description="Type of knowledge")
    tags: List[str] = Field(default_factory=list, description="Knowledge tags")
    priority: int = Field(default=1, ge=1, le=10, description="Knowledge priority (1-10)")
    is_active: bool = Field(default=True, description="Whether knowledge is active")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )

    def update_content(self, new_content: str) -> None:
        """Update knowledge content."""
        self.content = new_content
        self.last_updated = datetime.now(timezone.utc)
        self.version += 1


class ConversationSession(BaseEntity):
    """Conversation session between user and brand agent."""

    brand_agent_id: str = Field(description="Brand agent ID")
    user_id: Optional[str] = Field(default=None, description="User ID (if authenticated)")
    session_token: str = Field(description="Session token for anonymous users")
    channel: ConversationChannel = Field(description="Communication channel")
    status: str = Field(default="active", description="Session status")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Session start time"
    )
    ended_at: Optional[datetime] = Field(default=None, description="Session end time")
    message_count: int = Field(default=0, description="Number of messages in session")
    user_satisfaction: Optional[int] = Field(
        default=None, ge=1, le=5, description="User satisfaction rating (1-5)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    def add_message(self) -> None:
        """Increment message count."""
        self.message_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def end_session(self, satisfaction_rating: Optional[int] = None) -> None:
        """End the conversation session."""
        self.status = "ended"
        self.ended_at = datetime.now(timezone.utc)
        if satisfaction_rating:
            self.user_satisfaction = satisfaction_rating
        self.version += 1


class ConversationMessage(BaseEntity):
    """Individual message in a conversation."""

    session_id: str = Field(description="Conversation session ID")
    sender_type: str = Field(description="Sender type: 'user' or 'agent'")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Message timestamp"
    )
    message_type: str = Field(default="text", description="Message type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")

    @field_validator("sender_type")
    @classmethod
    def validate_sender_type(cls, v):
        if v not in ["user", "agent"]:
            raise ValueError("Sender type must be 'user' or 'agent'")
        return v


class BrandAgentConfiguration(BaseValueObject):
    """Brand agent configuration."""

    max_response_length: int = Field(default=500, description="Maximum response length")
    response_timeout_seconds: int = Field(default=30, description="Response timeout")
    supported_channels: List[ConversationChannel] = Field(
        default_factory=list, description="Supported channels"
    )
    knowledge_base_ids: List[str] = Field(
        default_factory=list, description="Associated knowledge base IDs"
    )
    escalation_triggers: List[str] = Field(
        default_factory=list, description="Triggers for human escalation"
    )
    business_hours: Dict[str, Any] = Field(
        default_factory=dict, description="Business hours configuration"
    )
    auto_responses: Dict[str, str] = Field(
        default_factory=dict, description="Automatic responses for common scenarios"
    )


class BrandAgentMetrics(BaseValueObject):
    """Brand agent performance metrics."""

    total_conversations: int = Field(default=0, description="Total conversations handled")
    successful_conversations: int = Field(default=0, description="Successful conversations")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    user_satisfaction_avg: float = Field(default=0.0, description="Average user satisfaction")
    escalation_rate: float = Field(default=0.0, description="Rate of escalations to humans")
    knowledge_usage_stats: Dict[str, int] = Field(
        default_factory=dict, description="Knowledge usage statistics"
    )
    channel_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Performance by channel"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last metrics update"
    )


class BrandAgent(AggregateRoot):
    """Brand Agent aggregate root."""

    name: str = Field(description="Brand agent name")
    brand_id: str = Field(description="Associated brand/company ID")
    agent_type: BrandAgentType = Field(description="Type of brand agent")
    description: Optional[str] = Field(default=None, description="Agent description")

    # Configuration
    personality: BrandPersonality = Field(
        default_factory=BrandPersonality, description="Agent personality"
    )
    configuration: BrandAgentConfiguration = Field(
        default_factory=BrandAgentConfiguration, description="Agent configuration"
    )

    # State
    is_active: bool = Field(default=True, description="Whether agent is active")
    is_deployed: bool = Field(default=False, description="Whether agent is deployed")
    deployment_channels: List[ConversationChannel] = Field(
        default_factory=list, description="Currently deployed channels"
    )

    # Performance
    metrics: BrandAgentMetrics = Field(
        default_factory=BrandAgentMetrics, description="Agent metrics"
    )

    # Knowledge
    knowledge_items: List[str] = Field(
        default_factory=list, description="Associated knowledge item IDs"
    )

    # Metadata
    owner_id: str = Field(description="Owner user ID")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def activate(self) -> None:
        """Activate the brand agent."""
        if not self.is_active:
            self.is_active = True
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def deactivate(self) -> None:
        """Deactivate the brand agent."""
        if self.is_active:
            self.is_active = False
            self.is_deployed = False
            self.deployment_channels = []
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def deploy_to_channel(self, channel: ConversationChannel) -> None:
        """Deploy agent to a specific channel."""
        if not self.is_active:
            raise ValidationError("Cannot deploy inactive agent")

        if channel not in self.configuration.supported_channels:
            raise ValidationError(f"Agent does not support channel: {channel}")

        if channel not in self.deployment_channels:
            self.deployment_channels.append(channel)
            self.is_deployed = True
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def remove_from_channel(self, channel: ConversationChannel) -> None:
        """Remove agent from a specific channel."""
        if channel in self.deployment_channels:
            self.deployment_channels.remove(channel)
            if not self.deployment_channels:
                self.is_deployed = False
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def add_knowledge_item(self, knowledge_id: str) -> None:
        """Add knowledge item to agent."""
        if knowledge_id not in self.knowledge_items:
            self.knowledge_items.append(knowledge_id)
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def remove_knowledge_item(self, knowledge_id: str) -> None:
        """Remove knowledge item from agent."""
        if knowledge_id in self.knowledge_items:
            self.knowledge_items.remove(knowledge_id)
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1

    def update_personality(self, personality: BrandPersonality) -> None:
        """Update agent personality."""
        self.personality = personality
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1

    def update_metrics(self, metrics: BrandAgentMetrics) -> None:
        """Update agent metrics."""
        self.metrics = metrics
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.metrics.total_conversations == 0:
            return 0.0
        return (self.metrics.successful_conversations / self.metrics.total_conversations) * 100

    @property
    def is_performing_well(self) -> bool:
        """Check if agent is performing well."""
        return (
            self.success_rate >= 80.0
            and self.metrics.user_satisfaction_avg >= 4.0
            and self.metrics.escalation_rate <= 0.1
        )


# Domain Events
class BrandAgentCreated(DomainEvent):
    """Event raised when a brand agent is created."""

    agent_id: str
    brand_id: str
    agent_type: BrandAgentType
    owner_id: str


class BrandAgentDeployed(DomainEvent):
    """Event raised when a brand agent is deployed."""

    agent_id: str
    channel: ConversationChannel
    deployed_at: datetime


class ConversationStarted(DomainEvent):
    """Event raised when a conversation starts."""

    session_id: str
    agent_id: str
    channel: ConversationChannel
    user_id: Optional[str]


class ConversationEnded(DomainEvent):
    """Event raised when a conversation ends."""

    session_id: str
    agent_id: str
    duration_seconds: int
    message_count: int
    user_satisfaction: Optional[int]
