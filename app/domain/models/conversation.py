"""
Conversation domain models for real-time chat functionality.
Extends the basic conversation models with advanced features.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import Field, field_validator

from .base import AggregateRoot, BaseEntity, BaseValueObject, DomainEvent
from .brand_agent import ConversationChannel


class MessageType(str, Enum):
    """Types of messages in a conversation."""
    
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    SYSTEM = "system"
    TYPING_INDICATOR = "typing_indicator"
    QUICK_REPLY = "quick_reply"
    CARD = "card"
    CAROUSEL = "carousel"


class MessageStatus(str, Enum):
    """Status of a message."""
    
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class ConversationStatus(str, Enum):
    """Status of a conversation."""
    
    ACTIVE = "active"
    WAITING = "waiting"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    TIMEOUT = "timeout"


class SentimentType(str, Enum):
    """Sentiment analysis results."""
    
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


class IntentType(str, Enum):
    """User intent classification."""
    
    QUESTION = "question"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    REQUEST = "request"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    SUPPORT = "support"
    SALES_INQUIRY = "sales_inquiry"
    PRODUCT_INFO = "product_info"
    PRICING = "pricing"
    TECHNICAL_ISSUE = "technical_issue"
    GENERAL_CHAT = "general_chat"


class MessageContext(BaseValueObject):
    """Context information for a message."""
    
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    ip_address: Optional[str] = Field(default=None, description="User IP address")
    location: Optional[Dict[str, Any]] = Field(default=None, description="User location data")
    device_info: Optional[Dict[str, Any]] = Field(default=None, description="Device information")
    referrer: Optional[str] = Field(default=None, description="Referrer URL")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session data")
    previous_messages: List[str] = Field(default_factory=list, description="Previous message IDs")


class MessageAnalysis(BaseValueObject):
    """Analysis results for a message."""
    
    sentiment: Optional[SentimentType] = Field(default=None, description="Sentiment analysis")
    intent: Optional[IntentType] = Field(default=None, description="Intent classification")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Analysis confidence")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    language: Optional[str] = Field(default=None, description="Detected language")
    toxicity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Toxicity score")


class QuickReply(BaseValueObject):
    """Quick reply option for users."""
    
    text: str = Field(description="Display text")
    payload: str = Field(description="Payload to send when selected")
    image_url: Optional[str] = Field(default=None, description="Optional image URL")


class MessageAttachment(BaseValueObject):
    """File attachment for a message."""
    
    filename: str = Field(description="Original filename")
    content_type: str = Field(description="MIME content type")
    size_bytes: int = Field(description="File size in bytes")
    url: str = Field(description="URL to access the file")
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationMessage(BaseEntity):
    """Enhanced conversation message with analysis and context."""
    
    conversation_id: str = Field(description="Conversation ID")
    sender_type: str = Field(description="Sender type: 'user' or 'agent'")
    sender_id: Optional[str] = Field(default=None, description="Sender ID")
    
    # Content
    content: str = Field(description="Message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Message type")
    
    # Status and timing
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Message status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp"
    )
    
    # Analysis
    analysis: Optional[MessageAnalysis] = Field(default=None, description="Message analysis")
    context: Optional[MessageContext] = Field(default=None, description="Message context")
    
    # Rich content
    attachments: List[MessageAttachment] = Field(default_factory=list, description="File attachments")
    quick_replies: List[QuickReply] = Field(default_factory=list, description="Quick reply options")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Response generation
    response_time_ms: Optional[int] = Field(default=None, description="Response generation time")
    knowledge_sources: List[str] = Field(default_factory=list, description="Knowledge sources used")
    
    @field_validator('sender_type')
    @classmethod
    def validate_sender_type(cls, v):
        if v not in ['user', 'agent', 'system']:
            raise ValueError("Sender type must be 'user', 'agent', or 'system'")
        return v
    
    def mark_as_read(self) -> None:
        """Mark message as read."""
        self.status = MessageStatus.READ
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
    
    def add_analysis(self, analysis: MessageAnalysis) -> None:
        """Add analysis results to message."""
        self.analysis = analysis
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1


class ConversationSummary(BaseValueObject):
    """Summary of a conversation."""
    
    total_messages: int = Field(default=0, description="Total message count")
    user_messages: int = Field(default=0, description="User message count")
    agent_messages: int = Field(default=0, description="Agent message count")
    
    duration_seconds: int = Field(default=0, description="Conversation duration")
    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    
    primary_intent: Optional[IntentType] = Field(default=None, description="Primary user intent")
    overall_sentiment: Optional[SentimentType] = Field(default=None, description="Overall sentiment")
    
    resolution_status: str = Field(default="unresolved", description="Resolution status")
    escalation_reason: Optional[str] = Field(default=None, description="Escalation reason")
    
    topics_discussed: List[str] = Field(default_factory=list, description="Main topics")
    knowledge_used: List[str] = Field(default_factory=list, description="Knowledge sources used")


class ConversationMetrics(BaseEntity):
    """Real-time metrics for a conversation."""

    message_count: int = Field(default=0, description="Current message count")
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp"
    )

    user_satisfaction: Optional[int] = Field(default=None, ge=1, le=5, description="User satisfaction")
    agent_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Agent confidence")

    response_times: List[int] = Field(default_factory=list, description="Response times in ms")
    sentiment_scores: List[float] = Field(default_factory=list, description="Sentiment scores")

    escalation_triggers: List[str] = Field(default_factory=list, description="Triggered escalations")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified knowledge gaps")


class LiveConversation(AggregateRoot):
    """Live conversation aggregate for real-time chat."""
    
    brand_agent_id: str = Field(description="Brand agent ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_token: str = Field(description="Session token")
    
    # Channel and context
    channel: ConversationChannel = Field(description="Communication channel")
    channel_metadata: Dict[str, Any] = Field(default_factory=dict, description="Channel-specific data")
    
    # Status and timing
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE, description="Conversation status")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Start timestamp"
    )
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp"
    )
    ended_at: Optional[datetime] = Field(default=None, description="End timestamp")
    
    # Participants
    participants: List[str] = Field(default_factory=list, description="Participant IDs")
    current_agent_id: Optional[str] = Field(default=None, description="Current handling agent")
    
    # Conversation data
    messages: List[str] = Field(default_factory=list, description="Message IDs in order")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    
    # Analytics
    summary: Optional[ConversationSummary] = Field(default=None, description="Conversation summary")
    metrics: ConversationMetrics = Field(
        default_factory=ConversationMetrics, description="Real-time metrics"
    )
    
    # Configuration
    auto_close_timeout: int = Field(default=1800, description="Auto-close timeout in seconds")
    escalation_enabled: bool = Field(default=True, description="Whether escalation is enabled")
    
    def add_message(self, message_id: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(message_id)
        self.metrics.message_count += 1
        self.last_activity_at = datetime.now(timezone.utc)
        self.metrics.last_activity = self.last_activity_at
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
    
    def update_status(self, status: ConversationStatus, reason: Optional[str] = None) -> None:
        """Update conversation status."""
        old_status = self.status
        self.status = status
        self.last_activity_at = datetime.now(timezone.utc)
        
        if status in [ConversationStatus.RESOLVED, ConversationStatus.CLOSED]:
            self.ended_at = datetime.now(timezone.utc)
        
        if status == ConversationStatus.ESCALATED and reason:
            if self.summary:
                self.summary.escalation_reason = reason
            self.metrics.escalation_triggers.append(reason)
        
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
    
    def add_participant(self, participant_id: str) -> None:
        """Add a participant to the conversation."""
        if participant_id not in self.participants:
            self.participants.append(participant_id)
            self.updated_at = datetime.now(timezone.utc)
            self.version += 1
    
    def set_current_agent(self, agent_id: str) -> None:
        """Set the current handling agent."""
        self.current_agent_id = agent_id
        self.add_participant(agent_id)
    
    def update_metrics(self, metrics: ConversationMetrics) -> None:
        """Update conversation metrics."""
        self.metrics = metrics
        self.last_activity_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
    
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self.status == ConversationStatus.ACTIVE
    
    def is_timeout(self) -> bool:
        """Check if conversation has timed out."""
        if not self.is_active():
            return False
        
        timeout_threshold = datetime.now(timezone.utc).timestamp() - self.auto_close_timeout
        return self.last_activity_at.timestamp() < timeout_threshold
    
    @property
    def duration_seconds(self) -> int:
        """Get conversation duration in seconds."""
        end_time = self.ended_at or datetime.now(timezone.utc)
        return int((end_time - self.started_at).total_seconds())


# Domain Events
class MessageSent(DomainEvent):
    """Event raised when a message is sent."""

    conversation_id: str
    message_id: str
    sender_type: str
    message_type: MessageType
    content_preview: str  # First 100 chars

    def __init__(self, **data):
        super().__init__(
            event_type="message_sent",
            aggregate_id=data.get("conversation_id"),
            aggregate_type="conversation",
            version=1,
            **data
        )


class ConversationStatusChanged(DomainEvent):
    """Event raised when conversation status changes."""

    conversation_id: str
    old_status: ConversationStatus
    new_status: ConversationStatus
    reason: Optional[str]

    def __init__(self, **data):
        super().__init__(
            event_type="conversation_status_changed",
            aggregate_id=data.get("conversation_id"),
            aggregate_type="conversation",
            version=1,
            **data
        )


class ConversationEscalated(DomainEvent):
    """Event raised when conversation is escalated."""

    conversation_id: str
    brand_agent_id: str
    escalation_reason: str
    escalated_to: Optional[str]

    def __init__(self, **data):
        super().__init__(
            event_type="conversation_escalated",
            aggregate_id=data.get("conversation_id"),
            aggregate_type="conversation",
            version=1,
            **data
        )


class UserSatisfactionReceived(DomainEvent):
    """Event raised when user provides satisfaction rating."""

    conversation_id: str
    rating: int
    feedback: Optional[str]

    def __init__(self, **data):
        super().__init__(
            event_type="user_satisfaction_received",
            aggregate_id=data.get("conversation_id"),
            aggregate_type="conversation",
            version=1,
            **data
        )
