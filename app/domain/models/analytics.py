"""
Analytics domain models for comprehensive conversation and performance analytics.
Provides detailed insights into agent performance, user behavior, and system metrics.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .base import BaseEntity, BaseValueObject, DomainEvent


class MetricType(str, Enum):
    """Types of metrics collected."""

    CONVERSATION_DURATION = "conversation_duration"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    RESOLUTION_RATE = "resolution_rate"
    ESCALATION_RATE = "escalation_rate"
    MESSAGE_COUNT = "message_count"
    KNOWLEDGE_USAGE = "knowledge_usage"
    SENTIMENT_SCORE = "sentiment_score"
    INTENT_ACCURACY = "intent_accuracy"
    AGENT_UTILIZATION = "agent_utilization"


class TimeGranularity(str, Enum):
    """Time granularity for metrics aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AnalyticsScope(str, Enum):
    """Scope of analytics data."""

    GLOBAL = "global"
    BRAND = "brand"
    AGENT = "agent"
    CHANNEL = "channel"
    USER = "user"
    CONVERSATION = "conversation"


class MetricValue(BaseValueObject):
    """A single metric value with metadata."""

    value: Union[int, float, str, bool] = Field(description="Metric value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the metric")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TimeSeriesPoint(BaseValueObject):
    """A single point in a time series."""

    timestamp: datetime = Field(description="Timestamp of the data point")
    value: MetricValue = Field(description="Metric value at this timestamp")
    tags: Dict[str, str] = Field(
        default_factory=dict, description="Tags for filtering and grouping"
    )


class AnalyticsMetric(BaseEntity):
    """A metric with its time series data."""

    metric_type: MetricType = Field(description="Type of metric")
    scope: AnalyticsScope = Field(description="Scope of the metric")
    scope_id: str = Field(description="ID of the scope (brand_id, agent_id, etc.)")

    # Time series data
    data_points: List[TimeSeriesPoint] = Field(default_factory=list, description="Time series data")

    # Aggregated values
    current_value: Optional[MetricValue] = Field(default=None, description="Current metric value")
    average_value: Optional[MetricValue] = Field(default=None, description="Average value")
    min_value: Optional[MetricValue] = Field(default=None, description="Minimum value")
    max_value: Optional[MetricValue] = Field(default=None, description="Maximum value")

    # Metadata
    collection_start: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When metric collection started",
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )

    def add_data_point(
        self,
        value: MetricValue,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a new data point to the metric."""
        point = TimeSeriesPoint(
            timestamp=timestamp or datetime.now(timezone.utc), value=value, tags=tags or {}
        )
        self.data_points.append(point)
        self.last_updated = datetime.now(timezone.utc)
        self.version += 1

        # Update aggregated values
        self._update_aggregated_values()

    def _update_aggregated_values(self) -> None:
        """Update aggregated metric values."""
        if not self.data_points:
            return

        # Get numeric values only
        numeric_values = []
        for point in self.data_points:
            if isinstance(point.value.value, (int, float)):
                numeric_values.append(point.value.value)

        if numeric_values:
            self.current_value = self.data_points[-1].value
            self.average_value = MetricValue(value=sum(numeric_values) / len(numeric_values))
            self.min_value = MetricValue(value=min(numeric_values))
            self.max_value = MetricValue(value=max(numeric_values))

    def get_data_for_period(self, start: datetime, end: datetime) -> List[TimeSeriesPoint]:
        """Get data points for a specific time period."""
        return [point for point in self.data_points if start <= point.timestamp <= end]

    def aggregate_by_granularity(self, granularity: TimeGranularity) -> List[TimeSeriesPoint]:
        """Aggregate data points by time granularity."""
        # This would implement time-based aggregation
        # For now, return the raw data points
        return self.data_points


class ConversationAnalytics(BaseEntity):
    """Analytics for a specific conversation."""

    conversation_id: str = Field(description="Conversation ID")
    brand_agent_id: str = Field(description="Brand agent ID")
    user_id: Optional[str] = Field(default=None, description="User ID")

    # Basic metrics
    duration_seconds: int = Field(default=0, description="Conversation duration")
    message_count: int = Field(default=0, description="Total messages")
    user_message_count: int = Field(default=0, description="User messages")
    agent_message_count: int = Field(default=0, description="Agent messages")

    # Quality metrics
    user_satisfaction: Optional[int] = Field(
        default=None, ge=1, le=5, description="User satisfaction rating"
    )
    resolution_status: str = Field(default="unresolved", description="Resolution status")
    escalated: bool = Field(default=False, description="Whether conversation was escalated")

    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    first_response_time_ms: Optional[float] = Field(default=None, description="First response time")

    # Content analysis
    primary_intent: Optional[str] = Field(default=None, description="Primary user intent")
    sentiment_scores: List[float] = Field(
        default_factory=list, description="Sentiment scores over time"
    )
    topics_discussed: List[str] = Field(default_factory=list, description="Topics discussed")

    # Knowledge usage
    knowledge_items_used: List[str] = Field(
        default_factory=list, description="Knowledge items used"
    )
    knowledge_effectiveness: Dict[str, float] = Field(
        default_factory=dict, description="Knowledge effectiveness scores"
    )

    # Channel and context
    channel: str = Field(description="Communication channel")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context data")

    def calculate_satisfaction_score(self) -> float:
        """Calculate overall satisfaction score."""
        if self.user_satisfaction:
            return self.user_satisfaction / 5.0

        # Calculate based on other metrics if no explicit rating
        score = 0.5  # Base score

        # Resolution bonus
        if self.resolution_status == "resolved":
            score += 0.3
        elif self.resolution_status == "partially_resolved":
            score += 0.1

        # Response time penalty
        if self.avg_response_time_ms > 5000:  # > 5 seconds
            score -= 0.2
        elif self.avg_response_time_ms < 2000:  # < 2 seconds
            score += 0.1

        # Escalation penalty
        if self.escalated:
            score -= 0.2

        return max(0.0, min(1.0, score))


class AgentPerformanceAnalytics(BaseEntity):
    """Performance analytics for a brand agent."""

    brand_agent_id: str = Field(description="Brand agent ID")
    brand_id: str = Field(description="Brand ID")

    # Time period
    period_start: datetime = Field(description="Analytics period start")
    period_end: datetime = Field(description="Analytics period end")

    # Conversation metrics
    total_conversations: int = Field(default=0, description="Total conversations")
    active_conversations: int = Field(default=0, description="Currently active conversations")
    completed_conversations: int = Field(default=0, description="Completed conversations")

    # Quality metrics
    avg_satisfaction: float = Field(default=0.0, description="Average user satisfaction")
    resolution_rate: float = Field(default=0.0, description="Resolution rate")
    escalation_rate: float = Field(default=0.0, description="Escalation rate")

    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    avg_conversation_duration: float = Field(
        default=0.0, description="Average conversation duration"
    )
    messages_per_conversation: float = Field(
        default=0.0, description="Average messages per conversation"
    )

    # Usage metrics
    utilization_rate: float = Field(default=0.0, description="Agent utilization rate")
    peak_concurrent_conversations: int = Field(
        default=0, description="Peak concurrent conversations"
    )

    # Knowledge metrics
    knowledge_usage_rate: float = Field(default=0.0, description="Knowledge usage rate")
    top_knowledge_items: List[str] = Field(
        default_factory=list, description="Most used knowledge items"
    )

    # Trend data
    satisfaction_trend: List[float] = Field(
        default_factory=list, description="Satisfaction trend over time"
    )
    response_time_trend: List[float] = Field(
        default_factory=list, description="Response time trend"
    )
    volume_trend: List[int] = Field(default_factory=list, description="Conversation volume trend")

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        score = 0.0
        weight_sum = 0.0

        # Satisfaction (30% weight)
        if self.avg_satisfaction > 0:
            score += (self.avg_satisfaction / 5.0) * 0.3
            weight_sum += 0.3

        # Resolution rate (25% weight)
        score += self.resolution_rate * 0.25
        weight_sum += 0.25

        # Response time (20% weight) - inverse relationship
        if self.avg_response_time_ms > 0:
            response_score = max(0, 1 - (self.avg_response_time_ms / 10000))  # 10s = 0 score
            score += response_score * 0.2
            weight_sum += 0.2

        # Utilization (15% weight)
        score += min(1.0, self.utilization_rate) * 0.15
        weight_sum += 0.15

        # Low escalation rate (10% weight)
        escalation_score = max(0, 1 - self.escalation_rate)
        score += escalation_score * 0.1
        weight_sum += 0.1

        return score / weight_sum if weight_sum > 0 else 0.0


class SystemPerformanceMetrics(BaseEntity):
    """System-wide performance metrics."""

    # Time period
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Metrics timestamp"
    )

    # System metrics
    total_active_conversations: int = Field(default=0, description="Total active conversations")
    total_agents: int = Field(default=0, description="Total agents")
    active_agents: int = Field(default=0, description="Active agents")

    # Performance metrics
    avg_system_response_time_ms: float = Field(
        default=0.0, description="Average system response time"
    )
    system_uptime_percentage: float = Field(default=100.0, description="System uptime percentage")
    error_rate: float = Field(default=0.0, description="System error rate")

    # Resource usage
    cpu_usage_percentage: float = Field(default=0.0, description="CPU usage percentage")
    memory_usage_percentage: float = Field(default=0.0, description="Memory usage percentage")
    database_connections: int = Field(default=0, description="Active database connections")
    websocket_connections: int = Field(default=0, description="Active WebSocket connections")

    # Throughput metrics
    messages_per_minute: float = Field(default=0.0, description="Messages processed per minute")
    conversations_started_per_hour: float = Field(
        default=0.0, description="Conversations started per hour"
    )
    ai_requests_per_minute: float = Field(default=0.0, description="AI requests per minute")

    # Quality metrics
    avg_ai_response_quality: float = Field(default=0.0, description="Average AI response quality")
    knowledge_hit_rate: float = Field(default=0.0, description="Knowledge base hit rate")


class AnalyticsEvent(DomainEvent):
    """Event raised when analytics data is collected."""

    metric_type: MetricType
    scope: AnalyticsScope
    scope_id: str
    value: MetricValue

    def __init__(self, **data):
        super().__init__(
            event_type="analytics_data_collected",
            aggregate_id=data.get("scope_id"),
            aggregate_type="analytics",
            version=1,
            **data,
        )


class PerformanceAlert(DomainEvent):
    """Event raised when performance threshold is exceeded."""

    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metric_type: MetricType
    current_value: float
    threshold_value: float

    def __init__(self, **data):
        super().__init__(
            event_type="performance_alert",
            aggregate_id=data.get("scope_id", "system"),
            aggregate_type="performance",
            version=1,
            **data,
        )
