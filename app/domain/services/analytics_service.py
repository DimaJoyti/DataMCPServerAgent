"""
Analytics Service - Comprehensive analytics and metrics collection.
Provides real-time insights into conversation performance, agent effectiveness, and system health.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.analytics import (
    AgentPerformanceAnalytics,
    AnalyticsEvent,
    AnalyticsMetric,
    AnalyticsScope,
    ConversationAnalytics,
    MetricType,
    MetricValue,
    PerformanceAlert,
    SystemPerformanceMetrics,
)
from app.domain.models.base import DomainService
from app.domain.models.conversation import ConversationMessage, LiveConversation

logger = get_logger(__name__)


class AnalyticsService(DomainService, LoggerMixin):
    """Service for collecting, processing, and analyzing conversation and performance metrics."""

    def __init__(self):
        super().__init__()
        self._metrics_cache: Dict[str, AnalyticsMetric] = {}
        self._performance_thresholds = self._setup_performance_thresholds()
        self._alert_cooldowns: Dict[str, datetime] = {}

    def _setup_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup performance alert thresholds."""
        return {
            MetricType.RESPONSE_TIME: {
                "warning": 3000.0,  # 3 seconds
                "critical": 10000.0,  # 10 seconds
            },
            MetricType.USER_SATISFACTION: {
                "warning": 3.0,  # Below 3/5
                "critical": 2.0,  # Below 2/5
            },
            MetricType.ESCALATION_RATE: {
                "warning": 0.2,  # 20%
                "critical": 0.4,  # 40%
            },
            MetricType.RESOLUTION_RATE: {
                "warning": 0.7,  # Below 70%
                "critical": 0.5,  # Below 50%
            },
        }

    async def collect_conversation_metrics(
        self, conversation: LiveConversation, messages: List[ConversationMessage]
    ) -> ConversationAnalytics:
        """Collect comprehensive metrics for a conversation."""
        analytics = ConversationAnalytics(
            conversation_id=conversation.id,
            brand_agent_id=conversation.brand_agent_id,
            user_id=conversation.user_id,
            channel=conversation.channel,
        )

        # Basic metrics
        analytics.duration_seconds = conversation.duration_seconds
        analytics.message_count = len(messages)
        analytics.user_message_count = len([m for m in messages if m.sender_type == "user"])
        analytics.agent_message_count = len([m for m in messages if m.sender_type == "agent"])

        # Performance metrics
        response_times = []
        sentiment_scores = []
        topics = set()
        knowledge_items = set()

        for i, message in enumerate(messages):
            if message.response_time_ms:
                response_times.append(message.response_time_ms)

            if message.analysis:
                if message.analysis.sentiment:
                    # Convert sentiment to numeric score
                    sentiment_score = self._sentiment_to_score(message.analysis.sentiment)
                    sentiment_scores.append(sentiment_score)

                if message.analysis.intent:
                    analytics.primary_intent = message.analysis.intent

                if message.analysis.keywords:
                    topics.update(message.analysis.keywords)

            if message.knowledge_sources:
                knowledge_items.update(message.knowledge_sources)

        # Calculate averages
        if response_times:
            analytics.avg_response_time_ms = sum(response_times) / len(response_times)
            analytics.first_response_time_ms = response_times[0] if response_times else None

        analytics.sentiment_scores = sentiment_scores
        analytics.topics_discussed = list(topics)
        analytics.knowledge_items_used = list(knowledge_items)

        # Resolution and satisfaction
        analytics.user_satisfaction = conversation.metrics.user_satisfaction
        analytics.escalated = conversation.status.value == "escalated"
        analytics.resolution_status = self._determine_resolution_status(conversation, messages)

        # Store analytics
        await self._store_conversation_analytics(analytics)

        # Emit analytics event
        await self.publish_event(
            AnalyticsEvent(
                metric_type=MetricType.CONVERSATION_DURATION,
                scope=AnalyticsScope.CONVERSATION,
                scope_id=conversation.id,
                value=MetricValue(value=analytics.duration_seconds, unit="seconds"),
            )
        )

        self.logger.info(f"Collected analytics for conversation {conversation.id}")
        return analytics

    async def collect_agent_performance(
        self, agent_id: str, period_start: datetime, period_end: datetime
    ) -> AgentPerformanceAnalytics:
        """Collect performance analytics for an agent over a time period."""
        # Get conversation analytics for the period
        conversation_analytics = await self._get_conversation_analytics_for_agent(
            agent_id, period_start, period_end
        )

        performance = AgentPerformanceAnalytics(
            brand_agent_id=agent_id,
            brand_id="",  # Would be fetched from agent data
            period_start=period_start,
            period_end=period_end,
        )

        if not conversation_analytics:
            return performance

        # Calculate metrics
        performance.total_conversations = len(conversation_analytics)
        performance.completed_conversations = len(
            [
                ca
                for ca in conversation_analytics
                if ca.resolution_status in ["resolved", "partially_resolved"]
            ]
        )

        # Quality metrics
        satisfactions = [
            ca.user_satisfaction for ca in conversation_analytics if ca.user_satisfaction
        ]
        if satisfactions:
            performance.avg_satisfaction = sum(satisfactions) / len(satisfactions)

        performance.resolution_rate = (
            performance.completed_conversations / performance.total_conversations
            if performance.total_conversations > 0
            else 0.0
        )

        escalated_count = len([ca for ca in conversation_analytics if ca.escalated])
        performance.escalation_rate = (
            escalated_count / performance.total_conversations
            if performance.total_conversations > 0
            else 0.0
        )

        # Performance metrics
        response_times = [
            ca.avg_response_time_ms for ca in conversation_analytics if ca.avg_response_time_ms > 0
        ]
        if response_times:
            performance.avg_response_time_ms = sum(response_times) / len(response_times)

        durations = [ca.duration_seconds for ca in conversation_analytics]
        if durations:
            performance.avg_conversation_duration = sum(durations) / len(durations)

        message_counts = [ca.message_count for ca in conversation_analytics]
        if message_counts:
            performance.messages_per_conversation = sum(message_counts) / len(message_counts)

        # Knowledge metrics
        knowledge_usage = [ca for ca in conversation_analytics if ca.knowledge_items_used]
        performance.knowledge_usage_rate = (
            len(knowledge_usage) / performance.total_conversations
            if performance.total_conversations > 0
            else 0.0
        )

        # Store performance analytics
        await self._store_agent_performance(performance)

        # Check for performance alerts
        await self._check_performance_alerts(performance)

        self.logger.info(f"Collected performance analytics for agent {agent_id}")
        return performance

    async def collect_system_metrics(self) -> SystemPerformanceMetrics:
        """Collect system-wide performance metrics."""
        metrics = SystemPerformanceMetrics()

        # Get current system state
        metrics.total_active_conversations = await self._get_active_conversation_count()
        metrics.total_agents = await self._get_total_agent_count()
        metrics.active_agents = await self._get_active_agent_count()

        # Performance metrics (would integrate with actual monitoring)
        metrics.avg_system_response_time_ms = await self._get_avg_system_response_time()
        metrics.system_uptime_percentage = await self._get_system_uptime()
        metrics.error_rate = await self._get_error_rate()

        # Resource usage (would integrate with system monitoring)
        metrics.cpu_usage_percentage = await self._get_cpu_usage()
        metrics.memory_usage_percentage = await self._get_memory_usage()
        metrics.database_connections = await self._get_db_connection_count()
        metrics.websocket_connections = await self._get_websocket_connection_count()

        # Throughput metrics
        metrics.messages_per_minute = await self._get_messages_per_minute()
        metrics.conversations_started_per_hour = await self._get_conversations_per_hour()
        metrics.ai_requests_per_minute = await self._get_ai_requests_per_minute()

        # Quality metrics
        metrics.avg_ai_response_quality = await self._get_avg_response_quality()
        metrics.knowledge_hit_rate = await self._get_knowledge_hit_rate()

        # Store system metrics
        await self._store_system_metrics(metrics)

        self.logger.info("Collected system performance metrics")
        return metrics

    async def get_analytics_dashboard_data(
        self, scope: AnalyticsScope, scope_id: str, time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get comprehensive analytics data for dashboard."""
        start_time, end_time = time_range

        dashboard_data = {
            "scope": scope,
            "scope_id": scope_id,
            "time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "metrics": {},
            "trends": {},
            "alerts": [],
        }

        if scope == AnalyticsScope.AGENT:
            # Agent-specific analytics
            performance = await self.collect_agent_performance(scope_id, start_time, end_time)
            dashboard_data["metrics"] = {
                "total_conversations": performance.total_conversations,
                "avg_satisfaction": performance.avg_satisfaction,
                "resolution_rate": performance.resolution_rate,
                "escalation_rate": performance.escalation_rate,
                "avg_response_time_ms": performance.avg_response_time_ms,
                "utilization_rate": performance.utilization_rate,
                "performance_score": performance.calculate_performance_score(),
            }

            dashboard_data["trends"] = {
                "satisfaction": performance.satisfaction_trend,
                "response_time": performance.response_time_trend,
                "volume": performance.volume_trend,
            }

        elif scope == AnalyticsScope.GLOBAL:
            # System-wide analytics
            system_metrics = await self.collect_system_metrics()
            dashboard_data["metrics"] = {
                "active_conversations": system_metrics.total_active_conversations,
                "active_agents": system_metrics.active_agents,
                "avg_response_time": system_metrics.avg_system_response_time_ms,
                "system_uptime": system_metrics.system_uptime_percentage,
                "error_rate": system_metrics.error_rate,
                "messages_per_minute": system_metrics.messages_per_minute,
            }

        # Get recent alerts
        dashboard_data["alerts"] = await self._get_recent_alerts(
            scope, scope_id, start_time, end_time
        )

        return dashboard_data

    async def _check_performance_alerts(self, performance: AgentPerformanceAnalytics) -> None:
        """Check performance metrics against thresholds and raise alerts."""
        agent_id = performance.brand_agent_id

        # Check response time
        if performance.avg_response_time_ms > 0:
            await self._check_metric_threshold(
                MetricType.RESPONSE_TIME,
                performance.avg_response_time_ms,
                agent_id,
                "high_response_time",
            )

        # Check satisfaction
        if performance.avg_satisfaction > 0:
            await self._check_metric_threshold(
                MetricType.USER_SATISFACTION,
                performance.avg_satisfaction,
                agent_id,
                "low_satisfaction",
                inverse=True,  # Lower values are worse
            )

        # Check escalation rate
        await self._check_metric_threshold(
            MetricType.ESCALATION_RATE,
            performance.escalation_rate,
            agent_id,
            "high_escalation_rate",
        )

        # Check resolution rate
        await self._check_metric_threshold(
            MetricType.RESOLUTION_RATE,
            performance.resolution_rate,
            agent_id,
            "low_resolution_rate",
            inverse=True,
        )

    async def _check_metric_threshold(
        self,
        metric_type: MetricType,
        value: float,
        scope_id: str,
        alert_type: str,
        inverse: bool = False,
    ) -> None:
        """Check if a metric value exceeds thresholds."""
        thresholds = self._performance_thresholds.get(metric_type, {})
        if not thresholds:
            return

        # Check cooldown
        cooldown_key = f"{alert_type}_{scope_id}"
        if cooldown_key in self._alert_cooldowns:
            if datetime.now(timezone.utc) - self._alert_cooldowns[cooldown_key] < timedelta(
                minutes=15
            ):
                return  # Still in cooldown

        severity = None
        threshold_value = None

        if inverse:
            # For metrics where lower is worse (satisfaction, resolution rate)
            if value < thresholds.get("critical", 0):
                severity = "critical"
                threshold_value = thresholds["critical"]
            elif value < thresholds.get("warning", 0):
                severity = "warning"
                threshold_value = thresholds["warning"]
        else:
            # For metrics where higher is worse (response time, escalation rate)
            if value > thresholds.get("critical", float("inf")):
                severity = "critical"
                threshold_value = thresholds["critical"]
            elif value > thresholds.get("warning", float("inf")):
                severity = "warning"
                threshold_value = thresholds["warning"]

        if severity:
            # Raise alert
            alert = PerformanceAlert(
                alert_type=alert_type,
                severity=severity,
                message=f"{metric_type.value} {severity} threshold exceeded",
                metric_type=metric_type,
                current_value=value,
                threshold_value=threshold_value,
                scope_id=scope_id,
            )

            await self.publish_event(alert)

            # Set cooldown
            self._alert_cooldowns[cooldown_key] = datetime.now(timezone.utc)

            self.logger.warning(
                f"Performance alert: {alert_type} for {scope_id} - {value} vs {threshold_value}"
            )

    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment to numeric score."""
        sentiment_scores = {
            "positive": 0.8,
            "satisfied": 0.9,
            "neutral": 0.5,
            "negative": 0.2,
            "frustrated": 0.1,
            "confused": 0.3,
        }
        return sentiment_scores.get(sentiment.lower(), 0.5)

    def _determine_resolution_status(
        self, conversation: LiveConversation, messages: List[ConversationMessage]
    ) -> str:
        """Determine conversation resolution status."""
        if conversation.status.value == "resolved":
            return "resolved"
        elif conversation.status.value == "closed":
            return "resolved"
        elif conversation.status.value == "escalated":
            return "escalated"
        elif len(messages) > 5:  # Heuristic for partial resolution
            return "partially_resolved"
        else:
            return "unresolved"

    # Mock implementations for system metrics (would integrate with actual monitoring)
    async def _get_active_conversation_count(self) -> int:
        return 42  # Mock value

    async def _get_total_agent_count(self) -> int:
        return 10  # Mock value

    async def _get_active_agent_count(self) -> int:
        return 8  # Mock value

    async def _get_avg_system_response_time(self) -> float:
        return 1250.0  # Mock value

    async def _get_system_uptime(self) -> float:
        return 99.9  # Mock value

    async def _get_error_rate(self) -> float:
        return 0.01  # Mock value

    async def _get_cpu_usage(self) -> float:
        return 45.0  # Mock value

    async def _get_memory_usage(self) -> float:
        return 60.0  # Mock value

    async def _get_db_connection_count(self) -> int:
        return 25  # Mock value

    async def _get_websocket_connection_count(self) -> int:
        return 150  # Mock value

    async def _get_messages_per_minute(self) -> float:
        return 120.0  # Mock value

    async def _get_conversations_per_hour(self) -> float:
        return 45.0  # Mock value

    async def _get_ai_requests_per_minute(self) -> float:
        return 80.0  # Mock value

    async def _get_avg_response_quality(self) -> float:
        return 0.85  # Mock value

    async def _get_knowledge_hit_rate(self) -> float:
        return 0.75  # Mock value

    # Storage methods (would integrate with actual repositories)
    async def _store_conversation_analytics(self, analytics: ConversationAnalytics) -> None:
        """Store conversation analytics."""
        # Mock implementation
        pass

    async def _store_agent_performance(self, performance: AgentPerformanceAnalytics) -> None:
        """Store agent performance analytics."""
        # Mock implementation
        pass

    async def _store_system_metrics(self, metrics: SystemPerformanceMetrics) -> None:
        """Store system metrics."""
        # Mock implementation
        pass

    async def _get_conversation_analytics_for_agent(
        self, agent_id: str, start: datetime, end: datetime
    ) -> List[ConversationAnalytics]:
        """Get conversation analytics for an agent in a time period."""
        # Mock implementation
        return []

    async def _get_recent_alerts(
        self, scope: AnalyticsScope, scope_id: str, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Get recent alerts for scope."""
        # Mock implementation
        return []
