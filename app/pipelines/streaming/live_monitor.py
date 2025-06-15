"""
Live Monitoring for Streaming Pipeline.

This module provides comprehensive real-time monitoring capabilities:
- Performance metrics collection
- Health monitoring
- Alert management
- Resource usage tracking
- Real-time dashboards
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.logging import get_logger


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Represents a system alert."""

    # Alert identification
    alert_id: str
    level: AlertLevel
    timestamp: float

    # Alert content
    title: str
    message: str
    source: str

    # Alert metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

    # Timing
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics snapshot."""

    # Timestamp
    timestamp: float = Field(..., description="Metrics timestamp")

    # Processing metrics
    events_per_second: float = Field(default=0.0, description="Events processed per second")
    avg_processing_time_ms: float = Field(default=0.0, description="Average processing time")
    success_rate: float = Field(default=0.0, description="Success rate (0-1)")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")

    # Queue metrics
    queue_size: int = Field(default=0, description="Current queue size")
    queue_usage: float = Field(default=0.0, description="Queue usage percentage")

    # Worker metrics
    active_workers: int = Field(default=0, description="Number of active workers")
    total_workers: int = Field(default=0, description="Total number of workers")
    worker_utilization: float = Field(default=0.0, description="Worker utilization percentage")

    # Resource metrics
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")

    # Latency metrics
    p50_latency_ms: float = Field(default=0.0, description="50th percentile latency")
    p95_latency_ms: float = Field(default=0.0, description="95th percentile latency")
    p99_latency_ms: float = Field(default=0.0, description="99th percentile latency")


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics collector."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.collection_interval = self.config.get("collection_interval", 1.0)
        self.retention_period = self.config.get("retention_period", 3600)  # 1 hour
        self.max_samples = self.config.get("max_samples", 1000)

        # Metrics storage
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=self.max_samples)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))

        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.last_collection_time = time.time()

        self.logger.info("MetricsCollector initialized")

    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.is_collecting:
            return

        self.is_collecting = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Metrics collection stopped")

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value

    def record_timer(self, name: str, value: float) -> None:
        """Record a timer metric."""
        self.timers[name].append(value)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        now = time.time()

        # Calculate rates
        time_delta = now - self.last_collection_time
        events_per_second = self.counters.get("events_processed", 0) / max(time_delta, 1.0)

        # Calculate success rate
        total_events = self.counters.get("events_processed", 0) + self.counters.get(
            "events_failed", 0
        )
        success_rate = self.counters.get("events_processed", 0) / max(total_events, 1)
        error_rate = self.counters.get("events_failed", 0) / max(total_events, 1)

        # Calculate latency percentiles
        processing_times = list(self.timers.get("processing_time", []))
        if processing_times:
            processing_times.sort()
            p50_latency = self._percentile(processing_times, 50)
            p95_latency = self._percentile(processing_times, 95)
            p99_latency = self._percentile(processing_times, 99)
            avg_processing_time = sum(processing_times) / len(processing_times)
        else:
            p50_latency = p95_latency = p99_latency = avg_processing_time = 0.0

        return PerformanceMetrics(
            timestamp=now,
            events_per_second=events_per_second,
            avg_processing_time_ms=avg_processing_time,
            success_rate=success_rate,
            error_rate=error_rate,
            queue_size=int(self.gauges.get("queue_size", 0)),
            queue_usage=self.gauges.get("queue_usage", 0.0),
            active_workers=int(self.gauges.get("active_workers", 0)),
            total_workers=int(self.gauges.get("total_workers", 0)),
            worker_utilization=self.gauges.get("worker_utilization", 0.0),
            memory_usage_mb=self.gauges.get("memory_usage_mb", 0.0),
            cpu_usage_percent=self.gauges.get("cpu_usage_percent", 0.0),
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
        )

    def get_metrics_history(
        self, duration_seconds: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """Get metrics history for a specified duration."""
        if duration_seconds is None:
            return list(self.metrics_history)

        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                # Collect current metrics
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)

                # Reset counters for rate calculations
                self._reset_rate_counters()
                self.last_collection_time = time.time()

                # Clean up old timer data
                self._cleanup_timers()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)

    def _reset_rate_counters(self) -> None:
        """Reset counters used for rate calculations."""
        rate_counters = ["events_processed", "events_failed"]
        for counter in rate_counters:
            self.counters[counter] = 0

    def _cleanup_timers(self) -> None:
        """Clean up old timer data."""
        # Timer data is automatically cleaned up by deque maxlen
        pass

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        index = int((percentile / 100.0) * len(data))
        index = min(index, len(data) - 1)
        return data[index]


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.max_alerts = self.config.get("max_alerts", 1000)
        self.alert_retention = self.config.get("alert_retention", 86400)  # 24 hours

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque[Alert] = deque(maxlen=self.max_alerts)

        # Alert handlers
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)

        # Thresholds
        self.thresholds = self.config.get(
            "thresholds",
            {
                "error_rate": 0.05,  # 5% error rate
                "queue_usage": 0.9,  # 90% queue usage
                "worker_utilization": 0.95,  # 95% worker utilization
                "memory_usage_mb": 1000,  # 1GB memory usage
                "cpu_usage_percent": 90,  # 90% CPU usage
                "p99_latency_ms": 5000,  # 5 second p99 latency
            },
        )

        self.logger.info("AlertManager initialized")

    def add_alert_handler(self, level: AlertLevel, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler for a specific level."""
        self.alert_handlers[level].append(handler)
        self.logger.info(f"Added alert handler for level: {level}")

    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            level=level,
            timestamp=time.time(),
            title=title,
            message=message,
            source=source,
            metadata=metadata or {},
        )

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Trigger handlers
        await self._trigger_handlers(alert)

        self.logger.info(f"Created {level} alert: {title}")
        return alert

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_at = time.time()

        self.logger.info(f"Acknowledged alert: {alert_id}")
        return True

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = time.time()

        # Remove from active alerts
        del self.active_alerts[alert_id]

        self.logger.info(f"Resolved alert: {alert_id}")
        return True

    async def check_metrics_for_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against thresholds and create alerts if needed."""
        # Error rate check
        if metrics.error_rate > self.thresholds["error_rate"]:
            await self.create_alert(
                AlertLevel.WARNING,
                "High Error Rate",
                f"Error rate is {metrics.error_rate:.2%}, threshold is {self.thresholds['error_rate']:.2%}",
                "metrics_monitor",
                {"error_rate": metrics.error_rate},
            )

        # Queue usage check
        if metrics.queue_usage > self.thresholds["queue_usage"]:
            await self.create_alert(
                AlertLevel.WARNING,
                "High Queue Usage",
                f"Queue usage is {metrics.queue_usage:.2%}, threshold is {self.thresholds['queue_usage']:.2%}",
                "metrics_monitor",
                {"queue_usage": metrics.queue_usage},
            )

        # Worker utilization check
        if metrics.worker_utilization > self.thresholds["worker_utilization"]:
            await self.create_alert(
                AlertLevel.WARNING,
                "High Worker Utilization",
                f"Worker utilization is {metrics.worker_utilization:.2%}",
                "metrics_monitor",
                {"worker_utilization": metrics.worker_utilization},
            )

        # Memory usage check
        if metrics.memory_usage_mb > self.thresholds["memory_usage_mb"]:
            await self.create_alert(
                AlertLevel.ERROR,
                "High Memory Usage",
                f"Memory usage is {metrics.memory_usage_mb:.1f}MB",
                "metrics_monitor",
                {"memory_usage_mb": metrics.memory_usage_mb},
            )

        # CPU usage check
        if metrics.cpu_usage_percent > self.thresholds["cpu_usage_percent"]:
            await self.create_alert(
                AlertLevel.ERROR,
                "High CPU Usage",
                f"CPU usage is {metrics.cpu_usage_percent:.1f}%",
                "metrics_monitor",
                {"cpu_usage_percent": metrics.cpu_usage_percent},
            )

        # Latency check
        if metrics.p99_latency_ms > self.thresholds["p99_latency_ms"]:
            await self.create_alert(
                AlertLevel.WARNING,
                "High Latency",
                f"P99 latency is {metrics.p99_latency_ms:.1f}ms",
                "metrics_monitor",
                {"p99_latency_ms": metrics.p99_latency_ms},
            )

    async def _trigger_handlers(self, alert: Alert) -> None:
        """Trigger alert handlers for an alert."""
        handlers = self.alert_handlers.get(alert.level, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, duration_seconds: Optional[int] = None) -> List[Alert]:
        """Get alert history for a specified duration."""
        if duration_seconds is None:
            return list(self.alert_history)

        cutoff_time = time.time() - duration_seconds
        return [a for a in self.alert_history if a.timestamp >= cutoff_time]


class LiveMonitor:
    """Main live monitoring coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize live monitor."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Components
        self.metrics_collector = MetricsCollector(self.config.get("metrics", {}))
        self.alert_manager = AlertManager(self.config.get("alerts", {}))

        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Configuration
        self.monitor_interval = self.config.get("monitor_interval", 5.0)

        self.logger.info("LiveMonitor initialized")

    async def start_monitoring(self) -> None:
        """Start live monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start components
        await self.metrics_collector.start_collection()

        # Start monitoring loop
        self.monitor_task = asyncio.create_task(self._monitor_loop())

        self.logger.info("Live monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop live monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Stop components
        await self.metrics_collector.stop_collection()

        # Stop monitoring loop
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Live monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current metrics
                metrics = self.metrics_collector.get_current_metrics()

                # Check for alerts
                await self.alert_manager.check_metrics_for_alerts(metrics)

                # Log metrics summary
                self.logger.debug(
                    f"Metrics: {metrics.events_per_second:.1f} eps, "
                    f"{metrics.success_rate:.2%} success, "
                    f"{metrics.queue_usage:.2%} queue usage"
                )

                await asyncio.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.monitor_interval)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        current_metrics = self.metrics_collector.get_current_metrics()
        metrics_history = self.metrics_collector.get_metrics_history(
            duration_seconds=300
        )  # Last 5 minutes
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "current_metrics": current_metrics.dict(),
            "metrics_history": [m.dict() for m in metrics_history],
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "level": a.level,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "acknowledged": a.acknowledged,
                }
                for a in active_alerts
            ],
            "system_health": self._calculate_system_health(current_metrics),
        }

    def _calculate_system_health(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall system health."""
        health_score = 100

        # Deduct points for various issues
        if metrics.error_rate > 0.01:  # 1%
            health_score -= 20
        if metrics.queue_usage > 0.8:  # 80%
            health_score -= 15
        if metrics.worker_utilization > 0.9:  # 90%
            health_score -= 15
        if metrics.p99_latency_ms > 1000:  # 1 second
            health_score -= 10

        if health_score >= 90:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        else:
            return "poor"
