"""
Analytics and monitoring for Reinforcement Learning system.
"""

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from app.core.logging_improved import get_logger

logger = get_logger(__name__)


@dataclass
class RLMetric:
    """Single RL metric data point."""
    timestamp: float
    metric_name: str
    value: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RLEvent:
    """RL system event."""
    timestamp: float
    event_type: str
    event_data: Dict[str, Any]
    severity: str = "info"  # info, warning, error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RLMetricsCollector:
    """Collects and stores RL metrics."""

    def __init__(self, max_metrics: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to store
        """
        self.max_metrics = max_metrics
        self.metrics = deque(maxlen=max_metrics)
        self.events = deque(maxlen=max_metrics)

        # Aggregated metrics
        self.metric_aggregates = defaultdict(list)
        self.event_counts = defaultdict(int)

        # Real-time tracking
        self.current_session = {
            "start_time": time.time(),
            "requests_processed": 0,
            "training_episodes": 0,
            "errors": 0,
            "warnings": 0,
        }

    def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metric = RLMetric(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            metadata=metadata or {}
        )

        self.metrics.append(metric)
        self.metric_aggregates[name].append(value)

        # Keep aggregates bounded
        if len(self.metric_aggregates[name]) > 1000:
            self.metric_aggregates[name] = self.metric_aggregates[name][-1000:]

    def record_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        severity: str = "info"
    ):
        """Record an event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            severity: Event severity
        """
        event = RLEvent(
            timestamp=time.time(),
            event_type=event_type,
            event_data=event_data,
            severity=severity
        )

        self.events.append(event)
        self.event_counts[event_type] += 1

        # Update session tracking
        if severity == "error":
            self.current_session["errors"] += 1
        elif severity == "warning":
            self.current_session["warnings"] += 1

    def get_metrics_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get metrics summary.
        
        Args:
            time_window: Time window in seconds (None for all time)
            
        Returns:
            Metrics summary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0

        # Filter metrics by time window
        filtered_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]

        if not filtered_metrics:
            return {"error": "No metrics in time window"}

        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in filtered_metrics:
            metric_groups[metric.metric_name].append(metric.value)

        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                }

        return {
            "time_window": time_window,
            "total_metrics": len(filtered_metrics),
            "metric_types": len(metric_groups),
            "metrics": summary,
        }

    def get_events_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get events summary.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Events summary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0

        # Filter events by time window
        filtered_events = [
            e for e in self.events
            if e.timestamp >= cutoff_time
        ]

        # Group by event type and severity
        event_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for event in filtered_events:
            event_type_counts[event.event_type] += 1
            severity_counts[event.severity] += 1

        return {
            "time_window": time_window,
            "total_events": len(filtered_events),
            "event_types": dict(event_type_counts),
            "severity_distribution": dict(severity_counts),
            "recent_events": [e.to_dict() for e in list(filtered_events)[-10:]],
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary.
        
        Returns:
            Session summary
        """
        session_duration = time.time() - self.current_session["start_time"]

        return {
            "session_duration": session_duration,
            "session_duration_formatted": f"{session_duration/3600:.1f}h",
            **self.current_session,
            "requests_per_hour": self.current_session["requests_processed"] / max(session_duration/3600, 0.001),
            "error_rate": self.current_session["errors"] / max(self.current_session["requests_processed"], 1),
        }


class RLPerformanceAnalyzer:
    """Analyzes RL system performance."""

    def __init__(self, metrics_collector: RLMetricsCollector):
        """Initialize performance analyzer.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector

    def analyze_training_performance(self) -> Dict[str, Any]:
        """Analyze training performance.
        
        Returns:
            Training performance analysis
        """
        # Get training-related metrics
        training_metrics = []
        for metric in self.metrics_collector.metrics:
            if any(keyword in metric.metric_name.lower() for keyword in
                   ['loss', 'reward', 'episode', 'training']):
                training_metrics.append(metric)

        if not training_metrics:
            return {"error": "No training metrics available"}

        # Analyze trends
        recent_metrics = training_metrics[-100:]  # Last 100 metrics

        # Group by metric type
        metric_trends = defaultdict(list)
        for metric in recent_metrics:
            metric_trends[metric.metric_name].append({
                "timestamp": metric.timestamp,
                "value": metric.value
            })

        # Calculate trends
        trend_analysis = {}
        for name, values in metric_trends.items():
            if len(values) >= 2:
                # Simple linear trend
                timestamps = [v["timestamp"] for v in values]
                metric_values = [v["value"] for v in values]

                # Normalize timestamps
                min_time = min(timestamps)
                norm_timestamps = [(t - min_time) for t in timestamps]

                # Calculate trend
                if len(norm_timestamps) > 1:
                    trend = np.polyfit(norm_timestamps, metric_values, 1)[0]
                    trend_analysis[name] = {
                        "trend": "improving" if trend > 0 else "declining" if trend < 0 else "stable",
                        "slope": trend,
                        "recent_value": metric_values[-1],
                        "data_points": len(values),
                    }

        return {
            "total_training_metrics": len(training_metrics),
            "recent_metrics_analyzed": len(recent_metrics),
            "trend_analysis": trend_analysis,
        }

    def analyze_response_performance(self) -> Dict[str, Any]:
        """Analyze response performance.
        
        Returns:
            Response performance analysis
        """
        # Get response time metrics
        response_metrics = []
        for metric in self.metrics_collector.metrics:
            if 'response_time' in metric.metric_name.lower():
                response_metrics.append(metric)

        if not response_metrics:
            return {"error": "No response time metrics available"}

        # Recent performance (last hour)
        current_time = time.time()
        recent_metrics = [
            m for m in response_metrics
            if current_time - m.timestamp <= 3600
        ]

        if not recent_metrics:
            return {"error": "No recent response time metrics"}

        response_times = [m.value for m in recent_metrics]

        # Performance analysis
        analysis = {
            "total_responses": len(response_times),
            "mean_response_time": np.mean(response_times),
            "median_response_time": np.median(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "p99_response_time": np.percentile(response_times, 99),
            "max_response_time": np.max(response_times),
            "min_response_time": np.min(response_times),
        }

        # Performance classification
        mean_time = analysis["mean_response_time"]
        if mean_time < 1.0:
            performance_class = "excellent"
        elif mean_time < 3.0:
            performance_class = "good"
        elif mean_time < 5.0:
            performance_class = "acceptable"
        else:
            performance_class = "poor"

        analysis["performance_classification"] = performance_class

        # SLA compliance (assuming 5s SLA)
        sla_violations = sum(1 for t in response_times if t > 5.0)
        analysis["sla_compliance"] = {
            "sla_threshold": 5.0,
            "violations": sla_violations,
            "compliance_rate": 1.0 - (sla_violations / len(response_times)),
        }

        return analysis

    def analyze_safety_performance(self) -> Dict[str, Any]:
        """Analyze safety performance.
        
        Returns:
            Safety performance analysis
        """
        # Get safety-related events
        safety_events = []
        for event in self.metrics_collector.events:
            if any(keyword in event.event_type.lower() for keyword in
                   ['safety', 'constraint', 'violation', 'risk']):
                safety_events.append(event)

        # Get safety metrics
        safety_metrics = []
        for metric in self.metrics_collector.metrics:
            if any(keyword in metric.metric_name.lower() for keyword in
                   ['safety', 'risk', 'constraint', 'violation']):
                safety_metrics.append(metric)

        # Analyze violations
        violation_events = [
            e for e in safety_events
            if 'violation' in e.event_type.lower()
        ]

        # Recent safety performance (last 24 hours)
        current_time = time.time()
        recent_violations = [
            e for e in violation_events
            if current_time - e.timestamp <= 86400
        ]

        # Safety score trends
        safety_scores = [
            m.value for m in safety_metrics
            if 'safety_score' in m.metric_name.lower()
        ]

        analysis = {
            "total_safety_events": len(safety_events),
            "total_violations": len(violation_events),
            "recent_violations_24h": len(recent_violations),
            "safety_metrics_count": len(safety_metrics),
        }

        if safety_scores:
            analysis["safety_score_stats"] = {
                "mean": np.mean(safety_scores),
                "min": np.min(safety_scores),
                "max": np.max(safety_scores),
                "recent_score": safety_scores[-1] if safety_scores else None,
            }

        # Safety classification
        if len(recent_violations) == 0:
            safety_class = "excellent"
        elif len(recent_violations) <= 5:
            safety_class = "good"
        elif len(recent_violations) <= 20:
            safety_class = "acceptable"
        else:
            safety_class = "concerning"

        analysis["safety_classification"] = safety_class

        return analysis

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Comprehensive report
        """
        return {
            "timestamp": time.time(),
            "session_summary": self.metrics_collector.get_session_summary(),
            "metrics_summary": self.metrics_collector.get_metrics_summary(3600),  # Last hour
            "events_summary": self.metrics_collector.get_events_summary(3600),
            "training_analysis": self.analyze_training_performance(),
            "response_analysis": self.analyze_response_performance(),
            "safety_analysis": self.analyze_safety_performance(),
        }


class RLDashboard:
    """Real-time dashboard for RL system."""

    def __init__(self, metrics_collector: RLMetricsCollector):
        """Initialize dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.analyzer = RLPerformanceAnalyzer(metrics_collector)
        self.dashboard_data = {}
        self.update_interval = 30  # seconds
        self.last_update = 0

    async def get_dashboard_data(self, force_update: bool = False) -> Dict[str, Any]:
        """Get dashboard data.
        
        Args:
            force_update: Force data update
            
        Returns:
            Dashboard data
        """
        current_time = time.time()

        # Update if needed
        if force_update or (current_time - self.last_update) > self.update_interval:
            await self._update_dashboard_data()
            self.last_update = current_time

        return self.dashboard_data

    async def _update_dashboard_data(self):
        """Update dashboard data."""
        try:
            # Get comprehensive report
            report = self.analyzer.generate_comprehensive_report()

            # Extract key metrics for dashboard
            session = report["session_summary"]
            metrics = report.get("metrics_summary", {})
            events = report.get("events_summary", {})

            # Real-time status
            status = {
                "uptime": session["session_duration_formatted"],
                "requests_processed": session["requests_processed"],
                "requests_per_hour": session["requests_per_hour"],
                "error_rate": session["error_rate"],
                "training_episodes": session["training_episodes"],
            }

            # Performance indicators
            response_analysis = report.get("response_analysis", {})
            performance = {
                "avg_response_time": response_analysis.get("mean_response_time", 0),
                "p95_response_time": response_analysis.get("p95_response_time", 0),
                "performance_class": response_analysis.get("performance_classification", "unknown"),
                "sla_compliance": response_analysis.get("sla_compliance", {}).get("compliance_rate", 0),
            }

            # Safety indicators
            safety_analysis = report.get("safety_analysis", {})
            safety = {
                "recent_violations": safety_analysis.get("recent_violations_24h", 0),
                "safety_class": safety_analysis.get("safety_classification", "unknown"),
                "safety_score": safety_analysis.get("safety_score_stats", {}).get("recent_score", 0),
            }

            # Training indicators
            training_analysis = report.get("training_analysis", {})
            training = {
                "total_metrics": training_analysis.get("total_training_metrics", 0),
                "trend_analysis": training_analysis.get("trend_analysis", {}),
            }

            self.dashboard_data = {
                "last_updated": time.time(),
                "status": status,
                "performance": performance,
                "safety": safety,
                "training": training,
                "recent_events": events.get("recent_events", []),
                "full_report": report,
            }

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}", exc_info=True)
            self.dashboard_data = {
                "error": str(e),
                "last_updated": time.time(),
            }


# Global instances
_metrics_collector: Optional[RLMetricsCollector] = None
_dashboard: Optional[RLDashboard] = None


def get_metrics_collector() -> RLMetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = RLMetricsCollector()
    return _metrics_collector


def get_dashboard() -> RLDashboard:
    """Get global dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = RLDashboard(get_metrics_collector())
    return _dashboard
