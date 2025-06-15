"""
Auto-Scaling System for DataMCPServerAgent.
This module implements intelligent auto-scaling based on workload patterns,
performance metrics, and predictive analytics.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from app.core.logging import get_logger
except ImportError:
    from app.core.simple_logging import get_logger

try:
    from app.monitoring.rl_analytics import get_metrics_collector
except ImportError:
    # Create a simple fallback metrics collector
    class SimpleMetricsCollector:
        def record_metric(self, name, value, tags=None):
            pass
        def record_event(self, name, data, level="info"):
            pass

    def get_metrics_collector():
        return SimpleMetricsCollector()

logger = get_logger(__name__)


class ScalingDirection(str, Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingPolicy(str, Enum):
    """Scaling policies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


class ResourceMetric(str, Enum):
    """Resource metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    ACTIVE_CONNECTIONS = "active_connections"


@dataclass
class ScalingRule:
    """Represents a scaling rule."""
    rule_id: str
    name: str
    metric: ResourceMetric
    threshold_up: float
    threshold_down: float
    scale_up_by: int
    scale_down_by: int
    cooldown_period: int  # seconds
    min_instances: int
    max_instances: int
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["metric"] = self.metric.value
        return result


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    timestamp: float
    direction: ScalingDirection
    trigger_metric: str
    trigger_value: float
    threshold: float
    instances_before: int
    instances_after: int
    rule_id: str
    success: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["direction"] = self.direction.value
        return result


class WorkloadPredictor:
    """Predicts future workload patterns for proactive scaling."""

    def __init__(self, history_window: int = 1440):  # 24 hours in minutes
        """Initialize workload predictor.
        
        Args:
            history_window: History window in minutes
        """
        self.history_window = history_window
        self.metric_history = defaultdict(lambda: deque(maxlen=history_window))
        self.patterns = {}

    def record_metric(self, metric: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value.
        
        Args:
            metric: Metric name
            value: Metric value
            timestamp: Timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        self.metric_history[metric].append({
            "value": value,
            "timestamp": timestamp,
        })

        # Update patterns periodically
        if len(self.metric_history[metric]) % 60 == 0:  # Every hour
            self._update_patterns(metric)

    def _update_patterns(self, metric: str):
        """Update patterns for a metric.
        
        Args:
            metric: Metric name
        """
        history = list(self.metric_history[metric])

        if len(history) < 60:  # Need at least 1 hour of data
            return

        # Extract hourly patterns
        hourly_values = defaultdict(list)
        daily_values = defaultdict(list)

        for entry in history:
            dt = datetime.fromtimestamp(entry["timestamp"])
            hour = dt.hour
            day_of_week = dt.weekday()

            hourly_values[hour].append(entry["value"])
            daily_values[day_of_week].append(entry["value"])

        # Calculate average patterns
        hourly_pattern = {
            hour: np.mean(values) for hour, values in hourly_values.items()
        }

        daily_pattern = {
            day: np.mean(values) for day, values in daily_values.items()
        }

        self.patterns[metric] = {
            "hourly": hourly_pattern,
            "daily": daily_pattern,
            "overall_mean": np.mean([entry["value"] for entry in history]),
            "overall_std": np.std([entry["value"] for entry in history]),
        }

    def predict_workload(
        self,
        metric: str,
        horizon_minutes: int = 60
    ) -> Tuple[float, float]:
        """Predict future workload.
        
        Args:
            metric: Metric to predict
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if metric not in self.patterns:
            # No patterns available, use recent average
            recent_values = [
                entry["value"] for entry in list(self.metric_history[metric])[-10:]
            ]
            if recent_values:
                return np.mean(recent_values), 0.5
            else:
                return 0.0, 0.0

        pattern = self.patterns[metric]

        # Get future time
        future_time = datetime.fromtimestamp(time.time() + horizon_minutes * 60)
        future_hour = future_time.hour
        future_day = future_time.weekday()

        # Combine hourly and daily patterns
        hourly_pred = pattern["hourly"].get(future_hour, pattern["overall_mean"])
        daily_factor = pattern["daily"].get(future_day, pattern["overall_mean"]) / pattern["overall_mean"]

        predicted_value = hourly_pred * daily_factor

        # Calculate confidence based on pattern stability
        hourly_std = np.std(list(pattern["hourly"].values())) if pattern["hourly"] else 0
        confidence = max(0.1, 1.0 - (hourly_std / pattern["overall_mean"]))

        return predicted_value, min(confidence, 0.9)

    def detect_anomalies(self, metric: str, current_value: float) -> bool:
        """Detect if current value is anomalous.
        
        Args:
            metric: Metric name
            current_value: Current metric value
            
        Returns:
            True if anomalous
        """
        if metric not in self.patterns:
            return False

        pattern = self.patterns[metric]
        mean = pattern["overall_mean"]
        std = pattern["overall_std"]

        # Use 3-sigma rule for anomaly detection
        threshold = 3 * std

        return abs(current_value - mean) > threshold


class AutoScaler:
    """Intelligent auto-scaling system."""

    def __init__(
        self,
        service_name: str,
        scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID,
        min_instances: int = 1,
        max_instances: int = 10
    ):
        """Initialize auto-scaler.
        
        Args:
            service_name: Name of service to scale
            scaling_policy: Scaling policy to use
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
        """
        self.service_name = service_name
        self.scaling_policy = scaling_policy
        self.min_instances = min_instances
        self.max_instances = max_instances

        # Current state
        self.current_instances = min_instances
        self.target_instances = min_instances

        # Scaling rules
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.last_scaling_time = 0
        self.scaling_events: List[ScalingEvent] = []

        # Workload prediction
        self.predictor = WorkloadPredictor()

        # Metrics and monitoring
        self.metrics_collector = get_metrics_collector()
        self.current_metrics = {}

        # Background tasks
        self.monitoring_task = None
        self.scaling_task = None
        self.is_running = False

        # Initialize default scaling rules
        self._initialize_default_rules()

        logger.info(f"🔧 Initialized auto-scaler for {service_name}")

    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        # CPU utilization rule
        self.add_scaling_rule(
            ScalingRule(
                rule_id="cpu_rule",
                name="CPU Utilization",
                metric=ResourceMetric.CPU_UTILIZATION,
                threshold_up=80.0,
                threshold_down=30.0,
                scale_up_by=1,
                scale_down_by=1,
                cooldown_period=300,  # 5 minutes
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            )
        )

        # Response time rule
        self.add_scaling_rule(
            ScalingRule(
                rule_id="response_time_rule",
                name="Response Time",
                metric=ResourceMetric.RESPONSE_TIME,
                threshold_up=2000.0,  # 2 seconds
                threshold_down=500.0,  # 0.5 seconds
                scale_up_by=2,  # Scale up faster for response time
                scale_down_by=1,
                cooldown_period=180,  # 3 minutes
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            )
        )

        # Request rate rule
        self.add_scaling_rule(
            ScalingRule(
                rule_id="request_rate_rule",
                name="Request Rate",
                metric=ResourceMetric.REQUEST_RATE,
                threshold_up=100.0,  # requests per second
                threshold_down=20.0,
                scale_up_by=1,
                scale_down_by=1,
                cooldown_period=240,  # 4 minutes
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            )
        )

    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule.
        
        Args:
            rule: Scaling rule to add
        """
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"📏 Added scaling rule: {rule.name}")

    def remove_scaling_rule(self, rule_id: str) -> bool:
        """Remove a scaling rule.
        
        Args:
            rule_id: Rule ID to remove
            
        Returns:
            True if removed successfully
        """
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            logger.info(f"🗑️ Removed scaling rule: {rule_id}")
            return True
        return False

    async def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if self.is_running:
            logger.warning("Auto-scaling already running")
            return

        self.is_running = True
        logger.info(f"🚀 Starting auto-scaling for {self.service_name}")

        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())

        logger.info("✅ Auto-scaling started")

    async def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info(f"🛑 Stopping auto-scaling for {self.service_name}")

        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.scaling_task:
            self.scaling_task.cancel()

        logger.info("✅ Auto-scaling stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(30)  # Collect metrics every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _scaling_loop(self):
        """Background scaling decision loop."""
        while self.is_running:
            try:
                await self._make_scaling_decision()
                await asyncio.sleep(60)  # Make decisions every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(120)

    async def _collect_metrics(self):
        """Collect current metrics."""
        try:
            # Simulate metric collection (in real implementation, get from monitoring system)
            current_time = time.time()

            # Generate realistic metrics with some patterns
            hour = datetime.fromtimestamp(current_time).hour

            # CPU utilization with daily pattern
            base_cpu = 40 + 30 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            cpu_noise = np.random.normal(0, 10)
            cpu_utilization = max(0, min(100, base_cpu + cpu_noise))

            # Response time correlated with CPU
            response_time = 500 + (cpu_utilization - 50) * 20 + np.random.normal(0, 100)
            response_time = max(100, response_time)

            # Request rate with business hours pattern
            if 9 <= hour <= 17:  # Business hours
                base_requests = 80 + np.random.normal(0, 20)
            else:
                base_requests = 20 + np.random.normal(0, 10)
            request_rate = max(0, base_requests)

            # Memory utilization
            memory_utilization = 60 + np.random.normal(0, 15)
            memory_utilization = max(0, min(100, memory_utilization))

            # Error rate
            error_rate = max(0, np.random.normal(2, 1))  # 2% average

            # Update current metrics
            self.current_metrics = {
                ResourceMetric.CPU_UTILIZATION.value: cpu_utilization,
                ResourceMetric.MEMORY_UTILIZATION.value: memory_utilization,
                ResourceMetric.REQUEST_RATE.value: request_rate,
                ResourceMetric.RESPONSE_TIME.value: response_time,
                ResourceMetric.ERROR_RATE.value: error_rate,
                ResourceMetric.QUEUE_LENGTH.value: max(0, np.random.normal(5, 2)),
                ResourceMetric.ACTIVE_CONNECTIONS.value: max(0, np.random.normal(50, 15)),
            }

            # Record metrics for prediction
            for metric, value in self.current_metrics.items():
                self.predictor.record_metric(metric, value, current_time)

                # Record in metrics collector
                self.metrics_collector.record_metric(
                    f"autoscaler_{metric}",
                    value,
                    {"service": self.service_name}
                )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _make_scaling_decision(self):
        """Make scaling decision based on current metrics and predictions."""
        try:
            if not self.current_metrics:
                return

            current_time = time.time()

            # Check cooldown period
            if current_time - self.last_scaling_time < 60:  # Minimum 1 minute between decisions
                return

            scaling_decisions = []

            # Evaluate each scaling rule
            for rule in self.scaling_rules.values():
                if not rule.enabled:
                    continue

                decision = await self._evaluate_scaling_rule(rule)
                if decision:
                    scaling_decisions.append(decision)

            # Apply scaling decisions
            if scaling_decisions:
                await self._apply_scaling_decisions(scaling_decisions)

        except Exception as e:
            logger.error(f"Error making scaling decision: {e}")

    async def _evaluate_scaling_rule(self, rule: ScalingRule) -> Optional[Dict[str, Any]]:
        """Evaluate a scaling rule.
        
        Args:
            rule: Scaling rule to evaluate
            
        Returns:
            Scaling decision or None
        """
        metric_value = self.current_metrics.get(rule.metric.value, 0)
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scaling_time < rule.cooldown_period:
            return None

        # Determine scaling direction
        direction = ScalingDirection.STABLE
        scale_by = 0
        threshold = 0

        if metric_value > rule.threshold_up and self.current_instances < rule.max_instances:
            direction = ScalingDirection.UP
            scale_by = rule.scale_up_by
            threshold = rule.threshold_up
        elif metric_value < rule.threshold_down and self.current_instances > rule.min_instances:
            direction = ScalingDirection.DOWN
            scale_by = rule.scale_down_by
            threshold = rule.threshold_down

        if direction == ScalingDirection.STABLE:
            return None

        # Consider predictive scaling for hybrid policy
        if self.scaling_policy in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID]:
            predicted_value, confidence = self.predictor.predict_workload(
                rule.metric.value, horizon_minutes=15
            )

            # Adjust scaling decision based on prediction
            if confidence > 0.7:
                if direction == ScalingDirection.UP and predicted_value < rule.threshold_up * 0.8:
                    # Don't scale up if prediction shows decrease
                    return None
                elif direction == ScalingDirection.DOWN and predicted_value > rule.threshold_down * 1.2:
                    # Don't scale down if prediction shows increase
                    return None

        return {
            "rule_id": rule.rule_id,
            "direction": direction,
            "scale_by": scale_by,
            "metric_value": metric_value,
            "threshold": threshold,
            "priority": 1 if rule.metric == ResourceMetric.RESPONSE_TIME else 2,
        }

    async def _apply_scaling_decisions(self, decisions: List[Dict[str, Any]]):
        """Apply scaling decisions.
        
        Args:
            decisions: List of scaling decisions
        """
        # Sort by priority (response time rules have higher priority)
        decisions.sort(key=lambda d: d["priority"])

        # Apply the highest priority decision
        decision = decisions[0]

        direction = decision["direction"]
        scale_by = decision["scale_by"]

        # Calculate new instance count
        if direction == ScalingDirection.UP:
            new_instances = min(self.max_instances, self.current_instances + scale_by)
        else:
            new_instances = max(self.min_instances, self.current_instances - scale_by)

        if new_instances == self.current_instances:
            return  # No change needed

        # Execute scaling
        success = await self._execute_scaling(new_instances)

        # Record scaling event
        event = ScalingEvent(
            event_id=f"scale_{int(time.time())}",
            timestamp=time.time(),
            direction=direction,
            trigger_metric=decision["rule_id"],
            trigger_value=decision["metric_value"],
            threshold=decision["threshold"],
            instances_before=self.current_instances,
            instances_after=new_instances if success else self.current_instances,
            rule_id=decision["rule_id"],
            success=success,
            reason=f"Metric {decision['rule_id']} triggered scaling {direction.value}",
        )

        self.scaling_events.append(event)
        self.last_scaling_time = time.time()

        # Record metrics
        self.metrics_collector.record_event(
            "autoscaler_scaling_event",
            {
                "service": self.service_name,
                "direction": direction.value,
                "instances_before": self.current_instances,
                "instances_after": new_instances if success else self.current_instances,
                "trigger_metric": decision["rule_id"],
                "success": success,
            },
            "info" if success else "warning"
        )

        if success:
            self.current_instances = new_instances
            self.target_instances = new_instances
            logger.info(f"📈 Scaled {direction.value}: {event.instances_before} → {event.instances_after} instances")
        else:
            logger.error(f"❌ Failed to scale {direction.value}")

    async def _execute_scaling(self, target_instances: int) -> bool:
        """Execute the actual scaling operation.
        
        Args:
            target_instances: Target number of instances
            
        Returns:
            True if scaling successful
        """
        try:
            # Simulate scaling operation
            logger.info(f"🔄 Scaling {self.service_name} to {target_instances} instances")

            # In real implementation, this would call cloud provider APIs
            # or container orchestration systems (Kubernetes, Docker Swarm, etc.)

            await asyncio.sleep(2)  # Simulate scaling time

            return True

        except Exception as e:
            logger.error(f"Error executing scaling: {e}")
            return False

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status.
        
        Returns:
            Scaling status
        """
        recent_events = [
            event.to_dict() for event in self.scaling_events[-10:]
        ]

        # Calculate scaling efficiency
        successful_events = [e for e in self.scaling_events if e.success]
        efficiency = len(successful_events) / len(self.scaling_events) if self.scaling_events else 1.0

        return {
            "service_name": self.service_name,
            "is_running": self.is_running,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scaling_policy": self.scaling_policy.value,
            "current_metrics": self.current_metrics,
            "scaling_rules": {rule_id: rule.to_dict() for rule_id, rule in self.scaling_rules.items()},
            "recent_events": recent_events,
            "total_scaling_events": len(self.scaling_events),
            "scaling_efficiency": efficiency,
            "last_scaling_time": self.last_scaling_time,
        }

    def get_predictions(self, horizon_minutes: int = 60) -> Dict[str, Tuple[float, float]]:
        """Get workload predictions.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Dictionary of metric predictions (value, confidence)
        """
        predictions = {}

        for metric in ResourceMetric:
            prediction = self.predictor.predict_workload(metric.value, horizon_minutes)
            predictions[metric.value] = prediction

        return predictions


# Global auto-scalers
_auto_scalers: Dict[str, AutoScaler] = {}


def create_auto_scaler(
    service_name: str,
    scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID,
    min_instances: int = 1,
    max_instances: int = 10
) -> AutoScaler:
    """Create a new auto-scaler.
    
    Args:
        service_name: Name of service to scale
        scaling_policy: Scaling policy to use
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        
    Returns:
        Auto-scaler instance
    """
    global _auto_scalers

    if service_name in _auto_scalers:
        return _auto_scalers[service_name]

    scaler = AutoScaler(
        service_name=service_name,
        scaling_policy=scaling_policy,
        min_instances=min_instances,
        max_instances=max_instances,
    )

    _auto_scalers[service_name] = scaler

    return scaler


def get_auto_scaler(service_name: str) -> Optional[AutoScaler]:
    """Get existing auto-scaler.
    
    Args:
        service_name: Service name
        
    Returns:
        Auto-scaler instance or None
    """
    return _auto_scalers.get(service_name)
