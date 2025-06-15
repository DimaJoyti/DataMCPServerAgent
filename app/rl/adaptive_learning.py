"""
Adaptive Learning System for DataMCPServerAgent.
This module implements continuous learning and adaptation capabilities.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging_improved import get_logger
from app.monitoring.rl_analytics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class LearningEvent:
    """Represents a learning event in the system."""
    timestamp: float
    event_type: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    feedback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AdaptationStrategy:
    """Represents an adaptation strategy."""
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    adaptation_actions: List[str]
    priority: int = 1
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceTracker:
    """Tracks system performance metrics for adaptive learning."""

    def __init__(self, window_size: int = 1000):
        """Initialize performance tracker.
        
        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.performance_baselines = {}
        self.trend_analysis = {}

    def record_metric(self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None):
        """Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Additional context
        """
        timestamp = time.time()
        metric_data = {
            "value": value,
            "timestamp": timestamp,
            "context": context or {}
        }

        self.metrics_history[metric_name].append(metric_data)

        # Update trend analysis
        self._update_trend_analysis(metric_name)

    def _update_trend_analysis(self, metric_name: str):
        """Update trend analysis for a metric.
        
        Args:
            metric_name: Name of the metric
        """
        history = self.metrics_history[metric_name]

        if len(history) < 10:  # Need minimum data points
            return

        # Get recent values
        recent_values = [item["value"] for item in list(history)[-10:]]
        older_values = [item["value"] for item in list(history)[-20:-10]] if len(history) >= 20 else []

        # Calculate trend
        if older_values:
            recent_avg = np.mean(recent_values)
            older_avg = np.mean(older_values)

            trend_direction = "improving" if recent_avg > older_avg else "declining"
            trend_magnitude = abs(recent_avg - older_avg) / max(abs(older_avg), 1e-6)

            self.trend_analysis[metric_name] = {
                "direction": trend_direction,
                "magnitude": trend_magnitude,
                "recent_avg": recent_avg,
                "older_avg": older_avg,
                "confidence": min(len(history) / self.window_size, 1.0),
                "last_updated": time.time(),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Performance summary
        """
        summary = {}

        for metric_name, history in self.metrics_history.items():
            if not history:
                continue

            values = [item["value"] for item in history]

            summary[metric_name] = {
                "current": values[-1] if values else 0,
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "trend": self.trend_analysis.get(metric_name, {}),
                "data_points": len(values),
            }

        return summary

    def detect_performance_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect performance anomalies.
        
        Args:
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []

        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Need sufficient data
                continue

            values = [item["value"] for item in history]
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:  # No variation
                continue

            # Check recent values for anomalies
            recent_values = values[-5:]  # Last 5 values

            for i, value in enumerate(recent_values):
                z_score = abs(value - mean_val) / std_val

                if z_score > threshold:
                    anomalies.append({
                        "metric": metric_name,
                        "value": value,
                        "expected_range": (mean_val - threshold * std_val, mean_val + threshold * std_val),
                        "z_score": z_score,
                        "severity": "high" if z_score > 3.0 else "medium",
                        "timestamp": history[-(5-i)]["timestamp"],
                    })

        return anomalies


class AdaptiveLearningEngine:
    """Main adaptive learning engine."""

    def __init__(self):
        """Initialize adaptive learning engine."""
        self.settings = get_settings()
        self.performance_tracker = PerformanceTracker()
        self.metrics_collector = get_metrics_collector()

        # Learning state
        self.learning_events = deque(maxlen=10000)
        self.adaptation_strategies = self._initialize_strategies()
        self.active_adaptations = {}

        # Learning parameters
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.min_confidence = 0.7

        # Background tasks
        self.monitoring_task = None
        self.adaptation_task = None
        self.is_running = False

    def _initialize_strategies(self) -> List[AdaptationStrategy]:
        """Initialize adaptation strategies.
        
        Returns:
            List of adaptation strategies
        """
        strategies = [
            AdaptationStrategy(
                name="performance_degradation",
                description="Adapt when performance degrades",
                trigger_conditions={
                    "response_time_increase": 0.5,  # 50% increase
                    "error_rate_increase": 0.2,     # 20% increase
                    "confidence": 0.8,
                },
                adaptation_actions=[
                    "reduce_model_complexity",
                    "increase_safety_weight",
                    "enable_conservative_mode",
                ],
                priority=1,
            ),
            AdaptationStrategy(
                name="high_accuracy_opportunity",
                description="Adapt when high accuracy is achievable",
                trigger_conditions={
                    "accuracy_trend": "improving",
                    "confidence_increase": 0.3,
                    "stability_period": 100,  # episodes
                },
                adaptation_actions=[
                    "increase_model_complexity",
                    "enable_advanced_features",
                    "reduce_exploration",
                ],
                priority=2,
            ),
            AdaptationStrategy(
                name="safety_violations",
                description="Adapt when safety violations occur",
                trigger_conditions={
                    "safety_violations": 5,  # violations in window
                    "violation_trend": "increasing",
                },
                adaptation_actions=[
                    "increase_safety_constraints",
                    "enable_conservative_mode",
                    "reduce_action_space",
                ],
                priority=1,
            ),
            AdaptationStrategy(
                name="user_feedback_negative",
                description="Adapt based on negative user feedback",
                trigger_conditions={
                    "negative_feedback_ratio": 0.3,
                    "feedback_confidence": 0.7,
                },
                adaptation_actions=[
                    "adjust_reward_function",
                    "increase_explanation_detail",
                    "enable_human_in_loop",
                ],
                priority=2,
            ),
            AdaptationStrategy(
                name="resource_optimization",
                description="Optimize resource usage",
                trigger_conditions={
                    "resource_usage_high": 0.9,
                    "performance_stable": True,
                },
                adaptation_actions=[
                    "optimize_model_size",
                    "enable_caching",
                    "reduce_computation",
                ],
                priority=3,
            ),
        ]

        return strategies

    async def start_adaptive_learning(self):
        """Start the adaptive learning system."""
        if self.is_running:
            logger.warning("Adaptive learning already running")
            return

        self.is_running = True
        logger.info("🧠 Starting adaptive learning engine")

        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.adaptation_task = asyncio.create_task(self._adaptation_loop())

        logger.info("✅ Adaptive learning engine started")

    async def stop_adaptive_learning(self):
        """Stop the adaptive learning system."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("🛑 Stopping adaptive learning engine")

        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.adaptation_task:
            self.adaptation_task.cancel()

        logger.info("✅ Adaptive learning engine stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                await self._collect_performance_metrics()
                await self._analyze_learning_events()
                await asyncio.sleep(30)  # Monitor every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error

    async def _adaptation_loop(self):
        """Background adaptation loop."""
        while self.is_running:
            try:
                await self._evaluate_adaptation_strategies()
                await self._apply_adaptations()
                await asyncio.sleep(60)  # Adapt every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}", exc_info=True)
                await asyncio.sleep(120)  # Wait longer on error

    async def _collect_performance_metrics(self):
        """Collect performance metrics."""
        try:
            # Get metrics from collector
            metrics_summary = self.metrics_collector.get_metrics_summary(time_window=300)  # Last 5 minutes

            if "error" not in metrics_summary:
                for metric_name, metric_data in metrics_summary.get("metrics", {}).items():
                    self.performance_tracker.record_metric(
                        metric_name,
                        metric_data.get("mean", 0),
                        {"source": "metrics_collector"}
                    )

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    async def _analyze_learning_events(self):
        """Analyze recent learning events."""
        try:
            # Analyze recent events for patterns
            recent_events = list(self.learning_events)[-100:]  # Last 100 events

            if not recent_events:
                return

            # Analyze event patterns
            event_types = defaultdict(int)
            outcomes = defaultdict(list)

            for event in recent_events:
                event_types[event.event_type] += 1
                if "success" in event.outcome:
                    outcomes[event.event_type].append(event.outcome["success"])

            # Record analysis results
            for event_type, count in event_types.items():
                success_rate = np.mean(outcomes[event_type]) if outcomes[event_type] else 0

                self.performance_tracker.record_metric(
                    f"event_{event_type}_success_rate",
                    success_rate,
                    {"event_count": count}
                )

        except Exception as e:
            logger.error(f"Error analyzing learning events: {e}")

    async def _evaluate_adaptation_strategies(self):
        """Evaluate which adaptation strategies should be triggered."""
        try:
            performance_summary = self.performance_tracker.get_performance_summary()
            anomalies = self.performance_tracker.detect_performance_anomalies()

            for strategy in self.adaptation_strategies:
                if not strategy.enabled:
                    continue

                should_trigger = await self._check_strategy_conditions(
                    strategy, performance_summary, anomalies
                )

                if should_trigger:
                    await self._trigger_adaptation_strategy(strategy)

        except Exception as e:
            logger.error(f"Error evaluating adaptation strategies: {e}")

    async def _check_strategy_conditions(
        self,
        strategy: AdaptationStrategy,
        performance_summary: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> bool:
        """Check if strategy conditions are met.
        
        Args:
            strategy: Adaptation strategy
            performance_summary: Performance summary
            anomalies: Detected anomalies
            
        Returns:
            True if conditions are met
        """
        conditions = strategy.trigger_conditions

        # Check performance degradation
        if strategy.name == "performance_degradation":
            response_time_metric = performance_summary.get("response_time", {})
            error_rate_metric = performance_summary.get("request_success", {})

            if response_time_metric.get("trend", {}).get("direction") == "declining":
                trend_magnitude = response_time_metric.get("trend", {}).get("magnitude", 0)
                if trend_magnitude > conditions.get("response_time_increase", 0.5):
                    return True

            if error_rate_metric.get("trend", {}).get("direction") == "declining":
                return True

        # Check safety violations
        elif strategy.name == "safety_violations":
            safety_anomalies = [a for a in anomalies if "safety" in a["metric"]]
            if len(safety_anomalies) >= conditions.get("safety_violations", 5):
                return True

        # Check high accuracy opportunity
        elif strategy.name == "high_accuracy_opportunity":
            accuracy_metrics = [
                m for name, m in performance_summary.items()
                if "accuracy" in name or "success" in name
            ]

            improving_trends = sum(
                1 for m in accuracy_metrics
                if m.get("trend", {}).get("direction") == "improving"
            )

            if improving_trends > 0:
                return True

        return False

    async def _trigger_adaptation_strategy(self, strategy: AdaptationStrategy):
        """Trigger an adaptation strategy.
        
        Args:
            strategy: Strategy to trigger
        """
        if strategy.name in self.active_adaptations:
            logger.debug(f"Strategy {strategy.name} already active")
            return

        logger.info(f"🔄 Triggering adaptation strategy: {strategy.name}")

        # Record adaptation event
        adaptation_event = LearningEvent(
            timestamp=time.time(),
            event_type="adaptation_triggered",
            context={"strategy": strategy.name, "actions": strategy.adaptation_actions},
            outcome={"status": "initiated"}
        )

        self.learning_events.append(adaptation_event)

        # Mark as active
        self.active_adaptations[strategy.name] = {
            "strategy": strategy,
            "started_at": time.time(),
            "actions_completed": [],
        }

        # Execute adaptation actions
        for action in strategy.adaptation_actions:
            try:
                await self._execute_adaptation_action(action, strategy)
                self.active_adaptations[strategy.name]["actions_completed"].append(action)

            except Exception as e:
                logger.error(f"Error executing adaptation action {action}: {e}")

        # Record metrics
        self.metrics_collector.record_event(
            "adaptation_strategy_triggered",
            {"strategy": strategy.name, "priority": strategy.priority},
            "info"
        )

    async def _execute_adaptation_action(self, action: str, strategy: AdaptationStrategy):
        """Execute a specific adaptation action.
        
        Args:
            action: Action to execute
            strategy: Parent strategy
        """
        logger.info(f"🎯 Executing adaptation action: {action}")

        if action == "reduce_model_complexity":
            # Reduce model complexity
            logger.info("Reducing model complexity for better performance")

        elif action == "increase_safety_weight":
            # Increase safety weight in reward function
            logger.info("Increasing safety weight in reward function")

        elif action == "enable_conservative_mode":
            # Enable conservative decision making
            logger.info("Enabling conservative decision making mode")

        elif action == "increase_model_complexity":
            # Increase model complexity for better accuracy
            logger.info("Increasing model complexity for better accuracy")

        elif action == "enable_advanced_features":
            # Enable advanced RL features
            logger.info("Enabling advanced RL features")

        elif action == "adjust_reward_function":
            # Adjust reward function based on feedback
            logger.info("Adjusting reward function based on user feedback")

        elif action == "optimize_model_size":
            # Optimize model size for resource efficiency
            logger.info("Optimizing model size for resource efficiency")

        else:
            logger.warning(f"Unknown adaptation action: {action}")

    async def _apply_adaptations(self):
        """Apply active adaptations."""
        try:
            completed_adaptations = []

            for strategy_name, adaptation_info in self.active_adaptations.items():
                strategy = adaptation_info["strategy"]
                started_at = adaptation_info["started_at"]

                # Check if adaptation should be completed
                if time.time() - started_at > 300:  # 5 minutes
                    completed_adaptations.append(strategy_name)

                    # Record completion
                    completion_event = LearningEvent(
                        timestamp=time.time(),
                        event_type="adaptation_completed",
                        context={"strategy": strategy_name},
                        outcome={"status": "completed", "duration": time.time() - started_at}
                    )

                    self.learning_events.append(completion_event)
                    logger.info(f"✅ Completed adaptation strategy: {strategy_name}")

            # Remove completed adaptations
            for strategy_name in completed_adaptations:
                del self.active_adaptations[strategy_name]

        except Exception as e:
            logger.error(f"Error applying adaptations: {e}")

    def record_learning_event(self, event: LearningEvent):
        """Record a learning event.
        
        Args:
            event: Learning event to record
        """
        self.learning_events.append(event)

        # Record in metrics collector
        self.metrics_collector.record_event(
            event.event_type,
            event.context,
            "info"
        )

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status.
        
        Returns:
            Adaptation status
        """
        return {
            "is_running": self.is_running,
            "active_adaptations": len(self.active_adaptations),
            "total_strategies": len(self.adaptation_strategies),
            "learning_events": len(self.learning_events),
            "performance_metrics": len(self.performance_tracker.metrics_history),
            "active_strategy_details": {
                name: {
                    "strategy_name": info["strategy"].name,
                    "started_at": info["started_at"],
                    "actions_completed": len(info["actions_completed"]),
                    "total_actions": len(info["strategy"].adaptation_actions),
                }
                for name, info in self.active_adaptations.items()
            },
        }


# Global adaptive learning engine instance
_adaptive_engine: Optional[AdaptiveLearningEngine] = None


def get_adaptive_learning_engine() -> AdaptiveLearningEngine:
    """Get global adaptive learning engine."""
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveLearningEngine()
    return _adaptive_engine
