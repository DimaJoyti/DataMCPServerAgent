"""
Dynamic Optimizer for Pipeline Performance.

This module provides dynamic optimization capabilities for pipeline performance
based on real-time metrics and adaptive tuning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.logging import LoggerMixin, get_logger


class OptimizationStrategy(str, Enum):
    """Optimization strategies."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ACCURACY = "accuracy"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    BALANCED = "balanced"


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""

    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str


class DynamicOptimizer(LoggerMixin):
    """Dynamic optimizer for pipeline performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dynamic optimizer."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Optimization configuration
        self.strategy = OptimizationStrategy(
            self.config.get("strategy", OptimizationStrategy.BALANCED)
        )
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.min_samples = self.config.get("min_samples", 10)

        # Performance history
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_history: List[OptimizationRecommendation] = []

        self.logger.info(f"DynamicOptimizer initialized with strategy: {self.strategy}")

    async def analyze_performance(
        self, metrics: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze performance metrics and provide optimization recommendations."""

        # Store metrics
        self.performance_history.append(metrics)

        # Need minimum samples for optimization
        if len(self.performance_history) < self.min_samples:
            self.logger.debug(
                f"Need {self.min_samples - len(self.performance_history)} more samples"
            )
            return []

        recommendations = []

        # Analyze different aspects based on strategy
        if self.strategy in [OptimizationStrategy.THROUGHPUT, OptimizationStrategy.BALANCED]:
            throughput_recs = await self._optimize_throughput(metrics)
            recommendations.extend(throughput_recs)

        if self.strategy in [OptimizationStrategy.LATENCY, OptimizationStrategy.BALANCED]:
            latency_recs = await self._optimize_latency(metrics)
            recommendations.extend(latency_recs)

        if self.strategy in [
            OptimizationStrategy.RESOURCE_EFFICIENCY,
            OptimizationStrategy.BALANCED,
        ]:
            resource_recs = await self._optimize_resources(metrics)
            recommendations.extend(resource_recs)

        # Store recommendations
        self.optimization_history.extend(recommendations)

        return recommendations

    async def _optimize_throughput(
        self, metrics: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Optimize for throughput."""
        recommendations = []

        current_throughput = metrics.get("events_per_second", 0)
        queue_usage = metrics.get("queue_usage", 0)
        worker_utilization = metrics.get("worker_utilization", 0)

        # If queue is backing up, increase workers
        if queue_usage > 0.8 and worker_utilization > 0.9:
            recommendations.append(
                OptimizationRecommendation(
                    parameter="worker_count",
                    current_value=metrics.get("active_workers", 1),
                    recommended_value=min(metrics.get("active_workers", 1) + 2, 20),
                    expected_improvement=0.3,
                    confidence=0.8,
                    reasoning="High queue usage and worker utilization indicate need for more workers",
                )
            )

        # If workers are underutilized, decrease batch timeout
        elif worker_utilization < 0.5 and queue_usage < 0.3:
            recommendations.append(
                OptimizationRecommendation(
                    parameter="batch_timeout",
                    current_value=metrics.get("batch_timeout", 5.0),
                    recommended_value=max(metrics.get("batch_timeout", 5.0) * 0.8, 1.0),
                    expected_improvement=0.2,
                    confidence=0.7,
                    reasoning="Low utilization suggests faster batch processing could improve throughput",
                )
            )

        return recommendations

    async def _optimize_latency(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Optimize for latency."""
        recommendations = []

        p99_latency = metrics.get("p99_latency_ms", 0)
        avg_latency = metrics.get("avg_processing_time_ms", 0)

        # If latency is high, reduce batch size
        if p99_latency > 1000:  # 1 second
            recommendations.append(
                OptimizationRecommendation(
                    parameter="batch_size",
                    current_value=metrics.get("batch_size", 10),
                    recommended_value=max(metrics.get("batch_size", 10) - 2, 1),
                    expected_improvement=0.25,
                    confidence=0.8,
                    reasoning="High P99 latency suggests smaller batches would reduce processing time",
                )
            )

        # If average latency is high, increase concurrency
        if avg_latency > 500:  # 500ms
            recommendations.append(
                OptimizationRecommendation(
                    parameter="max_concurrent_tasks",
                    current_value=metrics.get("max_concurrent_tasks", 5),
                    recommended_value=min(metrics.get("max_concurrent_tasks", 5) + 3, 20),
                    expected_improvement=0.2,
                    confidence=0.7,
                    reasoning="High average latency suggests more concurrent processing could help",
                )
            )

        return recommendations

    async def _optimize_resources(
        self, metrics: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Optimize for resource efficiency."""
        recommendations = []

        memory_usage = metrics.get("memory_usage_mb", 0)
        cpu_usage = metrics.get("cpu_usage_percent", 0)
        success_rate = metrics.get("success_rate", 1.0)

        # If memory usage is high but success rate is good, reduce batch size
        if memory_usage > 1000 and success_rate > 0.95:  # 1GB
            recommendations.append(
                OptimizationRecommendation(
                    parameter="batch_size",
                    current_value=metrics.get("batch_size", 10),
                    recommended_value=max(metrics.get("batch_size", 10) - 1, 1),
                    expected_improvement=0.15,
                    confidence=0.6,
                    reasoning="High memory usage with good success rate suggests smaller batches",
                )
            )

        # If CPU usage is low, reduce workers
        if cpu_usage < 30 and metrics.get("active_workers", 1) > 2:
            recommendations.append(
                OptimizationRecommendation(
                    parameter="worker_count",
                    current_value=metrics.get("active_workers", 1),
                    recommended_value=max(metrics.get("active_workers", 1) - 1, 2),
                    expected_improvement=0.1,
                    confidence=0.5,
                    reasoning="Low CPU usage suggests fewer workers could maintain performance",
                )
            )

        return recommendations

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        if not self.optimization_history:
            return {"total_recommendations": 0}

        # Group by parameter
        by_parameter = {}
        for rec in self.optimization_history:
            if rec.parameter not in by_parameter:
                by_parameter[rec.parameter] = []
            by_parameter[rec.parameter].append(rec)

        # Calculate average improvements
        total_expected_improvement = sum(
            rec.expected_improvement for rec in self.optimization_history
        )
        avg_confidence = sum(rec.confidence for rec in self.optimization_history) / len(
            self.optimization_history
        )

        return {
            "total_recommendations": len(self.optimization_history),
            "parameters_optimized": list(by_parameter.keys()),
            "total_expected_improvement": total_expected_improvement,
            "average_confidence": avg_confidence,
            "strategy": self.strategy,
            "recent_recommendations": self.optimization_history[-5:],  # Last 5
        }
