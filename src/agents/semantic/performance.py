"""
Performance and Scalability Management

Provides performance monitoring, optimization, and scalability features
for semantic agents and the coordination system.
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for agents and operations."""

    agent_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks and analyzes performance metrics for semantic agents.

    Features:
    - Real-time performance monitoring
    - Resource usage tracking
    - Performance trend analysis
    - Bottleneck identification
    - Optimization recommendations
    """

    def __init__(self, max_metrics_history: int = 10000):
        """Initialize the performance tracker."""
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.system_stats: Dict[str, Any] = {}

        self.logger = logging.getLogger("performance_tracker")

    def start_operation(
        self,
        agent_id: str,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start tracking a new operation."""
        operation_id = str(uuid.uuid4())

        # Get current system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        metrics = PerformanceMetrics(
            agent_id=agent_id,
            operation_type=operation_type,
            start_time=datetime.now(),
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            metadata=metadata or {},
        )

        self.active_operations[operation_id] = metrics

        self.logger.debug(f"Started tracking operation {operation_id} for agent {agent_id}")
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[PerformanceMetrics]:
        """End tracking an operation."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Operation {operation_id} not found in active operations")
            return None

        metrics = self.active_operations.pop(operation_id)
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.success = success
        metrics.error_message = error_message

        if result_metadata:
            metrics.metadata.update(result_metadata)

        # Add to history
        self.metrics_history.append(metrics)

        # Update agent statistics
        self._update_agent_stats(metrics)

        self.logger.debug(f"Completed tracking operation {operation_id}")
        return metrics

    def get_agent_performance(
        self,
        agent_id: str,
        time_window: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """Get performance statistics for a specific agent."""
        time_window = time_window or timedelta(hours=1)
        cutoff_time = datetime.now() - time_window

        # Filter metrics for this agent and time window
        agent_metrics = [
            m for m in self.metrics_history if m.agent_id == agent_id and m.start_time > cutoff_time
        ]

        if not agent_metrics:
            return {"agent_id": agent_id, "metrics_count": 0}

        # Calculate statistics
        total_operations = len(agent_metrics)
        successful_operations = sum(1 for m in agent_metrics if m.success)
        failed_operations = total_operations - successful_operations

        durations = [m.duration_ms for m in agent_metrics if m.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0

        memory_usage = [m.memory_usage_mb for m in agent_metrics if m.memory_usage_mb is not None]
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

        cpu_usage = [m.cpu_usage_percent for m in agent_metrics if m.cpu_usage_percent is not None]
        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

        # Group by operation type
        operation_stats = defaultdict(list)
        for metric in agent_metrics:
            operation_stats[metric.operation_type].append(metric)

        operation_summary = {}
        for op_type, ops in operation_stats.items():
            op_durations = [op.duration_ms for op in ops if op.duration_ms is not None]
            operation_summary[op_type] = {
                "count": len(ops),
                "success_rate": sum(1 for op in ops if op.success) / len(ops),
                "avg_duration_ms": sum(op_durations) / len(op_durations) if op_durations else 0,
            }

        return {
            "agent_id": agent_id,
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / total_operations,
            "avg_duration_ms": avg_duration,
            "avg_memory_usage_mb": avg_memory,
            "avg_cpu_usage_percent": avg_cpu,
            "operation_breakdown": operation_summary,
        }

    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Active operations
        active_ops_by_agent = defaultdict(int)
        for metrics in self.active_operations.values():
            active_ops_by_agent[metrics.agent_id] += 1

        # Recent performance trends
        recent_metrics = [
            m for m in self.metrics_history if m.start_time > datetime.now() - timedelta(minutes=5)
        ]

        recent_success_rate = 0
        if recent_metrics:
            recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)

        return {
            "timestamp": datetime.now(),
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            },
            "active_operations": {
                "total": len(self.active_operations),
                "by_agent": dict(active_ops_by_agent),
            },
            "recent_performance": {
                "operations_last_5min": len(recent_metrics),
                "success_rate_last_5min": recent_success_rate,
            },
            "metrics_history_size": len(self.metrics_history),
        }

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the system."""
        bottlenecks = []

        # Check for slow operations
        recent_metrics = [
            m
            for m in self.metrics_history
            if m.start_time > datetime.now() - timedelta(hours=1) and m.duration_ms is not None
        ]

        if recent_metrics:
            durations = [m.duration_ms for m in recent_metrics]
            avg_duration = sum(durations) / len(durations)

            # Find operations significantly slower than average
            slow_threshold = avg_duration * 2
            slow_operations = [m for m in recent_metrics if m.duration_ms > slow_threshold]

            if slow_operations:
                slow_agents = defaultdict(int)
                slow_operation_types = defaultdict(int)

                for op in slow_operations:
                    slow_agents[op.agent_id] += 1
                    slow_operation_types[op.operation_type] += 1

                bottlenecks.append(
                    {
                        "type": "slow_operations",
                        "description": f"Found {len(slow_operations)} slow operations",
                        "threshold_ms": slow_threshold,
                        "affected_agents": dict(slow_agents),
                        "affected_operation_types": dict(slow_operation_types),
                    }
                )

        # Check for high failure rates
        failed_metrics = [m for m in recent_metrics if not m.success]
        if failed_metrics and recent_metrics:
            failure_rate = len(failed_metrics) / len(recent_metrics)
            if failure_rate > 0.1:  # More than 10% failure rate
                failed_agents = defaultdict(int)
                failed_operation_types = defaultdict(int)

                for op in failed_metrics:
                    failed_agents[op.agent_id] += 1
                    failed_operation_types[op.operation_type] += 1

                bottlenecks.append(
                    {
                        "type": "high_failure_rate",
                        "description": f"High failure rate: {failure_rate:.2%}",
                        "failure_rate": failure_rate,
                        "affected_agents": dict(failed_agents),
                        "affected_operation_types": dict(failed_operation_types),
                    }
                )

        # Check system resources
        system_perf = self.get_system_performance()
        resources = system_perf["system_resources"]

        if resources["cpu_percent"] > 80:
            bottlenecks.append(
                {
                    "type": "high_cpu_usage",
                    "description": f"High CPU usage: {resources['cpu_percent']:.1f}%",
                    "cpu_percent": resources["cpu_percent"],
                }
            )

        if resources["memory_percent"] > 80:
            bottlenecks.append(
                {
                    "type": "high_memory_usage",
                    "description": f"High memory usage: {resources['memory_percent']:.1f}%",
                    "memory_percent": resources["memory_percent"],
                }
            )

        return bottlenecks

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on performance analysis."""
        recommendations = []
        bottlenecks = self.identify_bottlenecks()

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_operations":
                recommendations.append(
                    {
                        "type": "performance_optimization",
                        "priority": "high",
                        "description": "Optimize slow operations",
                        "actions": [
                            "Review and optimize slow operation types",
                            "Consider caching for frequently accessed data",
                            "Implement parallel processing where possible",
                        ],
                        "affected_components": bottleneck["affected_operation_types"],
                    }
                )

            elif bottleneck["type"] == "high_failure_rate":
                recommendations.append(
                    {
                        "type": "reliability_improvement",
                        "priority": "critical",
                        "description": "Reduce failure rate",
                        "actions": [
                            "Implement better error handling",
                            "Add retry mechanisms",
                            "Review and fix failing operations",
                        ],
                        "affected_components": bottleneck["affected_agents"],
                    }
                )

            elif bottleneck["type"] == "high_cpu_usage":
                recommendations.append(
                    {
                        "type": "resource_optimization",
                        "priority": "medium",
                        "description": "Reduce CPU usage",
                        "actions": [
                            "Implement CPU-intensive operation queuing",
                            "Consider horizontal scaling",
                            "Optimize algorithms and data structures",
                        ],
                    }
                )

            elif bottleneck["type"] == "high_memory_usage":
                recommendations.append(
                    {
                        "type": "memory_optimization",
                        "priority": "medium",
                        "description": "Reduce memory usage",
                        "actions": [
                            "Implement memory cleanup routines",
                            "Optimize data structures",
                            "Consider memory-efficient algorithms",
                        ],
                    }
                )

        # General recommendations
        if not bottlenecks:
            recommendations.append(
                {
                    "type": "general_optimization",
                    "priority": "low",
                    "description": "System performing well",
                    "actions": [
                        "Continue monitoring performance",
                        "Consider proactive scaling",
                        "Implement performance testing",
                    ],
                }
            )

        return recommendations

    def _update_agent_stats(self, metrics: PerformanceMetrics) -> None:
        """Update agent statistics with new metrics."""
        agent_id = metrics.agent_id

        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "total_operations": 0,
                "successful_operations": 0,
                "total_duration_ms": 0,
                "last_operation": None,
            }

        stats = self.agent_stats[agent_id]
        stats["total_operations"] += 1

        if metrics.success:
            stats["successful_operations"] += 1

        if metrics.duration_ms:
            stats["total_duration_ms"] += metrics.duration_ms

        stats["last_operation"] = metrics.end_time

    def clear_old_metrics(self, older_than: timedelta = timedelta(days=7)) -> int:
        """Clear old metrics to free memory."""
        cutoff_time = datetime.now() - older_than

        original_count = len(self.metrics_history)

        # Filter out old metrics
        self.metrics_history = deque(
            (m for m in self.metrics_history if m.start_time > cutoff_time),
            maxlen=self.max_metrics_history,
        )

        cleared_count = original_count - len(self.metrics_history)

        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} old metrics")

        return cleared_count


class CacheManager:
    """
    Manages caching for improved performance.

    Features:
    - LRU cache implementation
    - TTL-based expiration
    - Cache hit/miss tracking
    - Memory-aware cache sizing
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize the cache manager."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: deque = deque()
        self.hit_count = 0
        self.miss_count = 0

        self.logger = logging.getLogger("cache_manager")

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key not in self.cache:
            self.miss_count += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if entry["expires_at"] < time.time():
            del self.cache[key]
            self.miss_count += 1
            return None

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        self.hit_count += 1
        return entry["value"]

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        # Remove if already exists
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
        else:
            # Check size limit
            while len(self.cache) >= self.max_size:
                # Remove least recently used
                if self.access_order:
                    lru_key = self.access_order.popleft()
                    if lru_key in self.cache:
                        del self.cache[lru_key]
                else:
                    break

        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time(),
        }

        self.access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_usage_estimate": len(str(self.cache)),
        }
