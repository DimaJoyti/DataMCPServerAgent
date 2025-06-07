"""
Pipeline Metrics for monitoring and observability.

This module provides comprehensive metrics collection and reporting
for data pipeline operations and performance monitoring.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import defaultdict, deque
import time

import structlog
from pydantic import BaseModel, Field

from ...core.pipeline_models import PipelineRun, PipelineStatus

class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_retention_hours: int = Field(default=24, description="Metrics retention in hours")
    aggregation_interval: int = Field(default=60, description="Aggregation interval in seconds")

    # Prometheus integration
    enable_prometheus: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")

    # Custom metrics
    enable_custom_metrics: bool = Field(default=True, description="Enable custom metrics")
    max_metric_samples: int = Field(default=1000, description="Maximum metric samples to keep")

class MetricSample(BaseModel):
    """Represents a single metric sample."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)

class PipelineMetrics:
    """
    Pipeline metrics collector and reporter.

    Provides comprehensive metrics collection for pipeline operations,
    performance monitoring, and observability.
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the pipeline metrics.

        Args:
            config: Metrics configuration
            logger: Logger instance
        """
        self.config = config or MetricsConfig()
        self.logger = logger or structlog.get_logger("pipeline_metrics")

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_metric_samples))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Pipeline-specific metrics
        self.pipeline_runs: Dict[str, PipelineRun] = {}
        self.pipeline_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # System metrics
        self.system_metrics: Dict[str, Any] = {}

        self.logger.info("Pipeline metrics initialized")

    async def record_pipeline_start(self, pipeline_run: PipelineRun) -> None:
        """
        Record pipeline start event.

        Args:
            pipeline_run: Pipeline run information
        """
        try:
            self.pipeline_runs[pipeline_run.run_id] = pipeline_run

            # Record metrics
            await self._record_counter("pipeline_starts_total", 1, {
                "pipeline_id": pipeline_run.pipeline_id,
                "triggered_by": pipeline_run.triggered_by or "unknown"
            })

            await self._record_gauge("active_pipelines", len([
                run for run in self.pipeline_runs.values()
                if run.status == PipelineStatus.RUNNING
            ]))

            self.logger.debug(
                "Pipeline start recorded",
                run_id=pipeline_run.run_id,
                pipeline_id=pipeline_run.pipeline_id
            )

        except Exception as e:
            self.logger.error("Failed to record pipeline start", error=str(e))

    async def record_pipeline_completion(self, pipeline_run: PipelineRun) -> None:
        """
        Record pipeline completion event.

        Args:
            pipeline_run: Completed pipeline run
        """
        try:
            # Update stored run
            self.pipeline_runs[pipeline_run.run_id] = pipeline_run

            # Record completion metrics
            await self._record_counter("pipeline_completions_total", 1, {
                "pipeline_id": pipeline_run.pipeline_id,
                "status": pipeline_run.status.value,
                "triggered_by": pipeline_run.triggered_by or "unknown"
            })

            # Record duration
            if pipeline_run.duration:
                await self._record_histogram("pipeline_duration_seconds", pipeline_run.duration, {
                    "pipeline_id": pipeline_run.pipeline_id,
                    "status": pipeline_run.status.value
                })

            # Record task metrics
            for task in pipeline_run.tasks:
                await self._record_counter("task_completions_total", 1, {
                    "pipeline_id": pipeline_run.pipeline_id,
                    "task_id": task.task_id,
                    "task_type": task.config.task_type.value,
                    "status": task.status.value
                })

                if task.duration:
                    await self._record_histogram("task_duration_seconds", task.duration, {
                        "pipeline_id": pipeline_run.pipeline_id,
                        "task_id": task.task_id,
                        "task_type": task.config.task_type.value
                    })

            # Update pipeline statistics
            await self._update_pipeline_stats(pipeline_run)

            # Update active pipelines gauge
            await self._record_gauge("active_pipelines", len([
                run for run in self.pipeline_runs.values()
                if run.status == PipelineStatus.RUNNING
            ]))

            self.logger.debug(
                "Pipeline completion recorded",
                run_id=pipeline_run.run_id,
                pipeline_id=pipeline_run.pipeline_id,
                status=pipeline_run.status,
                duration=pipeline_run.duration
            )

        except Exception as e:
            self.logger.error("Failed to record pipeline completion", error=str(e))

    async def record_data_volume(
        self,
        pipeline_id: str,
        task_id: str,
        records_processed: int,
        bytes_processed: int
    ) -> None:
        """
        Record data volume metrics.

        Args:
            pipeline_id: Pipeline identifier
            task_id: Task identifier
            records_processed: Number of records processed
            bytes_processed: Number of bytes processed
        """
        try:
            await self._record_counter("records_processed_total", records_processed, {
                "pipeline_id": pipeline_id,
                "task_id": task_id
            })

            await self._record_counter("bytes_processed_total", bytes_processed, {
                "pipeline_id": pipeline_id,
                "task_id": task_id
            })

        except Exception as e:
            self.logger.error("Failed to record data volume", error=str(e))

    async def record_error(
        self,
        pipeline_id: str,
        task_id: Optional[str],
        error_type: str,
        error_message: str
    ) -> None:
        """
        Record error metrics.

        Args:
            pipeline_id: Pipeline identifier
            task_id: Task identifier (optional)
            error_type: Type of error
            error_message: Error message
        """
        try:
            labels = {
                "pipeline_id": pipeline_id,
                "error_type": error_type
            }

            if task_id:
                labels["task_id"] = task_id

            await self._record_counter("errors_total", 1, labels)

            self.logger.debug(
                "Error recorded",
                pipeline_id=pipeline_id,
                task_id=task_id,
                error_type=error_type
            )

        except Exception as e:
            self.logger.error("Failed to record error", error=str(e))

    async def update_orchestrator_metrics(
        self,
        active_pipelines: int,
        active_tasks: int,
        registered_pipelines: int
    ) -> None:
        """
        Update orchestrator metrics.

        Args:
            active_pipelines: Number of active pipelines
            active_tasks: Number of active tasks
            registered_pipelines: Number of registered pipelines
        """
        try:
            await self._record_gauge("active_pipelines", active_pipelines)
            await self._record_gauge("active_tasks", active_tasks)
            await self._record_gauge("registered_pipelines", registered_pipelines)

        except Exception as e:
            self.logger.error("Failed to update orchestrator metrics", error=str(e))

    async def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline metrics
        """
        try:
            stats = self.pipeline_stats.get(pipeline_id, {})

            # Get recent runs for this pipeline
            recent_runs = [
                run for run in self.pipeline_runs.values()
                if run.pipeline_id == pipeline_id
            ]

            # Calculate success rate
            if recent_runs:
                successful_runs = len([
                    run for run in recent_runs
                    if run.status == PipelineStatus.SUCCESS
                ])
                success_rate = successful_runs / len(recent_runs)
            else:
                success_rate = 0.0

            # Calculate average duration
            completed_runs = [
                run for run in recent_runs
                if run.duration is not None
            ]

            if completed_runs:
                avg_duration = sum(run.duration for run in completed_runs) / len(completed_runs)
            else:
                avg_duration = 0.0

            return {
                "pipeline_id": pipeline_id,
                "total_runs": len(recent_runs),
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "last_run_time": max(
                    (run.start_time for run in recent_runs if run.start_time),
                    default=None
                ),
                "stats": stats
            }

        except Exception as e:
            self.logger.error("Failed to get pipeline metrics", pipeline_id=pipeline_id, error=str(e))
            return {}

    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system-wide metrics.

        Returns:
            System metrics
        """
        try:
            total_runs = len(self.pipeline_runs)
            active_runs = len([
                run for run in self.pipeline_runs.values()
                if run.status == PipelineStatus.RUNNING
            ])

            successful_runs = len([
                run for run in self.pipeline_runs.values()
                if run.status == PipelineStatus.SUCCESS
            ])

            failed_runs = len([
                run for run in self.pipeline_runs.values()
                if run.status == PipelineStatus.FAILED
            ])

            return {
                "total_pipeline_runs": total_runs,
                "active_pipeline_runs": active_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error("Failed to get system metrics", error=str(e))
            return {}

    async def _record_counter(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        metric_key = self._build_metric_key(name, labels or {})
        self.counters[metric_key] += value

        # Also store as time series
        sample = MetricSample(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.metrics[metric_key].append(sample)

    async def _record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        metric_key = self._build_metric_key(name, labels or {})
        self.gauges[metric_key] = value

        # Also store as time series
        sample = MetricSample(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.metrics[metric_key].append(sample)

    async def _record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram metric."""
        metric_key = self._build_metric_key(name, labels or {})
        self.histograms[metric_key].append(value)

        # Keep only recent samples
        if len(self.histograms[metric_key]) > self.config.max_metric_samples:
            self.histograms[metric_key] = self.histograms[metric_key][-self.config.max_metric_samples:]

        # Also store as time series
        sample = MetricSample(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.metrics[metric_key].append(sample)

    def _build_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Build a metric key from name and labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    async def _update_pipeline_stats(self, pipeline_run: PipelineRun) -> None:
        """Update pipeline statistics."""
        pipeline_id = pipeline_run.pipeline_id

        if pipeline_id not in self.pipeline_stats:
            self.pipeline_stats[pipeline_id] = {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_duration": 0.0,
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0
            }

        stats = self.pipeline_stats[pipeline_id]
        stats["total_runs"] += 1

        if pipeline_run.status == PipelineStatus.SUCCESS:
            stats["successful_runs"] += 1
        elif pipeline_run.status == PipelineStatus.FAILED:
            stats["failed_runs"] += 1

        if pipeline_run.duration:
            stats["total_duration"] += pipeline_run.duration

        # Task statistics
        for task in pipeline_run.tasks:
            stats["total_tasks"] += 1
            if task.status.value == "success":
                stats["successful_tasks"] += 1
            elif task.status.value == "failed":
                stats["failed_tasks"] += 1

    async def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus formatted metrics
        """
        lines = []

        # Export counters
        for metric_key, value in self.counters.items():
            lines.append(f"# TYPE {metric_key.split('{')[0]} counter")
            lines.append(f"{metric_key} {value}")

        # Export gauges
        for metric_key, value in self.gauges.items():
            lines.append(f"# TYPE {metric_key.split('{')[0]} gauge")
            lines.append(f"{metric_key} {value}")

        # Export histograms (simplified)
        for metric_key, values in self.histograms.items():
            if values:
                lines.append(f"# TYPE {metric_key.split('{')[0]} histogram")
                lines.append(f"{metric_key}_sum {sum(values)}")
                lines.append(f"{metric_key}_count {len(values)}")

        return "\n".join(lines)

    async def cleanup_old_metrics(self) -> None:
        """Clean up old metrics based on retention policy."""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (self.config.metrics_retention_hours * 3600)

            for metric_key, samples in self.metrics.items():
                # Remove old samples
                while samples and samples[0].timestamp.timestamp() < cutoff_time:
                    samples.popleft()

            # Clean up old pipeline runs
            old_runs = [
                run_id for run_id, run in self.pipeline_runs.items()
                if (run.start_time and
                    run.start_time.timestamp() < cutoff_time and
                    run.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.CANCELLED])
            ]

            for run_id in old_runs:
                del self.pipeline_runs[run_id]

            self.logger.debug(f"Cleaned up {len(old_runs)} old pipeline runs")

        except Exception as e:
            self.logger.error("Failed to cleanup old metrics", error=str(e))
