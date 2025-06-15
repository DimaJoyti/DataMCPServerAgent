"""
Pipeline Coordinator for Orchestration.

This module coordinates multiple pipelines and manages their interactions,
resource allocation, and overall system performance.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.logging import LoggerMixin, get_logger

from .optimizer import DynamicOptimizer, OptimizationRecommendation
from .router import PipelineRouter, RoutingDecision


class CoordinatorStatus(str, Enum):
    """Coordinator status."""

    IDLE = "idle"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    ERROR = "error"


@dataclass
class PipelineInstance:
    """Running pipeline instance."""

    pipeline_id: str
    pipeline_type: str
    status: str
    created_at: float
    last_activity: float
    processed_items: int = 0
    failed_items: int = 0
    resource_usage: Dict[str, Any] = None

    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}


class PipelineCoordinator(LoggerMixin):
    """Coordinates multiple pipelines and optimizes system performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline coordinator."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Components
        self.router = PipelineRouter(self.config.get("router", {}))
        self.optimizer = DynamicOptimizer(self.config.get("optimizer", {}))

        # Coordinator state
        self.status = CoordinatorStatus.IDLE
        self.active_pipelines: Dict[str, PipelineInstance] = {}
        self.coordination_metrics: Dict[str, Any] = {}

        # Configuration
        self.max_concurrent_pipelines = self.config.get("max_concurrent_pipelines", 10)
        self.optimization_interval = self.config.get("optimization_interval", 60.0)  # seconds
        self.cleanup_interval = self.config.get("cleanup_interval", 300.0)  # 5 minutes

        # Tasks
        self.is_running = False
        self.coordination_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        self.logger.info("PipelineCoordinator initialized")

    async def start(self) -> None:
        """Start the coordinator."""
        if self.is_running:
            return

        self.is_running = True
        self.status = CoordinatorStatus.ACTIVE

        # Start coordination tasks
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("PipelineCoordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self.is_running:
            return

        self.is_running = False
        self.status = CoordinatorStatus.IDLE

        # Cancel tasks
        for task in [self.coordination_task, self.optimization_task, self.cleanup_task]:
            if task:
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            *[
                task
                for task in [self.coordination_task, self.optimization_task, self.cleanup_task]
                if task
            ],
            return_exceptions=True,
        )

        self.logger.info("PipelineCoordinator stopped")

    async def coordinate_request(
        self, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Coordinate processing of a request."""

        try:
            # Route to appropriate pipeline
            routing_decision = await self.router.route_content(content, metadata)

            # Check resource availability
            if not await self._check_resource_availability(routing_decision):
                return {
                    "status": "rejected",
                    "reason": "Insufficient resources",
                    "routing_decision": routing_decision.__dict__,
                }

            # Create or get pipeline instance
            pipeline_instance = await self._get_or_create_pipeline(routing_decision)

            # Process request (placeholder - in real implementation, this would delegate to
            # actual pipeline)
            result = await self._process_with_pipeline(pipeline_instance, content, metadata)

            # Update metrics
            await self._update_coordination_metrics(pipeline_instance, result)

            return {
                "status": "completed",
                "pipeline_id": pipeline_instance.pipeline_id,
                "pipeline_type": pipeline_instance.pipeline_type,
                "routing_decision": routing_decision.__dict__,
                "result": result,
            }

        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _check_resource_availability(self, routing_decision: RoutingDecision) -> bool:
        """Check if resources are available for the routing decision."""

        # Check concurrent pipeline limit
        if len(self.active_pipelines) >= self.max_concurrent_pipelines:
            self.logger.warning("Maximum concurrent pipelines reached")
            return False

        # Check resource requirements (simplified)
        required_memory = routing_decision.resource_requirements.get("memory_mb", 0)
        required_cores = routing_decision.resource_requirements.get("cpu_cores", 0)

        # In a real implementation, this would check actual system resources
        # For now, assume resources are available if requirements are reasonable
        if required_memory > 2000 or required_cores > 8:  # 2GB, 8 cores
            self.logger.warning(
                f"Resource requirements too high: {routing_decision.resource_requirements}"
            )
            return False

        return True

    async def _get_or_create_pipeline(self, routing_decision: RoutingDecision) -> PipelineInstance:
        """Get existing or create new pipeline instance."""

        # Look for existing pipeline of the same type
        for pipeline in self.active_pipelines.values():
            if (
                pipeline.pipeline_type == routing_decision.pipeline_type.value
                and pipeline.status == "idle"
            ):
                pipeline.status = "active"
                pipeline.last_activity = time.time()
                return pipeline

        # Create new pipeline instance
        pipeline_id = f"{routing_decision.pipeline_type.value}_{int(time.time() * 1000)}"

        pipeline_instance = PipelineInstance(
            pipeline_id=pipeline_id,
            pipeline_type=routing_decision.pipeline_type.value,
            status="active",
            created_at=time.time(),
            last_activity=time.time(),
            resource_usage=routing_decision.resource_requirements,
        )

        self.active_pipelines[pipeline_id] = pipeline_instance

        self.logger.info(f"Created new pipeline instance: {pipeline_id}")
        return pipeline_instance

    async def _process_with_pipeline(
        self,
        pipeline_instance: PipelineInstance,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process content with the specified pipeline instance."""

        # Simulate processing (in real implementation, this would delegate to actual pipeline)
        processing_time = 0.1 + (hash(str(content)) % 100) / 1000  # 0.1-0.2 seconds
        await asyncio.sleep(processing_time)

        # Update pipeline metrics
        pipeline_instance.processed_items += 1
        pipeline_instance.last_activity = time.time()

        # Simulate occasional failures
        if hash(str(content)) % 20 == 0:  # 5% failure rate
            pipeline_instance.failed_items += 1
            raise Exception("Simulated processing failure")

        return {
            "processed_by": pipeline_instance.pipeline_id,
            "processing_time": processing_time,
            "pipeline_type": pipeline_instance.pipeline_type,
            "success": True,
        }

    async def _update_coordination_metrics(
        self, pipeline_instance: PipelineInstance, result: Dict[str, Any]
    ) -> None:
        """Update coordination metrics."""

        current_time = time.time()

        if "coordination_metrics" not in self.coordination_metrics:
            self.coordination_metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_processing_time": 0.0,
                "pipeline_usage": {},
                "last_updated": current_time,
            }

        metrics = self.coordination_metrics
        metrics["total_requests"] += 1
        metrics["last_updated"] = current_time

        if result.get("success", False):
            metrics["successful_requests"] += 1
            metrics["total_processing_time"] += result.get("processing_time", 0)
        else:
            metrics["failed_requests"] += 1

        # Update pipeline usage
        pipeline_type = pipeline_instance.pipeline_type
        if pipeline_type not in metrics["pipeline_usage"]:
            metrics["pipeline_usage"][pipeline_type] = 0
        metrics["pipeline_usage"][pipeline_type] += 1

    async def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                # Monitor active pipelines
                await self._monitor_pipelines()

                # Update system metrics
                await self._update_system_metrics()

                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5.0)

    async def _optimization_loop(self) -> None:
        """Optimization loop."""
        while self.is_running:
            try:
                self.status = CoordinatorStatus.OPTIMIZING

                # Collect system metrics
                system_metrics = await self._collect_system_metrics()

                # Get optimization recommendations
                recommendations = await self.optimizer.analyze_performance(system_metrics)

                # Apply recommendations (simplified)
                if recommendations:
                    await self._apply_optimizations(recommendations)

                self.status = CoordinatorStatus.ACTIVE

                await asyncio.sleep(self.optimization_interval)

            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                self.status = CoordinatorStatus.ERROR
                await asyncio.sleep(self.optimization_interval)

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for idle pipelines."""
        while self.is_running:
            try:
                current_time = time.time()
                idle_threshold = 300.0  # 5 minutes

                # Find idle pipelines
                idle_pipelines = []
                for pipeline_id, pipeline in self.active_pipelines.items():
                    if (
                        current_time - pipeline.last_activity > idle_threshold
                        and pipeline.status == "idle"
                    ):
                        idle_pipelines.append(pipeline_id)

                # Remove idle pipelines
                for pipeline_id in idle_pipelines:
                    del self.active_pipelines[pipeline_id]
                    self.logger.info(f"Cleaned up idle pipeline: {pipeline_id}")

                await asyncio.sleep(self.cleanup_interval)

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _monitor_pipelines(self) -> None:
        """Monitor active pipelines."""
        for pipeline in self.active_pipelines.values():
            # Mark pipelines as idle if not recently active
            if time.time() - pipeline.last_activity > 60.0:  # 1 minute
                pipeline.status = "idle"

    async def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        # This would collect real system metrics in production
        pass

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics for optimization."""

        # Calculate aggregate metrics from active pipelines
        total_processed = sum(p.processed_items for p in self.active_pipelines.values())
        total_failed = sum(p.failed_items for p in self.active_pipelines.values())

        success_rate = total_processed / max(total_processed + total_failed, 1)

        return {
            "active_pipelines": len(self.active_pipelines),
            "total_processed": total_processed,
            "total_failed": total_failed,
            "success_rate": success_rate,
            "coordination_metrics": self.coordination_metrics,
        }

    async def _apply_optimizations(self, recommendations: List[OptimizationRecommendation]) -> None:
        """Apply optimization recommendations."""

        for rec in recommendations:
            self.logger.info(f"Applying optimization: {rec.parameter} -> {rec.recommended_value}")
            # In a real implementation, this would apply the optimization
            # For now, just log the recommendation

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "status": self.status,
            "active_pipelines": len(self.active_pipelines),
            "pipeline_details": [
                {
                    "id": p.pipeline_id,
                    "type": p.pipeline_type,
                    "status": p.status,
                    "processed": p.processed_items,
                    "failed": p.failed_items,
                }
                for p in self.active_pipelines.values()
            ],
            "coordination_metrics": self.coordination_metrics,
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }
