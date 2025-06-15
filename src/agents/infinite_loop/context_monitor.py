"""
Context Monitor

Monitors context usage across the infinite loop system to prevent context
window exhaustion and optimize resource allocation.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class ContextUsageSnapshot:
    """Snapshot of context usage at a point in time."""

    timestamp: datetime
    estimated_tokens: int
    memory_usage_mb: float
    active_agents: int
    active_tasks: int
    context_percentage: float
    system_load: float


class ContextMonitor:
    """
    Monitors context usage and system resources for the infinite loop system.

    Features:
    - Token usage estimation and tracking
    - Memory usage monitoring
    - System resource monitoring
    - Context threshold management
    - Usage prediction and optimization
    - Resource cleanup recommendations
    """

    def __init__(self, config: Any):
        """Initialize the context monitor."""
        self.config = config
        self.logger = logging.getLogger("context_monitor")

        # Context tracking
        self.max_context_tokens = 200000  # Estimated max context window
        self.current_estimated_tokens = 0
        self.context_threshold = config.context_threshold

        # Usage history
        self.usage_snapshots: List[ContextUsageSnapshot] = []
        self.max_snapshots = 100

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 10.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None

        # Token estimation factors
        self.token_estimation_factors = {
            "system_prompt": 500,
            "user_prompt": 300,
            "spec_analysis": 200,
            "directory_state": 150,
            "innovation_context": 100,
            "response_overhead": 200,
        }

    async def start_monitoring(self) -> None:
        """Start continuous context monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started context monitoring")

    async def stop_monitoring(self) -> None:
        """Stop context monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        self.logger.info("Stopped context monitoring")

    async def get_context_usage(self) -> float:
        """
        Get current context usage as a percentage (0.0 to 1.0).

        Returns:
            Context usage percentage
        """
        # Update current estimation
        await self._update_context_estimation()

        # Calculate percentage
        usage_percentage = self.current_estimated_tokens / self.max_context_tokens
        return min(usage_percentage, 1.0)

    async def estimate_task_context_cost(
        self,
        spec_analysis: Dict[str, Any],
        innovation_dimension: str,
    ) -> int:
        """
        Estimate context token cost for a single task.

        Args:
            spec_analysis: Specification analysis
            innovation_dimension: Innovation dimension

        Returns:
            Estimated token cost
        """
        base_cost = (
            self.token_estimation_factors["system_prompt"]
            + self.token_estimation_factors["user_prompt"]
            + self.token_estimation_factors["response_overhead"]
        )

        # Add cost based on spec complexity
        spec_complexity = len(str(spec_analysis))
        spec_cost = min(spec_complexity // 4, 500)  # Rough estimation

        # Add cost for innovation dimension
        dimension_cost = len(innovation_dimension) * 2

        total_cost = base_cost + spec_cost + dimension_cost

        return total_cost

    async def estimate_wave_context_cost(
        self,
        wave_size: int,
        spec_analysis: Dict[str, Any],
        innovation_dimensions: List[str],
    ) -> int:
        """
        Estimate context token cost for an entire wave.

        Args:
            wave_size: Number of agents in the wave
            spec_analysis: Specification analysis
            innovation_dimensions: Innovation dimensions for the wave

        Returns:
            Estimated total token cost for the wave
        """
        # Calculate cost per task
        avg_dimension = innovation_dimensions[0] if innovation_dimensions else "default"
        task_cost = await self.estimate_task_context_cost(spec_analysis, avg_dimension)

        # Total cost for the wave
        wave_cost = task_cost * wave_size

        # Add overhead for coordination
        coordination_overhead = wave_size * 50  # 50 tokens per agent for coordination

        return wave_cost + coordination_overhead

    async def can_execute_wave(
        self,
        wave_size: int,
        spec_analysis: Dict[str, Any],
        innovation_dimensions: List[str],
    ) -> Dict[str, Any]:
        """
        Check if a wave can be executed within context limits.

        Args:
            wave_size: Number of agents in the wave
            spec_analysis: Specification analysis
            innovation_dimensions: Innovation dimensions

        Returns:
            Execution feasibility analysis
        """
        current_usage = await self.get_context_usage()
        wave_cost = await self.estimate_wave_context_cost(
            wave_size, spec_analysis, innovation_dimensions
        )

        # Calculate projected usage
        projected_tokens = self.current_estimated_tokens + wave_cost
        projected_usage = projected_tokens / self.max_context_tokens

        can_execute = projected_usage <= self.context_threshold

        return {
            "can_execute": can_execute,
            "current_usage": current_usage,
            "projected_usage": projected_usage,
            "wave_cost_tokens": wave_cost,
            "remaining_capacity": max(0, self.context_threshold - current_usage),
            "recommendation": self._get_execution_recommendation(
                current_usage, projected_usage, wave_size
            ),
        }

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Take usage snapshot
                snapshot = await self._take_usage_snapshot()
                self.usage_snapshots.append(snapshot)

                # Limit snapshot history
                if len(self.usage_snapshots) > self.max_snapshots:
                    self.usage_snapshots.pop(0)

                # Log warnings if usage is high
                if snapshot.context_percentage > 0.8:
                    self.logger.warning(f"High context usage: {snapshot.context_percentage:.1%}")

                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _take_usage_snapshot(self) -> ContextUsageSnapshot:
        """Take a snapshot of current usage."""
        await self._update_context_estimation()

        # Get system metrics
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        system_load = psutil.cpu_percent(interval=0.1)

        # Create snapshot
        snapshot = ContextUsageSnapshot(
            timestamp=datetime.now(),
            estimated_tokens=self.current_estimated_tokens,
            memory_usage_mb=memory_usage,
            active_agents=0,  # Would be updated by agent pool manager
            active_tasks=0,  # Would be updated by agent pool manager
            context_percentage=self.current_estimated_tokens / self.max_context_tokens,
            system_load=system_load,
        )

        return snapshot

    async def _update_context_estimation(self) -> None:
        """Update current context token estimation."""
        # This is a simplified estimation
        # In a real implementation, this would track actual token usage

        # Base context for system
        base_context = 1000

        # Add estimation based on recent activity
        recent_snapshots = self.usage_snapshots[-10:] if self.usage_snapshots else []
        activity_factor = len(recent_snapshots) * 100

        self.current_estimated_tokens = base_context + activity_factor

    def _get_execution_recommendation(
        self,
        current_usage: float,
        projected_usage: float,
        wave_size: int,
    ) -> str:
        """Get recommendation for wave execution."""
        if projected_usage <= 0.5:
            return "safe_to_execute"
        elif projected_usage <= 0.7:
            return "proceed_with_caution"
        elif projected_usage <= self.context_threshold:
            return "reduce_wave_size"
        else:
            return "defer_execution"

    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get context usage statistics."""
        if not self.usage_snapshots:
            return {
                "current_usage": 0.0,
                "average_usage": 0.0,
                "peak_usage": 0.0,
                "trend": "stable",
                "snapshots_count": 0,
            }

        current_usage = await self.get_context_usage()

        # Calculate statistics from snapshots
        usage_values = [s.context_percentage for s in self.usage_snapshots]
        average_usage = sum(usage_values) / len(usage_values)
        peak_usage = max(usage_values)

        # Calculate trend
        if len(usage_values) >= 5:
            recent_avg = sum(usage_values[-5:]) / 5
            older_avg = sum(usage_values[-10:-5]) / 5 if len(usage_values) >= 10 else recent_avg

            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "current_usage": current_usage,
            "average_usage": average_usage,
            "peak_usage": peak_usage,
            "trend": trend,
            "snapshots_count": len(self.usage_snapshots),
            "estimated_tokens": self.current_estimated_tokens,
            "max_tokens": self.max_context_tokens,
            "threshold": self.context_threshold,
        }

    async def optimize_for_context(self) -> Dict[str, Any]:
        """Provide optimization recommendations for context usage."""
        current_usage = await self.get_context_usage()

        recommendations = []

        if current_usage > 0.8:
            recommendations.extend(
                [
                    "Reduce wave size to minimum",
                    "Consider context cleanup",
                    "Defer non-critical tasks",
                ]
            )
        elif current_usage > 0.6:
            recommendations.extend(
                [
                    "Reduce wave size by 25%",
                    "Monitor usage closely",
                ]
            )
        elif current_usage < 0.3:
            recommendations.extend(
                [
                    "Can increase wave size",
                    "Good capacity for complex tasks",
                ]
            )

        return {
            "current_usage": current_usage,
            "optimization_needed": current_usage > 0.7,
            "recommendations": recommendations,
            "suggested_wave_size": self._suggest_optimal_wave_size(current_usage),
        }

    def _suggest_optimal_wave_size(self, current_usage: float) -> int:
        """Suggest optimal wave size based on current usage."""
        if current_usage > 0.8:
            return 1
        elif current_usage > 0.6:
            return 2
        elif current_usage > 0.4:
            return 3
        else:
            return self.config.wave_size_max

    async def cleanup_context(self) -> Dict[str, Any]:
        """Perform context cleanup to free up space."""
        # This would implement actual context cleanup
        # For now, just reset estimation

        old_tokens = self.current_estimated_tokens
        self.current_estimated_tokens = max(1000, self.current_estimated_tokens // 2)

        freed_tokens = old_tokens - self.current_estimated_tokens

        self.logger.info(f"Context cleanup freed {freed_tokens} estimated tokens")

        return {
            "cleanup_performed": True,
            "tokens_freed": freed_tokens,
            "new_usage": await self.get_context_usage(),
        }
