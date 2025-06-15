"""
Wave Manager

Manages wave-based execution for infinite mode, coordinating multiple waves
of parallel agents with progressive sophistication and context monitoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class WaveInfo:
    """Information about a wave of execution."""

    wave_number: int
    wave_size: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    tasks: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    execution_time: Optional[float] = None


class WaveManager:
    """
    Manages wave-based execution for infinite mode.

    Features:
    - Wave planning and sizing optimization
    - Progressive sophistication across waves
    - Context capacity monitoring
    - Wave coordination and synchronization
    - Performance tracking and optimization
    """

    def __init__(self, config: Any):
        """Initialize the wave manager."""
        self.config = config
        self.logger = logging.getLogger("wave_manager")

        # Wave tracking
        self.waves: List[WaveInfo] = []
        self.current_wave: Optional[WaveInfo] = None
        self.wave_counter = 0

        # Performance metrics
        self.total_waves = 0
        self.total_tasks = 0
        self.total_execution_time = 0.0
        self.average_wave_time = 0.0

    async def plan_next_wave(
        self,
        context_usage: float,
        previous_wave_performance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plan the next wave based on context usage and performance.

        Args:
            context_usage: Current context usage (0.0 to 1.0)
            previous_wave_performance: Performance data from previous wave

        Returns:
            Wave plan with size and configuration
        """
        self.wave_counter += 1

        # Calculate optimal wave size
        wave_size = self._calculate_optimal_wave_size(context_usage, previous_wave_performance)

        if wave_size == 0:
            return {
                "can_execute": False,
                "reason": "Context capacity exceeded",
                "wave_number": self.wave_counter,
            }

        # Determine sophistication level
        sophistication_level = self._determine_sophistication_level(self.wave_counter)

        wave_plan = {
            "can_execute": True,
            "wave_number": self.wave_counter,
            "wave_size": wave_size,
            "sophistication_level": sophistication_level,
            "estimated_context_usage": self._estimate_wave_context_usage(wave_size),
            "recommended_timeout": self._calculate_wave_timeout(wave_size),
        }

        self.logger.info(
            f"Planned wave {self.wave_counter}: size={wave_size}, sophistication={sophistication_level}"
        )

        return wave_plan

    async def start_wave(self, wave_plan: Dict[str, Any]) -> WaveInfo:
        """Start a new wave of execution."""
        wave_info = WaveInfo(
            wave_number=wave_plan["wave_number"],
            wave_size=wave_plan["wave_size"],
            start_time=datetime.now(),
            status="running",
        )

        self.current_wave = wave_info
        self.waves.append(wave_info)

        self.logger.info(f"Started wave {wave_info.wave_number} with {wave_info.wave_size} agents")

        return wave_info

    async def complete_wave(
        self,
        wave_info: WaveInfo,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Complete a wave and update statistics."""
        wave_info.end_time = datetime.now()
        wave_info.results = results
        wave_info.execution_time = (wave_info.end_time - wave_info.start_time).total_seconds()

        # Count successes and failures
        wave_info.success_count = sum(1 for r in results if r.get("success", False))
        wave_info.failure_count = len(results) - wave_info.success_count

        # Update status
        if wave_info.failure_count == 0:
            wave_info.status = "completed"
        elif wave_info.success_count > 0:
            wave_info.status = "partially_completed"
        else:
            wave_info.status = "failed"

        # Update global statistics
        self.total_waves += 1
        self.total_tasks += len(results)
        self.total_execution_time += wave_info.execution_time
        self.average_wave_time = self.total_execution_time / self.total_waves

        # Clear current wave
        if self.current_wave == wave_info:
            self.current_wave = None

        completion_summary = {
            "wave_number": wave_info.wave_number,
            "status": wave_info.status,
            "execution_time": wave_info.execution_time,
            "success_count": wave_info.success_count,
            "failure_count": wave_info.failure_count,
            "success_rate": wave_info.success_count / len(results) if results else 0.0,
        }

        self.logger.info(f"Completed wave {wave_info.wave_number}: {completion_summary}")

        return completion_summary

    def _calculate_optimal_wave_size(
        self,
        context_usage: float,
        previous_performance: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Calculate optimal wave size based on context and performance."""
        # Base calculation on remaining context capacity
        remaining_capacity = 1.0 - context_usage

        if remaining_capacity < 0.1:  # Less than 10% capacity
            return 0

        # Calculate base size from remaining capacity
        base_size = int(remaining_capacity * self.config.wave_size_max)
        base_size = max(self.config.wave_size_min, base_size)

        # Adjust based on previous performance
        if previous_performance:
            success_rate = previous_performance.get("success_rate", 1.0)
            avg_time = previous_performance.get("average_execution_time", 1.0)

            # Reduce size if previous wave had low success rate
            if success_rate < 0.7:
                base_size = max(1, int(base_size * 0.8))

            # Reduce size if previous wave was slow
            if avg_time > 60.0:  # More than 1 minute per task
                base_size = max(1, int(base_size * 0.9))

        # Ensure within bounds
        return min(base_size, self.config.wave_size_max)

    def _determine_sophistication_level(self, wave_number: int) -> str:
        """Determine sophistication level for the wave."""
        if wave_number <= 2:
            return "basic"
        elif wave_number <= 5:
            return "intermediate"
        elif wave_number <= 10:
            return "advanced"
        else:
            return "expert"

    def _estimate_wave_context_usage(self, wave_size: int) -> float:
        """Estimate context usage for a wave."""
        # Base estimation: each agent uses some context
        base_usage_per_agent = 0.05  # 5% per agent
        return min(wave_size * base_usage_per_agent, 0.8)  # Cap at 80%

    def _calculate_wave_timeout(self, wave_size: int) -> float:
        """Calculate recommended timeout for a wave."""
        # Base timeout plus scaling factor
        base_timeout = 60.0  # 1 minute base
        scaling_factor = wave_size * 10.0  # 10 seconds per agent
        return base_timeout + scaling_factor

    async def get_wave_statistics(self) -> Dict[str, Any]:
        """Get statistics about wave execution."""
        if not self.waves:
            return {
                "total_waves": 0,
                "total_tasks": 0,
                "average_wave_time": 0.0,
                "overall_success_rate": 0.0,
                "current_wave": None,
            }

        total_successes = sum(wave.success_count for wave in self.waves)
        total_tasks = sum(len(wave.results) for wave in self.waves)
        overall_success_rate = total_successes / total_tasks if total_tasks > 0 else 0.0

        current_wave_info = None
        if self.current_wave:
            current_wave_info = {
                "wave_number": self.current_wave.wave_number,
                "wave_size": self.current_wave.wave_size,
                "status": self.current_wave.status,
                "start_time": self.current_wave.start_time.isoformat(),
            }

        return {
            "total_waves": len(self.waves),
            "total_tasks": total_tasks,
            "average_wave_time": self.average_wave_time,
            "overall_success_rate": overall_success_rate,
            "current_wave": current_wave_info,
            "wave_history": [
                {
                    "wave_number": wave.wave_number,
                    "wave_size": wave.wave_size,
                    "status": wave.status,
                    "success_count": wave.success_count,
                    "failure_count": wave.failure_count,
                    "execution_time": wave.execution_time,
                }
                for wave in self.waves[-10:]  # Last 10 waves
            ],
        }

    async def shutdown(self) -> None:
        """Shutdown the wave manager."""
        self.logger.info("Shutting down wave manager")

        # Mark current wave as cancelled if running
        if self.current_wave and self.current_wave.status == "running":
            self.current_wave.status = "cancelled"
            self.current_wave.end_time = datetime.now()

        self.logger.info(f"Wave manager shutdown complete. Processed {self.total_waves} waves")
