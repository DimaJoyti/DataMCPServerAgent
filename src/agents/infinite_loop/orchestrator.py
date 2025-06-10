"""
Infinite Agentic Loop Orchestrator

Main coordinator for the infinite generation system that manages the entire
lifecycle of specification-driven content generation with parallel agents.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from .specification_parser import SpecificationParser
from .directory_analyzer import DirectoryAnalyzer
from .agent_pool_manager import AgentPoolManager
from .wave_manager import WaveManager
from .context_monitor import ContextMonitor
from .task_assignment_engine import TaskAssignmentEngine
from .progress_tracker import ProgressTracker
from .quality_controller import QualityController
from .parallel_executor import StatePersistence, ErrorRecoveryManager


@dataclass
class InfiniteLoopConfig:
    """Configuration for the infinite agentic loop system."""
    
    # Core settings
    max_parallel_agents: int = 5
    wave_size_min: int = 3
    wave_size_max: int = 5
    context_threshold: float = 0.8
    max_iterations: Optional[int] = None
    
    # Quality control
    quality_threshold: float = 0.7
    uniqueness_threshold: float = 0.8
    validation_enabled: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    error_recovery_enabled: bool = True
    
    # Performance
    batch_processing: bool = True
    async_execution: bool = True
    memory_optimization: bool = True
    
    # Logging
    log_level: str = "INFO"
    detailed_logging: bool = False


@dataclass
class ExecutionState:
    """Current state of the infinite loop execution."""
    
    # Execution metadata
    session_id: str
    start_time: datetime
    current_wave: int = 0
    total_iterations: int = 0
    
    # Status tracking
    is_running: bool = False
    is_infinite: bool = False
    context_usage: float = 0.0
    
    # Results
    completed_iterations: List[str] = field(default_factory=list)
    failed_iterations: List[str] = field(default_factory=list)
    active_agents: Dict[str, str] = field(default_factory=dict)
    
    # Performance metrics
    average_iteration_time: float = 0.0
    success_rate: float = 1.0
    quality_score: float = 0.0


class InfiniteAgenticLoopOrchestrator:
    """
    Main orchestrator for the infinite agentic loop system.
    
    Coordinates specification analysis, directory reconnaissance, agent management,
    wave-based execution, and context monitoring for infinite content generation.
    """
    
    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        config: Optional[InfiniteLoopConfig] = None,
    ):
        """Initialize the infinite loop orchestrator."""
        self.model = model
        self.tools = tools
        self.config = config or InfiniteLoopConfig()
        
        # Setup logging
        self.logger = logging.getLogger("infinite_loop_orchestrator")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Initialize core components
        self.spec_parser = SpecificationParser()
        self.directory_analyzer = DirectoryAnalyzer()
        self.agent_pool_manager = AgentPoolManager(model, tools, self.config)
        self.wave_manager = WaveManager(self.config)
        self.context_monitor = ContextMonitor(self.config)
        self.task_assignment_engine = TaskAssignmentEngine()
        self.progress_tracker = ProgressTracker()
        self.quality_controller = QualityController(self.config)
        self.state_persistence = StatePersistence()
        self.error_recovery = ErrorRecoveryManager(self.config)
        
        # Execution state
        self.execution_state: Optional[ExecutionState] = None
        self.is_shutting_down = False
        
    async def execute_infinite_loop(
        self,
        spec_file: Union[str, Path],
        output_dir: Union[str, Path],
        count: Union[int, str],
    ) -> Dict[str, Any]:
        """
        Execute the infinite agentic loop with the given parameters.
        
        Args:
            spec_file: Path to the specification file
            output_dir: Directory for output iterations
            count: Number of iterations (integer or "infinite")
            
        Returns:
            Execution results and statistics
        """
        try:
            # Initialize execution
            session_id = f"infinite_loop_{int(time.time())}"
            self.execution_state = ExecutionState(
                session_id=session_id,
                start_time=datetime.now(),
                is_infinite=(count == "infinite"),
            )
            
            self.logger.info(f"Starting infinite loop execution: {session_id}")
            self.logger.info(f"Spec file: {spec_file}")
            self.logger.info(f"Output dir: {output_dir}")
            self.logger.info(f"Count: {count}")
            
            # Phase 1: Specification Analysis
            spec_analysis = await self._analyze_specification(spec_file)
            
            # Phase 2: Directory Reconnaissance
            directory_state = await self._analyze_directory(output_dir)
            
            # Phase 3: Iteration Strategy
            iteration_strategy = await self._plan_iteration_strategy(
                spec_analysis, directory_state, count
            )
            
            # Phase 4 & 5: Execute based on mode
            if self.execution_state.is_infinite:
                results = await self._execute_infinite_mode(
                    spec_analysis, directory_state, iteration_strategy, output_dir
                )
            else:
                results = await self._execute_finite_mode(
                    spec_analysis, directory_state, iteration_strategy, output_dir, int(count)
                )
            
            # Finalize execution
            self.execution_state.is_running = False
            
            return {
                "success": True,
                "session_id": session_id,
                "execution_state": self.execution_state,
                "results": results,
                "statistics": self._generate_statistics(),
            }
            
        except Exception as e:
            self.logger.error(f"Infinite loop execution failed: {str(e)}")
            if self.execution_state:
                self.execution_state.is_running = False
            
            return {
                "success": False,
                "error": str(e),
                "session_id": getattr(self.execution_state, "session_id", "unknown"),
                "execution_state": self.execution_state,
            }
    
    async def _analyze_specification(self, spec_file: Union[str, Path]) -> Dict[str, Any]:
        """Phase 1: Analyze the specification file."""
        self.logger.info("Phase 1: Analyzing specification file")
        
        spec_analysis = await self.spec_parser.parse_specification(spec_file)
        
        self.logger.info(f"Specification analysis complete:")
        self.logger.info(f"- Content type: {spec_analysis.get('content_type', 'unknown')}")
        self.logger.info(f"- Format: {spec_analysis.get('format', 'unknown')}")
        self.logger.info(f"- Evolution pattern: {spec_analysis.get('evolution_pattern', 'unknown')}")
        
        return spec_analysis
    
    async def _analyze_directory(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Phase 2: Analyze the output directory."""
        self.logger.info("Phase 2: Analyzing output directory")
        
        directory_state = await self.directory_analyzer.analyze_directory(output_dir)
        
        self.logger.info(f"Directory analysis complete:")
        self.logger.info(f"- Existing files: {len(directory_state.get('existing_files', []))}")
        self.logger.info(f"- Highest iteration: {directory_state.get('highest_iteration', 0)}")
        self.logger.info(f"- Content evolution: {directory_state.get('evolution_summary', 'none')}")
        
        return directory_state
    
    async def _plan_iteration_strategy(
        self,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        count: Union[int, str],
    ) -> Dict[str, Any]:
        """Phase 3: Plan the iteration strategy."""
        self.logger.info("Phase 3: Planning iteration strategy")
        
        starting_iteration = directory_state.get("highest_iteration", 0) + 1
        
        strategy = {
            "starting_iteration": starting_iteration,
            "target_count": count,
            "is_infinite": (count == "infinite"),
            "wave_strategy": self._determine_wave_strategy(count),
            "innovation_dimensions": self._extract_innovation_dimensions(spec_analysis),
            "quality_requirements": spec_analysis.get("quality_requirements", {}),
        }
        
        self.logger.info(f"Iteration strategy planned:")
        self.logger.info(f"- Starting iteration: {starting_iteration}")
        self.logger.info(f"- Wave strategy: {strategy['wave_strategy']}")
        self.logger.info(f"- Innovation dimensions: {len(strategy['innovation_dimensions'])}")
        
        return strategy
    
    def _determine_wave_strategy(self, count: Union[int, str]) -> Dict[str, Any]:
        """Determine the wave execution strategy based on count."""
        if count == "infinite":
            return {
                "type": "infinite_waves",
                "wave_size": self.config.wave_size_min,
                "max_waves": None,
                "context_monitoring": True,
            }
        elif isinstance(count, int):
            if count <= 5:
                return {
                    "type": "single_wave",
                    "wave_size": count,
                    "max_waves": 1,
                    "context_monitoring": False,
                }
            elif count <= 20:
                return {
                    "type": "batched_waves",
                    "wave_size": 5,
                    "max_waves": (count + 4) // 5,
                    "context_monitoring": True,
                }
            else:
                return {
                    "type": "large_batched_waves",
                    "wave_size": self.config.wave_size_max,
                    "max_waves": (count + self.config.wave_size_max - 1) // self.config.wave_size_max,
                    "context_monitoring": True,
                }
        
        return {"type": "unknown", "wave_size": 1, "max_waves": 1}
    
    def _extract_innovation_dimensions(self, spec_analysis: Dict[str, Any]) -> List[str]:
        """Extract innovation dimensions from specification analysis."""
        dimensions = []
        
        # Extract from specification
        if "innovation_areas" in spec_analysis:
            dimensions.extend(spec_analysis["innovation_areas"])
        
        # Default dimensions
        default_dimensions = [
            "functional_enhancement",
            "structural_innovation",
            "interaction_patterns",
            "performance_optimization",
            "user_experience",
            "integration_capabilities",
            "scalability_improvements",
            "security_enhancements",
            "accessibility_features",
            "paradigm_shifts",
        ]
        
        # Combine and deduplicate
        all_dimensions = list(set(dimensions + default_dimensions))
        
        return all_dimensions
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate execution statistics."""
        if not self.execution_state:
            return {}
        
        execution_time = (datetime.now() - self.execution_state.start_time).total_seconds()
        
        return {
            "execution_time_seconds": execution_time,
            "total_iterations": self.execution_state.total_iterations,
            "completed_iterations": len(self.execution_state.completed_iterations),
            "failed_iterations": len(self.execution_state.failed_iterations),
            "success_rate": self.execution_state.success_rate,
            "average_iteration_time": self.execution_state.average_iteration_time,
            "quality_score": self.execution_state.quality_score,
            "waves_completed": self.execution_state.current_wave,
            "context_usage": self.execution_state.context_usage,
        }
    
    async def _execute_infinite_mode(
        self,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        iteration_strategy: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """Execute infinite mode with wave-based generation."""
        self.logger.info("Phase 4-5: Executing infinite mode")

        self.execution_state.is_running = True
        results = {"waves": [], "total_iterations": 0}

        wave_number = 1
        current_iteration = iteration_strategy["starting_iteration"]

        while not self.is_shutting_down:
            # Check context capacity
            context_usage = await self.context_monitor.get_context_usage()
            if context_usage > self.config.context_threshold:
                self.logger.info(f"Context threshold reached: {context_usage:.2f}")
                break

            # Plan next wave
            wave_size = self._calculate_wave_size(context_usage, wave_number)
            if wave_size == 0:
                break

            self.logger.info(f"Starting wave {wave_number} with {wave_size} agents")

            # Execute wave
            wave_result = await self._execute_wave(
                wave_number,
                current_iteration,
                wave_size,
                spec_analysis,
                directory_state,
                iteration_strategy,
                output_dir,
            )

            results["waves"].append(wave_result)
            results["total_iterations"] += wave_result.get("completed_iterations", 0)

            # Update state
            self.execution_state.current_wave = wave_number
            self.execution_state.total_iterations = results["total_iterations"]
            current_iteration += wave_size
            wave_number += 1

            # Progressive sophistication
            iteration_strategy["innovation_dimensions"] = self._evolve_innovation_dimensions(
                iteration_strategy["innovation_dimensions"], wave_number
            )

            # Brief pause between waves
            await asyncio.sleep(0.1)

        self.logger.info(f"Infinite mode completed: {results['total_iterations']} iterations across {wave_number-1} waves")
        return results

    async def _execute_finite_mode(
        self,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        iteration_strategy: Dict[str, Any],
        output_dir: Union[str, Path],
        count: int,
    ) -> Dict[str, Any]:
        """Execute finite mode with predetermined iteration count."""
        self.logger.info(f"Phase 4: Executing finite mode ({count} iterations)")

        self.execution_state.is_running = True
        results = {"waves": [], "total_iterations": 0}

        wave_strategy = iteration_strategy["wave_strategy"]
        wave_size = wave_strategy["wave_size"]
        max_waves = wave_strategy["max_waves"]

        current_iteration = iteration_strategy["starting_iteration"]
        remaining_iterations = count

        for wave_number in range(1, max_waves + 1):
            if remaining_iterations <= 0 or self.is_shutting_down:
                break

            # Calculate actual wave size for this wave
            actual_wave_size = min(wave_size, remaining_iterations)

            self.logger.info(f"Starting wave {wave_number}/{max_waves} with {actual_wave_size} agents")

            # Execute wave
            wave_result = await self._execute_wave(
                wave_number,
                current_iteration,
                actual_wave_size,
                spec_analysis,
                directory_state,
                iteration_strategy,
                output_dir,
            )

            results["waves"].append(wave_result)
            completed = wave_result.get("completed_iterations", 0)
            results["total_iterations"] += completed

            # Update state
            self.execution_state.current_wave = wave_number
            self.execution_state.total_iterations = results["total_iterations"]
            current_iteration += actual_wave_size
            remaining_iterations -= actual_wave_size

        self.logger.info(f"Finite mode completed: {results['total_iterations']} iterations")
        return results

    async def _execute_wave(
        self,
        wave_number: int,
        starting_iteration: int,
        wave_size: int,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        iteration_strategy: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """Execute a single wave of parallel agents."""
        wave_start_time = time.time()

        # Create tasks for this wave
        tasks = []
        for i in range(wave_size):
            iteration_number = starting_iteration + i
            innovation_dimension = self._assign_innovation_dimension(
                iteration_strategy["innovation_dimensions"], i, wave_number
            )

            task = self.task_assignment_engine.create_task(
                iteration_number=iteration_number,
                spec_analysis=spec_analysis,
                directory_state=directory_state,
                innovation_dimension=innovation_dimension,
                output_dir=output_dir,
            )
            tasks.append(task)

        # Execute tasks in parallel
        wave_results = await self.agent_pool_manager.execute_parallel_tasks(tasks)

        # Process results
        completed_iterations = 0
        failed_iterations = 0

        for result in wave_results:
            if result.get("success", False):
                completed_iterations += 1
                self.execution_state.completed_iterations.append(str(result.get("iteration_number")))
            else:
                failed_iterations += 1
                self.execution_state.failed_iterations.append(str(result.get("iteration_number")))

        # Update performance metrics
        wave_time = time.time() - wave_start_time
        self.execution_state.average_iteration_time = (
            (self.execution_state.average_iteration_time * self.execution_state.total_iterations + wave_time) /
            (self.execution_state.total_iterations + completed_iterations)
        ) if completed_iterations > 0 else self.execution_state.average_iteration_time

        self.execution_state.success_rate = (
            len(self.execution_state.completed_iterations) /
            (len(self.execution_state.completed_iterations) + len(self.execution_state.failed_iterations))
        ) if (len(self.execution_state.completed_iterations) + len(self.execution_state.failed_iterations)) > 0 else 1.0

        return {
            "wave_number": wave_number,
            "completed_iterations": completed_iterations,
            "failed_iterations": failed_iterations,
            "execution_time": wave_time,
            "results": wave_results,
        }

    def _calculate_wave_size(self, context_usage: float, wave_number: int) -> int:
        """Calculate optimal wave size based on context usage and wave number."""
        if context_usage > self.config.context_threshold:
            return 0

        # Reduce wave size as context usage increases
        remaining_capacity = 1.0 - context_usage
        max_possible_size = int(remaining_capacity * self.config.wave_size_max)

        # Ensure minimum wave size
        return max(self.config.wave_size_min, min(max_possible_size, self.config.wave_size_max))

    def _assign_innovation_dimension(
        self, innovation_dimensions: List[str], agent_index: int, wave_number: int
    ) -> str:
        """Assign a unique innovation dimension to an agent."""
        # Cycle through dimensions with some randomization based on wave
        dimension_index = (agent_index + wave_number * 3) % len(innovation_dimensions)
        return innovation_dimensions[dimension_index]

    def _evolve_innovation_dimensions(
        self, current_dimensions: List[str], wave_number: int
    ) -> List[str]:
        """Evolve innovation dimensions for progressive sophistication."""
        if wave_number <= 2:
            return current_dimensions

        # Add more advanced dimensions as waves progress
        advanced_dimensions = [
            "paradigm_revolution",
            "cross_domain_synthesis",
            "emergent_behaviors",
            "adaptive_intelligence",
            "quantum_improvements",
            "meta_optimization",
            "holistic_integration",
            "future_proofing",
        ]

        # Gradually introduce advanced dimensions
        new_dimensions = current_dimensions.copy()
        for i, dim in enumerate(advanced_dimensions):
            if wave_number > (i + 3) and dim not in new_dimensions:
                new_dimensions.append(dim)

        return new_dimensions

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down infinite loop orchestrator")
        self.is_shutting_down = True

        if self.execution_state and self.execution_state.is_running:
            self.execution_state.is_running = False

        # Shutdown components
        await self.agent_pool_manager.shutdown()
        await self.wave_manager.shutdown()
        await self.state_persistence.save_final_state(self.execution_state)

        self.logger.info("Infinite loop orchestrator shutdown complete")
