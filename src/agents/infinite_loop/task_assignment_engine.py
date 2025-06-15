"""
Task Assignment Engine

Creates and manages task assignments for iteration generation agents.
Handles task creation, priority assignment, and distribution optimization.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskSpecification:
    """Specification for a task to be executed by an agent."""

    task_id: str
    iteration_number: int
    spec_analysis: Dict[str, Any]
    directory_state: Dict[str, Any]
    innovation_dimension: str
    output_dir: Union[str, Path]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    estimated_complexity: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskAssignmentEngine:
    """
    Creates and manages task assignments for iteration generation.

    Features:
    - Task specification creation and validation
    - Priority assignment based on complexity and dependencies
    - Innovation dimension distribution optimization
    - Task dependency management
    - Load balancing considerations
    - Quality requirement propagation
    """

    def __init__(self):
        """Initialize the task assignment engine."""
        self.logger = logging.getLogger("task_assignment_engine")

        # Task tracking
        self.created_tasks: List[TaskSpecification] = []
        self.task_counter = 0

        # Innovation dimension tracking
        self.dimension_usage: Dict[str, int] = {}

        # Complexity estimation factors
        self.complexity_factors = {
            "content_type": {
                "code": 1.5,
                "documentation": 1.0,
                "configuration": 0.8,
                "data": 1.2,
                "template": 0.9,
                "test": 1.3,
                "api": 1.4,
            },
            "format": {
                "python": 1.4,
                "javascript": 1.3,
                "json": 0.7,
                "yaml": 0.8,
                "markdown": 0.9,
                "html": 1.1,
                "css": 1.0,
                "xml": 1.0,
            },
            "innovation_dimension": {
                "functional_enhancement": 1.2,
                "structural_innovation": 1.4,
                "interaction_patterns": 1.3,
                "performance_optimization": 1.5,
                "user_experience": 1.1,
                "integration_capabilities": 1.3,
                "scalability_improvements": 1.4,
                "security_enhancements": 1.3,
                "accessibility_features": 1.2,
                "paradigm_shifts": 1.8,
                "paradigm_revolution": 2.0,
                "cross_domain_synthesis": 1.7,
                "emergent_behaviors": 1.6,
                "adaptive_intelligence": 1.8,
                "quantum_improvements": 1.9,
                "meta_optimization": 1.7,
                "holistic_integration": 1.6,
                "future_proofing": 1.5,
            },
        }

    def create_task(
        self,
        iteration_number: int,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        innovation_dimension: str,
        output_dir: Union[str, Path],
        priority: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a task specification for iteration generation.

        Args:
            iteration_number: The iteration number to generate
            spec_analysis: Parsed specification analysis
            directory_state: Current directory state
            innovation_dimension: Assigned innovation dimension
            output_dir: Output directory path
            priority: Optional priority override

        Returns:
            Task specification dictionary
        """
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{uuid.uuid4().hex[:8]}"

        # Calculate complexity
        complexity = self._estimate_complexity(
            spec_analysis, innovation_dimension, iteration_number
        )

        # Determine priority
        if priority is None:
            priority = self._calculate_priority(iteration_number, complexity, directory_state)

        # Create task specification
        task_spec = TaskSpecification(
            task_id=task_id,
            iteration_number=iteration_number,
            spec_analysis=spec_analysis,
            directory_state=directory_state,
            innovation_dimension=innovation_dimension,
            output_dir=output_dir,
            priority=priority,
            estimated_complexity=complexity,
            metadata={
                "content_type": spec_analysis.get("content_type", "unknown"),
                "format": spec_analysis.get("format", "unknown"),
                "evolution_pattern": spec_analysis.get("evolution_pattern", "incremental"),
                "existing_iterations": len(directory_state.get("iteration_files", [])),
            },
        )

        # Track task creation
        self.created_tasks.append(task_spec)
        self._update_dimension_usage(innovation_dimension)

        self.logger.debug(f"Created task {task_id} for iteration {iteration_number}")
        self.logger.debug(f"- Innovation dimension: {innovation_dimension}")
        self.logger.debug(f"- Estimated complexity: {complexity:.2f}")
        self.logger.debug(f"- Priority: {priority}")

        # Convert to dictionary format expected by agent pool
        return {
            "task_id": task_id,
            "iteration_number": iteration_number,
            "spec_analysis": spec_analysis,
            "directory_state": directory_state,
            "innovation_dimension": innovation_dimension,
            "output_dir": str(output_dir),
            "priority": priority,
            "estimated_complexity": complexity,
            "metadata": task_spec.metadata,
        }

    def create_batch_tasks(
        self,
        starting_iteration: int,
        count: int,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        innovation_dimensions: List[str],
        output_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Create a batch of tasks for parallel execution.

        Args:
            starting_iteration: Starting iteration number
            count: Number of tasks to create
            spec_analysis: Parsed specification analysis
            directory_state: Current directory state
            innovation_dimensions: Available innovation dimensions
            output_dir: Output directory path

        Returns:
            List of task specifications
        """
        self.logger.info(
            f"Creating batch of {count} tasks starting from iteration {starting_iteration}"
        )

        tasks = []

        for i in range(count):
            iteration_number = starting_iteration + i

            # Assign innovation dimension
            dimension = self._assign_innovation_dimension(innovation_dimensions, i, len(tasks))

            # Create task
            task = self.create_task(
                iteration_number=iteration_number,
                spec_analysis=spec_analysis,
                directory_state=directory_state,
                innovation_dimension=dimension,
                output_dir=output_dir,
            )

            tasks.append(task)

        self.logger.info(f"Created {len(tasks)} tasks for batch execution")
        return tasks

    def _estimate_complexity(
        self,
        spec_analysis: Dict[str, Any],
        innovation_dimension: str,
        iteration_number: int,
    ) -> float:
        """Estimate task complexity based on various factors."""
        base_complexity = 1.0

        # Content type factor
        content_type = spec_analysis.get("content_type", "unknown")
        content_factor = self.complexity_factors["content_type"].get(content_type, 1.0)

        # Format factor
        format_type = spec_analysis.get("format", "unknown")
        format_factor = self.complexity_factors["format"].get(format_type, 1.0)

        # Innovation dimension factor
        dimension_factor = self.complexity_factors["innovation_dimension"].get(
            innovation_dimension, 1.0
        )

        # Iteration number factor (later iterations may be more complex)
        iteration_factor = 1.0 + (iteration_number - 1) * 0.05  # 5% increase per iteration
        iteration_factor = min(iteration_factor, 2.0)  # Cap at 2x

        # Requirements complexity
        requirements = spec_analysis.get("requirements", [])
        requirements_factor = 1.0 + len(requirements) * 0.1

        # Constraints complexity
        constraints = spec_analysis.get("constraints", [])
        constraints_factor = 1.0 + len(constraints) * 0.05

        # Calculate final complexity
        complexity = (
            base_complexity
            * content_factor
            * format_factor
            * dimension_factor
            * iteration_factor
            * requirements_factor
            * constraints_factor
        )

        return round(complexity, 2)

    def _calculate_priority(
        self,
        iteration_number: int,
        complexity: float,
        directory_state: Dict[str, Any],
    ) -> int:
        """Calculate task priority based on various factors."""
        base_priority = 1

        # Lower iteration numbers get higher priority
        iteration_priority = max(1, 10 - iteration_number // 10)

        # Higher complexity gets lower priority (to balance load)
        complexity_priority = max(1, int(5 - complexity))

        # Fill gaps get higher priority
        gaps = directory_state.get("gaps", [])
        if iteration_number in gaps:
            gap_priority = 3
        else:
            gap_priority = 1

        # Calculate final priority
        priority = base_priority + iteration_priority + complexity_priority + gap_priority

        return min(priority, 10)  # Cap at 10

    def _assign_innovation_dimension(
        self,
        innovation_dimensions: List[str],
        task_index: int,
        total_tasks: int,
    ) -> str:
        """Assign innovation dimension to balance distribution."""
        if not innovation_dimensions:
            return "functional_enhancement"  # Default

        # Use round-robin with some variation
        dimension_index = task_index % len(innovation_dimensions)

        # Add some variation based on total tasks to avoid patterns
        variation = (total_tasks * 3 + task_index * 7) % len(innovation_dimensions)
        final_index = (dimension_index + variation) % len(innovation_dimensions)

        return innovation_dimensions[final_index]

    def _update_dimension_usage(self, dimension: str) -> None:
        """Update dimension usage tracking."""
        if dimension not in self.dimension_usage:
            self.dimension_usage[dimension] = 0
        self.dimension_usage[dimension] += 1

    def get_dimension_distribution(self) -> Dict[str, int]:
        """Get current distribution of innovation dimensions."""
        return self.dimension_usage.copy()

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about created tasks."""
        if not self.created_tasks:
            return {
                "total_tasks": 0,
                "average_complexity": 0.0,
                "complexity_distribution": {},
                "priority_distribution": {},
                "dimension_distribution": {},
            }

        # Calculate statistics
        total_tasks = len(self.created_tasks)
        complexities = [task.estimated_complexity for task in self.created_tasks]
        priorities = [task.priority for task in self.created_tasks]

        average_complexity = sum(complexities) / len(complexities)

        # Distribution calculations
        complexity_ranges = {
            "low (0.5-1.0)": len([c for c in complexities if 0.5 <= c <= 1.0]),
            "medium (1.0-1.5)": len([c for c in complexities if 1.0 < c <= 1.5]),
            "high (1.5-2.0)": len([c for c in complexities if 1.5 < c <= 2.0]),
            "very_high (>2.0)": len([c for c in complexities if c > 2.0]),
        }

        priority_distribution = {}
        for priority in priorities:
            priority_distribution[str(priority)] = priority_distribution.get(str(priority), 0) + 1

        return {
            "total_tasks": total_tasks,
            "average_complexity": round(average_complexity, 2),
            "complexity_distribution": complexity_ranges,
            "priority_distribution": priority_distribution,
            "dimension_distribution": self.dimension_usage.copy(),
        }

    def optimize_task_distribution(
        self,
        tasks: List[Dict[str, Any]],
        available_agents: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Optimize task distribution across available agents.

        Args:
            tasks: List of tasks to distribute
            available_agents: Number of available agents

        Returns:
            List of task batches for each agent
        """
        if not tasks or available_agents <= 0:
            return []

        # Sort tasks by priority (higher first) and complexity (lower first for balance)
        sorted_tasks = sorted(tasks, key=lambda t: (-t["priority"], t["estimated_complexity"]))

        # Initialize agent batches
        agent_batches = [[] for _ in range(available_agents)]
        agent_loads = [0.0] * available_agents

        # Distribute tasks using a greedy approach
        for task in sorted_tasks:
            # Find agent with lowest current load
            min_load_agent = min(range(available_agents), key=lambda i: agent_loads[i])

            # Assign task to agent
            agent_batches[min_load_agent].append(task)
            agent_loads[min_load_agent] += task["estimated_complexity"]

        # Filter out empty batches
        non_empty_batches = [batch for batch in agent_batches if batch]

        self.logger.info(f"Distributed {len(tasks)} tasks across {len(non_empty_batches)} agents")
        for i, batch in enumerate(non_empty_batches):
            total_complexity = sum(task["estimated_complexity"] for task in batch)
            self.logger.debug(f"Agent {i}: {len(batch)} tasks, complexity: {total_complexity:.2f}")

        return non_empty_batches

    def validate_task_specification(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a task specification.

        Args:
            task: Task specification to validate

        Returns:
            Validation result with success status and any issues
        """
        validation_result = {
            "valid": True,
            "issues": [],
        }

        # Required fields
        required_fields = [
            "task_id",
            "iteration_number",
            "spec_analysis",
            "directory_state",
            "innovation_dimension",
            "output_dir",
        ]

        for field in required_fields:
            if field not in task:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")

        # Type validation
        if "iteration_number" in task and not isinstance(task["iteration_number"], int):
            validation_result["valid"] = False
            validation_result["issues"].append("iteration_number must be an integer")

        if "priority" in task and not isinstance(task["priority"], int):
            validation_result["valid"] = False
            validation_result["issues"].append("priority must be an integer")

        # Value validation
        if "iteration_number" in task and task["iteration_number"] < 1:
            validation_result["valid"] = False
            validation_result["issues"].append("iteration_number must be positive")

        if "priority" in task and not (1 <= task["priority"] <= 10):
            validation_result["valid"] = False
            validation_result["issues"].append("priority must be between 1 and 10")

        return validation_result
