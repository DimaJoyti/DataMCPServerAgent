"""
Safe reinforcement learning module for DataMCPServerAgent.
This module implements safety constraints and risk-aware RL algorithms.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase


class SafetyConstraint:
    """Represents a safety constraint for RL agents."""

    def __init__(
        self,
        name: str,
        constraint_type: str,
        threshold: float,
        violation_penalty: float = -10.0,
        description: str = "",
    ):
        """Initialize safety constraint.
        
        Args:
            name: Constraint name
            constraint_type: Type of constraint ('hard', 'soft', 'probabilistic')
            threshold: Constraint threshold
            violation_penalty: Penalty for constraint violation
            description: Human-readable description
        """
        self.name = name
        self.constraint_type = constraint_type
        self.threshold = threshold
        self.violation_penalty = violation_penalty
        self.description = description

        # Violation tracking
        self.violation_count = 0
        self.total_evaluations = 0
        self.violation_history = []

    def evaluate(self, state: np.ndarray, action: int, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate constraint satisfaction.
        
        Args:
            state: Current state
            action: Proposed action
            context: Additional context
            
        Returns:
            Tuple of (is_satisfied, constraint_value)
        """
        self.total_evaluations += 1

        # Implement specific constraint logic
        constraint_value = self._compute_constraint_value(state, action, context)

        if self.constraint_type == "hard":
            is_satisfied = constraint_value <= self.threshold
        elif self.constraint_type == "soft":
            is_satisfied = constraint_value <= self.threshold
        elif self.constraint_type == "probabilistic":
            # For probabilistic constraints, we need to track violation probability
            is_satisfied = constraint_value <= self.threshold
        else:
            is_satisfied = True

        if not is_satisfied:
            self.violation_count += 1
            self.violation_history.append({
                "timestamp": time.time(),
                "state": state.tolist(),
                "action": action,
                "constraint_value": constraint_value,
                "context": context,
            })

        return is_satisfied, constraint_value

    def _compute_constraint_value(self, state: np.ndarray, action: int, context: Dict[str, Any]) -> float:
        """Compute constraint value (to be overridden by specific constraints).
        
        Args:
            state: Current state
            action: Proposed action
            context: Additional context
            
        Returns:
            Constraint value
        """
        # Default implementation - override in subclasses
        return 0.0

    def get_violation_rate(self) -> float:
        """Get constraint violation rate.
        
        Returns:
            Violation rate (0.0 to 1.0)
        """
        if self.total_evaluations == 0:
            return 0.0
        return self.violation_count / self.total_evaluations

    def reset_statistics(self):
        """Reset constraint statistics."""
        self.violation_count = 0
        self.total_evaluations = 0
        self.violation_history.clear()


class ResourceUsageConstraint(SafetyConstraint):
    """Constraint on resource usage (CPU, memory, etc.)."""

    def __init__(self, max_resource_usage: float = 0.8, **kwargs):
        """Initialize resource usage constraint.
        
        Args:
            max_resource_usage: Maximum allowed resource usage (0.0 to 1.0)
            **kwargs: Additional constraint arguments
        """
        super().__init__(
            name="resource_usage",
            constraint_type="hard",
            threshold=max_resource_usage,
            description=f"Resource usage must not exceed {max_resource_usage*100}%",
            **kwargs
        )

    def _compute_constraint_value(self, state: np.ndarray, action: int, context: Dict[str, Any]) -> float:
        """Compute resource usage constraint value."""
        # Simulate resource usage based on action complexity
        base_usage = 0.1  # Base resource usage

        # Different actions have different resource requirements
        action_multipliers = {0: 1.0, 1: 1.5, 2: 2.0, 3: 1.2, 4: 0.8}
        action_usage = action_multipliers.get(action, 1.0)

        # State complexity affects resource usage
        state_complexity = np.linalg.norm(state) / len(state)

        # Context factors
        context_factor = 1.0
        if context.get("high_priority", False):
            context_factor = 1.3
        if context.get("batch_processing", False):
            context_factor = 0.7

        total_usage = base_usage * action_usage * (1 + state_complexity) * context_factor
        return min(total_usage, 1.0)  # Cap at 100%


class ResponseTimeConstraint(SafetyConstraint):
    """Constraint on response time."""

    def __init__(self, max_response_time: float = 5.0, **kwargs):
        """Initialize response time constraint.
        
        Args:
            max_response_time: Maximum allowed response time in seconds
            **kwargs: Additional constraint arguments
        """
        super().__init__(
            name="response_time",
            constraint_type="soft",
            threshold=max_response_time,
            description=f"Response time should not exceed {max_response_time} seconds",
            **kwargs
        )

    def _compute_constraint_value(self, state: np.ndarray, action: int, context: Dict[str, Any]) -> float:
        """Compute response time constraint value."""
        # Simulate response time based on action and context
        base_time = 0.5  # Base response time

        # Different actions have different time requirements
        action_times = {0: 0.5, 1: 1.0, 2: 2.0, 3: 1.5, 4: 0.3}
        action_time = action_times.get(action, 1.0)

        # Context factors
        if context.get("complex_query", False):
            action_time *= 2.0
        if context.get("cached_result", False):
            action_time *= 0.3

        return base_time + action_time


class SafetyMonitor:
    """Monitors safety constraints during RL training and execution."""

    def __init__(self, constraints: List[SafetyConstraint]):
        """Initialize safety monitor.
        
        Args:
            constraints: List of safety constraints to monitor
        """
        self.constraints = {constraint.name: constraint for constraint in constraints}
        self.safety_violations = []
        self.safety_score_history = []

    def check_safety(
        self,
        state: np.ndarray,
        action: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if action satisfies all safety constraints.
        
        Args:
            state: Current state
            action: Proposed action
            context: Additional context
            
        Returns:
            Tuple of (is_safe, constraint_results)
        """
        constraint_results = {}
        is_safe = True
        total_penalty = 0.0

        for constraint_name, constraint in self.constraints.items():
            satisfied, value = constraint.evaluate(state, action, context)

            constraint_results[constraint_name] = {
                "satisfied": satisfied,
                "value": value,
                "threshold": constraint.threshold,
                "type": constraint.constraint_type,
            }

            if not satisfied:
                is_safe = False
                total_penalty += constraint.violation_penalty

                # Record violation
                self.safety_violations.append({
                    "timestamp": time.time(),
                    "constraint": constraint_name,
                    "state": state.tolist(),
                    "action": action,
                    "value": value,
                    "threshold": constraint.threshold,
                    "context": context,
                })

        # Compute overall safety score
        safety_score = self._compute_safety_score(constraint_results)
        self.safety_score_history.append(safety_score)

        return is_safe, {
            "constraints": constraint_results,
            "safety_score": safety_score,
            "total_penalty": total_penalty,
        }

    def _compute_safety_score(self, constraint_results: Dict[str, Any]) -> float:
        """Compute overall safety score.
        
        Args:
            constraint_results: Results from constraint evaluation
            
        Returns:
            Safety score (0.0 to 1.0, higher is safer)
        """
        if not constraint_results:
            return 1.0

        scores = []
        for result in constraint_results.values():
            if result["satisfied"]:
                scores.append(1.0)
            else:
                # Partial score based on how close to threshold
                value = result["value"]
                threshold = result["threshold"]
                if threshold > 0:
                    score = max(0.0, 1.0 - (value - threshold) / threshold)
                else:
                    score = 0.0
                scores.append(score)

        return np.mean(scores)

    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics.
        
        Returns:
            Safety statistics
        """
        total_checks = sum(constraint.total_evaluations for constraint in self.constraints.values())
        total_violations = len(self.safety_violations)

        constraint_stats = {}
        for name, constraint in self.constraints.items():
            constraint_stats[name] = {
                "violation_rate": constraint.get_violation_rate(),
                "total_evaluations": constraint.total_evaluations,
                "violation_count": constraint.violation_count,
            }

        recent_safety_score = np.mean(self.safety_score_history[-100:]) if self.safety_score_history else 1.0

        return {
            "total_safety_checks": total_checks,
            "total_violations": total_violations,
            "overall_violation_rate": total_violations / max(1, total_checks),
            "recent_safety_score": recent_safety_score,
            "constraint_statistics": constraint_stats,
        }


class SafeRLAgent:
    """Safe reinforcement learning agent with constraint satisfaction."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        base_agent: Any,
        safety_monitor: SafetyMonitor,
        safety_weight: float = 0.5,
        constraint_learning: bool = True,
    ):
        """Initialize safe RL agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            base_agent: Base RL agent
            safety_monitor: Safety constraint monitor
            safety_weight: Weight for safety in reward function
            constraint_learning: Whether to learn from constraint violations
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.base_agent = base_agent
        self.safety_monitor = safety_monitor
        self.safety_weight = safety_weight
        self.constraint_learning = constraint_learning

        # Safety-aware modifications
        self.safe_action_history = []
        self.constraint_violation_memory = []

        # Risk assessment
        self.risk_threshold = 0.3
        self.conservative_mode = False

    async def select_safe_action(
        self,
        state: np.ndarray,
        context: Dict[str, Any],
        training: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action considering safety constraints.
        
        Args:
            state: Current state
            context: Additional context
            training: Whether in training mode
            
        Returns:
            Tuple of (safe_action, safety_info)
        """
        # Get action from base agent
        if hasattr(self.base_agent, 'select_action'):
            proposed_action = self.base_agent.select_action(state, training)
        else:
            proposed_action = np.random.randint(0, 5)  # Fallback

        # Check safety of proposed action
        is_safe, safety_results = self.safety_monitor.check_safety(
            state, proposed_action, context
        )

        if is_safe or not training:
            # Action is safe or we're not training (use as-is)
            selected_action = proposed_action
            safety_info = {
                "action_modified": False,
                "original_action": proposed_action,
                "safety_results": safety_results,
            }
        else:
            # Action is unsafe, find safe alternative
            safe_action = await self._find_safe_action(state, context, proposed_action)
            selected_action = safe_action
            safety_info = {
                "action_modified": True,
                "original_action": proposed_action,
                "safe_action": safe_action,
                "safety_results": safety_results,
            }

        # Record safe action
        self.safe_action_history.append({
            "state": state.tolist(),
            "original_action": proposed_action,
            "selected_action": selected_action,
            "safety_score": safety_results["safety_score"],
            "timestamp": time.time(),
        })

        return selected_action, safety_info

    async def _find_safe_action(
        self,
        state: np.ndarray,
        context: Dict[str, Any],
        original_action: int
    ) -> int:
        """Find a safe alternative action.
        
        Args:
            state: Current state
            context: Additional context
            original_action: Original unsafe action
            
        Returns:
            Safe action
        """
        # Try all possible actions to find a safe one
        action_dim = getattr(self.base_agent, 'action_dim', 5)

        best_action = original_action
        best_safety_score = 0.0

        for action in range(action_dim):
            if action == original_action:
                continue  # Skip the unsafe action

            is_safe, safety_results = self.safety_monitor.check_safety(
                state, action, context
            )

            safety_score = safety_results["safety_score"]

            if is_safe and safety_score > best_safety_score:
                best_action = action
                best_safety_score = safety_score

        # If no safe action found, use the safest available
        if best_safety_score == 0.0:
            # Conservative fallback - choose action 0 (usually safest)
            best_action = 0

        return best_action

    def compute_safe_reward(
        self,
        original_reward: float,
        safety_results: Dict[str, Any]
    ) -> float:
        """Compute safety-adjusted reward.
        
        Args:
            original_reward: Original reward from environment
            safety_results: Safety constraint evaluation results
            
        Returns:
            Safety-adjusted reward
        """
        safety_score = safety_results["safety_score"]
        total_penalty = safety_results["total_penalty"]

        # Combine original reward with safety considerations
        safe_reward = (
            (1 - self.safety_weight) * original_reward +
            self.safety_weight * safety_score +
            total_penalty  # Penalty is negative for violations
        )

        return safe_reward

    async def train_with_safety(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        safety_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train agent with safety considerations.
        
        Args:
            state: Current state
            action: Action taken
            reward: Original reward
            next_state: Next state
            done: Whether episode is done
            safety_results: Safety evaluation results
            
        Returns:
            Training metrics
        """
        # Compute safety-adjusted reward
        safe_reward = self.compute_safe_reward(reward, safety_results)

        # Store experience in base agent
        if hasattr(self.base_agent, 'store_experience'):
            self.base_agent.store_experience(state, action, safe_reward, next_state, done)

        # Train base agent
        training_metrics = {}
        if hasattr(self.base_agent, 'train'):
            training_metrics = self.base_agent.train()

        # Add safety metrics
        training_metrics.update({
            "original_reward": reward,
            "safe_reward": safe_reward,
            "safety_score": safety_results["safety_score"],
            "safety_penalty": safety_results["total_penalty"],
        })

        # Learn from constraint violations if enabled
        if self.constraint_learning and safety_results["total_penalty"] < 0:
            await self._learn_from_violation(state, action, safety_results)

        return training_metrics

    async def _learn_from_violation(
        self,
        state: np.ndarray,
        action: int,
        safety_results: Dict[str, Any]
    ):
        """Learn from constraint violations to avoid them in the future.
        
        Args:
            state: State where violation occurred
            action: Action that caused violation
            safety_results: Safety evaluation results
        """
        violation_data = {
            "state": state.tolist(),
            "action": action,
            "violated_constraints": [
                name for name, result in safety_results["constraints"].items()
                if not result["satisfied"]
            ],
            "timestamp": time.time(),
        }

        self.constraint_violation_memory.append(violation_data)

        # Keep memory bounded
        if len(self.constraint_violation_memory) > 1000:
            self.constraint_violation_memory.pop(0)

        # Adjust risk threshold based on violations
        recent_violations = len([
            v for v in self.constraint_violation_memory
            if time.time() - v["timestamp"] < 3600  # Last hour
        ])

        if recent_violations > 10:
            self.risk_threshold = max(0.1, self.risk_threshold - 0.05)
            self.conservative_mode = True
        elif recent_violations == 0:
            self.risk_threshold = min(0.5, self.risk_threshold + 0.01)
            self.conservative_mode = False

    def get_safety_performance(self) -> Dict[str, Any]:
        """Get safety performance metrics.
        
        Returns:
            Safety performance metrics
        """
        if not self.safe_action_history:
            return {"error": "No action history available"}

        # Action modification rate
        modified_actions = sum(
            1 for record in self.safe_action_history
            if record["original_action"] != record["selected_action"]
        )
        modification_rate = modified_actions / len(self.safe_action_history)

        # Average safety score
        avg_safety_score = np.mean([
            record["safety_score"] for record in self.safe_action_history
        ])

        # Recent performance
        recent_records = self.safe_action_history[-100:]
        recent_safety_score = np.mean([
            record["safety_score"] for record in recent_records
        ])

        # Safety monitoring stats
        monitor_stats = self.safety_monitor.get_safety_statistics()

        return {
            "action_modification_rate": modification_rate,
            "avg_safety_score": avg_safety_score,
            "recent_safety_score": recent_safety_score,
            "total_actions": len(self.safe_action_history),
            "constraint_violations": len(self.constraint_violation_memory),
            "conservative_mode": self.conservative_mode,
            "risk_threshold": self.risk_threshold,
            "monitor_statistics": monitor_stats,
        }


# Factory function to create safe RL agent
async def create_safe_rl_agent(
    model: ChatAnthropic,
    db: MemoryDatabase,
    base_agent: Any,
    safety_constraints: Optional[List[SafetyConstraint]] = None,
    safety_weight: float = 0.5,
) -> SafeRLAgent:
    """Create safe RL agent with safety constraints.
    
    Args:
        model: Language model
        db: Memory database
        base_agent: Base RL agent to make safe
        safety_constraints: List of safety constraints
        safety_weight: Weight for safety in reward function
        
    Returns:
        Safe RL agent
    """
    # Create default safety constraints if none provided
    if safety_constraints is None:
        safety_constraints = [
            ResourceUsageConstraint(max_resource_usage=0.8),
            ResponseTimeConstraint(max_response_time=5.0),
        ]

    # Create safety monitor
    safety_monitor = SafetyMonitor(safety_constraints)

    # Create reward system
    reward_system = RewardSystem(db)

    # Create safe RL agent
    safe_agent = SafeRLAgent(
        name="safe_rl_agent",
        model=model,
        db=db,
        reward_system=reward_system,
        base_agent=base_agent,
        safety_monitor=safety_monitor,
        safety_weight=safety_weight,
        constraint_learning=True,
    )

    return safe_agent
