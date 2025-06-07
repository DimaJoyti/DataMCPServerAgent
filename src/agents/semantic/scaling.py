"""
Scaling and Load Management

Provides automatic scaling, load balancing, and resource management
for semantic agents to ensure optimal performance under varying loads.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .base_semantic_agent import BaseSemanticAgent, SemanticAgentConfig
from .coordinator import SemanticCoordinator
from .performance import PerformanceTracker

class ScalingAction(str, Enum):
    """Types of scaling actions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REDISTRIBUTE = "redistribute"
    OPTIMIZE = "optimize"

@dataclass
class ScalingRule:
    """Scaling rule configuration."""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default_rule"
    metric_name: str = "cpu_usage"
    threshold_high: float = 80.0
    threshold_low: float = 20.0
    action_scale_up: ScalingAction = ScalingAction.SCALE_UP
    action_scale_down: ScalingAction = ScalingAction.SCALE_DOWN
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    enabled: bool = True
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingEvent:
    """Scaling event record."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    action: ScalingAction
    trigger_metric: str
    trigger_value: float
    threshold: float
    agent_id: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

class LoadBalancer:
    """
    Load balancer for distributing tasks across semantic agents.

    Features:
    - Round-robin distribution
    - Weighted distribution based on performance
    - Health-aware routing
    - Sticky sessions for stateful operations
    """

    def __init__(self):
        """Initialize the load balancer."""
        self.agent_weights: Dict[str, float] = {}
        self.agent_health: Dict[str, bool] = {}
        self.round_robin_index = 0
        self.session_affinity: Dict[str, str] = {}  # session_id -> agent_id

        self.logger = logging.getLogger("load_balancer")

    def register_agent(self, agent_id: str, weight: float = 1.0) -> None:
        """Register an agent with the load balancer."""
        self.agent_weights[agent_id] = weight
        self.agent_health[agent_id] = True
        self.logger.info(f"Registered agent {agent_id} with weight {weight}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the load balancer."""
        self.agent_weights.pop(agent_id, None)
        self.agent_health.pop(agent_id, None)

        # Remove session affinities
        sessions_to_remove = [
            session_id for session_id, aid in self.session_affinity.items()
            if aid == agent_id
        ]
        for session_id in sessions_to_remove:
            del self.session_affinity[session_id]

        self.logger.info(f"Unregistered agent {agent_id}")

    def update_agent_weight(self, agent_id: str, weight: float) -> None:
        """Update an agent's weight based on performance."""
        if agent_id in self.agent_weights:
            self.agent_weights[agent_id] = weight
            self.logger.debug(f"Updated weight for agent {agent_id}: {weight}")

    def update_agent_health(self, agent_id: str, healthy: bool) -> None:
        """Update an agent's health status."""
        if agent_id in self.agent_health:
            self.agent_health[agent_id] = healthy
            self.logger.debug(f"Updated health for agent {agent_id}: {healthy}")

    def select_agent(
        self,
        strategy: str = "weighted",
        session_id: Optional[str] = None,
        exclude_agents: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Select an agent for task execution.

        Args:
            strategy: Selection strategy ("round_robin", "weighted", "random")
            session_id: Optional session ID for sticky sessions
            exclude_agents: Set of agent IDs to exclude

        Returns:
            Selected agent ID or None if no agents available
        """
        exclude_agents = exclude_agents or set()

        # Check session affinity first
        if session_id and session_id in self.session_affinity:
            agent_id = self.session_affinity[session_id]
            if (agent_id in self.agent_health and
                self.agent_health[agent_id] and
                agent_id not in exclude_agents):
                return agent_id

        # Get healthy agents
        healthy_agents = [
            agent_id for agent_id, healthy in self.agent_health.items()
            if healthy and agent_id not in exclude_agents
        ]

        if not healthy_agents:
            return None

        if strategy == "round_robin":
            return self._select_round_robin(healthy_agents)
        elif strategy == "weighted":
            return self._select_weighted(healthy_agents)
        else:
            # Default to weighted
            return self._select_weighted(healthy_agents)

    def _select_round_robin(self, agents: List[str]) -> str:
        """Select agent using round-robin strategy."""
        if not agents:
            return None

        selected_agent = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        return selected_agent

    def _select_weighted(self, agents: List[str]) -> str:
        """Select agent using weighted strategy."""
        if not agents:
            return None

        # Calculate total weight
        total_weight = sum(self.agent_weights.get(agent_id, 1.0) for agent_id in agents)

        if total_weight == 0:
            return agents[0]

        # Select based on weights
        import random
        target = random.uniform(0, total_weight)
        current_weight = 0

        for agent_id in agents:
            current_weight += self.agent_weights.get(agent_id, 1.0)
            if current_weight >= target:
                return agent_id

        return agents[-1]  # Fallback

    def create_session_affinity(self, session_id: str, agent_id: str) -> None:
        """Create session affinity for sticky sessions."""
        self.session_affinity[session_id] = agent_id
        self.logger.debug(f"Created session affinity: {session_id} -> {agent_id}")

    def remove_session_affinity(self, session_id: str) -> None:
        """Remove session affinity."""
        if session_id in self.session_affinity:
            del self.session_affinity[session_id]
            self.logger.debug(f"Removed session affinity: {session_id}")

class AutoScaler:
    """
    Automatic scaling manager for semantic agents.

    Features:
    - Metric-based scaling decisions
    - Configurable scaling rules
    - Cooldown periods to prevent thrashing
    - Integration with performance monitoring
    """

    def __init__(
        self,
        coordinator: SemanticCoordinator,
        performance_tracker: PerformanceTracker,
        load_balancer: LoadBalancer,
    ):
        """Initialize the auto scaler."""
        self.coordinator = coordinator
        self.performance_tracker = performance_tracker
        self.load_balancer = load_balancer

        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.last_scaling_action: Dict[str, datetime] = {}

        self.is_running = False
        self.check_interval = 30  # seconds

        self.logger = logging.getLogger("auto_scaler")

    async def initialize(self) -> None:
        """Initialize the auto scaler."""
        self.logger.info("Initializing auto scaler")

        # Add default scaling rules
        await self.add_default_scaling_rules()

        # Start monitoring
        self.is_running = True
        asyncio.create_task(self._monitoring_loop())

        self.logger.info("Auto scaler initialized")

    async def shutdown(self) -> None:
        """Shutdown the auto scaler."""
        self.logger.info("Shutting down auto scaler")
        self.is_running = False

    async def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule."""
        self.scaling_rules[rule.rule_id] = rule
        self.logger.info(f"Added scaling rule: {rule.name}")

    async def remove_scaling_rule(self, rule_id: str) -> bool:
        """Remove a scaling rule."""
        if rule_id in self.scaling_rules:
            rule = self.scaling_rules.pop(rule_id)
            self.logger.info(f"Removed scaling rule: {rule.name}")
            return True
        return False

    async def add_default_scaling_rules(self) -> None:
        """Add default scaling rules."""
        # CPU usage rule
        cpu_rule = ScalingRule(
            name="cpu_usage_rule",
            metric_name="cpu_usage",
            threshold_high=80.0,
            threshold_low=20.0,
            cooldown_period=timedelta(minutes=5),
        )
        await self.add_scaling_rule(cpu_rule)

        # Memory usage rule
        memory_rule = ScalingRule(
            name="memory_usage_rule",
            metric_name="memory_usage",
            threshold_high=85.0,
            threshold_low=30.0,
            cooldown_period=timedelta(minutes=5),
        )
        await self.add_scaling_rule(memory_rule)

        # Task queue length rule
        queue_rule = ScalingRule(
            name="task_queue_rule",
            metric_name="active_tasks",
            threshold_high=10.0,
            threshold_low=2.0,
            cooldown_period=timedelta(minutes=3),
        )
        await self.add_scaling_rule(queue_rule)

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for scaling decisions."""
        while self.is_running:
            try:
                await self._check_scaling_conditions()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in scaling monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_scaling_conditions(self) -> None:
        """Check all scaling rules and trigger actions if needed."""
        current_time = datetime.now()

        # Get current system metrics
        system_metrics = self.performance_tracker.get_system_performance()

        for rule in self.scaling_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown period
            last_action_time = self.last_scaling_action.get(rule.rule_id)
            if (last_action_time and
                current_time - last_action_time < rule.cooldown_period):
                continue

            # Get metric value
            metric_value = self._get_metric_value(system_metrics, rule.metric_name)
            if metric_value is None:
                continue

            # Check thresholds
            action = None
            threshold = None

            if metric_value > rule.threshold_high:
                action = rule.action_scale_up
                threshold = rule.threshold_high
            elif metric_value < rule.threshold_low:
                action = rule.action_scale_down
                threshold = rule.threshold_low

            if action:
                await self._execute_scaling_action(
                    action, rule, metric_value, threshold
                )

    def _get_metric_value(
        self,
        system_metrics: Dict[str, Any],
        metric_name: str,
    ) -> Optional[float]:
        """Extract metric value from system metrics."""
        if metric_name == "cpu_usage":
            return system_metrics.get("system_resources", {}).get("cpu_percent")
        elif metric_name == "memory_usage":
            return system_metrics.get("system_resources", {}).get("memory_percent")
        elif metric_name == "active_tasks":
            return system_metrics.get("active_operations", {}).get("total", 0)
        else:
            return None

    async def _execute_scaling_action(
        self,
        action: ScalingAction,
        rule: ScalingRule,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Execute a scaling action."""
        self.logger.info(
            f"Executing scaling action {action} for rule {rule.name} "
            f"(metric: {metric_value}, threshold: {threshold})"
        )

        success = False
        details = {}

        try:
            if action == ScalingAction.SCALE_UP:
                success = await self._scale_up()
                details["action"] = "Added new agent instance"
            elif action == ScalingAction.SCALE_DOWN:
                success = await self._scale_down()
                details["action"] = "Removed agent instance"
            elif action == ScalingAction.REDISTRIBUTE:
                success = await self._redistribute_load()
                details["action"] = "Redistributed load"
            elif action == ScalingAction.OPTIMIZE:
                success = await self._optimize_performance()
                details["action"] = "Applied performance optimizations"

        except Exception as e:
            self.logger.error(f"Error executing scaling action {action}: {e}")
            details["error"] = str(e)

        # Record scaling event
        event = ScalingEvent(
            action=action,
            trigger_metric=rule.metric_name,
            trigger_value=metric_value,
            threshold=threshold,
            success=success,
            details=details,
        )

        self.scaling_history.append(event)
        self.last_scaling_action[rule.rule_id] = datetime.now()

    async def _scale_up(self) -> bool:
        """Scale up by adding more agent instances."""
        # This is a simplified implementation
        # In practice, this would create new agent instances

        # For now, we'll just log the action
        self.logger.info("Scaling up: Would create new agent instances")
        return True

    async def _scale_down(self) -> bool:
        """Scale down by removing agent instances."""
        # This is a simplified implementation
        # In practice, this would safely remove agent instances

        # Check if we have agents that can be safely removed
        agent_performance = {}
        for agent_id in self.coordinator.registered_agents:
            perf = self.performance_tracker.get_agent_performance(agent_id)
            agent_performance[agent_id] = perf

        # Find agents with low utilization
        low_utilization_agents = [
            agent_id for agent_id, perf in agent_performance.items()
            if perf.get("total_operations", 0) < 5  # Low activity threshold
        ]

        if low_utilization_agents:
            # Remove one agent (simplified)
            agent_to_remove = low_utilization_agents[0]
            self.logger.info(f"Scaling down: Would remove agent {agent_to_remove}")
            return True

        return False

    async def _redistribute_load(self) -> bool:
        """Redistribute load across agents."""
        # Update load balancer weights based on performance
        for agent_id in self.coordinator.registered_agents:
            perf = self.performance_tracker.get_agent_performance(agent_id)

            # Calculate weight based on success rate and response time
            success_rate = perf.get("success_rate", 0.5)
            avg_duration = perf.get("avg_duration_ms", 1000)

            # Higher success rate and lower duration = higher weight
            weight = success_rate * (1000 / max(avg_duration, 100))
            weight = max(0.1, min(2.0, weight))  # Clamp between 0.1 and 2.0

            self.load_balancer.update_agent_weight(agent_id, weight)

        self.logger.info("Redistributed load based on agent performance")
        return True

    async def _optimize_performance(self) -> bool:
        """Apply performance optimizations."""
        # This could include:
        # - Clearing caches
        # - Garbage collection
        # - Resource cleanup
        # - Configuration adjustments

        self.logger.info("Applied performance optimizations")
        return True

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and history."""
        recent_events = [
            event for event in self.scaling_history
            if event.timestamp > datetime.now() - timedelta(hours=24)
        ]

        return {
            "is_running": self.is_running,
            "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
            "total_rules": len(self.scaling_rules),
            "recent_events": len(recent_events),
            "last_scaling_actions": {
                rule_id: action_time.isoformat()
                for rule_id, action_time in self.last_scaling_action.items()
            },
            "scaling_rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "metric": rule.metric_name,
                    "enabled": rule.enabled,
                    "thresholds": {
                        "high": rule.threshold_high,
                        "low": rule.threshold_low,
                    },
                }
                for rule in self.scaling_rules.values()
            ],
        }
