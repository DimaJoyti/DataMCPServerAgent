"""
Agent domain services.
Contains business logic for agent management, scaling, and orchestration.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.agent import Agent, AgentConfiguration, AgentMetrics, AgentStatus, AgentType
from app.domain.models.base import BusinessRuleError, DomainService, ValidationError
from app.domain.models.task import Task, TaskStatus

logger = get_logger(__name__)


class AgentService(DomainService, LoggerMixin):
    """Core agent management service."""

    async def create_agent(
        self,
        name: str,
        agent_type: AgentType,
        configuration: AgentConfiguration = None,
        description: str = None,
    ) -> Agent:
        """Create a new agent."""
        self.logger.info(f"Creating agent: {name} of type {agent_type}")

        # Validate agent name uniqueness
        agent_repo = self.get_repository("agent")
        existing_agents = await agent_repo.list(name=name)
        if existing_agents:
            raise ValidationError(f"Agent with name '{name}' already exists")

        # Create agent
        agent = Agent(
            name=name,
            agent_type=agent_type,
            configuration=configuration or AgentConfiguration(),
            description=description,
            status=AgentStatus.INITIALIZING,
        )

        # Save agent
        saved_agent = await agent_repo.save(agent)

        self.logger.info(f"Agent created successfully: {saved_agent.id}")
        return saved_agent

    async def update_agent_status(self, agent_id: str, new_status: AgentStatus) -> Agent:
        """Update agent status."""
        agent_repo = self.get_repository("agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Agent not found: {agent_id}")

        self.logger.info(f"Updating agent {agent_id} status from {agent.status} to {new_status}")

        agent.change_status(new_status)
        return await agent_repo.save(agent)

    async def update_agent_metrics(self, agent_id: str, metrics: AgentMetrics) -> Agent:
        """Update agent metrics."""
        agent_repo = self.get_repository("agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Agent not found: {agent_id}")

        agent.update_metrics(metrics)
        return await agent_repo.save(agent)

    async def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get all agents of a specific type."""
        agent_repo = self.get_repository("agent")
        return await agent_repo.list(agent_type=agent_type)

    async def get_healthy_agents(self) -> List[Agent]:
        """Get all healthy agents."""
        agent_repo = self.get_repository("agent")
        all_agents = await agent_repo.list()

        return [agent for agent in all_agents if agent.metrics.is_healthy]

    async def get_agents_needing_scaling(self) -> List[Agent]:
        """Get agents that need scaling (desired != current instances)."""
        agent_repo = self.get_repository("agent")
        all_agents = await agent_repo.list()

        return [
            agent
            for agent in all_agents
            if agent.desired_instances != agent.current_instances and agent.is_scalable()
        ]

    async def assign_task_to_agent(self, task: Task) -> Optional[Agent]:
        """Find and assign a task to the best available agent."""
        # Get agents that can handle this task type
        suitable_agents = await self._find_suitable_agents_for_task(task)

        if not suitable_agents:
            self.logger.warning(f"No suitable agents found for task {task.id}")
            return None

        # Select best agent based on load and capabilities
        best_agent = await self._select_best_agent(suitable_agents, task)

        self.logger.info(f"Assigned task {task.id} to agent {best_agent.id}")
        return best_agent

    async def _find_suitable_agents_for_task(self, task: Task) -> List[Agent]:
        """Find agents suitable for a specific task."""
        agent_repo = self.get_repository("agent")

        # Get all active agents
        active_agents = await agent_repo.list(status=AgentStatus.ACTIVE)

        # Filter by capability and availability
        suitable_agents = []
        for agent in active_agents:
            if agent.can_accept_tasks():
                # Check if agent has required capabilities for task type
                if await self._agent_can_handle_task(agent, task):
                    suitable_agents.append(agent)

        return suitable_agents

    async def _agent_can_handle_task(self, agent: Agent, task: Task) -> bool:
        """Check if an agent can handle a specific task."""
        # Basic type matching
        task_type_mapping = {
            "data_analysis": ["analytics", "worker"],
            "email_processing": ["email", "worker"],
            "webrtc_call": ["webrtc", "worker"],
            "state_sync": ["worker"],
            "scaling_operation": ["orchestrator", "worker"],
            "health_check": ["worker"],
        }

        required_types = task_type_mapping.get(task.task_type.value, ["worker"])
        agent_type_matches = agent.agent_type.value in required_types

        # Check specific capabilities if needed
        if task.task_type.value == "data_analysis":
            return agent_type_matches and agent.has_capability("data_processing")
        elif task.task_type.value == "email_processing":
            return agent_type_matches and agent.has_capability("email_handling")
        elif task.task_type.value == "webrtc_call":
            return agent_type_matches and agent.has_capability("webrtc_communication")

        return agent_type_matches

    async def _select_best_agent(self, agents: List[Agent], task: Task) -> Agent:
        """Select the best agent from a list of suitable agents."""
        if not agents:
            raise BusinessRuleError("No agents available for task assignment")

        # Score agents based on various factors
        scored_agents = []
        for agent in agents:
            score = await self._calculate_agent_score(agent, task)
            scored_agents.append((agent, score))

        # Sort by score (higher is better)
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        return scored_agents[0][0]

    async def _calculate_agent_score(self, agent: Agent, task: Task) -> float:
        """Calculate a score for agent suitability."""
        score = 0.0

        # Base score for being active and healthy
        if agent.status == AgentStatus.ACTIVE and agent.metrics.is_healthy:
            score += 100.0

        # Prefer agents with lower CPU usage
        cpu_score = max(0, 100 - agent.metrics.cpu_usage_percent)
        score += cpu_score * 0.3

        # Prefer agents with lower memory usage
        memory_usage_percent = (
            agent.metrics.memory_usage_mb / agent.configuration.memory_limit_mb
        ) * 100
        memory_score = max(0, 100 - memory_usage_percent)
        score += memory_score * 0.2

        # Prefer agents with higher success rate
        score += agent.metrics.success_rate * 0.3

        # Prefer agents with lower response time
        if agent.metrics.average_response_time_ms > 0:
            response_time_score = max(0, 100 - (agent.metrics.average_response_time_ms / 1000))
            score += response_time_score * 0.2

        # Priority bonus for matching task priority
        if task.priority.value == "critical":
            score += 50.0
        elif task.priority.value == "high":
            score += 25.0

        return score


class AgentScalingService(DomainService, LoggerMixin):
    """Service for agent scaling operations."""

    async def scale_agent(self, agent_id: str, target_instances: int) -> Agent:
        """Scale an agent to target number of instances."""
        agent_repo = self.get_repository("agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Agent not found: {agent_id}")

        if not agent.is_scalable():
            raise BusinessRuleError(
                f"Agent {agent_id} cannot be scaled in current status: {agent.status}"
            )

        self.logger.info(
            f"Scaling agent {agent_id} from {agent.current_instances} to {target_instances} instances"
        )

        agent.scale_to(target_instances)
        return await agent_repo.save(agent)

    async def auto_scale_agents(self) -> List[Agent]:
        """Automatically scale agents based on load and metrics."""
        agent_repo = self.get_repository("agent")
        all_agents = await agent_repo.list()

        scaled_agents = []

        for agent in all_agents:
            if not agent.is_scalable():
                continue

            scaling_decision = await self._make_scaling_decision(agent)

            if scaling_decision["action"] == "scale_up":
                new_instances = min(
                    agent.current_instances + scaling_decision["instances"],
                    10,  # Max instances limit
                )
                agent.scale_to(new_instances)
                scaled_agents.append(agent)

            elif scaling_decision["action"] == "scale_down":
                new_instances = max(
                    agent.current_instances - scaling_decision["instances"],
                    1,  # Min instances limit
                )
                agent.scale_to(new_instances)
                scaled_agents.append(agent)

        # Save all scaled agents
        for agent in scaled_agents:
            await agent_repo.save(agent)

        return scaled_agents

    async def _make_scaling_decision(self, agent: Agent) -> Dict[str, Any]:
        """Make scaling decision for an agent based on metrics."""
        # Get recent tasks for this agent
        task_repo = self.get_repository("task")
        recent_tasks = await task_repo.list(
            agent_id=agent.id, created_at_gte=datetime.now(timezone.utc) - timedelta(minutes=15)
        )

        # Calculate load metrics
        running_tasks = len([t for t in recent_tasks if t.status == TaskStatus.RUNNING])
        queued_tasks = len([t for t in recent_tasks if t.status == TaskStatus.QUEUED])

        cpu_usage = agent.metrics.cpu_usage_percent
        memory_usage = (agent.metrics.memory_usage_mb / agent.configuration.memory_limit_mb) * 100

        # Scale up conditions
        if (
            cpu_usage > 80
            or memory_usage > 80
            or queued_tasks > 5
            or running_tasks > agent.configuration.max_concurrent_tasks * 0.8
        ):
            return {
                "action": "scale_up",
                "instances": 1,
                "reason": f"High load: CPU={cpu_usage}%, Memory={memory_usage}%, Queued={queued_tasks}",
            }

        # Scale down conditions
        if (
            agent.current_instances > 1
            and cpu_usage < 20
            and memory_usage < 30
            and queued_tasks == 0
            and running_tasks < agent.configuration.max_concurrent_tasks * 0.2
        ):
            return {
                "action": "scale_down",
                "instances": 1,
                "reason": f"Low load: CPU={cpu_usage}%, Memory={memory_usage}%, Queued={queued_tasks}",
            }

        return {"action": "maintain", "instances": 0, "reason": "Load within normal range"}

    async def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations for all agents."""
        agent_repo = self.get_repository("agent")
        all_agents = await agent_repo.list()

        recommendations = []

        for agent in all_agents:
            if not agent.is_scalable():
                continue

            decision = await self._make_scaling_decision(agent)

            if decision["action"] != "maintain":
                recommendations.append(
                    {
                        "agent_id": agent.id,
                        "agent_name": agent.name,
                        "current_instances": agent.current_instances,
                        "recommended_action": decision["action"],
                        "recommended_instances": decision["instances"],
                        "reason": decision["reason"],
                        "priority": "high" if decision["action"] == "scale_up" else "low",
                    }
                )

        return recommendations
