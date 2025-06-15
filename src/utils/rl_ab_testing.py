"""
A/B testing framework for reinforcement learning strategies in DataMCPServerAgent.
This module provides utilities for comparing different RL approaches and automatically
selecting the best one based on performance metrics.
"""

import random
import time
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic

from src.agents.advanced_rl_decision_making import (
    AdvancedRLCoordinatorAgent,
    create_advanced_rl_agent_architecture,
)
from src.agents.multi_objective_rl import (
    MultiObjectiveRLCoordinatorAgent,
    create_multi_objective_rl_agent_architecture,
)
from src.agents.reinforcement_learning import (
    RLCoordinatorAgent,
    create_rl_agent_architecture,
)
from src.memory.memory_persistence import MemoryDatabase


class RLStrategyVariant:
    """Represents a variant of a reinforcement learning strategy for A/B testing."""

    def __init__(
        self,
        name: str,
        agent: Union[
            RLCoordinatorAgent, AdvancedRLCoordinatorAgent, MultiObjectiveRLCoordinatorAgent
        ],
        variant_type: str,
        config: Dict[str, Any],
    ):
        """Initialize the RL strategy variant.

        Args:
            name: Name of the variant
            agent: RL agent instance
            variant_type: Type of variant ("basic", "advanced", "multi_objective")
            config: Configuration parameters
        """
        self.name = name
        self.agent = agent
        self.variant_type = variant_type
        self.config = config
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_reward": 0.0,
            "avg_response_time": 0.0,
            "avg_tool_usage": 0.0,
        }
        self.request_history = []

    async def process_request(self, request: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a request using this variant.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Process the request using the agent
        result = await self.agent.process_request(request, history)

        # Update performance metrics
        self.performance_metrics["total_requests"] += 1
        if result.get("success", False):
            self.performance_metrics["successful_requests"] += 1

        # Update reward
        reward = result.get("reward", 0.0)
        if isinstance(reward, dict) and "total" in reward:
            reward = reward["total"]
        self.performance_metrics["total_reward"] += reward

        # Update response time
        response_time = result.get("performance_metrics", {}).get("response_time", 0.0)
        current_avg = self.performance_metrics["avg_response_time"]
        count = self.performance_metrics["total_requests"]
        self.performance_metrics["avg_response_time"] = (
            current_avg * (count - 1) + response_time
        ) / count

        # Update tool usage
        tool_usage = result.get("performance_metrics", {}).get("tool_usage", 0)
        current_avg = self.performance_metrics["avg_tool_usage"]
        self.performance_metrics["avg_tool_usage"] = (
            current_avg * (count - 1) + tool_usage
        ) / count

        # Add to request history
        self.request_history.append(
            {
                "timestamp": time.time(),
                "request": request,
                "success": result.get("success", False),
                "reward": reward,
                "response_time": response_time,
                "tool_usage": tool_usage,
            }
        )

        # Return the result
        return result

    def get_success_rate(self) -> float:
        """Get the success rate for this variant.

        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.performance_metrics["total_requests"] == 0:
            return 0.0
        return (
            self.performance_metrics["successful_requests"]
            / self.performance_metrics["total_requests"]
        )

    def get_avg_reward(self) -> float:
        """Get the average reward for this variant.

        Returns:
            Average reward
        """
        if self.performance_metrics["total_requests"] == 0:
            return 0.0
        return self.performance_metrics["total_reward"] / self.performance_metrics["total_requests"]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics for this variant.

        Returns:
            Performance summary
        """
        return {
            "name": self.name,
            "variant_type": self.variant_type,
            "config": self.config,
            "success_rate": self.get_success_rate(),
            "avg_reward": self.get_avg_reward(),
            "avg_response_time": self.performance_metrics["avg_response_time"],
            "avg_tool_usage": self.performance_metrics["avg_tool_usage"],
            "total_requests": self.performance_metrics["total_requests"],
        }


class RLABTestingFramework:
    """Framework for A/B testing different reinforcement learning strategies."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        sub_agents: Dict[str, Any],
        tools: List[Any],
        exploration_rate: float = 0.2,
    ):
        """Initialize the RL A/B testing framework.

        Args:
            model: Language model to use
            db: Memory database for persistence
            sub_agents: Dictionary of sub-agents
            tools: List of available tools
            exploration_rate: Exploration rate for variant selection
        """
        self.model = model
        self.db = db
        self.sub_agents = sub_agents
        self.tools = tools
        self.exploration_rate = exploration_rate
        self.variants = {}
        self.default_variant = None
        self.test_results = []

    async def add_variant(
        self,
        name: str,
        variant_type: str,
        config: Dict[str, Any],
        set_as_default: bool = False,
    ) -> None:
        """Add a variant for testing.

        Args:
            name: Name of the variant
            variant_type: Type of variant ("basic", "advanced", "multi_objective")
            config: Configuration parameters
            set_as_default: Whether to set this variant as the default
        """
        # Create the agent based on variant type
        if variant_type == "basic":
            agent = await create_rl_agent_architecture(
                model=self.model,
                db=self.db,
                sub_agents=self.sub_agents,
                rl_agent_type=config.get("rl_agent_type", "q_learning"),
            )
        elif variant_type == "advanced":
            agent = await create_advanced_rl_agent_architecture(
                model=self.model,
                db=self.db,
                sub_agents=self.sub_agents,
                tools=self.tools,
                rl_agent_type=config.get("rl_agent_type", "deep_rl"),
            )
        elif variant_type == "multi_objective":
            agent = await create_multi_objective_rl_agent_architecture(
                model=self.model,
                db=self.db,
                sub_agents=self.sub_agents,
                objectives=config.get(
                    "objectives",
                    ["user_satisfaction", "task_completion", "efficiency", "accuracy"],
                ),
                objective_weights=config.get("objective_weights"),
            )
        else:
            raise ValueError(f"Unknown variant type: {variant_type}")

        # Create the variant
        variant = RLStrategyVariant(name, agent, variant_type, config)

        # Add to variants
        self.variants[name] = variant

        # Set as default if requested
        if set_as_default or self.default_variant is None:
            self.default_variant = name

    def select_variant(self, request: str) -> str:
        """Select a variant for a request using epsilon-greedy strategy.

        Args:
            request: User request

        Returns:
            Name of the selected variant
        """
        # If no variants, return None
        if not self.variants:
            return None

        # Exploration: random variant
        if random.random() < self.exploration_rate:
            return random.choice(list(self.variants.keys()))

        # Exploitation: best variant by average reward
        best_variant = max(self.variants.values(), key=lambda v: v.get_avg_reward())
        return best_variant.name

    async def process_request(self, request: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a request using the A/B testing framework.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Select a variant
        variant_name = self.select_variant(request)

        # If no variant selected, use default
        if variant_name is None:
            if self.default_variant is None:
                raise ValueError("No variants available for testing")
            variant_name = self.default_variant

        # Get the variant
        variant = self.variants[variant_name]

        # Process the request using the variant
        result = await variant.process_request(request, history)

        # Add variant information to the result
        result["variant"] = variant_name

        return result

    def get_test_results(self) -> Dict[str, Any]:
        """Get the results of the A/B test.

        Returns:
            Test results
        """
        # Get performance summaries for all variants
        variant_summaries = {
            name: variant.get_performance_summary() for name, variant in self.variants.items()
        }

        # Find the best variant by different metrics
        if variant_summaries:
            best_by_success_rate = max(variant_summaries.values(), key=lambda v: v["success_rate"])
            best_by_avg_reward = max(variant_summaries.values(), key=lambda v: v["avg_reward"])
            best_by_response_time = min(
                variant_summaries.values(), key=lambda v: v["avg_response_time"]
            )
        else:
            best_by_success_rate = None
            best_by_avg_reward = None
            best_by_response_time = None

        # Create test results
        test_results = {
            "variant_summaries": variant_summaries,
            "best_variants": {
                "by_success_rate": best_by_success_rate["name"] if best_by_success_rate else None,
                "by_avg_reward": best_by_avg_reward["name"] if best_by_avg_reward else None,
                "by_response_time": (
                    best_by_response_time["name"] if best_by_response_time else None
                ),
            },
            "total_requests": sum(v["total_requests"] for v in variant_summaries.values()),
        }

        # Save test results
        self.test_results.append(test_results)

        return test_results

    def get_best_variant(self, metric: str = "avg_reward") -> Optional[str]:
        """Get the name of the best variant based on a metric.

        Args:
            metric: Metric to use for comparison ("success_rate", "avg_reward", "avg_response_time")

        Returns:
            Name of the best variant
        """
        if not self.variants:
            return None

        if metric == "success_rate":
            return max(self.variants.items(), key=lambda x: x[1].get_success_rate())[0]
        elif metric == "avg_reward":
            return max(self.variants.items(), key=lambda x: x[1].get_avg_reward())[0]
        elif metric == "avg_response_time":
            return min(
                self.variants.items(),
                key=lambda x: x[1].performance_metrics["avg_response_time"],
            )[0]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def set_default_variant(self, variant_name: str) -> None:
        """Set the default variant.

        Args:
            variant_name: Name of the variant to set as default
        """
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")
        self.default_variant = variant_name

    def auto_optimize(self, metric: str = "avg_reward") -> str:
        """Automatically set the best variant as the default based on a metric.

        Args:
            metric: Metric to use for optimization ("success_rate", "avg_reward", "avg_response_time")

        Returns:
            Name of the selected best variant
        """
        best_variant = self.get_best_variant(metric)
        if best_variant:
            self.set_default_variant(best_variant)
        return best_variant
