"""
Multi-objective reinforcement learning module for DataMCPServerAgent.
This module provides mechanisms for agents to learn from multiple objectives
and balance different goals in decision making.
"""

import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase


class MultiObjectiveRewardSystem(RewardSystem):
    """System for calculating rewards based on multiple objectives."""

    def __init__(self, db: MemoryDatabase, objective_weights: Optional[Dict[str, float]] = None):
        """Initialize the multi-objective reward system.

        Args:
            db: Memory database for persistence
            objective_weights: Dictionary of objective weights
        """
        super().__init__(db)

        # Define default objective weights if not provided
        self.objective_weights = objective_weights or {
            "user_satisfaction": 0.4,  # Weight for user satisfaction
            "task_completion": 0.3,  # Weight for task completion
            "efficiency": 0.2,  # Weight for efficiency (time, resources)
            "accuracy": 0.1,  # Weight for accuracy
        }

        # Initialize reward history for each objective
        self.objective_reward_history = {}

    def calculate_reward(
        self,
        agent_name: str,
        feedback: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate rewards for multiple objectives.

        Args:
            agent_name: Name of the agent
            feedback: User feedback and self-evaluation
            performance_metrics: Performance metrics

        Returns:
            Dictionary of rewards for each objective
        """
        # Calculate rewards for each objective
        objective_rewards = {
            "user_satisfaction": self._calculate_user_satisfaction(feedback),
            "task_completion": self._calculate_task_completion(performance_metrics),
            "efficiency": self._calculate_efficiency(performance_metrics),
            "accuracy": self._calculate_accuracy(feedback, performance_metrics),
        }

        # Calculate weighted sum for total reward
        total_reward = sum(
            self.objective_weights[obj] * reward
            for obj, reward in objective_rewards.items()
        )

        # Store the rewards in history
        if agent_name not in self.objective_reward_history:
            self.objective_reward_history[agent_name] = []

        self.objective_reward_history[agent_name].append(
            {
                "timestamp": time.time(),
                "total_reward": total_reward,
                "objective_rewards": objective_rewards,
            }
        )

        # Store the rewards in the database
        self.db.save_agent_multi_objective_reward(
            agent_name, total_reward, objective_rewards
        )

        # Return both total reward and objective rewards
        return {"total": total_reward, **objective_rewards}

    def _calculate_accuracy(
        self, feedback: Dict[str, Any], performance_metrics: Dict[str, Any]
    ) -> float:
        """Calculate accuracy reward component.

        Args:
            feedback: User feedback and self-evaluation
            performance_metrics: Performance metrics

        Returns:
            Accuracy reward component
        """
        # Check for explicit accuracy metric
        if "accuracy" in performance_metrics:
            return performance_metrics["accuracy"]

        # Check for self-evaluation accuracy
        self_evaluation = feedback.get("self_evaluation", {})
        if "accuracy" in self_evaluation:
            return self_evaluation["accuracy"]

        # Check for factual correctness in feedback
        user_feedback = feedback.get("user_feedback", {})
        feedback_text = user_feedback.get("feedback", "").lower()

        # Simple keyword analysis for accuracy
        accuracy_keywords = {
            "positive": ["correct", "accurate", "right", "factual", "true"],
            "negative": ["incorrect", "inaccurate", "wrong", "false", "mistake", "error"],
        }

        positive_count = sum(
            1 for word in accuracy_keywords["positive"] if word in feedback_text
        )
        negative_count = sum(
            1 for word in accuracy_keywords["negative"] if word in feedback_text
        )

        # Calculate accuracy score
        if positive_count + negative_count > 0:
            accuracy = positive_count / (positive_count + negative_count)
        else:
            accuracy = 0.5  # Neutral

        return accuracy

    def update_objective_weights(
        self, new_weights: Dict[str, float], normalize: bool = True
    ) -> None:
        """Update the weights for different objectives.

        Args:
            new_weights: New weights for objectives
            normalize: Whether to normalize weights to sum to 1
        """
        # Update weights
        for objective, weight in new_weights.items():
            if objective in self.objective_weights:
                self.objective_weights[objective] = weight

        # Normalize weights if requested
        if normalize:
            total_weight = sum(self.objective_weights.values())
            if total_weight > 0:
                self.objective_weights = {
                    obj: weight / total_weight
                    for obj, weight in self.objective_weights.items()
                }

    def get_objective_rewards(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent objective rewards for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to return

        Returns:
            List of recent objective rewards
        """
        if agent_name in self.objective_reward_history:
            return sorted(
                self.objective_reward_history[agent_name],
                key=lambda x: x["timestamp"],
                reverse=True,
            )[:limit]

        return []


class MOQLearningAgent:
    """Agent that learns using multi-objective Q-learning algorithm."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: MultiObjectiveRewardSystem,
        state_extractor: Callable[[Dict[str, Any]], str],
        actions: List[str],
        objectives: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
    ):
        """Initialize the multi-objective Q-learning agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Multi-objective reward system
            state_extractor: Function to extract state from context
            actions: List of possible actions
            objectives: List of objectives
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Exploration rate (epsilon)
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_extractor = state_extractor
        self.actions = actions
        self.objectives = objectives
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize Q-tables for each objective
        self.q_tables = self.db.get_mo_q_tables(name) or {
            objective: {} for objective in objectives
        }

    def select_action(self, state: str) -> str:
        """Select an action using scalarized Q-values.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Exploitation: best action from scalarized Q-values
        return self._get_best_action(state)

    def _get_best_action(self, state: str) -> str:
        """Get the best action for a state using scalarized Q-values.

        Args:
            state: Current state

        Returns:
            Best action
        """
        # Initialize state in Q-tables if not present
        for objective in self.objectives:
            if state not in self.q_tables[objective]:
                self.q_tables[objective][state] = {
                    action: 0.0 for action in self.actions
                }

        # Calculate scalarized Q-values
        scalarized_q_values = {}
        for action in self.actions:
            # Weighted sum of Q-values across objectives
            scalarized_q_values[action] = sum(
                self.reward_system.objective_weights.get(objective, 1.0 / len(self.objectives))
                * self.q_tables[objective][state][action]
                for objective in self.objectives
            )

        # Get action with highest scalarized Q-value
        return max(scalarized_q_values, key=scalarized_q_values.get)

    def update_q_values(
        self, state: str, action: str, rewards: Dict[str, float], next_state: str
    ) -> None:
        """Update Q-values for each objective.

        Args:
            state: Current state
            action: Action taken
            rewards: Dictionary of rewards for each objective
            next_state: Next state
        """
        # Initialize states in Q-tables if not present
        for objective in self.objectives:
            if state not in self.q_tables[objective]:
                self.q_tables[objective][state] = {
                    action: 0.0 for action in self.actions
                }
            if next_state not in self.q_tables[objective]:
                self.q_tables[objective][next_state] = {
                    action: 0.0 for action in self.actions
                }

        # Update Q-values for each objective
        for objective in self.objectives:
            if objective in rewards:
                # Get current Q-value
                current_q = self.q_tables[objective][state].get(action, 0.0)

                # Get max Q-value for next state
                max_next_q = max(self.q_tables[objective][next_state].values())

                # Calculate new Q-value
                new_q = current_q + self.learning_rate * (
                    rewards[objective] + self.discount_factor * max_next_q - current_q
                )

                # Update Q-table
                self.q_tables[objective][state][action] = new_q

        # Save Q-tables to database
        self.db.save_mo_q_tables(self.name, self.q_tables)


class MultiObjectiveRLCoordinatorAgent:
    """Coordinator agent that uses multi-objective reinforcement learning for decision making."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: MultiObjectiveRewardSystem,
        sub_agents: Dict[str, Any],
        objectives: List[str],
    ):
        """Initialize the multi-objective RL coordinator agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Multi-objective reward system
            sub_agents: Dictionary of sub-agents
            objectives: List of objectives
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.sub_agents = sub_agents
        self.objectives = objectives

        # Define actions (which sub-agents to use)
        self.actions = list(sub_agents.keys())

        # Create multi-objective Q-learning agent
        self.rl_agent = MOQLearningAgent(
            name=f"{name}_moql",
            model=model,
            db=db,
            reward_system=reward_system,
            state_extractor=self._extract_state,
            actions=self.actions,
            objectives=objectives,
        )

        # Create the prompt for state extraction
        self.state_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a state extraction agent responsible for converting user requests into state representations.
Your job is to analyze a user request and extract key features that can be used to determine the appropriate agent to handle it.

For each request, you should:
1. Identify the main task type (search, scraping, analysis, etc.)
2. Recognize entities mentioned in the request
3. Determine the complexity level
4. Identify any special requirements

Respond with a concise state identifier that captures these key aspects.
"""
                ),
                HumanMessage(
                    content="""
User request:
{request}

Recent conversation:
{history}

Extract a state identifier for this request.
"""
                ),
            ]
        )

    async def _extract_state(self, context: Dict[str, Any]) -> str:
        """Extract state from context.

        Args:
            context: Context dictionary

        Returns:
            State identifier
        """
        # Extract request and history from context
        request = context.get("request", "")
        history = context.get("history", [])

        # Format history
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-3:]]
        )

        # Prepare the input for the state extraction prompt
        input_values = {"request": request, "history": formatted_history}

        # Get the state identifier from the model
        messages = self.state_extraction_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Return the state identifier
        return response.content.strip()

    async def process_request(
        self, request: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a user request using multi-objective reinforcement learning for agent selection.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Create context for state extraction
        context = {"request": request, "history": history}

        # Extract state
        state = await self._extract_state(context)

        # Select sub-agent using multi-objective RL
        selected_agent_name = self.rl_agent.select_action(state)

        # Get the selected sub-agent
        selected_agent = self.sub_agents[selected_agent_name]

        # Record start time for performance tracking
        start_time = time.time()

        # Execute the selected sub-agent
        result = await selected_agent.execute(request, self.db)

        # Record end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # Calculate performance metrics
        performance_metrics = {
            "success_rate": 1.0 if result["success"] else 0.0,
            "response_time": duration,
            "tool_usage": len(result.get("tool_calls", []))
            if "tool_calls" in result
            else 0,
            "accuracy": result.get("accuracy", 0.5),  # Default to neutral
        }

        # Calculate rewards for multiple objectives
        feedback = {
            "user_feedback": {},  # Will be updated later with actual user feedback
            "self_evaluation": result.get("self_evaluation", {}),
        }

        rewards = self.reward_system.calculate_reward(
            selected_agent_name, feedback, performance_metrics
        )

        # For multi-objective Q-learning, we need the next state
        next_context = {
            "request": request,
            "history": history
            + [
                {"role": "user", "content": request},
                {
                    "role": "assistant",
                    "content": result["response"]
                    if result["success"]
                    else result["error"],
                },
            ],
        }
        next_state = await self._extract_state(next_context)

        # Update Q-values
        self.rl_agent.update_q_values(state, selected_agent_name, rewards, next_state)

        # Return the result with additional RL information
        return {
            **result,
            "selected_agent": selected_agent_name,
            "rewards": rewards,
            "performance_metrics": performance_metrics,
        }


# Factory function to create multi-objective RL-based agent architecture
async def create_multi_objective_rl_agent_architecture(
    model: ChatAnthropic,
    db: MemoryDatabase,
    sub_agents: Dict[str, Any],
    objectives: Optional[List[str]] = None,
    objective_weights: Optional[Dict[str, float]] = None,
) -> MultiObjectiveRLCoordinatorAgent:
    """Create a multi-objective reinforcement learning-based agent architecture.

    Args:
        model: Language model to use
        db: Memory database for persistence
        sub_agents: Dictionary of sub-agents
        objectives: List of objectives (default: ["user_satisfaction", "task_completion", "efficiency", "accuracy"])
        objective_weights: Dictionary of objective weights

    Returns:
        Multi-objective RL coordinator agent
    """
    # Define default objectives if not provided
    if objectives is None:
        objectives = ["user_satisfaction", "task_completion", "efficiency", "accuracy"]

    # Create multi-objective reward system
    reward_system = MultiObjectiveRewardSystem(db, objective_weights)

    # Create multi-objective RL coordinator agent
    mo_rl_coordinator = MultiObjectiveRLCoordinatorAgent(
        name="mo_rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        objectives=objectives,
    )

    return mo_rl_coordinator
