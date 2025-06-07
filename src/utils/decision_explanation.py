"""
Decision explanation module for DataMCPServerAgent.
This module provides utilities for explaining reinforcement learning-based decisions.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.memory.memory_persistence import MemoryDatabase

class DecisionExplainer:
    """Utility for explaining reinforcement learning-based decisions."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        explanation_level: str = "detailed",
    ):
        """Initialize the decision explainer.

        Args:
            model: Language model to use
            db: Memory database for persistence
            explanation_level: Level of explanation detail ("simple", "moderate", "detailed")
        """
        self.model = model
        self.db = db
        self.explanation_level = explanation_level

        # Create the explanation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a decision explanation agent responsible for explaining reinforcement learning-based decisions.
Your job is to analyze the decision-making process and provide clear, understandable explanations.

For each decision, you should:
1. Explain the factors that influenced the decision
2. Describe the expected outcomes
3. Compare with alternative choices
4. Provide confidence level and reasoning

Adjust your explanation based on the requested detail level:
- Simple: Brief, high-level explanation focusing on the main factors
- Moderate: More detailed explanation including alternatives
- Detailed: Comprehensive explanation with technical details and confidence metrics
"""
                ),
                HumanMessage(
                    content="""
Decision context:
{context}

Selected action: {selected_action}
Alternative actions: {alternative_actions}
State representation: {state}
Q-values or probabilities: {q_values}
Reward history: {reward_history}
Explanation level: {explanation_level}

Explain this decision in a way that is understandable to the user.
"""
                ),
            ]
        )

    async def explain_decision(
        self,
        context: Dict[str, Any],
        selected_action: str,
        alternative_actions: List[str],
        state: Union[str, List[float]],
        q_values: Dict[str, float],
        reward_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate an explanation for a reinforcement learning-based decision.

        Args:
            context: Decision context
            selected_action: Selected action
            alternative_actions: Alternative actions
            state: State representation
            q_values: Q-values or probabilities for actions
            reward_history: Optional reward history

        Returns:
            Explanation text
        """
        # Format the input for the explanation prompt
        input_values = {
            "context": json.dumps(context, indent=2),
            "selected_action": selected_action,
            "alternative_actions": ", ".join(alternative_actions),
            "state": state if isinstance(state, str) else f"[{', '.join(map(str, state))}]",
            "q_values": json.dumps(q_values, indent=2),
            "reward_history": json.dumps(reward_history or [], indent=2),
            "explanation_level": self.explanation_level,
        }

        # Get the explanation from the model
        messages = self.explanation_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Return the explanation
        return response.content.strip()

class QValueVisualizer:
    """Utility for visualizing Q-values for better understanding."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the Q-value visualizer.

        Args:
            db: Memory database for persistence
        """
        self.db = db

    def get_q_value_summary(
        self, agent_name: str, state: str, actions: List[str]
    ) -> Dict[str, Any]:
        """Get a summary of Q-values for a state.

        Args:
            agent_name: Name of the agent
            state: State to get Q-values for
            actions: List of possible actions

        Returns:
            Q-value summary
        """
        # Get Q-table from database
        q_table = self.db.get_q_table(agent_name) or {}

        # Get Q-values for the state
        state_q_values = q_table.get(state, {})

        # Fill in missing actions with zero values
        for action in actions:
            if action not in state_q_values:
                state_q_values[action] = 0.0

        # Calculate statistics
        if state_q_values:
            max_action = max(state_q_values, key=state_q_values.get)
            min_action = min(state_q_values, key=state_q_values.get)
            avg_value = sum(state_q_values.values()) / len(state_q_values)
            value_range = max(state_q_values.values()) - min(state_q_values.values())
        else:
            max_action = ""
            min_action = ""
            avg_value = 0.0
            value_range = 0.0

        # Create summary
        return {
            "state": state,
            "q_values": state_q_values,
            "max_action": max_action,
            "min_action": min_action,
            "avg_value": avg_value,
            "value_range": value_range,
        }

    def get_multi_objective_q_value_summary(
        self, agent_name: str, state: str, actions: List[str], objectives: List[str]
    ) -> Dict[str, Any]:
        """Get a summary of multi-objective Q-values for a state.

        Args:
            agent_name: Name of the agent
            state: State to get Q-values for
            actions: List of possible actions
            objectives: List of objectives

        Returns:
            Multi-objective Q-value summary
        """
        # Get multi-objective Q-tables from database
        mo_q_tables = self.db.get_mo_q_tables(agent_name) or {
            objective: {} for objective in objectives
        }

        # Get Q-values for each objective
        objective_q_values = {}
        for objective in objectives:
            objective_q_table = mo_q_tables.get(objective, {})
            state_q_values = objective_q_table.get(state, {})

            # Fill in missing actions with zero values
            for action in actions:
                if action not in state_q_values:
                    state_q_values[action] = 0.0

            objective_q_values[objective] = state_q_values

        # Calculate best action for each objective
        best_actions = {
            objective: max(q_values, key=q_values.get)
            for objective, q_values in objective_q_values.items()
            if q_values
        }

        # Create summary
        return {
            "state": state,
            "objective_q_values": objective_q_values,
            "best_actions": best_actions,
        }

class PolicyExplainer:
    """Utility for explaining reinforcement learning policies."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
    ):
        """Initialize the policy explainer.

        Args:
            model: Language model to use
            db: Memory database for persistence
        """
        self.model = model
        self.db = db

        # Create the policy explanation prompt
        self.policy_explanation_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a policy explanation agent responsible for explaining reinforcement learning policies.
Your job is to analyze the learned policy and provide clear, understandable explanations.

For each policy, you should:
1. Explain the general strategy the agent has learned
2. Identify patterns in state-action mappings
3. Highlight strengths and weaknesses of the policy
4. Suggest potential improvements

Focus on making the explanation accessible to non-technical users while still being accurate.
"""
                ),
                HumanMessage(
                    content="""
Agent name: {agent_name}
Policy type: {policy_type}
Policy data: {policy_data}
Recent rewards: {recent_rewards}

Explain this reinforcement learning policy in a way that is understandable to the user.
"""
                ),
            ]
        )

    async def explain_policy(
        self,
        agent_name: str,
        policy_type: str,
        policy_data: Dict[str, Any],
        recent_rewards: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate an explanation for a reinforcement learning policy.

        Args:
            agent_name: Name of the agent
            policy_type: Type of policy ("q_learning", "policy_gradient", "deep_rl", "multi_objective")
            policy_data: Policy data
            recent_rewards: Optional recent rewards

        Returns:
            Explanation text
        """
        # Get recent rewards if not provided
        if recent_rewards is None:
            recent_rewards = self.db.get_agent_rewards(agent_name, limit=5)

        # Format the input for the policy explanation prompt
        input_values = {
            "agent_name": agent_name,
            "policy_type": policy_type,
            "policy_data": json.dumps(policy_data, indent=2),
            "recent_rewards": json.dumps(recent_rewards, indent=2),
        }

        # Get the explanation from the model
        messages = self.policy_explanation_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Return the explanation
        return response.content.strip()

class DecisionTracker:
    """Utility for tracking and analyzing reinforcement learning decisions over time."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the decision tracker.

        Args:
            db: Memory database for persistence
        """
        self.db = db
        self.decision_history = {}

    def record_decision(
        self,
        agent_name: str,
        state: Union[str, List[float]],
        selected_action: str,
        q_values: Dict[str, float],
        reward: Union[float, Dict[str, float]],
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a decision for later analysis.

        Args:
            agent_name: Name of the agent
            state: State representation
            selected_action: Selected action
            q_values: Q-values or probabilities for actions
            reward: Reward received
            timestamp: Optional timestamp (defaults to current time)
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()

        # Initialize agent history if not present
        if agent_name not in self.decision_history:
            self.decision_history[agent_name] = []

        # Record the decision
        self.decision_history[agent_name].append(
            {
                "timestamp": timestamp,
                "state": state,
                "selected_action": selected_action,
                "q_values": q_values,
                "reward": reward,
            }
        )

        # Save to database
        self.db.save_agent_decision(
            agent_name,
            {
                "timestamp": timestamp,
                "state": state if isinstance(state, str) else json.dumps(state),
                "selected_action": selected_action,
                "q_values": q_values,
                "reward": reward if isinstance(reward, float) else json.dumps(reward),
            },
        )

    def get_decision_history(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent decisions for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of decisions to return

        Returns:
            List of recent decisions
        """
        # Get from database
        return self.db.get_agent_decisions(agent_name, limit=limit)

    def analyze_decision_patterns(
        self, agent_name: str, window: int = 20
    ) -> Dict[str, Any]:
        """Analyze patterns in recent decisions.

        Args:
            agent_name: Name of the agent
            window: Number of recent decisions to analyze

        Returns:
            Analysis results
        """
        # Get recent decisions
        decisions = self.get_decision_history(agent_name, limit=window)

        if not decisions:
            return {"status": "No decisions found for analysis"}

        # Count action frequencies
        action_counts = {}
        for decision in decisions:
            action = decision["selected_action"]
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1

        # Calculate average rewards
        action_rewards = {}
        for decision in decisions:
            action = decision["selected_action"]
            reward = decision["reward"]
            if isinstance(reward, str):
                try:
                    reward = json.loads(reward)
                    if isinstance(reward, dict) and "total" in reward:
                        reward = reward["total"]
                    else:
                        reward = 0.0
                except:
                    reward = 0.0

            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(reward)

        avg_rewards = {
            action: sum(rewards) / len(rewards)
            for action, rewards in action_rewards.items()
        }

        # Identify most and least used actions
        most_used = max(action_counts, key=action_counts.get) if action_counts else ""
        least_used = min(action_counts, key=action_counts.get) if action_counts else ""

        # Identify best and worst performing actions
        best_performing = (
            max(avg_rewards, key=avg_rewards.get) if avg_rewards else ""
        )
        worst_performing = (
            min(avg_rewards, key=avg_rewards.get) if avg_rewards else ""
        )

        # Return analysis
        return {
            "action_counts": action_counts,
            "avg_rewards": avg_rewards,
            "most_used_action": most_used,
            "least_used_action": least_used,
            "best_performing_action": best_performing,
            "worst_performing_action": worst_performing,
        }
