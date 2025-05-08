"""
Reinforcement learning module for DataMCPServerAgent.
This module provides mechanisms for agents to learn from rewards and improve through experience.
"""

import random
import time
from typing import Any, Callable, Dict, List

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.memory.memory_persistence import MemoryDatabase


class RewardSystem:
    """System for calculating rewards based on agent performance and feedback."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the reward system.

        Args:
            db: Memory database for persistence
        """
        self.db = db

        # Define reward weights
        self.reward_weights = {
            "user_satisfaction": 0.5,  # Weight for user satisfaction
            "task_completion": 0.3,  # Weight for task completion
            "efficiency": 0.2,  # Weight for efficiency (time, resources)
        }

        # Initialize reward history
        self.reward_history = {}

    def calculate_reward(
        self,
        agent_name: str,
        feedback: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> float:
        """Calculate a reward based on feedback and performance metrics.

        Args:
            agent_name: Name of the agent
            feedback: User feedback and self-evaluation
            performance_metrics: Performance metrics

        Returns:
            Calculated reward value
        """
        # Calculate user satisfaction reward
        user_satisfaction = self._calculate_user_satisfaction(feedback)

        # Calculate task completion reward
        task_completion = self._calculate_task_completion(performance_metrics)

        # Calculate efficiency reward
        efficiency = self._calculate_efficiency(performance_metrics)

        # Calculate total reward
        total_reward = (
            self.reward_weights["user_satisfaction"] * user_satisfaction
            + self.reward_weights["task_completion"] * task_completion
            + self.reward_weights["efficiency"] * efficiency
        )

        # Store the reward in history
        if agent_name not in self.reward_history:
            self.reward_history[agent_name] = []

        self.reward_history[agent_name].append(
            {
                "timestamp": time.time(),
                "reward": total_reward,
                "components": {
                    "user_satisfaction": user_satisfaction,
                    "task_completion": task_completion,
                    "efficiency": efficiency,
                },
            }
        )

        # Store the reward in the database
        self.db.save_agent_reward(
            agent_name,
            total_reward,
            {
                "user_satisfaction": user_satisfaction,
                "task_completion": task_completion,
                "efficiency": efficiency,
            },
        )

        return total_reward

    def _calculate_user_satisfaction(self, feedback: Dict[str, Any]) -> float:
        """Calculate user satisfaction reward component.

        Args:
            feedback: User feedback and self-evaluation

        Returns:
            User satisfaction reward component
        """
        # Extract user feedback
        user_feedback = feedback.get("user_feedback", {})

        # Check for explicit ratings
        if "rating" in user_feedback:
            # Normalize rating to [0, 1]
            return user_feedback["rating"] / 5.0

        # Check for sentiment in feedback text
        feedback_text = user_feedback.get("feedback", "")

        # Simple sentiment analysis
        positive_words = [
            "good",
            "great",
            "excellent",
            "helpful",
            "thanks",
            "thank",
            "perfect",
            "awesome",
        ]
        negative_words = [
            "bad",
            "poor",
            "unhelpful",
            "wrong",
            "incorrect",
            "error",
            "mistake",
            "not",
        ]

        positive_count = sum(
            1 for word in positive_words if word in feedback_text.lower()
        )
        negative_count = sum(
            1 for word in negative_words if word in feedback_text.lower()
        )

        # Calculate sentiment score
        if positive_count + negative_count > 0:
            sentiment = positive_count / (positive_count + negative_count)
        else:
            sentiment = 0.5  # Neutral

        return sentiment

    def _calculate_task_completion(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate task completion reward component.

        Args:
            performance_metrics: Performance metrics

        Returns:
            Task completion reward component
        """
        # Check for success rate
        if "success_rate" in performance_metrics:
            return performance_metrics["success_rate"]

        # Check for task completion flag
        if "task_completed" in performance_metrics:
            return 1.0 if performance_metrics["task_completed"] else 0.0

        # Default to neutral
        return 0.5

    def _calculate_efficiency(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate efficiency reward component.

        Args:
            performance_metrics: Performance metrics

        Returns:
            Efficiency reward component
        """
        # Check for response time
        if "response_time" in performance_metrics:
            # Normalize response time (assuming 10 seconds is optimal)
            response_time = performance_metrics["response_time"]
            if response_time <= 10:
                return 1.0
            else:
                return max(0.0, 1.0 - (response_time - 10) / 50)

        # Check for tool usage efficiency
        if "tool_usage" in performance_metrics:
            tool_usage = performance_metrics["tool_usage"]
            # Normalize tool usage (assuming 5 tool calls is optimal)
            if tool_usage <= 5:
                return 1.0
            else:
                return max(0.0, 1.0 - (tool_usage - 5) / 15)

        # Default to neutral
        return 0.5

    def get_agent_rewards(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent rewards for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to return

        Returns:
            List of recent rewards
        """
        if agent_name in self.reward_history:
            return sorted(
                self.reward_history[agent_name],
                key=lambda x: x["timestamp"],
                reverse=True,
            )[:limit]

        return []


class QLearningAgent:
    """Agent that learns using Q-learning algorithm."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_extractor: Callable[[Dict[str, Any]], str],
        actions: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
    ):
        """Initialize the Q-learning agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Reward system
            state_extractor: Function to extract state from context
            actions: List of possible actions
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
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize Q-table
        self.q_table = self.db.get_q_table(name) or {}

    def select_action(self, state: str) -> str:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Exploitation: best action from Q-table
        return self._get_best_action(state)

    def _get_best_action(self, state: str) -> str:
        """Get the best action for a state from the Q-table.

        Args:
            state: Current state

        Returns:
            Best action
        """
        # If state not in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Get action with highest Q-value
        state_actions = self.q_table[state]
        return max(state_actions, key=state_actions.get)

    def update_q_value(
        self, state: str, action: str, reward: float, next_state: str
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # If state not in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # If next_state not in Q-table, initialize it
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        # Get current Q-value
        current_q = self.q_table[state].get(action, 0.0)

        # Get max Q-value for next state
        max_next_q = max(self.q_table[next_state].values())

        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-table
        self.q_table[state][action] = new_q

        # Save Q-table to database
        self.db.save_q_table(self.name, self.q_table)


class PolicyGradientAgent:
    """Agent that learns using policy gradient algorithm."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_extractor: Callable[[Dict[str, Any]], List[float]],
        actions: List[str],
        learning_rate: float = 0.01,
    ):
        """Initialize the policy gradient agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Reward system
            state_extractor: Function to extract state features from context
            actions: List of possible actions
            learning_rate: Learning rate
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_extractor = state_extractor
        self.actions = actions
        self.learning_rate = learning_rate

        # Initialize policy parameters
        self.policy_params = (
            self.db.get_policy_params(name) or self._initialize_policy_params()
        )

        # Initialize episode history
        self.episode_history = []

    def _initialize_policy_params(self) -> Dict[str, List[float]]:
        """Initialize policy parameters.

        Returns:
            Policy parameters
        """
        # Initialize with small random values
        return {
            action: [
                random.uniform(-0.1, 0.1) for _ in range(10)
            ]  # Assuming 10 state features
            for action in self.actions
        }

    def select_action(self, state_features: List[float]) -> str:
        """Select an action using the current policy.

        Args:
            state_features: Features representing the current state

        Returns:
            Selected action
        """
        # Calculate action probabilities
        action_probs = self._calculate_action_probabilities(state_features)

        # Sample action based on probabilities
        return self._sample_action(action_probs)

    def _calculate_action_probabilities(
        self, state_features: List[float]
    ) -> Dict[str, float]:
        """Calculate action probabilities using softmax.

        Args:
            state_features: Features representing the current state

        Returns:
            Dictionary of action probabilities
        """
        # Calculate action values
        action_values = {}
        for action in self.actions:
            # Dot product of state features and policy parameters
            value = sum(
                f * p for f, p in zip(state_features, self.policy_params[action])
            )
            action_values[action] = value

        # Apply softmax
        max_value = max(action_values.values())
        exp_values = {a: np.exp(v - max_value) for a, v in action_values.items()}
        total_exp = sum(exp_values.values())

        return {a: v / total_exp for a, v in exp_values.items()}

    def _sample_action(self, action_probs: Dict[str, float]) -> str:
        """Sample an action based on probabilities.

        Args:
            action_probs: Dictionary of action probabilities

        Returns:
            Sampled action
        """
        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        return np.random.choice(actions, p=probs)

    def record_step(
        self, state_features: List[float], action: str, reward: float
    ) -> None:
        """Record a step in the episode history.

        Args:
            state_features: Features representing the state
            action: Action taken
            reward: Reward received
        """
        self.episode_history.append(
            {"state_features": state_features, "action": action, "reward": reward}
        )

    def update_policy(self) -> None:
        """Update policy parameters using policy gradient."""
        if not self.episode_history:
            return

        # Calculate returns (sum of rewards from each step to the end)
        returns = []
        G = 0
        for step in reversed(self.episode_history):
            G = step["reward"] + G
            returns.insert(0, G)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)

        # Update policy parameters
        for i, step in enumerate(self.episode_history):
            state_features = step["state_features"]
            action = step["action"]
            G = returns[i]

            # Calculate action probabilities
            action_probs = self._calculate_action_probabilities(state_features)

            # Update parameters for the taken action
            for j, feature in enumerate(state_features):
                # Gradient ascent
                self.policy_params[action][j] += (
                    self.learning_rate * G * feature * (1 - action_probs[action])
                )

                # Update parameters for other actions
                for other_action in self.actions:
                    if other_action != action:
                        self.policy_params[other_action][j] -= (
                            self.learning_rate
                            * G
                            * feature
                            * action_probs[other_action]
                        )

        # Save policy parameters to database
        self.db.save_policy_params(self.name, self.policy_params)

        # Clear episode history
        self.episode_history = []


class RLCoordinatorAgent:
    """Coordinator agent that uses reinforcement learning for decision making."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        sub_agents: Dict[str, Any],
        rl_agent_type: str = "q_learning",
    ):
        """Initialize the RL coordinator agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Reward system
            sub_agents: Dictionary of sub-agents
            rl_agent_type: Type of RL agent to use ("q_learning" or "policy_gradient")
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.sub_agents = sub_agents

        # Define actions (which sub-agents to use)
        self.actions = list(sub_agents.keys())

        # Create RL agent based on type
        if rl_agent_type == "q_learning":
            self.rl_agent = QLearningAgent(
                name=f"{name}_q_learning",
                model=model,
                db=db,
                reward_system=reward_system,
                state_extractor=self._extract_state,
                actions=self.actions,
            )
        else:  # policy_gradient
            self.rl_agent = PolicyGradientAgent(
                name=f"{name}_policy_gradient",
                model=model,
                db=db,
                reward_system=reward_system,
                state_extractor=self._extract_state_features,
                actions=self.actions,
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

    async def _extract_state_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract state features from context.

        Args:
            context: Context dictionary

        Returns:
            List of state features
        """
        # This is a simplified version that would be expanded in a real implementation
        # In practice, you would extract meaningful numerical features from the context

        # Extract request and history from context
        request = context.get("request", "")

        # Simple feature extraction
        features = [
            len(request) / 100,  # Normalized request length
            request.count("?") / 5,  # Normalized question count
            request.lower().count("search") / 2,  # Search-related terms
            request.lower().count("scrape") / 2,  # Scrape-related terms
            request.lower().count("analyze") / 2,  # Analysis-related terms
            request.lower().count("compare") / 2,  # Comparison-related terms
            request.lower().count("find") / 2,  # Finding-related terms
            request.lower().count("get") / 2,  # Getting-related terms
            request.lower().count("how") / 2,  # How-related terms
            request.lower().count("why") / 2,  # Why-related terms
        ]

        return features

    async def process_request(
        self, request: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a user request using reinforcement learning for agent selection.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Create context for state extraction
        context = {"request": request, "history": history}

        # Extract state or state features based on RL agent type
        if isinstance(self.rl_agent, QLearningAgent):
            state = await self._extract_state(context)
            # Select sub-agent using RL
            selected_agent_name = self.rl_agent.select_action(state)
        else:  # PolicyGradientAgent
            state_features = await self._extract_state_features(context)
            # Select sub-agent using RL
            selected_agent_name = self.rl_agent.select_action(state_features)

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
        }

        # Calculate reward
        feedback = {
            "user_feedback": {},  # Will be updated later with actual user feedback
            "self_evaluation": result.get("self_evaluation", {}),
        }

        reward = self.reward_system.calculate_reward(
            selected_agent_name, feedback, performance_metrics
        )

        # Update RL agent
        if isinstance(self.rl_agent, QLearningAgent):
            # For Q-learning, we need the next state
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

            # Update Q-value
            self.rl_agent.update_q_value(state, selected_agent_name, reward, next_state)
        else:  # PolicyGradientAgent
            # For policy gradient, we record the step
            self.rl_agent.record_step(state_features, selected_agent_name, reward)

        # Return the result with additional RL information
        return {
            **result,
            "selected_agent": selected_agent_name,
            "reward": reward,
            "performance_metrics": performance_metrics,
        }

    async def update_from_feedback(
        self, request: str, response: str, feedback: str
    ) -> None:
        """Update the RL agent based on user feedback.

        Args:
            request: Original user request
            response: Agent response
            feedback: User feedback
        """
        # Create context for state extraction
        context = {
            "request": request,
            "history": [
                {"role": "user", "content": request},
                {"role": "assistant", "content": response},
            ],
        }

        # Extract state or state features based on RL agent type
        if isinstance(self.rl_agent, QLearningAgent):
            state = await self._extract_state(context)

            # We don't have the action that was taken, so we'll use the best action
            action = self.rl_agent._get_best_action(state)

            # Calculate reward from feedback
            feedback_data = {"user_feedback": {"feedback": feedback}}

            reward = self.reward_system.calculate_reward(
                self.name,
                feedback_data,
                {"success_rate": 1.0},  # Assume success since we got feedback
            )

            # Update Q-value (using same state as next state since feedback is terminal)
            self.rl_agent.update_q_value(state, action, reward, state)
        else:  # PolicyGradientAgent
            # For policy gradient, we update the policy
            self.rl_agent.update_policy()

    async def learn_from_batch(self, batch_size: int = 10) -> Dict[str, Any]:
        """Learn from a batch of past interactions.

        Args:
            batch_size: Number of past interactions to learn from

        Returns:
            Learning results
        """
        # Get past interactions from the database
        interactions = self.db.get_agent_interactions(self.name, limit=batch_size)

        if not interactions:
            return {"status": "No interactions found for learning"}

        # Process each interaction
        for interaction in interactions:
            request = interaction.get("request", "")
            response = interaction.get("response", "")
            feedback = interaction.get("feedback", "")

            # Update from this interaction
            await self.update_from_feedback(request, response, feedback)

        # For policy gradient, update policy after processing all interactions
        if isinstance(self.rl_agent, PolicyGradientAgent):
            self.rl_agent.update_policy()

        return {
            "status": "Learning completed",
            "interactions_processed": len(interactions),
        }


# Factory function to create RL-based agent architecture
async def create_rl_agent_architecture(
    model: ChatAnthropic,
    db: MemoryDatabase,
    sub_agents: Dict[str, Any],
    rl_agent_type: str = "q_learning",
) -> RLCoordinatorAgent:
    """Create a reinforcement learning-based agent architecture.

    Args:
        model: Language model to use
        db: Memory database for persistence
        sub_agents: Dictionary of sub-agents
        rl_agent_type: Type of RL agent to use ("q_learning" or "policy_gradient")

    Returns:
        RL coordinator agent
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Create RL coordinator agent
    rl_coordinator = RLCoordinatorAgent(
        name="rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        rl_agent_type=rl_agent_type,
    )

    return rl_coordinator
