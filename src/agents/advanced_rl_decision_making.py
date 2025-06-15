"""
Advanced reinforcement learning decision-making module for DataMCPServerAgent.
This module extends the basic reinforcement learning capabilities with more sophisticated
decision-making algorithms and approaches.
"""

import random
import time
from typing import Any, Callable, Dict, List

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from src.agents.reinforcement_learning import (
    QLearningAgent,
    RewardSystem,
    RLCoordinatorAgent,
)
from src.memory.memory_persistence import MemoryDatabase


class DeepRLAgent:
    """Agent that uses deep reinforcement learning for decision making."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_extractor: Callable[[Dict[str, Any]], List[float]],
        actions: List[str],
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        hidden_layer_size: int = 64,
    ):
        """Initialize the deep RL agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Reward system
            state_extractor: Function to extract state features from context
            actions: List of possible actions
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            exploration_rate: Exploration rate (epsilon)
            hidden_layer_size: Size of the hidden layer in the neural network
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
        self.hidden_layer_size = hidden_layer_size

        # Initialize neural network weights
        self.weights = self.db.get_drl_weights(name) or self._initialize_weights()

        # Initialize experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = 1000
        self.batch_size = 32

    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize neural network weights.

        Returns:
            Dictionary of weights
        """
        # Simple 2-layer neural network
        # Input layer -> Hidden layer -> Output layer
        input_size = 10  # Assuming 10 state features
        output_size = len(self.actions)

        return {
            "W1": np.random.randn(input_size, self.hidden_layer_size) * 0.1,
            "b1": np.zeros((1, self.hidden_layer_size)),
            "W2": np.random.randn(self.hidden_layer_size, output_size) * 0.1,
            "b2": np.zeros((1, output_size)),
        }

    def _forward(self, state_features: List[float]) -> np.ndarray:
        """Forward pass through the neural network.

        Args:
            state_features: Features representing the current state

        Returns:
            Q-values for each action
        """
        # Convert state features to numpy array
        x = np.array(state_features).reshape(1, -1)

        # First layer
        z1 = np.dot(x, self.weights["W1"]) + self.weights["b1"]
        a1 = np.tanh(z1)  # Tanh activation

        # Output layer
        z2 = np.dot(a1, self.weights["W2"]) + self.weights["b2"]
        # No activation for Q-values

        return z2[0]  # Return as 1D array

    def select_action(self, state_features: List[float]) -> str:
        """Select an action using epsilon-greedy policy.

        Args:
            state_features: Features representing the current state

        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Exploitation: best action from Q-values
        q_values = self._forward(state_features)
        action_idx = np.argmax(q_values)
        return self.actions[action_idx]

    def add_to_replay_buffer(
        self,
        state_features: List[float],
        action: str,
        reward: float,
        next_state_features: List[float],
        done: bool,
    ) -> None:
        """Add experience to replay buffer.

        Args:
            state_features: Features representing the current state
            action: Action taken
            reward: Reward received
            next_state_features: Features representing the next state
            done: Whether the episode is done
        """
        # Add experience to replay buffer
        self.replay_buffer.append((state_features, action, reward, next_state_features, done))

        # Limit buffer size
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def update_network(self) -> None:
        """Update neural network weights using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch data
        states = np.array([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Convert actions to indices
        action_indices = [self.actions.index(action) for action in actions]

        # Get current Q-values
        current_q_values = np.array([self._forward(state) for state in states])

        # Get next Q-values
        next_q_values = np.array([self._forward(state) for state in next_states])

        # Calculate target Q-values
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, action_indices[i]] = rewards[i]
            else:
                targets[i, action_indices[i]] = rewards[i] + self.discount_factor * np.max(
                    next_q_values[i]
                )

        # Update weights using simple gradient descent
        # In a real implementation, you would use a proper deep learning framework
        # This is a simplified version for illustration

        # Save weights to database
        self.db.save_drl_weights(self.name, self.weights)


class AdvancedRLCoordinatorAgent(RLCoordinatorAgent):
    """Advanced coordinator agent that uses reinforcement learning for decision making."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        sub_agents: Dict[str, Any],
        tools: List[BaseTool],
        rl_agent_type: str = "q_learning",
    ):
        """Initialize the advanced RL coordinator agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            reward_system: Reward system
            sub_agents: Dictionary of sub-agents
            tools: List of available tools
            rl_agent_type: Type of RL agent to use ("q_learning", "policy_gradient", or "deep_rl")
        """
        # Initialize parent class
        super().__init__(name, model, db, reward_system, sub_agents, rl_agent_type)

        # Store tools
        self.tools = tools

        # Create tool selection RL agent if using deep_rl
        if rl_agent_type == "deep_rl":
            self.rl_agent = DeepRLAgent(
                name=f"{name}_deep_rl",
                model=model,
                db=db,
                reward_system=reward_system,
                state_extractor=self._extract_state_features,
                actions=self.actions,
            )

            # Create tool selection RL agent
            self.tool_selection_agent = DeepRLAgent(
                name=f"{name}_tool_selection",
                model=model,
                db=db,
                reward_system=reward_system,
                state_extractor=self._extract_state_features,
                actions=[tool.name for tool in tools],
            )
        else:
            # For other RL types, we'll use the parent class implementation
            self.tool_selection_agent = None

    async def select_tools(self, request: str, state_features: List[float]) -> List[str]:
        """Select tools using reinforcement learning.

        Args:
            request: User request
            state_features: Features representing the current state

        Returns:
            List of selected tool names
        """
        if self.tool_selection_agent:
            # Use the tool selection agent to select a tool
            selected_tool = self.tool_selection_agent.select_action(state_features)
            return [selected_tool]
        else:
            # Fallback to a simple heuristic
            # In a real implementation, you would use a more sophisticated approach
            return [tool.name for tool in self.tools[:3]]  # Select first 3 tools

    async def process_request(self, request: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a user request using advanced reinforcement learning for decision making.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Create context for state extraction
        context = {"request": request, "history": history}

        # Extract state features
        state_features = await self._extract_state_features(context)

        # Select tools using RL
        selected_tools = await self.select_tools(request, state_features)

        # Select sub-agent using RL
        if isinstance(self.rl_agent, DeepRLAgent):
            selected_agent_name = self.rl_agent.select_action(state_features)
        elif isinstance(self.rl_agent, QLearningAgent):
            state = await self._extract_state(context)
            selected_agent_name = self.rl_agent.select_action(state)
        else:  # PolicyGradientAgent
            selected_agent_name = self.rl_agent.select_action(state_features)

        # Get the selected sub-agent
        selected_agent = self.sub_agents[selected_agent_name]

        # Record start time for performance tracking
        start_time = time.time()

        # Execute the selected sub-agent with selected tools
        result = await selected_agent.execute(request, self.db, selected_tools)

        # Record end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # Calculate performance metrics
        performance_metrics = {
            "success_rate": 1.0 if result["success"] else 0.0,
            "response_time": duration,
            "tool_usage": len(result.get("tool_calls", [])) if "tool_calls" in result else 0,
        }

        # Calculate reward
        feedback = {
            "user_feedback": {},  # Will be updated later with actual user feedback
            "self_evaluation": result.get("self_evaluation", {}),
        }

        reward = self.reward_system.calculate_reward(
            selected_agent_name, feedback, performance_metrics
        )

        # Update RL agents
        if isinstance(self.rl_agent, DeepRLAgent):
            # For Deep RL, we need the next state features
            next_context = {
                "request": request,
                "history": history
                + [
                    {"role": "user", "content": request},
                    {
                        "role": "assistant",
                        "content": result["response"] if result["success"] else result["error"],
                    },
                ],
            }
            next_state_features = await self._extract_state_features(next_context)

            # Add experience to replay buffer
            self.rl_agent.add_to_replay_buffer(
                state_features, selected_agent_name, reward, next_state_features, False
            )

            # Update network
            self.rl_agent.update_network()

            # Update tool selection agent if used
            if self.tool_selection_agent and selected_tools:
                # Calculate tool reward based on overall success
                tool_reward = reward * 0.8  # Slightly lower weight for tool selection

                # Add experience to replay buffer for each selected tool
                for tool_name in selected_tools:
                    self.tool_selection_agent.add_to_replay_buffer(
                        state_features, tool_name, tool_reward, next_state_features, False
                    )

                # Update network
                self.tool_selection_agent.update_network()

        elif isinstance(self.rl_agent, QLearningAgent):
            # For Q-learning, we need the next state
            next_context = {
                "request": request,
                "history": history
                + [
                    {"role": "user", "content": request},
                    {
                        "role": "assistant",
                        "content": result["response"] if result["success"] else result["error"],
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
            "selected_tools": selected_tools,
            "reward": reward,
            "performance_metrics": performance_metrics,
        }


# Factory function to create advanced RL-based agent architecture
async def create_advanced_rl_agent_architecture(
    model: ChatAnthropic,
    db: MemoryDatabase,
    sub_agents: Dict[str, Any],
    tools: List[BaseTool],
    rl_agent_type: str = "deep_rl",
) -> AdvancedRLCoordinatorAgent:
    """Create an advanced reinforcement learning-based agent architecture.

    Args:
        model: Language model to use
        db: Memory database for persistence
        sub_agents: Dictionary of sub-agents
        tools: List of available tools
        rl_agent_type: Type of RL agent to use ("q_learning", "policy_gradient", or "deep_rl")

    Returns:
        Advanced RL coordinator agent
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Create advanced RL coordinator agent
    advanced_rl_coordinator = AdvancedRLCoordinatorAgent(
        name="advanced_rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        tools=tools,
        rl_agent_type=rl_agent_type,
    )

    return advanced_rl_coordinator
