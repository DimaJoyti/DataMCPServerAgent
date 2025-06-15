"""
Multi-agent reinforcement learning module for DataMCPServerAgent.
This module implements cooperative and competitive multi-agent RL algorithms.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import DQNNetwork


class CommunicationModule(nn.Module):
    """Communication module for multi-agent coordination."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        """Initialize communication module.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension (message size)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Message generation network
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # Message processing network
        self.process_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def generate_message(self, state: torch.Tensor) -> torch.Tensor:
        """Generate message from state.
        
        Args:
            state: Agent state
            
        Returns:
            Generated message
        """
        return self.message_net(state)

    def process_message(self, message: torch.Tensor) -> torch.Tensor:
        """Process received message.
        
        Args:
            message: Received message
            
        Returns:
            Processed message features
        """
        return self.process_net(message)


class MultiAgentDQN:
    """Multi-agent DQN with communication."""

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        communication: bool = True,
        message_dim: int = 32,
        learning_rate: float = 1e-4,
    ):
        """Initialize multi-agent DQN.
        
        Args:
            agent_id: Unique agent identifier
            state_dim: State space dimension
            action_dim: Action space dimension
            num_agents: Total number of agents
            communication: Whether to use communication
            message_dim: Message dimension
            learning_rate: Learning rate
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.communication = communication
        self.message_dim = message_dim

        # Adjust input dimension for communication
        input_dim = state_dim
        if communication:
            input_dim += message_dim * (num_agents - 1)  # Messages from other agents

        # Q-network
        self.q_network = DQNNetwork(input_dim, action_dim)
        self.target_network = DQNNetwork(input_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Communication module
        if communication:
            self.comm_module = CommunicationModule(
                state_dim, output_dim=message_dim
            )

        # Optimizer
        params = list(self.q_network.parameters())
        if communication:
            params.extend(list(self.comm_module.parameters()))
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 10000

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        if communication:
            self.comm_module.to(self.device)

    def generate_message(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Generate communication message.
        
        Args:
            state: Current state
            
        Returns:
            Generated message or None if no communication
        """
        if not self.communication:
            return None

        with torch.no_grad():
            message = self.comm_module.generate_message(state)
        return message

    def select_action(
        self,
        state: torch.Tensor,
        messages: Optional[List[torch.Tensor]] = None,
        epsilon: float = 0.1
    ) -> int:
        """Select action based on state and messages.
        
        Args:
            state: Current state
            messages: Messages from other agents
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        # Prepare input
        input_tensor = state
        if self.communication and messages:
            # Concatenate messages
            message_tensor = torch.cat(messages, dim=-1)
            input_tensor = torch.cat([state, message_tensor], dim=-1)

        with torch.no_grad():
            q_values = self.q_network(input_tensor.unsqueeze(0))
            action = q_values.argmax().item()

        return action

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        messages: Optional[List[torch.Tensor]] = None,
        next_messages: Optional[List[torch.Tensor]] = None
    ):
        """Store experience in buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            messages: Messages received
            next_messages: Next messages received
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "messages": messages,
            "next_messages": next_messages,
        }

        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)

        self.experience_buffer.append(experience)

    def train(self, batch_size: int = 32) -> Dict[str, float]:
        """Train the agent.
        
        Args:
            batch_size: Training batch size
            
        Returns:
            Training metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {}

        # Sample batch
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Prepare batch tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for exp in batch:
            # Prepare state input
            state_input = exp["state"]
            next_state_input = exp["next_state"]

            if self.communication and exp["messages"]:
                message_tensor = torch.cat(exp["messages"], dim=-1)
                state_input = torch.cat([state_input, message_tensor], dim=-1)

            if self.communication and exp["next_messages"]:
                next_message_tensor = torch.cat(exp["next_messages"], dim=-1)
                next_state_input = torch.cat([next_state_input, next_message_tensor], dim=-1)

            states.append(state_input)
            actions.append(exp["action"])
            rewards.append(exp["reward"])
            next_states.append(next_state_input)
            dones.append(exp["done"])

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class MultiAgentCoordinator:
    """Coordinator for multi-agent RL system."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        cooperation_mode: str = "cooperative",
        communication: bool = True,
    ):
        """Initialize multi-agent coordinator.
        
        Args:
            name: Coordinator name
            model: Language model
            db: Memory database
            reward_system: Reward system
            num_agents: Number of agents
            state_dim: State space dimension
            action_dim: Action space dimension
            cooperation_mode: "cooperative", "competitive", or "mixed"
            communication: Whether to enable communication
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cooperation_mode = cooperation_mode
        self.communication = communication

        # Create agents
        self.agents = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = MultiAgentDQN(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                communication=communication,
            )

        # Global state and metrics
        self.global_state = {}
        self.episode_rewards = {agent_id: [] for agent_id in self.agents.keys()}
        self.cooperation_metrics = []

    async def process_multi_agent_request(
        self,
        request: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process request using multi-agent system.
        
        Args:
            request: User request
            history: Conversation history
            
        Returns:
            Multi-agent processing result
        """
        # Extract global state
        global_state = await self._extract_global_state(request, history)

        # Generate messages if communication is enabled
        messages = {}
        if self.communication:
            for agent_id, agent in self.agents.items():
                state_tensor = torch.FloatTensor(global_state[agent_id])
                message = agent.generate_message(state_tensor)
                if message is not None:
                    messages[agent_id] = message

        # Select actions for all agents
        actions = {}
        for agent_id, agent in self.agents.items():
            state_tensor = torch.FloatTensor(global_state[agent_id])

            # Collect messages from other agents
            other_messages = []
            if self.communication:
                for other_id, message in messages.items():
                    if other_id != agent_id:
                        other_messages.append(message)

            action = agent.select_action(
                state_tensor,
                other_messages if other_messages else None
            )
            actions[agent_id] = action

        # Execute actions and compute rewards
        results = await self._execute_multi_agent_actions(
            actions, request, history
        )

        # Compute individual and team rewards
        rewards = self._compute_multi_agent_rewards(results, actions)

        # Store experiences
        for agent_id, agent in self.agents.items():
            state_tensor = torch.FloatTensor(global_state[agent_id])
            action = actions[agent_id]
            reward = rewards[agent_id]

            # For simplicity, use same state as next state
            # In real implementation, you'd compute actual next state
            next_state_tensor = state_tensor

            other_messages = []
            if self.communication:
                for other_id, message in messages.items():
                    if other_id != agent_id:
                        other_messages.append(message)

            agent.store_experience(
                state_tensor, action, reward, next_state_tensor, False,
                other_messages if other_messages else None,
                other_messages if other_messages else None  # Same for next
            )

        # Train agents
        training_metrics = {}
        for agent_id, agent in self.agents.items():
            metrics = agent.train()
            if metrics:
                training_metrics[agent_id] = metrics

        # Compute cooperation metrics
        cooperation_score = self._compute_cooperation_score(actions, rewards)
        self.cooperation_metrics.append(cooperation_score)

        return {
            "success": True,
            "response": self._format_multi_agent_response(results),
            "actions": actions,
            "rewards": rewards,
            "cooperation_score": cooperation_score,
            "training_metrics": training_metrics,
        }

    async def _extract_global_state(
        self,
        request: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract global state for all agents.
        
        Args:
            request: User request
            history: Conversation history
            
        Returns:
            Dictionary mapping agent IDs to state vectors
        """
        # Simple state extraction - can be enhanced
        base_features = []

        # Text features
        base_features.append(len(request) / 1000.0)
        base_features.append(len(request.split()) / 100.0)

        # History features
        base_features.append(len(history) / 10.0)

        # Pad to state dimension
        while len(base_features) < self.state_dim:
            base_features.append(0.0)

        base_features = base_features[:self.state_dim]

        # Create agent-specific states
        global_state = {}
        for i, agent_id in enumerate(self.agents.keys()):
            # Add agent-specific features
            agent_features = base_features.copy()
            agent_features[0] += i * 0.1  # Agent ID encoding
            global_state[agent_id] = agent_features

        return global_state

    async def _execute_multi_agent_actions(
        self,
        actions: Dict[str, int],
        request: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute actions for all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            request: User request
            history: Conversation history
            
        Returns:
            Execution results
        """
        # Simple action execution - can be enhanced
        results = {}

        for agent_id, action in actions.items():
            # Map action to behavior
            if action == 0:
                behavior = "search"
            elif action == 1:
                behavior = "analyze"
            elif action == 2:
                behavior = "create"
            elif action == 3:
                behavior = "communicate"
            else:
                behavior = "wait"

            results[agent_id] = {
                "behavior": behavior,
                "success": np.random.choice([True, False], p=[0.8, 0.2]),
                "output": f"Agent {agent_id} performed {behavior}",
            }

        return results

    def _compute_multi_agent_rewards(
        self,
        results: Dict[str, Any],
        actions: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute rewards for all agents.
        
        Args:
            results: Execution results
            actions: Actions taken
            
        Returns:
            Dictionary mapping agent IDs to rewards
        """
        rewards = {}

        # Individual rewards
        for agent_id, result in results.items():
            individual_reward = 1.0 if result["success"] else -0.5
            rewards[agent_id] = individual_reward

        # Team reward component
        if self.cooperation_mode == "cooperative":
            team_success_rate = sum(
                1 for result in results.values() if result["success"]
            ) / len(results)
            team_bonus = team_success_rate * 0.5

            for agent_id in rewards:
                rewards[agent_id] += team_bonus

        elif self.cooperation_mode == "competitive":
            # Competitive rewards - zero-sum
            total_success = sum(
                1 for result in results.values() if result["success"]
            )
            if total_success > 0:
                for agent_id, result in results.items():
                    if result["success"]:
                        rewards[agent_id] += 1.0 / total_success
                    else:
                        rewards[agent_id] -= 0.1

        return rewards

    def _compute_cooperation_score(
        self,
        actions: Dict[str, int],
        rewards: Dict[str, float]
    ) -> float:
        """Compute cooperation score.
        
        Args:
            actions: Actions taken by agents
            rewards: Rewards received by agents
            
        Returns:
            Cooperation score
        """
        # Simple cooperation metric based on action diversity and reward correlation
        action_diversity = len(set(actions.values())) / len(actions)
        reward_variance = np.var(list(rewards.values()))

        # Higher diversity and lower variance indicate better cooperation
        cooperation_score = action_diversity * (1.0 / (1.0 + reward_variance))

        return cooperation_score

    def _format_multi_agent_response(self, results: Dict[str, Any]) -> str:
        """Format multi-agent response.
        
        Args:
            results: Execution results
            
        Returns:
            Formatted response string
        """
        response_parts = []

        for agent_id, result in results.items():
            status = "✅" if result["success"] else "❌"
            response_parts.append(
                f"{status} {agent_id}: {result['behavior']} - {result['output']}"
            )

        return "\n".join(response_parts)

    def get_cooperation_metrics(self) -> Dict[str, float]:
        """Get cooperation metrics.
        
        Returns:
            Dictionary of cooperation metrics
        """
        if not self.cooperation_metrics:
            return {}

        return {
            "avg_cooperation": np.mean(self.cooperation_metrics),
            "cooperation_trend": np.mean(self.cooperation_metrics[-10:]) - np.mean(self.cooperation_metrics[:10]) if len(self.cooperation_metrics) >= 20 else 0.0,
            "cooperation_stability": 1.0 / (1.0 + np.var(self.cooperation_metrics)),
        }

    def update_target_networks(self):
        """Update target networks for all agents."""
        for agent in self.agents.values():
            agent.update_target_network()


# Factory function to create multi-agent RL architecture
async def create_multi_agent_rl_architecture(
    model: ChatAnthropic,
    db: MemoryDatabase,
    num_agents: int = 3,
    state_dim: int = 128,
    action_dim: int = 5,
    cooperation_mode: str = "cooperative",
    communication: bool = True,
) -> MultiAgentCoordinator:
    """Create a multi-agent RL architecture.
    
    Args:
        model: Language model to use
        db: Memory database for persistence
        num_agents: Number of agents
        state_dim: State space dimension
        action_dim: Action space dimension
        cooperation_mode: Cooperation mode
        communication: Whether to enable communication
        
    Returns:
        Multi-agent coordinator
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Create multi-agent coordinator
    coordinator = MultiAgentCoordinator(
        name="multi_agent_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        cooperation_mode=cooperation_mode,
        communication=communication,
    )

    return coordinator
