"""
Advanced reinforcement learning techniques for DataMCPServerAgent.
This module implements advanced RL techniques like Rainbow DQN, distributional RL, and more.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import NoisyLinear


class RainbowDQNNetwork(nn.Module):
    """Rainbow DQN network combining multiple improvements."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 512],
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
        dueling: bool = True,
    ):
        """Initialize Rainbow DQN network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
            noisy: Whether to use noisy networks
            dueling: Whether to use dueling architecture
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.noisy = noisy
        self.dueling = dueling

        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Feature layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            if noisy:
                layers.append(NoisyLinear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture with distributional outputs
            if noisy:
                self.value_head = NoisyLinear(input_dim, num_atoms)
                self.advantage_head = NoisyLinear(input_dim, action_dim * num_atoms)
            else:
                self.value_head = nn.Linear(input_dim, num_atoms)
                self.advantage_head = nn.Linear(input_dim, action_dim * num_atoms)
        else:
            # Standard distributional DQN
            if noisy:
                self.q_head = NoisyLinear(input_dim, action_dim * num_atoms)
            else:
                self.q_head = nn.Linear(input_dim, action_dim * num_atoms)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action-value distributions.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action-value distributions
        """
        batch_size = state.size(0)
        features = self.feature_layers(state)

        if self.dueling:
            # Dueling distributional architecture
            value_dist = self.value_head(features)  # (batch, num_atoms)
            advantage_dist = self.advantage_head(features)  # (batch, action_dim * num_atoms)

            # Reshape advantage
            advantage_dist = advantage_dist.view(batch_size, self.action_dim, self.num_atoms)

            # Dueling formula for distributions
            value_dist = value_dist.unsqueeze(1).expand_as(advantage_dist)
            advantage_mean = advantage_dist.mean(dim=1, keepdim=True)

            q_dist = value_dist + advantage_dist - advantage_mean
        else:
            # Standard distributional DQN
            q_dist = self.q_head(features)
            q_dist = q_dist.view(batch_size, self.action_dim, self.num_atoms)

        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_dist, dim=-1)

        return q_dist

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values by computing expected values of distributions.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        q_dist = self.forward(state)
        support = self.support.to(q_dist.device)
        q_values = (q_dist * support).sum(dim=-1)
        return q_values

    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for layer in self.modules():
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()


class RainbowDQNAgent:
    """Rainbow DQN agent with all improvements."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        target_update_freq: int = 8000,
        batch_size: int = 32,
        buffer_size: int = 1000000,
        multi_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        alpha: float = 0.5,  # Prioritized replay
        beta: float = 0.4,   # Importance sampling
    ):
        """Initialize Rainbow DQN agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            target_update_freq: Target network update frequency
            batch_size: Training batch size
            buffer_size: Experience replay buffer size
            multi_step: Number of steps for multi-step learning
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
            alpha: Prioritized replay exponent
            beta: Importance sampling exponent
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.multi_step = multi_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta

        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Neural networks
        self.q_network = RainbowDQNNetwork(
            state_dim, action_dim, num_atoms=num_atoms,
            v_min=v_min, v_max=v_max, noisy=True, dueling=True
        )
        self.target_network = RainbowDQNNetwork(
            state_dim, action_dim, num_atoms=num_atoms,
            v_min=v_min, v_max=v_max, noisy=True, dueling=True
        )

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Prioritized experience replay
        from src.agents.modern_deep_rl import ExperienceReplay
        self.replay_buffer = ExperienceReplay(buffer_size, prioritized=True)

        # Multi-step learning
        self.multi_step_buffer = []

        # Training counters
        self.steps = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.support = self.support.to(self.device)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using noisy networks (no epsilon-greedy needed).
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if training:
                self.q_network.reset_noise()

            q_values = self.q_network.get_q_values(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience with multi-step learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add to multi-step buffer
        self.multi_step_buffer.append((state, action, reward, next_state, done))

        # If buffer is full or episode is done, compute multi-step return
        if len(self.multi_step_buffer) >= self.multi_step or done:
            # Compute multi-step return
            multi_step_reward = 0
            for i, (_, _, r, _, _) in enumerate(self.multi_step_buffer):
                multi_step_reward += (self.gamma ** i) * r

            # Get first state and action, last next_state and done
            first_state, first_action = self.multi_step_buffer[0][:2]
            last_next_state, last_done = self.multi_step_buffer[-1][3:]

            # Store in replay buffer
            self.replay_buffer.push(
                first_state, first_action, multi_step_reward,
                last_next_state, last_done
            )

            # Remove first element for sliding window
            if not done:
                self.multi_step_buffer.pop(0)
            else:
                self.multi_step_buffer.clear()

    def train(self) -> Dict[str, float]:
        """Train the Rainbow DQN agent.
        
        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch with prioritized replay
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Current distributions
        current_dist = self.q_network(states)
        current_dist = current_dist[range(self.batch_size), actions]

        # Target distributions
        with torch.no_grad():
            # Double DQN: use main network to select actions
            next_q_values = self.q_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)

            # Use target network to evaluate
            target_dist = self.target_network(next_states)
            target_dist = target_dist[range(self.batch_size), next_actions]

            # Compute target support
            target_support = rewards.unsqueeze(1) + (self.gamma ** self.multi_step) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Distribute probability
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Fix disappearing probability mass
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability mass
            projected_dist = torch.zeros_like(target_dist)
            for i in range(self.batch_size):
                for j in range(self.num_atoms):
                    projected_dist[i, l[i, j]] += target_dist[i, j] * (u[i, j] - b[i, j])
                    projected_dist[i, u[i, j]] += target_dist[i, j] * (b[i, j] - l[i, j])

        # Compute loss (cross-entropy)
        loss = -(projected_dist * current_dist.log()).sum(1)

        # Apply importance sampling weights
        loss = (weights * loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        td_errors = (projected_dist * (self.support.unsqueeze(0) - (current_dist * self.support.unsqueeze(0)).sum(1, keepdim=True))).sum(1)
        priorities = td_errors.abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": self.q_network.get_q_values(states).mean().item(),
        }
