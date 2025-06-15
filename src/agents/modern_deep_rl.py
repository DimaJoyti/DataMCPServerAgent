"""
Modern deep reinforcement learning algorithms for DataMCPServerAgent.
This module implements state-of-the-art deep RL algorithms including DQN, PPO, A2C, and SAC.
"""

import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import ActorCriticNetwork, AttentionStateEncoder, DQNNetwork


class ExperienceReplay:
    """Experience replay buffer for deep RL algorithms."""

    def __init__(self, capacity: int = 100000, prioritized: bool = False):
        """Initialize experience replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            prioritized: Whether to use prioritized experience replay
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)

        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.alpha = 0.6  # Prioritization exponent
            self.beta = 0.4   # Importance sampling exponent
            self.beta_increment = 0.001
            self.epsilon = 1e-6  # Small constant to avoid zero priorities

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority for prioritized replay
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

        if self.prioritized:
            if priority is None:
                priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Batch of experiences as tensors
        """
        if self.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)

    def _sample_uniform(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample uniformly from buffer."""
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        return states, actions, rewards, next_states, dones

    def _sample_prioritized(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample with prioritized experience replay."""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        weights = torch.FloatTensor(weights)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent with modern improvements."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update_freq: int = 1000,
        batch_size: int = 32,
        buffer_size: int = 100000,
        double_dqn: bool = True,
        dueling: bool = True,
        noisy: bool = False,
        prioritized_replay: bool = True,
    ):
        """Initialize DQN agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
            target_update_freq: Target network update frequency
            batch_size: Training batch size
            buffer_size: Experience replay buffer size
            double_dqn: Whether to use Double DQN
            dueling: Whether to use Dueling DQN
            noisy: Whether to use Noisy Networks
            prioritized_replay: Whether to use prioritized experience replay
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.noisy = noisy

        # Neural networks
        self.q_network = DQNNetwork(
            state_dim, action_dim, dueling=dueling, noisy=noisy
        )
        self.target_network = DQNNetwork(
            state_dim, action_dim, dueling=dueling, noisy=noisy
        )

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.replay_buffer = ExperienceReplay(buffer_size, prioritized_replay)

        # Training counters
        self.steps = 0
        self.episodes = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and not self.noisy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if self.noisy:
                self.q_network.reset_noise()

            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> Dict[str, float]:
        """Train the DQN agent.
        
        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        if self.replay_buffer.prioritized:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))

        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        if self.replay_buffer.prioritized and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q_values.mean().item(),
        }

    def save_model(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)

    def load_model(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        continuous: bool = False,
    ):
        """Initialize PPO agent.

        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: Number of PPO epochs per update
            batch_size: Training batch size
            continuous: Whether action space is continuous
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.continuous = continuous

        # Neural network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, continuous=continuous
        )

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Storage for rollouts
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action_and_value(state_tensor)

            return action.item(), log_prob.item(), value.item()

    def store_experience(self, state: np.ndarray, action: int, log_prob: float,
                        reward: float, value: float, done: bool):
        """Store experience for training.

        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            value: Value estimate
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation.

        Args:
            next_value: Value of next state (for bootstrapping)

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_est = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_est = self.values[i + 1]

            delta = self.rewards[i] + self.gamma * next_value_est * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])

        return advantages, returns

    def train(self, next_value: float = 0.0) -> Dict[str, float]:
        """Train the PPO agent.

        Args:
            next_value: Value of next state

        Returns:
            Training metrics
        """
        if len(self.states) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            actor_output, values = self.network(states)

            if self.continuous:
                # Continuous actions
                mean, log_std = torch.chunk(actor_output, 2, dim=-1)
                std = torch.exp(log_std.clamp(-20, 2))
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions.float()).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                # Discrete actions
                dist = torch.distributions.Categorical(logits=actor_output)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            # Compute ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # Policy loss
            policy_loss_batch = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss_batch = F.mse_loss(values.squeeze(), returns)

            # Entropy loss
            entropy_loss_batch = -entropy

            # Total loss
            loss = (policy_loss_batch +
                   self.value_coef * value_loss_batch +
                   self.entropy_coef * entropy_loss_batch)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            policy_loss += policy_loss_batch.item()
            value_loss += value_loss_batch.item()
            entropy_loss += entropy_loss_batch.item()

        # Clear storage
        self.clear_storage()

        return {
            "total_loss": total_loss / self.ppo_epochs,
            "policy_loss": policy_loss / self.ppo_epochs,
            "value_loss": value_loss / self.ppo_epochs,
            "entropy_loss": entropy_loss / self.ppo_epochs,
        }

    def clear_storage(self):
        """Clear experience storage."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class A2CAgent:
    """Advantage Actor-Critic agent."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
    ):
        """Initialize A2C agent.

        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            continuous: Whether action space is continuous
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous

        # Neural network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, continuous=continuous
        )

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action_and_value(state_tensor)

            return action.item(), log_prob.item(), value.item()

    def train(self, states: List[np.ndarray], actions: List[int],
              rewards: List[float], next_value: float = 0.0) -> Dict[str, float]:
        """Train the A2C agent.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_value: Value of next state

        Returns:
            Training metrics
        """
        if len(states) == 0:
            return {}

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        # Compute returns
        returns = []
        R = next_value
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)

        # Get current policy outputs
        actor_output, values = self.network(states_tensor)
        values = values.squeeze()

        # Compute advantages
        advantages = returns - values

        if self.continuous:
            # Continuous actions
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            std = torch.exp(log_std.clamp(-20, 2))
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_tensor.float()).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            # Discrete actions
            dist = torch.distributions.Categorical(logits=actor_output)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy

        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }


class ModernDeepRLCoordinatorAgent:
    """Coordinator agent using modern deep RL algorithms."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        sub_agents: Dict[str, Any],
        tools: List[Any],
        rl_algorithm: str = "dqn",
        state_encoder: Optional[AttentionStateEncoder] = None,
        **kwargs
    ):
        """Initialize modern deep RL coordinator.

        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            sub_agents: Dictionary of sub-agents
            tools: List of available tools
            rl_algorithm: RL algorithm to use ("dqn", "ppo", "a2c")
            state_encoder: Optional state encoder
            **kwargs: Additional arguments for RL agent
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.sub_agents = sub_agents
        self.tools = tools
        self.rl_algorithm = rl_algorithm

        # Actions (sub-agents and tools)
        self.actions = list(sub_agents.keys()) + [tool.name for tool in tools]
        self.action_dim = len(self.actions)

        # State encoder
        if state_encoder is None:
            self.state_encoder = AttentionStateEncoder(input_dim=512, hidden_dim=256)
        else:
            self.state_encoder = state_encoder

        self.state_dim = self.state_encoder.hidden_dim

        # Create RL agent
        if rl_algorithm == "dqn":
            self.rl_agent = DQNAgent(
                name=f"{name}_dqn",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                **kwargs
            )
        elif rl_algorithm == "ppo":
            self.rl_agent = PPOAgent(
                name=f"{name}_ppo",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                **kwargs
            )
        elif rl_algorithm == "a2c":
            self.rl_agent = A2CAgent(
                name=f"{name}_a2c",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown RL algorithm: {rl_algorithm}")

        # Training data storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_dones = []

    async def _extract_state_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract state features from context.

        Args:
            context: Context dictionary containing request and history

        Returns:
            State feature vector
        """
        # Extract text features from request and history
        request = context.get("request", "")
        history = context.get("history", [])

        # Create text representation
        text_parts = [request]
        for msg in history[-5:]:  # Last 5 messages
            if isinstance(msg, dict):
                text_parts.append(msg.get("content", ""))
            else:
                text_parts.append(str(msg))

        text = " ".join(text_parts)

        # Simple feature extraction (can be enhanced with embeddings)
        features = []

        # Text length features
        features.append(len(text) / 1000.0)  # Normalized text length
        features.append(len(text.split()) / 100.0)  # Normalized word count

        # Keyword features
        keywords = ["search", "analyze", "create", "update", "delete", "help"]
        for keyword in keywords:
            features.append(1.0 if keyword.lower() in text.lower() else 0.0)

        # History length
        features.append(len(history) / 10.0)  # Normalized history length

        # Pad or truncate to fixed size
        target_size = 512
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        return np.array(features, dtype=np.float32)

    async def process_request(
        self, request: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process request using modern deep RL.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Extract state features
        context = {"request": request, "history": history}
        state_features = await self._extract_state_features(context)

        # Select action using RL agent
        if self.rl_algorithm == "dqn":
            action_idx = self.rl_agent.select_action(state_features)
            log_prob = None
            value = None
        else:  # PPO or A2C
            action_idx, log_prob, value = self.rl_agent.select_action(state_features)

        selected_action = self.actions[action_idx]

        # Execute action
        start_time = time.time()

        if selected_action in self.sub_agents:
            # Use sub-agent
            sub_agent = self.sub_agents[selected_action]
            result = await sub_agent.process_request(request, history)
        else:
            # Use tool
            tool = next((t for t in self.tools if t.name == selected_action), None)
            if tool:
                try:
                    result = await tool.arun(request)
                    result = {"success": True, "response": result}
                except Exception as e:
                    result = {"success": False, "error": str(e)}
            else:
                result = {"success": False, "error": "Action not found"}

        end_time = time.time()

        # Calculate reward
        reward = self.reward_system.calculate_reward(
            agent_name=self.name,
            task_id=f"task_{int(time.time())}",
            feedback={"self_evaluation": result},
            performance_metrics={
                "success_rate": 1.0 if result.get("success", False) else 0.0,
                "response_time": end_time - start_time,
            },
        )

        # Store experience
        self.episode_states.append(state_features)
        self.episode_actions.append(action_idx)
        self.episode_rewards.append(reward)

        if log_prob is not None:
            self.episode_log_probs.append(log_prob)
        if value is not None:
            self.episode_values.append(value)

        # Store in replay buffer for DQN
        if self.rl_algorithm == "dqn":
            # For simplicity, we'll store with next_state as current state
            # In a real implementation, you'd wait for the next state
            self.rl_agent.store_experience(
                state_features, action_idx, reward, state_features, False
            )

        return {
            "success": result.get("success", False),
            "response": result.get("response", result.get("error", "")),
            "selected_action": selected_action,
            "reward": reward,
            "state_features": state_features.tolist(),
        }

    async def train_episode(self) -> Dict[str, float]:
        """Train the RL agent on the current episode.

        Returns:
            Training metrics
        """
        if len(self.episode_states) == 0:
            return {}

        if self.rl_algorithm == "dqn":
            # DQN trains on each step
            return self.rl_agent.train()
        elif self.rl_algorithm == "ppo":
            # PPO trains on episodes
            for i, (state, action, reward, log_prob, value) in enumerate(
                zip(self.episode_states, self.episode_actions,
                    self.episode_rewards, self.episode_log_probs, self.episode_values)
            ):
                done = (i == len(self.episode_states) - 1)
                self.rl_agent.store_experience(state, action, log_prob, reward, value, done)

            metrics = self.rl_agent.train()
            self._clear_episode_data()
            return metrics
        elif self.rl_algorithm == "a2c":
            # A2C trains on episodes
            metrics = self.rl_agent.train(
                self.episode_states, self.episode_actions, self.episode_rewards
            )
            self._clear_episode_data()
            return metrics

        return {}

    def _clear_episode_data(self):
        """Clear episode data."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_values.clear()
        self.episode_dones.clear()


# Factory function to create modern deep RL agent architecture
async def create_modern_deep_rl_agent_architecture(
    model: ChatAnthropic,
    db: MemoryDatabase,
    sub_agents: Dict[str, Any],
    tools: List[Any],
    rl_algorithm: str = "dqn",
    **kwargs
) -> ModernDeepRLCoordinatorAgent:
    """Create a modern deep RL-based agent architecture.

    Args:
        model: Language model to use
        db: Memory database for persistence
        sub_agents: Dictionary of sub-agents
        tools: List of available tools
        rl_algorithm: RL algorithm to use ("dqn", "ppo", "a2c")
        **kwargs: Additional arguments for RL agent

    Returns:
        Modern deep RL coordinator agent
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Create modern deep RL coordinator agent
    coordinator = ModernDeepRLCoordinatorAgent(
        name="modern_deep_rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        tools=tools,
        rl_algorithm=rl_algorithm,
        **kwargs
    )

    return coordinator
