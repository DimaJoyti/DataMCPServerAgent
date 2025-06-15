"""
Neural network architectures for enhanced reinforcement learning in DataMCPServerAgent.
This module provides modern neural network architectures for deep RL algorithms.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based reinforcement learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.1,
        dueling: bool = False,
        noisy: bool = False,
    ):
        """Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
            dueling: Whether to use dueling architecture
            noisy: Whether to use noisy networks
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            if noisy:
                layers.append(NoisyLinear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture
            if noisy:
                self.value_head = NoisyLinear(input_dim, 1)
                self.advantage_head = NoisyLinear(input_dim, action_dim)
            else:
                self.value_head = nn.Linear(input_dim, 1)
                self.advantage_head = nn.Linear(input_dim, action_dim)
        else:
            # Standard DQN
            if noisy:
                self.q_head = NoisyLinear(input_dim, action_dim)
            else:
                self.q_head = nn.Linear(input_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layers(state)

        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values

    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for layer in self.modules():
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy-based reinforcement learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.1,
        continuous: bool = False,
    ):
        """Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
            continuous: Whether action space is continuous
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Shared feature layers
        shared_layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(self.activation)
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*shared_layers)

        # Actor head
        actor_layers = []
        if len(hidden_dims) > 0:
            actor_layers.append(nn.Linear(input_dim, hidden_dims[-1]))
            actor_layers.append(self.activation)
            if dropout > 0:
                actor_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dims[-1]

        if continuous:
            # For continuous actions, output mean and log_std
            actor_layers.append(nn.Linear(input_dim, action_dim * 2))
        else:
            # For discrete actions, output logits
            actor_layers.append(nn.Linear(input_dim, action_dim))

        self.actor_head = nn.Sequential(*actor_layers)

        # Critic head
        critic_layers = []
        input_dim = hidden_dims[0] if hidden_dims else state_dim

        if len(hidden_dims) > 0:
            critic_layers.append(nn.Linear(input_dim, hidden_dims[-1]))
            critic_layers.append(self.activation)
            if dropout > 0:
                critic_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dims[-1]

        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic_head = nn.Sequential(*critic_layers)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (actor_output, critic_value)
        """
        shared_features = self.shared_layers(state)

        actor_output = self.actor_head(shared_features)
        critic_value = self.critic_head(shared_features)

        return actor_output, critic_value

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        actor_output, value = self.forward(state)

        if self.continuous:
            # Split into mean and log_std
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            std = torch.exp(log_std.clamp(-20, 2))  # Clamp for stability

            # Sample action from normal distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Discrete actions
            dist = torch.distributions.Categorical(logits=actor_output)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in deep RL."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset network parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise buffers."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through noisy linear layer."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class AttentionStateEncoder(nn.Module):
    """Attention-based state encoder for complex state representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize attention-based state encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through attention encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Encoded state representation
        """
        # Project input
        x = self.input_projection(x)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)

        # Output projection
        x = self.output_projection(x)

        return x
