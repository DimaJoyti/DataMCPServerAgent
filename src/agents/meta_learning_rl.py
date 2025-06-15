"""
Meta-learning reinforcement learning module for DataMCPServerAgent.
This module implements Model-Agnostic Meta-Learning (MAML) and other meta-learning
techniques for fast adaptation to new tasks.
"""

import copy
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import ActorCriticNetwork, DQNNetwork


class MAMLAgent:
    """Model-Agnostic Meta-Learning agent for RL."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
        network_type: str = "actor_critic",
    ):
        """Initialize MAML agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            meta_batch_size: Meta batch size
            network_type: Type of network ("actor_critic" or "dqn")
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.network_type = network_type

        # Create meta-network
        if network_type == "actor_critic":
            self.meta_network = ActorCriticNetwork(
                state_dim, action_dim, continuous=False
            )
        else:  # dqn
            self.meta_network = DQNNetwork(state_dim, action_dim)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=meta_lr)

        # Task storage
        self.task_buffer = []
        self.max_tasks = 100

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_network.to(self.device)

    def add_task(self, task_data: Dict[str, Any]):
        """Add a task to the task buffer.
        
        Args:
            task_data: Dictionary containing task information
        """
        if len(self.task_buffer) >= self.max_tasks:
            self.task_buffer.pop(0)

        self.task_buffer.append(task_data)

    def sample_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Sample tasks for meta-learning.
        
        Args:
            num_tasks: Number of tasks to sample
            
        Returns:
            List of sampled tasks
        """
        if len(self.task_buffer) < num_tasks:
            return self.task_buffer.copy()

        indices = np.random.choice(len(self.task_buffer), num_tasks, replace=False)
        return [self.task_buffer[i] for i in indices]

    def inner_loop_update(
        self,
        network: nn.Module,
        support_data: List[Tuple[torch.Tensor, torch.Tensor, float]]
    ) -> nn.Module:
        """Perform inner loop update for a specific task.
        
        Args:
            network: Network to update
            support_data: Support set data for the task
            
        Returns:
            Updated network
        """
        # Create a copy of the network for inner loop updates
        adapted_network = copy.deepcopy(network)
        inner_optimizer = optim.SGD(adapted_network.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            total_loss = 0

            for state, action, reward in support_data:
                state = state.to(self.device)
                action = action.to(self.device)

                if self.network_type == "actor_critic":
                    actor_output, value = adapted_network(state.unsqueeze(0))

                    # Policy loss
                    if len(actor_output.shape) > 1:
                        log_probs = F.log_softmax(actor_output, dim=-1)
                        policy_loss = -log_probs[0, action] * reward
                    else:
                        policy_loss = torch.tensor(0.0, device=self.device)

                    # Value loss
                    value_loss = F.mse_loss(value.squeeze(), torch.tensor(reward, device=self.device))

                    loss = policy_loss + 0.5 * value_loss
                else:  # dqn
                    q_values = adapted_network(state.unsqueeze(0))
                    target_q = torch.tensor(reward, device=self.device)
                    loss = F.mse_loss(q_values[0, action], target_q)

                total_loss += loss

            # Inner loop update
            inner_optimizer.zero_grad()
            total_loss.backward()
            inner_optimizer.step()

        return adapted_network

    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform meta-update across multiple tasks.
        
        Args:
            tasks: List of tasks for meta-learning
            
        Returns:
            Meta-learning metrics
        """
        meta_loss = 0
        num_valid_tasks = 0

        for task in tasks:
            support_data = task.get("support_data", [])
            query_data = task.get("query_data", [])

            if not support_data or not query_data:
                continue

            # Inner loop adaptation
            adapted_network = self.inner_loop_update(self.meta_network, support_data)

            # Compute loss on query set
            query_loss = 0
            for state, action, reward in query_data:
                state = state.to(self.device)
                action = action.to(self.device)

                if self.network_type == "actor_critic":
                    actor_output, value = adapted_network(state.unsqueeze(0))

                    if len(actor_output.shape) > 1:
                        log_probs = F.log_softmax(actor_output, dim=-1)
                        policy_loss = -log_probs[0, action] * reward
                    else:
                        policy_loss = torch.tensor(0.0, device=self.device)

                    value_loss = F.mse_loss(value.squeeze(), torch.tensor(reward, device=self.device))
                    loss = policy_loss + 0.5 * value_loss
                else:  # dqn
                    q_values = adapted_network(state.unsqueeze(0))
                    target_q = torch.tensor(reward, device=self.device)
                    loss = F.mse_loss(q_values[0, action], target_q)

                query_loss += loss

            meta_loss += query_loss
            num_valid_tasks += 1

        if num_valid_tasks == 0:
            return {"meta_loss": 0.0, "num_tasks": 0}

        # Meta-update
        meta_loss = meta_loss / num_valid_tasks
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_network.parameters(), 1.0)
        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "num_tasks": num_valid_tasks,
        }

    def adapt_to_task(
        self,
        task_data: List[Tuple[torch.Tensor, torch.Tensor, float]]
    ) -> nn.Module:
        """Quickly adapt to a new task using few-shot learning.
        
        Args:
            task_data: Few-shot examples for the new task
            
        Returns:
            Adapted network for the task
        """
        return self.inner_loop_update(self.meta_network, task_data)

    def train_meta_learning(self) -> Dict[str, float]:
        """Train the meta-learning agent.
        
        Returns:
            Training metrics
        """
        if len(self.task_buffer) < self.meta_batch_size:
            return {"meta_loss": 0.0, "num_tasks": 0}

        # Sample tasks for meta-learning
        sampled_tasks = self.sample_tasks(self.meta_batch_size)

        # Perform meta-update
        metrics = self.meta_update(sampled_tasks)

        return metrics


class TransferLearningAgent:
    """Transfer learning agent for RL."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        source_agent: Any,
        target_state_dim: int,
        target_action_dim: int,
        transfer_method: str = "feature_extraction",
    ):
        """Initialize transfer learning agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            source_agent: Pre-trained source agent
            target_state_dim: Target task state dimension
            target_action_dim: Target task action dimension
            transfer_method: Transfer method ("feature_extraction", "fine_tuning", "progressive")
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.source_agent = source_agent
        self.target_state_dim = target_state_dim
        self.target_action_dim = target_action_dim
        self.transfer_method = transfer_method

        # Create target network based on transfer method
        self._create_target_network()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_network.to(self.device)

    def _create_target_network(self):
        """Create target network based on transfer method."""
        if self.transfer_method == "feature_extraction":
            # Freeze source features, only train new head
            if hasattr(self.source_agent, 'q_network'):
                source_network = self.source_agent.q_network

                # Copy feature layers
                self.target_network = DQNNetwork(
                    self.target_state_dim, self.target_action_dim
                )

                # Copy and freeze feature layers
                if hasattr(source_network, 'feature_layers'):
                    self.target_network.feature_layers.load_state_dict(
                        source_network.feature_layers.state_dict()
                    )

                    # Freeze feature layers
                    for param in self.target_network.feature_layers.parameters():
                        param.requires_grad = False

        elif self.transfer_method == "fine_tuning":
            # Initialize with source weights, fine-tune all parameters
            if hasattr(self.source_agent, 'q_network'):
                self.target_network = DQNNetwork(
                    self.target_state_dim, self.target_action_dim
                )

                # Copy compatible layers
                source_dict = self.source_agent.q_network.state_dict()
                target_dict = self.target_network.state_dict()

                # Copy compatible parameters
                for name, param in source_dict.items():
                    if name in target_dict and param.shape == target_dict[name].shape:
                        target_dict[name] = param

                self.target_network.load_state_dict(target_dict)

        else:  # progressive
            # Progressive neural networks approach
            self.target_network = DQNNetwork(
                self.target_state_dim, self.target_action_dim
            )

        # Create optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.target_network.parameters()),
            lr=1e-4
        )

    def compute_task_similarity(
        self,
        source_states: List[torch.Tensor],
        target_states: List[torch.Tensor]
    ) -> float:
        """Compute similarity between source and target tasks.
        
        Args:
            source_states: States from source task
            target_states: States from target task
            
        Returns:
            Task similarity score
        """
        if not source_states or not target_states:
            return 0.0

        # Simple similarity based on state statistics
        source_mean = torch.stack(source_states).mean(dim=0)
        target_mean = torch.stack(target_states).mean(dim=0)

        # Cosine similarity
        similarity = F.cosine_similarity(
            source_mean.flatten(), target_mean.flatten(), dim=0
        )

        return similarity.item()

    def transfer_knowledge(
        self,
        target_data: List[Tuple[torch.Tensor, torch.Tensor, float]]
    ) -> Dict[str, float]:
        """Transfer knowledge to target task.
        
        Args:
            target_data: Training data for target task
            
        Returns:
            Transfer learning metrics
        """
        total_loss = 0
        num_samples = len(target_data)

        if num_samples == 0:
            return {"transfer_loss": 0.0, "num_samples": 0}

        for state, action, reward in target_data:
            state = state.to(self.device)
            action = action.to(self.device)

            # Forward pass
            q_values = self.target_network(state.unsqueeze(0))
            target_q = torch.tensor(reward, device=self.device)

            # Compute loss
            loss = F.mse_loss(q_values[0, action], target_q)
            total_loss += loss

        # Backward pass
        avg_loss = total_loss / num_samples
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.target_network.parameters(), 1.0)
        self.optimizer.step()

        return {
            "transfer_loss": avg_loss.item(),
            "num_samples": num_samples,
        }


class FewShotLearningAgent:
    """Few-shot learning agent for RL."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        state_dim: int,
        action_dim: int,
        memory_size: int = 1000,
        k_shot: int = 5,
    ):
        """Initialize few-shot learning agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            state_dim: State space dimension
            action_dim: Action space dimension
            memory_size: Size of episodic memory
            k_shot: Number of shots for few-shot learning
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.k_shot = k_shot

        # Episodic memory
        self.episodic_memory = []

        # Base network
        self.base_network = DQNNetwork(state_dim, action_dim)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_network.to(self.device)

    def add_to_memory(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        context: Dict[str, Any]
    ):
        """Add experience to episodic memory.
        
        Args:
            state: State tensor
            action: Action taken
            reward: Reward received
            context: Additional context information
        """
        if len(self.episodic_memory) >= self.memory_size:
            self.episodic_memory.pop(0)

        self.episodic_memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "context": context,
            "timestamp": time.time(),
        })

    def retrieve_similar_experiences(
        self,
        query_state: torch.Tensor,
        query_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve similar experiences from episodic memory.
        
        Args:
            query_state: Query state
            query_context: Query context
            
        Returns:
            List of similar experiences
        """
        if not self.episodic_memory:
            return []

        similarities = []

        for experience in self.episodic_memory:
            # State similarity
            state_sim = F.cosine_similarity(
                query_state.flatten(),
                experience["state"].flatten(),
                dim=0
            ).item()

            # Context similarity (simple text matching)
            context_sim = 0.0
            if "request" in query_context and "request" in experience["context"]:
                query_words = set(query_context["request"].lower().split())
                exp_words = set(experience["context"]["request"].lower().split())
                if query_words and exp_words:
                    context_sim = len(query_words & exp_words) / len(query_words | exp_words)

            # Combined similarity
            total_sim = 0.7 * state_sim + 0.3 * context_sim
            similarities.append((total_sim, experience))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:self.k_shot]]

    def few_shot_predict(
        self,
        state: torch.Tensor,
        context: Dict[str, Any]
    ) -> Tuple[int, float]:
        """Make prediction using few-shot learning.
        
        Args:
            state: Current state
            context: Current context
            
        Returns:
            Tuple of (action, confidence)
        """
        # Retrieve similar experiences
        similar_experiences = self.retrieve_similar_experiences(state, context)

        if not similar_experiences:
            # Fallback to base network
            with torch.no_grad():
                q_values = self.base_network(state.unsqueeze(0))
                action = q_values.argmax().item()
                confidence = torch.softmax(q_values, dim=-1).max().item()
            return action, confidence

        # Aggregate predictions from similar experiences
        action_votes = {}
        total_weight = 0

        for exp in similar_experiences:
            action = exp["action"]
            reward = exp["reward"]
            weight = max(0, reward)  # Use reward as weight

            if action not in action_votes:
                action_votes[action] = 0
            action_votes[action] += weight
            total_weight += weight

        if total_weight == 0:
            # Fallback to base network
            with torch.no_grad():
                q_values = self.base_network(state.unsqueeze(0))
                action = q_values.argmax().item()
                confidence = torch.softmax(q_values, dim=-1).max().item()
            return action, confidence

        # Select action with highest vote
        best_action = max(action_votes.items(), key=lambda x: x[1])[0]
        confidence = action_votes[best_action] / total_weight

        return best_action, confidence
