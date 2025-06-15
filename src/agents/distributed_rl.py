"""
Distributed reinforcement learning module for DataMCPServerAgent.
This module implements distributed training with multiple workers and parameter servers.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import ActorCriticNetwork, DQNNetwork


class ParameterServer:
    """Parameter server for distributed RL training."""

    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        learning_rate: float = 1e-4,
        aggregation_method: str = "average",
    ):
        """Initialize parameter server.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Model initialization arguments
            learning_rate: Learning rate for parameter updates
            aggregation_method: Method for aggregating gradients
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method

        # Initialize global model
        self.global_model = model_class(**model_kwargs)
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=learning_rate)

        # Worker management
        self.workers = {}
        self.gradient_buffer = []
        self.update_lock = threading.Lock()

        # Statistics
        self.update_count = 0
        self.worker_contributions = {}

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)

    def register_worker(self, worker_id: str) -> Dict[str, Any]:
        """Register a new worker.
        
        Args:
            worker_id: Unique worker identifier
            
        Returns:
            Initial model parameters
        """
        with self.update_lock:
            self.workers[worker_id] = {
                "registered_at": time.time(),
                "last_update": time.time(),
                "gradient_count": 0,
            }
            self.worker_contributions[worker_id] = []

        # Return current model state
        return {
            "model_state_dict": self.global_model.state_dict(),
            "worker_id": worker_id,
        }

    def push_gradients(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Receive gradients from worker.
        
        Args:
            worker_id: Worker identifier
            gradients: Computed gradients
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if worker_id not in self.workers:
            return False

        with self.update_lock:
            # Add gradients to buffer
            self.gradient_buffer.append({
                "worker_id": worker_id,
                "gradients": gradients,
                "timestamp": time.time(),
                "metadata": metadata or {},
            })

            # Update worker stats
            self.workers[worker_id]["last_update"] = time.time()
            self.workers[worker_id]["gradient_count"] += 1

            # Track contribution
            if metadata and "loss" in metadata:
                self.worker_contributions[worker_id].append(metadata["loss"])

        return True

    def pull_parameters(self, worker_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Send current parameters to worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Current model parameters
        """
        if worker_id not in self.workers:
            return None

        return self.global_model.state_dict()

    def aggregate_and_update(self, min_gradients: int = 1) -> Dict[str, float]:
        """Aggregate gradients and update global model.
        
        Args:
            min_gradients: Minimum number of gradients to aggregate
            
        Returns:
            Update statistics
        """
        with self.update_lock:
            if len(self.gradient_buffer) < min_gradients:
                return {"updated": False, "gradient_count": len(self.gradient_buffer)}

            # Aggregate gradients
            aggregated_gradients = self._aggregate_gradients()

            if not aggregated_gradients:
                return {"updated": False, "error": "No valid gradients"}

            # Apply aggregated gradients
            self.optimizer.zero_grad()

            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param.grad = aggregated_gradients[name].to(self.device)

            # Update parameters
            self.optimizer.step()
            self.update_count += 1

            # Clear gradient buffer
            processed_count = len(self.gradient_buffer)
            self.gradient_buffer.clear()

            return {
                "updated": True,
                "gradient_count": processed_count,
                "update_count": self.update_count,
            }

    def _aggregate_gradients(self) -> Dict[str, torch.Tensor]:
        """Aggregate gradients from multiple workers.
        
        Returns:
            Aggregated gradients
        """
        if not self.gradient_buffer:
            return {}

        # Get parameter names from first gradient
        param_names = list(self.gradient_buffer[0]["gradients"].keys())
        aggregated = {}

        for param_name in param_names:
            gradients = []
            weights = []

            for grad_data in self.gradient_buffer:
                if param_name in grad_data["gradients"]:
                    grad = grad_data["gradients"][param_name]
                    gradients.append(grad)

                    # Weight by inverse loss (better performance = higher weight)
                    loss = grad_data["metadata"].get("loss", 1.0)
                    weight = 1.0 / (1.0 + abs(loss))
                    weights.append(weight)

            if gradients:
                if self.aggregation_method == "average":
                    # Simple average
                    aggregated[param_name] = torch.stack(gradients).mean(dim=0)
                elif self.aggregation_method == "weighted_average":
                    # Weighted average
                    weights_tensor = torch.tensor(weights)
                    weights_tensor = weights_tensor / weights_tensor.sum()

                    weighted_grads = []
                    for grad, weight in zip(gradients, weights_tensor):
                        weighted_grads.append(grad * weight)

                    aggregated[param_name] = torch.stack(weighted_grads).sum(dim=0)
                elif self.aggregation_method == "median":
                    # Median aggregation (robust to outliers)
                    aggregated[param_name] = torch.stack(gradients).median(dim=0)[0]

        return aggregated

    def get_statistics(self) -> Dict[str, Any]:
        """Get parameter server statistics.
        
        Returns:
            Server statistics
        """
        with self.update_lock:
            active_workers = sum(
                1 for worker_data in self.workers.values()
                if time.time() - worker_data["last_update"] < 300  # 5 minutes
            )

            avg_contributions = {}
            for worker_id, contributions in self.worker_contributions.items():
                if contributions:
                    avg_contributions[worker_id] = np.mean(contributions[-10:])  # Last 10

            return {
                "total_workers": len(self.workers),
                "active_workers": active_workers,
                "update_count": self.update_count,
                "gradient_buffer_size": len(self.gradient_buffer),
                "avg_worker_contributions": avg_contributions,
            }


class DistributedWorker:
    """Distributed worker for RL training."""

    def __init__(
        self,
        worker_id: str,
        parameter_server_address: str,
        model_class: type,
        model_kwargs: Dict[str, Any],
        environment_config: Dict[str, Any],
        sync_frequency: int = 10,
    ):
        """Initialize distributed worker.
        
        Args:
            worker_id: Unique worker identifier
            parameter_server_address: Parameter server address
            model_class: Model class
            model_kwargs: Model initialization arguments
            environment_config: Environment configuration
            sync_frequency: How often to sync with parameter server
        """
        self.worker_id = worker_id
        self.parameter_server_address = parameter_server_address
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.environment_config = environment_config
        self.sync_frequency = sync_frequency

        # Initialize local model
        self.local_model = model_class(**model_kwargs)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=1e-4)

        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.local_gradients = []

        # Statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0,
            "avg_loss": 0,
            "sync_count": 0,
        }

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model.to(self.device)

    async def initialize(self):
        """Initialize worker with parameter server."""
        try:
            # Register with parameter server
            init_data = await self._call_parameter_server("register_worker", {
                "worker_id": self.worker_id
            })

            if init_data and "model_state_dict" in init_data:
                self.local_model.load_state_dict(init_data["model_state_dict"])
                print(f"✅ Worker {self.worker_id} initialized successfully")
                return True

        except Exception as e:
            print(f"❌ Worker {self.worker_id} initialization failed: {e}")
            return False

        return False

    async def train_episode(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Train on a single episode.
        
        Args:
            episode_data: Episode training data
            
        Returns:
            Training metrics
        """
        # Simulate training step
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        rewards = episode_data.get("rewards", [])

        if not states or not actions or not rewards:
            return {"loss": 0.0, "reward": 0.0}

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # Forward pass
        if hasattr(self.local_model, 'get_action_and_value'):
            # Actor-Critic model
            _, log_probs, values = self.local_model.get_action_and_value(states_tensor)

            # Compute returns
            returns = []
            R = 0
            for reward in reversed(rewards):
                R = reward + 0.99 * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)

            # Compute loss
            advantages = returns - values.squeeze()
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            loss = policy_loss + 0.5 * value_loss
        else:
            # DQN model
            q_values = self.local_model(states_tensor)
            q_values_selected = q_values.gather(1, actions_tensor.unsqueeze(1))

            # Simple target (can be improved)
            targets = rewards_tensor.unsqueeze(1)
            loss = nn.MSELoss()(q_values_selected, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Store gradients for later synchronization
        gradients = {}
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().cpu()

        self.local_gradients.append({
            "gradients": gradients,
            "loss": loss.item(),
            "episode": self.episode_count,
        })

        # Apply local update
        self.optimizer.step()

        # Update statistics
        episode_reward = sum(rewards)
        self.training_stats["episodes"] += 1
        self.training_stats["total_reward"] += episode_reward
        self.training_stats["avg_loss"] = (
            self.training_stats["avg_loss"] * 0.9 + loss.item() * 0.1
        )

        self.step_count += len(states)
        self.episode_count += 1

        # Sync with parameter server if needed
        if self.episode_count % self.sync_frequency == 0:
            await self._sync_with_parameter_server()

        return {
            "loss": loss.item(),
            "reward": episode_reward,
            "episode": self.episode_count,
        }

    async def _sync_with_parameter_server(self):
        """Synchronize with parameter server."""
        try:
            # Push gradients
            if self.local_gradients:
                for grad_data in self.local_gradients:
                    await self._call_parameter_server("push_gradients", {
                        "worker_id": self.worker_id,
                        "gradients": grad_data["gradients"],
                        "metadata": {
                            "loss": grad_data["loss"],
                            "episode": grad_data["episode"],
                        }
                    })

                self.local_gradients.clear()

            # Pull updated parameters
            new_params = await self._call_parameter_server("pull_parameters", {
                "worker_id": self.worker_id
            })

            if new_params:
                self.local_model.load_state_dict(new_params)
                self.training_stats["sync_count"] += 1

        except Exception as e:
            print(f"⚠️ Worker {self.worker_id} sync failed: {e}")

    async def _call_parameter_server(self, method: str, params: Dict[str, Any]) -> Any:
        """Call parameter server method.
        
        Args:
            method: Method name
            params: Method parameters
            
        Returns:
            Method result
        """
        # Simulate RPC call (in real implementation, use actual RPC)
        # This is a placeholder for demonstration
        await asyncio.sleep(0.01)  # Simulate network delay

        if method == "register_worker":
            return {
                "model_state_dict": self.local_model.state_dict(),
                "worker_id": params["worker_id"],
            }
        elif method == "push_gradients":
            return True
        elif method == "pull_parameters":
            return self.local_model.state_dict()

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Worker statistics
        """
        avg_reward = (
            self.training_stats["total_reward"] / max(1, self.training_stats["episodes"])
        )

        return {
            "worker_id": self.worker_id,
            "episodes": self.training_stats["episodes"],
            "steps": self.step_count,
            "avg_reward": avg_reward,
            "avg_loss": self.training_stats["avg_loss"],
            "sync_count": self.training_stats["sync_count"],
        }


class DistributedRLCoordinator:
    """Coordinator for distributed RL training."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        num_workers: int = 4,
        model_class: type = DQNNetwork,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize distributed RL coordinator.
        
        Args:
            name: Coordinator name
            model: Language model
            db: Memory database
            reward_system: Reward system
            num_workers: Number of distributed workers
            model_class: Model class for training
            model_kwargs: Model initialization arguments
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.num_workers = num_workers
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {"state_dim": 128, "action_dim": 5}

        # Initialize parameter server
        self.parameter_server = ParameterServer(
            model_class=model_class,
            model_kwargs=self.model_kwargs,
            learning_rate=1e-4,
            aggregation_method="weighted_average",
        )

        # Initialize workers
        self.workers = []
        for i in range(num_workers):
            worker = DistributedWorker(
                worker_id=f"worker_{i}",
                parameter_server_address="localhost:8000",
                model_class=model_class,
                model_kwargs=self.model_kwargs,
                environment_config={},
                sync_frequency=10,
            )
            self.workers.append(worker)

        # Training state
        self.training_active = False
        self.global_episode_count = 0

    async def initialize_distributed_training(self):
        """Initialize distributed training system."""
        print(f"🚀 Initializing distributed RL training with {self.num_workers} workers...")

        # Initialize all workers
        initialization_results = []
        for worker in self.workers:
            result = await worker.initialize()
            initialization_results.append(result)

        successful_workers = sum(initialization_results)
        print(f"✅ {successful_workers}/{self.num_workers} workers initialized successfully")

        return successful_workers > 0

    async def train_distributed_episode(
        self,
        request: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train using distributed workers.
        
        Args:
            request: User request
            history: Conversation history
            
        Returns:
            Training results
        """
        # Generate episode data for each worker
        episode_tasks = []

        for i, worker in enumerate(self.workers):
            # Create slightly different episode data for each worker
            episode_data = await self._generate_episode_data(request, history, worker_id=i)

            # Train worker on episode
            task = worker.train_episode(episode_data)
            episode_tasks.append(task)

        # Wait for all workers to complete
        worker_results = await asyncio.gather(*episode_tasks, return_exceptions=True)

        # Aggregate results
        successful_results = [
            result for result in worker_results
            if isinstance(result, dict) and "loss" in result
        ]

        if not successful_results:
            return {"success": False, "error": "No successful worker results"}

        # Aggregate and update global model
        server_stats = self.parameter_server.aggregate_and_update(min_gradients=1)

        # Compute aggregate metrics
        avg_loss = np.mean([result["loss"] for result in successful_results])
        avg_reward = np.mean([result["reward"] for result in successful_results])

        self.global_episode_count += 1

        return {
            "success": True,
            "avg_loss": avg_loss,
            "avg_reward": avg_reward,
            "successful_workers": len(successful_results),
            "server_stats": server_stats,
            "global_episode": self.global_episode_count,
        }

    async def _generate_episode_data(
        self,
        request: str,
        history: List[Dict[str, Any]],
        worker_id: int
    ) -> Dict[str, Any]:
        """Generate episode data for a worker.
        
        Args:
            request: User request
            history: Conversation history
            worker_id: Worker identifier
            
        Returns:
            Episode data
        """
        # Generate synthetic episode data
        episode_length = np.random.randint(10, 20)

        states = []
        actions = []
        rewards = []

        for step in range(episode_length):
            # Generate state (simplified)
            state = np.random.randn(self.model_kwargs["state_dim"]).astype(np.float32)

            # Add worker-specific noise for diversity
            state += np.random.normal(0, 0.1 * worker_id, state.shape)

            # Generate action
            action = np.random.randint(0, self.model_kwargs["action_dim"])

            # Generate reward (with some correlation to request)
            base_reward = np.random.uniform(-1, 1)
            if "analyze" in request.lower():
                base_reward += 0.2  # Bonus for analysis tasks
            if "create" in request.lower():
                base_reward += 0.1  # Bonus for creation tasks

            states.append(state.tolist())
            actions.append(action)
            rewards.append(base_reward)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "request": request,
            "worker_id": worker_id,
        }

    def get_distributed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive distributed training statistics.
        
        Returns:
            Distributed training statistics
        """
        # Parameter server stats
        server_stats = self.parameter_server.get_statistics()

        # Worker stats
        worker_stats = []
        for worker in self.workers:
            worker_stats.append(worker.get_statistics())

        # Aggregate worker metrics
        total_episodes = sum(stats["episodes"] for stats in worker_stats)
        avg_reward = np.mean([stats["avg_reward"] for stats in worker_stats])
        avg_loss = np.mean([stats["avg_loss"] for stats in worker_stats])

        return {
            "server": server_stats,
            "workers": worker_stats,
            "aggregate": {
                "total_episodes": total_episodes,
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
                "global_episodes": self.global_episode_count,
            },
        }


# Factory function to create distributed RL system
async def create_distributed_rl_system(
    model: ChatAnthropic,
    db: MemoryDatabase,
    num_workers: int = 4,
    model_type: str = "dqn",
    **kwargs
) -> DistributedRLCoordinator:
    """Create distributed RL training system.
    
    Args:
        model: Language model
        db: Memory database
        num_workers: Number of distributed workers
        model_type: Type of model to use
        **kwargs: Additional arguments
        
    Returns:
        Distributed RL coordinator
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Select model class
    if model_type == "dqn":
        model_class = DQNNetwork
        model_kwargs = {
            "state_dim": kwargs.get("state_dim", 128),
            "action_dim": kwargs.get("action_dim", 5),
        }
    elif model_type == "actor_critic":
        model_class = ActorCriticNetwork
        model_kwargs = {
            "state_dim": kwargs.get("state_dim", 128),
            "action_dim": kwargs.get("action_dim", 5),
            "continuous": kwargs.get("continuous", False),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create distributed coordinator
    coordinator = DistributedRLCoordinator(
        name="distributed_rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        num_workers=num_workers,
        model_class=model_class,
        model_kwargs=model_kwargs,
    )

    # Initialize distributed training
    await coordinator.initialize_distributed_training()

    return coordinator
