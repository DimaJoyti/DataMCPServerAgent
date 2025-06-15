"""
Hyperparameter optimization for reinforcement learning in DataMCPServerAgent.
This module implements automated hyperparameter tuning using various optimization methods.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
from langchain_anthropic import ChatAnthropic
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import CmaEsSampler, TPESampler

from src.memory.memory_persistence import MemoryDatabase


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space."""

    name: str
    param_type: str  # 'float', 'int', 'categorical', 'bool'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None

    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggest a value for this hyperparameter.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Suggested hyperparameter value
        """
        if self.param_type == 'float':
            return trial.suggest_float(
                self.name, self.low, self.high, log=self.log, step=self.step
            )
        elif self.param_type == 'int':
            return trial.suggest_int(
                self.name, int(self.low), int(self.high), step=int(self.step) if self.step else None
            )
        elif self.param_type == 'categorical':
            return trial.suggest_categorical(self.name, self.choices)
        elif self.param_type == 'bool':
            return trial.suggest_categorical(self.name, [True, False])
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""

    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        objective_function: Callable,
        n_trials: int = 100,
        sampler: str = "tpe",
        pruner: str = "median",
        direction: str = "maximize",
    ):
        """Initialize Bayesian optimizer.
        
        Args:
            search_space: List of hyperparameter spaces
            objective_function: Function to optimize
            n_trials: Number of optimization trials
            sampler: Sampling strategy ('tpe', 'cmaes', 'random')
            pruner: Pruning strategy ('median', 'hyperband', 'none')
            direction: Optimization direction ('maximize', 'minimize')
        """
        self.search_space = search_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.direction = direction

        # Configure sampler
        if sampler == "tpe":
            self.sampler = TPESampler()
        elif sampler == "cmaes":
            self.sampler = CmaEsSampler()
        else:
            self.sampler = optuna.samplers.RandomSampler()

        # Configure pruner
        if pruner == "median":
            self.pruner = MedianPruner()
        elif pruner == "hyperband":
            self.pruner = HyperbandPruner()
        else:
            self.pruner = optuna.pruners.NopPruner()

        # Create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # Optimization history
        self.optimization_history = []

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function wrapper for Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value
        """
        # Suggest hyperparameters
        params = {}
        for param_space in self.search_space:
            params[param_space.name] = param_space.suggest(trial)

        # Evaluate objective function
        start_time = time.time()
        try:
            value = self.objective_function(params, trial)

            # Record trial
            self.optimization_history.append({
                "trial_number": trial.number,
                "params": params,
                "value": value,
                "duration": time.time() - start_time,
                "state": "completed",
            })

            return value

        except optuna.TrialPruned:
            # Trial was pruned
            self.optimization_history.append({
                "trial_number": trial.number,
                "params": params,
                "value": None,
                "duration": time.time() - start_time,
                "state": "pruned",
            })
            raise
        except Exception as e:
            # Trial failed
            self.optimization_history.append({
                "trial_number": trial.number,
                "params": params,
                "value": None,
                "duration": time.time() - start_time,
                "state": "failed",
                "error": str(e),
            })
            raise

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Returns:
            Optimization results
        """
        print(f"🔍 Starting Bayesian optimization with {self.n_trials} trials...")

        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)

        # Get results
        best_params = self.study.best_params
        best_value = self.study.best_value

        print("✅ Optimization completed!")
        print(f"   Best value: {best_value:.4f}")
        print(f"   Best params: {best_params}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
            "optimization_history": self.optimization_history,
        }

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Returns:
            Optimization statistics
        """
        completed_trials = [
            trial for trial in self.optimization_history
            if trial["state"] == "completed"
        ]

        if not completed_trials:
            return {"error": "No completed trials"}

        values = [trial["value"] for trial in completed_trials]
        durations = [trial["duration"] for trial in completed_trials]

        return {
            "total_trials": len(self.optimization_history),
            "completed_trials": len(completed_trials),
            "pruned_trials": sum(1 for t in self.optimization_history if t["state"] == "pruned"),
            "failed_trials": sum(1 for t in self.optimization_history if t["state"] == "failed"),
            "best_value": max(values) if self.direction == "maximize" else min(values),
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "mean_duration": np.mean(durations),
            "total_duration": sum(durations),
        }


class GridSearchOptimizer:
    """Grid search optimizer for exhaustive hyperparameter search."""

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        objective_function: Callable,
        direction: str = "maximize",
    ):
        """Initialize grid search optimizer.
        
        Args:
            search_space: Dictionary of parameter names to value lists
            objective_function: Function to optimize
            direction: Optimization direction
        """
        self.search_space = search_space
        self.objective_function = objective_function
        self.direction = direction

        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()
        self.results = []

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations.
        
        Returns:
            List of parameter combinations
        """
        import itertools

        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())

        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations

    def optimize(self) -> Dict[str, Any]:
        """Run grid search optimization.
        
        Returns:
            Optimization results
        """
        print(f"🔍 Starting grid search with {len(self.param_combinations)} combinations...")

        best_value = float('-inf') if self.direction == "maximize" else float('inf')
        best_params = None

        for i, params in enumerate(self.param_combinations):
            print(f"   Trial {i+1}/{len(self.param_combinations)}: {params}")

            start_time = time.time()
            try:
                value = self.objective_function(params)
                duration = time.time() - start_time

                # Check if this is the best result
                is_better = (
                    (self.direction == "maximize" and value > best_value) or
                    (self.direction == "minimize" and value < best_value)
                )

                if is_better:
                    best_value = value
                    best_params = params.copy()

                self.results.append({
                    "trial": i,
                    "params": params,
                    "value": value,
                    "duration": duration,
                    "is_best": is_better,
                })

            except Exception as e:
                print(f"   ❌ Trial {i+1} failed: {e}")
                self.results.append({
                    "trial": i,
                    "params": params,
                    "value": None,
                    "duration": time.time() - start_time,
                    "error": str(e),
                })

        print("✅ Grid search completed!")
        print(f"   Best value: {best_value:.4f}")
        print(f"   Best params: {best_params}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.param_combinations),
            "results": self.results,
        }


class RLHyperparameterOptimizer:
    """Specialized hyperparameter optimizer for RL agents."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        agent_factory: Callable,
        evaluation_episodes: int = 10,
        optimization_method: str = "bayesian",
    ):
        """Initialize RL hyperparameter optimizer.
        
        Args:
            model: Language model
            db: Memory database
            agent_factory: Function to create RL agent with given parameters
            evaluation_episodes: Number of episodes for evaluation
            optimization_method: Optimization method ('bayesian', 'grid', 'random')
        """
        self.model = model
        self.db = db
        self.agent_factory = agent_factory
        self.evaluation_episodes = evaluation_episodes
        self.optimization_method = optimization_method

        # Define common RL hyperparameter spaces
        self.rl_search_spaces = {
            "dqn": [
                HyperparameterSpace("learning_rate", "float", 1e-5, 1e-2, log=True),
                HyperparameterSpace("epsilon", "float", 0.01, 1.0),
                HyperparameterSpace("epsilon_decay", "float", 0.99, 0.999),
                HyperparameterSpace("target_update_freq", "int", 100, 2000),
                HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64, 128]),
                HyperparameterSpace("buffer_size", "categorical", choices=[1000, 5000, 10000, 50000]),
                HyperparameterSpace("gamma", "float", 0.9, 0.999),
                HyperparameterSpace("double_dqn", "bool"),
                HyperparameterSpace("dueling", "bool"),
            ],
            "ppo": [
                HyperparameterSpace("learning_rate", "float", 1e-5, 1e-2, log=True),
                HyperparameterSpace("clip_epsilon", "float", 0.1, 0.3),
                HyperparameterSpace("ppo_epochs", "int", 3, 10),
                HyperparameterSpace("batch_size", "categorical", choices=[32, 64, 128, 256]),
                HyperparameterSpace("gae_lambda", "float", 0.9, 0.99),
                HyperparameterSpace("value_coef", "float", 0.1, 1.0),
                HyperparameterSpace("entropy_coef", "float", 0.001, 0.1, log=True),
                HyperparameterSpace("max_grad_norm", "float", 0.1, 2.0),
            ],
            "a2c": [
                HyperparameterSpace("learning_rate", "float", 1e-5, 1e-2, log=True),
                HyperparameterSpace("value_coef", "float", 0.1, 1.0),
                HyperparameterSpace("entropy_coef", "float", 0.001, 0.1, log=True),
                HyperparameterSpace("max_grad_norm", "float", 0.1, 2.0),
                HyperparameterSpace("gamma", "float", 0.9, 0.999),
            ],
        }

        # Optimization results
        self.optimization_results = {}

    async def optimize_agent(
        self,
        agent_type: str,
        n_trials: int = 50,
        custom_search_space: Optional[List[HyperparameterSpace]] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific agent type.
        
        Args:
            agent_type: Type of RL agent ('dqn', 'ppo', 'a2c')
            n_trials: Number of optimization trials
            custom_search_space: Custom search space (overrides default)
            
        Returns:
            Optimization results
        """
        print(f"🎯 Optimizing {agent_type.upper()} hyperparameters...")

        # Get search space
        search_space = custom_search_space or self.rl_search_spaces.get(agent_type, [])

        if not search_space:
            raise ValueError(f"No search space defined for agent type: {agent_type}")

        # Define objective function
        async def objective_function(params: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> float:
            return await self._evaluate_agent_performance(agent_type, params, trial)

        # Create optimizer
        if self.optimization_method == "bayesian":
            optimizer = BayesianOptimizer(
                search_space=search_space,
                objective_function=lambda params, trial: asyncio.run(objective_function(params, trial)),
                n_trials=n_trials,
                direction="maximize",
            )
            results = optimizer.optimize()
        else:
            raise ValueError(f"Optimization method {self.optimization_method} not implemented")

        # Store results
        self.optimization_results[agent_type] = results

        return results

    async def _evaluate_agent_performance(
        self,
        agent_type: str,
        params: Dict[str, Any],
        trial: Optional[optuna.Trial] = None
    ) -> float:
        """Evaluate agent performance with given hyperparameters.
        
        Args:
            agent_type: Type of RL agent
            params: Hyperparameters to evaluate
            trial: Optuna trial for pruning
            
        Returns:
            Performance score
        """
        try:
            # Create agent with given parameters
            agent = await self.agent_factory(agent_type, params)

            # Evaluate agent performance
            episode_rewards = []
            episode_losses = []

            for episode in range(self.evaluation_episodes):
                # Simulate episode
                episode_data = await self._simulate_episode(agent)

                episode_rewards.append(episode_data["reward"])
                if "loss" in episode_data:
                    episode_losses.append(episode_data["loss"])

                # Report intermediate value for pruning
                if trial and episode > 2:  # Need some episodes for meaningful intermediate value
                    intermediate_value = np.mean(episode_rewards)
                    trial.report(intermediate_value, episode)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # Calculate performance metrics
            avg_reward = np.mean(episode_rewards)
            reward_std = np.std(episode_rewards)

            # Performance score (higher is better)
            # Combine average reward with stability (lower std is better)
            performance_score = avg_reward - 0.1 * reward_std

            return performance_score

        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            return float('-inf')  # Return worst possible score

    async def _simulate_episode(self, agent: Any) -> Dict[str, float]:
        """Simulate a single episode for evaluation.
        
        Args:
            agent: RL agent to evaluate
            
        Returns:
            Episode results
        """
        # Simulate episode (simplified)
        total_reward = 0
        episode_length = np.random.randint(10, 20)

        for step in range(episode_length):
            # Generate random state
            state = np.random.randn(128).astype(np.float32)

            # Agent selects action
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=False)
            else:
                action = np.random.randint(0, 5)

            # Simulate reward
            reward = np.random.uniform(-1, 1)
            total_reward += reward

            # Simulate training step
            if hasattr(agent, 'store_experience'):
                next_state = np.random.randn(128).astype(np.float32)
                agent.store_experience(state, action, reward, next_state, False)

            # Train agent
            if hasattr(agent, 'train') and step % 5 == 0:
                metrics = agent.train()
                if metrics and "loss" in metrics:
                    return {"reward": total_reward, "loss": metrics["loss"]}

        return {"reward": total_reward}

    def get_best_hyperparameters(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters for an agent type.
        
        Args:
            agent_type: Type of RL agent
            
        Returns:
            Best hyperparameters or None if not optimized
        """
        if agent_type in self.optimization_results:
            return self.optimization_results[agent_type]["best_params"]
        return None

    def save_optimization_results(self, filepath: str):
        """Save optimization results to file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)

        print(f"💾 Optimization results saved to {filepath}")

    def load_optimization_results(self, filepath: str):
        """Load optimization results from file.
        
        Args:
            filepath: Path to load results from
        """
        try:
            with open(filepath) as f:
                self.optimization_results = json.load(f)

            print(f"📂 Optimization results loaded from {filepath}")
        except FileNotFoundError:
            print(f"⚠️ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading results: {e}")


# Factory function to create hyperparameter optimizer
async def create_rl_hyperparameter_optimizer(
    model: ChatAnthropic,
    db: MemoryDatabase,
    agent_factory: Callable,
    optimization_method: str = "bayesian",
    evaluation_episodes: int = 10,
) -> RLHyperparameterOptimizer:
    """Create RL hyperparameter optimizer.
    
    Args:
        model: Language model
        db: Memory database
        agent_factory: Function to create RL agents
        optimization_method: Optimization method
        evaluation_episodes: Number of evaluation episodes
        
    Returns:
        RL hyperparameter optimizer
    """
    optimizer = RLHyperparameterOptimizer(
        model=model,
        db=db,
        agent_factory=agent_factory,
        evaluation_episodes=evaluation_episodes,
        optimization_method=optimization_method,
    )

    return optimizer
