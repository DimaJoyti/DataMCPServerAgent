"""
Reinforcement Learning trading agent.
"""

import logging
import numpy as np
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class RLTradingAgent:
    """
    Reinforcement Learning trading agent using Q-Learning.
    
    Features:
    - Q-Learning algorithm
    - Experience replay
    - Epsilon-greedy exploration
    - Portfolio management
    - Risk-aware rewards
    """
    
    def __init__(
        self,
        name: str = "RLTradingAgent",
        state_size: int = 20,
        action_size: int = 3,  # 0: Hold, 1: Buy, 2: Sell
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000
    ):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.logger = logging.getLogger(f"RLTradingAgent.{name}")
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Q-table (simplified - in practice would use neural network)
        self.q_table = {}
        
        # Trading state
        self.current_position = 0  # -1: Short, 0: Neutral, 1: Long
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.cash = 100000.0
        self.shares = 0
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.total_episodes = 0
        self.total_steps = 0
        
        # Training history
        self.training_history = []
    
    def get_state_key(self, state: np.ndarray) -> str:
        """Convert state array to string key for Q-table."""
        # Discretize continuous state values
        discretized = np.round(state * 100).astype(int)
        return str(discretized.tolist())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action
            action = np.argmax(self.q_table[state_key])
        
        return action
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32) -> None:
        """Train the agent using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            # Initialize Q-values if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            target = reward
            if not done:
                target += self.discount_factor * np.max(self.q_table[next_state_key])
            
            # Update Q-value
            self.q_table[state_key][action] += self.learning_rate * (
                target - self.q_table[state_key][action]
            )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(
        self,
        action: int,
        price_change: float,
        portfolio_change: float,
        risk_penalty: float = 0.0
    ) -> float:
        """
        Calculate reward for the action taken.
        
        Args:
            action: Action taken (0: Hold, 1: Buy, 2: Sell)
            price_change: Price change since last action
            portfolio_change: Portfolio value change
            risk_penalty: Risk penalty factor
            
        Returns:
            Reward value
        """
        base_reward = 0.0
        
        # Reward based on portfolio performance
        base_reward += portfolio_change * 100  # Scale portfolio change
        
        # Action-specific rewards
        if action == 1:  # Buy
            if price_change > 0:
                base_reward += price_change * 50  # Reward for buying before price increase
            else:
                base_reward += price_change * 25  # Penalty for buying before price decrease
        elif action == 2:  # Sell
            if price_change < 0:
                base_reward += abs(price_change) * 50  # Reward for selling before price decrease
            else:
                base_reward -= price_change * 25  # Penalty for selling before price increase
        
        # Risk penalty
        base_reward -= risk_penalty
        
        # Transaction cost penalty
        if action != 0:  # If not holding
            base_reward -= self.transaction_cost * 100
        
        return base_reward
    
    def execute_action(self, action: int, current_price: float) -> Dict[str, Any]:
        """
        Execute trading action and update portfolio.
        
        Args:
            action: Action to execute
            current_price: Current asset price
            
        Returns:
            Execution result
        """
        try:
            previous_value = self.portfolio_value
            transaction_cost = 0.0
            
            if action == 1:  # Buy
                if self.cash > current_price:
                    # Calculate how many shares to buy (use 10% of cash)
                    buy_amount = self.cash * 0.1
                    shares_to_buy = buy_amount / current_price
                    transaction_cost = buy_amount * self.transaction_cost
                    
                    self.shares += shares_to_buy
                    self.cash -= (buy_amount + transaction_cost)
                    self.current_position = 1
                    
            elif action == 2:  # Sell
                if self.shares > 0:
                    # Sell 10% of shares
                    shares_to_sell = self.shares * 0.1
                    sell_amount = shares_to_sell * current_price
                    transaction_cost = sell_amount * self.transaction_cost
                    
                    self.shares -= shares_to_sell
                    self.cash += (sell_amount - transaction_cost)
                    
                    if self.shares == 0:
                        self.current_position = 0
            
            # Update portfolio value
            self.portfolio_value = self.cash + (self.shares * current_price)
            portfolio_change = (self.portfolio_value - previous_value) / previous_value
            
            result = {
                'action': action,
                'action_name': ['HOLD', 'BUY', 'SELL'][action],
                'price': current_price,
                'shares': self.shares,
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'portfolio_change': portfolio_change,
                'transaction_cost': transaction_cost,
                'position': self.current_position
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return {
                'action': 0,
                'action_name': 'HOLD',
                'error': str(e)
            }
    
    def train_episode(
        self,
        price_data: pd.Series,
        features: pd.DataFrame,
        episode_length: int = 100
    ) -> Dict[str, Any]:
        """
        Train the agent for one episode.
        
        Args:
            price_data: Historical price data
            features: Feature data
            episode_length: Length of training episode
            
        Returns:
            Episode results
        """
        try:
            if len(price_data) < episode_length + 1:
                raise ValueError("Insufficient data for episode")
            
            # Reset portfolio for episode
            self.cash = 100000.0
            self.shares = 0
            self.current_position = 0
            self.portfolio_value = 100000.0
            
            episode_reward = 0.0
            episode_actions = []
            
            # Random starting point
            start_idx = random.randint(0, len(price_data) - episode_length - 1)
            
            for step in range(episode_length):
                current_idx = start_idx + step
                next_idx = current_idx + 1
                
                # Get current state (features)
                if current_idx < len(features):
                    current_state = features.iloc[current_idx].values
                    # Normalize state
                    current_state = (current_state - np.mean(current_state)) / (np.std(current_state) + 1e-8)
                    current_state = current_state[:self.state_size]  # Limit to state size
                else:
                    current_state = np.zeros(self.state_size)
                
                # Get action
                action = self.get_action(current_state, training=True)
                
                # Execute action
                current_price = price_data.iloc[current_idx]
                execution_result = self.execute_action(action, current_price)
                
                # Calculate reward
                if next_idx < len(price_data):
                    next_price = price_data.iloc[next_idx]
                    price_change = (next_price - current_price) / current_price
                    
                    reward = self.calculate_reward(
                        action,
                        price_change,
                        execution_result.get('portfolio_change', 0)
                    )
                else:
                    reward = 0.0
                
                # Get next state
                if next_idx < len(features):
                    next_state = features.iloc[next_idx].values
                    next_state = (next_state - np.mean(next_state)) / (np.std(next_state) + 1e-8)
                    next_state = next_state[:self.state_size]
                else:
                    next_state = np.zeros(self.state_size)
                
                # Store experience
                done = (step == episode_length - 1)
                self.remember(current_state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_actions.append(action)
                
                self.total_steps += 1
            
            # Train with experience replay
            self.replay()
            
            # Store episode results
            episode_result = {
                'episode': self.total_episodes,
                'total_reward': episode_reward,
                'final_portfolio_value': self.portfolio_value,
                'return_pct': (self.portfolio_value - 100000.0) / 100000.0 * 100,
                'actions_taken': episode_actions,
                'epsilon': self.epsilon,
                'steps': episode_length
            }
            
            self.episode_rewards.append(episode_reward)
            self.episode_actions.append(episode_actions)
            self.total_episodes += 1
            
            # Store in training history
            self.training_history.append(episode_result)
            
            self.logger.info(
                f"Episode {self.total_episodes}: Reward={episode_reward:.2f}, "
                f"Portfolio=${self.portfolio_value:.2f}, Return={episode_result['return_pct']:.2f}%"
            )
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"Error in training episode: {str(e)}")
            return {'error': str(e)}
    
    def predict_action(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Predict best action for given state (inference mode).
        
        Args:
            state: Current state
            
        Returns:
            Prediction result
        """
        try:
            # Normalize state
            normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)
            normalized_state = normalized_state[:self.state_size]
            
            # Get action (no exploration)
            action = self.get_action(normalized_state, training=False)
            
            # Get Q-values for confidence
            state_key = self.get_state_key(normalized_state)
            q_values = self.q_table.get(state_key, np.zeros(self.action_size))
            
            confidence = np.max(q_values) - np.mean(q_values) if np.std(q_values) > 0 else 0.0
            
            return {
                'action': action,
                'action_name': ['HOLD', 'BUY', 'SELL'][action],
                'confidence': confidence,
                'q_values': q_values.tolist(),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting action: {str(e)}")
            return {
                'action': 0,
                'action_name': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'average_reward': np.mean(self.episode_rewards),
            'recent_average_reward': np.mean(recent_rewards),
            'best_reward': max(self.episode_rewards),
            'worst_reward': min(self.episode_rewards),
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory),
            'current_portfolio_value': self.portfolio_value
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            import pickle
            
            model_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'training_history': self.training_history,
                'hyperparameters': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.epsilon = model_data['epsilon']
            self.training_history = model_data.get('training_history', [])
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
