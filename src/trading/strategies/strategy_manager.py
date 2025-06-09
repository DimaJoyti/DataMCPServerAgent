"""
Strategy Manager

Coordinates multiple algorithmic trading strategies, manages their execution,
performance tracking, and risk allocation.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base_strategy import (
    EnhancedBaseStrategy, StrategySignal, StrategySignalData, 
    StrategyState, StrategyMetrics
)
from ..core.base_models import MarketData
from ..core.enums import OrderSide


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration."""
    strategy_id: str
    allocation_percentage: float  # 0.0 to 1.0
    max_allocation: Decimal
    current_allocation: Decimal = Decimal('0')
    is_active: bool = True
    priority: int = 1  # 1 = highest priority


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""
    total_pnl: Decimal = Decimal('0')
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    active_strategies: int = 0
    total_allocation: Decimal = Decimal('0')
    
    def update_from_strategies(self, strategies: Dict[str, EnhancedBaseStrategy]) -> None:
        """Update portfolio metrics from strategy metrics."""
        self.total_pnl = Decimal('0')
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.active_strategies = 0
        
        for strategy in strategies.values():
            if strategy.state == StrategyState.ACTIVE:
                self.active_strategies += 1
            
            self.total_pnl += strategy.metrics.total_pnl
            self.total_trades += strategy.metrics.total_trades
            self.winning_trades += strategy.metrics.winning_trades
            self.losing_trades += strategy.metrics.losing_trades
        
        # Calculate derived metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Calculate profit factor (simplified)
        total_wins = sum(s.metrics.avg_win * s.metrics.winning_trades for s in strategies.values())
        total_losses = sum(s.metrics.avg_loss * s.metrics.losing_trades for s in strategies.values())
        self.profit_factor = float(total_wins / total_losses) if total_losses > 0 else float('inf')


class StrategyManager:
    """Manages multiple trading strategies with allocation and risk management."""
    
    def __init__(
        self,
        total_capital: Decimal,
        max_strategies: int = 10,
        rebalance_interval: int = 3600,  # seconds
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        self.total_capital = total_capital
        self.max_strategies = max_strategies
        self.rebalance_interval = rebalance_interval
        self.risk_parameters = risk_parameters or {}
        
        # Strategy management
        self.strategies: Dict[str, EnhancedBaseStrategy] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.portfolio_metrics = PortfolioMetrics()
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.price_history: Dict[str, pd.DataFrame] = {}
        
        # Risk management
        self.max_portfolio_drawdown = self.risk_parameters.get('max_portfolio_drawdown', Decimal('0.15'))
        self.max_correlation_threshold = self.risk_parameters.get('max_correlation_threshold', 0.7)
        self.min_strategy_allocation = self.risk_parameters.get('min_strategy_allocation', Decimal('0.05'))
        
        # State management
        self.is_running = False
        self.last_rebalance = datetime.now()
        
        # Logging
        self.logger = logging.getLogger("StrategyManager")
    
    async def add_strategy(
        self, 
        strategy: EnhancedBaseStrategy, 
        allocation_percentage: float,
        priority: int = 1
    ) -> bool:
        """Add a strategy to the manager."""
        try:
            if len(self.strategies) >= self.max_strategies:
                self.logger.warning(f"Maximum strategies ({self.max_strategies}) reached")
                return False
            
            if allocation_percentage <= 0 or allocation_percentage > 1:
                self.logger.error(f"Invalid allocation percentage: {allocation_percentage}")
                return False
            
            # Check if total allocation would exceed 100%
            total_allocation = sum(alloc.allocation_percentage for alloc in self.allocations.values())
            if total_allocation + allocation_percentage > 1.0:
                self.logger.error(f"Total allocation would exceed 100%: {total_allocation + allocation_percentage}")
                return False
            
            # Add strategy
            self.strategies[strategy.strategy_id] = strategy
            
            # Create allocation
            max_allocation = self.total_capital * Decimal(str(allocation_percentage))
            self.allocations[strategy.strategy_id] = StrategyAllocation(
                strategy_id=strategy.strategy_id,
                allocation_percentage=allocation_percentage,
                max_allocation=max_allocation,
                priority=priority
            )
            
            self.logger.info(f"Added strategy {strategy.name} with {allocation_percentage*100:.1f}% allocation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy.strategy_id}: {e}")
            return False
    
    async def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the manager."""
        try:
            if strategy_id not in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} not found")
                return False
            
            # Stop strategy if running
            strategy = self.strategies[strategy_id]
            if strategy.state == StrategyState.ACTIVE:
                await strategy.stop()
            
            # Remove from collections
            del self.strategies[strategy_id]
            del self.allocations[strategy_id]
            
            self.logger.info(f"Removed strategy {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing strategy {strategy_id}: {e}")
            return False
    
    async def start(self) -> None:
        """Start the strategy manager."""
        try:
            self.is_running = True
            
            # Start all strategies
            for strategy in self.strategies.values():
                if self.allocations[strategy.strategy_id].is_active:
                    await strategy.start()
            
            self.logger.info(f"Strategy Manager started with {len(self.strategies)} strategies")
            
            # Start background tasks
            asyncio.create_task(self._rebalance_loop())
            asyncio.create_task(self._monitor_risk())
            
        except Exception as e:
            self.logger.error(f"Error starting Strategy Manager: {e}")
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the strategy manager."""
        try:
            self.is_running = False
            
            # Stop all strategies
            for strategy in self.strategies.values():
                await strategy.stop()
            
            self.logger.info("Strategy Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Strategy Manager: {e}")
    
    async def update_market_data(self, symbol: str, market_data: MarketData) -> None:
        """Update market data for all strategies."""
        try:
            self.market_data_cache[symbol] = market_data
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = pd.DataFrame()
            
            # Add new data point
            new_data = pd.DataFrame({
                'timestamp': [market_data.timestamp],
                'open': [market_data.open_price],
                'high': [market_data.high_price],
                'low': [market_data.low_price],
                'close': [market_data.price],
                'volume': [market_data.volume or 0]
            })
            
            self.price_history[symbol] = pd.concat([self.price_history[symbol], new_data]).tail(1000)
            
            # Update strategies
            for strategy in self.strategies.values():
                if symbol in strategy.symbols:
                    await strategy.update_market_data(symbol, self.price_history[symbol])
            
        except Exception as e:
            self.logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def process_signals(self) -> List[Dict[str, Any]]:
        """Process signals from all active strategies."""
        orders = []
        
        try:
            for strategy in self.strategies.values():
                if (strategy.state != StrategyState.ACTIVE or 
                    not self.allocations[strategy.strategy_id].is_active):
                    continue
                
                for symbol in strategy.symbols:
                    if symbol not in self.market_data_cache:
                        continue
                    
                    # Generate signal
                    signal = await strategy.generate_signal(symbol, self.market_data_cache[symbol])
                    
                    if signal and signal.signal != StrategySignal.HOLD:
                        # Check allocation limits
                        allocation = self.allocations[strategy.strategy_id]
                        if allocation.current_allocation >= allocation.max_allocation:
                            continue
                        
                        # Process signal
                        order = await strategy.process_signal(symbol, signal)
                        
                        if order:
                            # Add strategy allocation info
                            order['allocation_used'] = allocation.current_allocation
                            order['max_allocation'] = allocation.max_allocation
                            order['strategy_priority'] = allocation.priority
                            
                            orders.append(order)
            
            # Sort orders by priority and signal strength
            orders.sort(key=lambda x: (x['strategy_priority'], -x['signal_strength']))
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            return []
    
    async def _rebalance_loop(self) -> None:
        """Background task for periodic rebalancing."""
        while self.is_running:
            try:
                await asyncio.sleep(self.rebalance_interval)
                
                if datetime.now() - self.last_rebalance > timedelta(seconds=self.rebalance_interval):
                    await self._rebalance_strategies()
                    self.last_rebalance = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error in rebalance loop: {e}")
    
    async def _rebalance_strategies(self) -> None:
        """Rebalance strategy allocations based on performance."""
        try:
            # Update portfolio metrics
            self.portfolio_metrics.update_from_strategies(self.strategies)
            
            # Calculate performance scores
            performance_scores = {}
            for strategy_id, strategy in self.strategies.items():
                if strategy.metrics.total_trades > 10:  # Minimum trades for evaluation
                    # Simple performance score (can be enhanced)
                    score = (
                        strategy.metrics.win_rate * 0.3 +
                        min(strategy.metrics.profit_factor / 2.0, 1.0) * 0.4 +
                        max(1.0 - float(strategy.metrics.max_drawdown), 0.0) * 0.3
                    )
                    performance_scores[strategy_id] = score
                else:
                    performance_scores[strategy_id] = 0.5  # Neutral score for new strategies
            
            # Adjust allocations based on performance (simplified)
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for strategy_id, allocation in self.allocations.items():
                    if strategy_id in performance_scores:
                        new_percentage = performance_scores[strategy_id] / total_score
                        
                        # Ensure minimum allocation
                        new_percentage = max(new_percentage, float(self.min_strategy_allocation))
                        
                        # Update allocation
                        allocation.allocation_percentage = new_percentage
                        allocation.max_allocation = self.total_capital * Decimal(str(new_percentage))
            
            self.logger.info("Strategy rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing strategies: {e}")
    
    async def _monitor_risk(self) -> None:
        """Background task for risk monitoring."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check portfolio drawdown
                if self.portfolio_metrics.max_drawdown > self.max_portfolio_drawdown:
                    self.logger.warning(f"Portfolio drawdown exceeded limit: {self.portfolio_metrics.max_drawdown}")
                    await self._emergency_stop()
                
                # Check strategy correlations (simplified)
                await self._check_strategy_correlations()
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
    
    async def _emergency_stop(self) -> None:
        """Emergency stop all strategies."""
        self.logger.critical("Emergency stop triggered!")
        
        for strategy in self.strategies.values():
            await strategy.stop()
        
        # Disable all allocations
        for allocation in self.allocations.values():
            allocation.is_active = False
    
    async def _check_strategy_correlations(self) -> None:
        """Check correlations between strategies (simplified implementation)."""
        try:
            # This is a simplified correlation check
            # In production, you'd want more sophisticated correlation analysis
            
            active_strategies = [s for s in self.strategies.values() if s.state == StrategyState.ACTIVE]
            
            if len(active_strategies) < 2:
                return
            
            # Check if too many strategies are generating similar signals
            recent_signals = {}
            for strategy in active_strategies:
                if strategy.signals_history:
                    recent_signal = strategy.signals_history[-1]
                    signal_type = recent_signal.signal
                    
                    if signal_type not in recent_signals:
                        recent_signals[signal_type] = 0
                    recent_signals[signal_type] += 1
            
            # If more than 70% of strategies have the same signal, reduce confidence
            total_strategies = len(active_strategies)
            for signal_type, count in recent_signals.items():
                if count / total_strategies > self.max_correlation_threshold:
                    self.logger.warning(f"High correlation detected: {count}/{total_strategies} strategies have {signal_type} signal")
            
        except Exception as e:
            self.logger.error(f"Error checking strategy correlations: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        self.portfolio_metrics.update_from_strategies(self.strategies)
        
        strategy_summaries = {}
        for strategy_id, strategy in self.strategies.items():
            allocation = self.allocations[strategy_id]
            strategy_summaries[strategy_id] = {
                **strategy.get_performance_summary(),
                'allocation_percentage': allocation.allocation_percentage,
                'max_allocation': float(allocation.max_allocation),
                'current_allocation': float(allocation.current_allocation),
                'is_active': allocation.is_active,
                'priority': allocation.priority
            }
        
        return {
            'portfolio_metrics': {
                'total_pnl': float(self.portfolio_metrics.total_pnl),
                'total_trades': self.portfolio_metrics.total_trades,
                'win_rate': self.portfolio_metrics.win_rate,
                'profit_factor': self.portfolio_metrics.profit_factor,
                'max_drawdown': float(self.portfolio_metrics.max_drawdown),
                'active_strategies': self.portfolio_metrics.active_strategies,
                'total_allocation': float(self.portfolio_metrics.total_allocation)
            },
            'strategies': strategy_summaries,
            'total_capital': float(self.total_capital),
            'is_running': self.is_running,
            'last_rebalance': self.last_rebalance.isoformat()
        }
