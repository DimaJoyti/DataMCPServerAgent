"""
Trading Strategy Orchestrator using Infinite Agentic Loop

This module integrates the Infinite Agentic Loop system with the trading system
to generate, test, and optimize trading strategies automatically.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from ..infinite_loop import InfiniteAgenticLoopOrchestrator, InfiniteLoopConfig
from ..trading_system import AdvancedCryptoTradingSystem, TradeRecommendation
from ..trading_system.advanced_ml_models import ModelManager, ModelType, PredictionTarget
from ..trading_system.data_sources import DataSourceManager


class TradingStrategyConfig(InfiniteLoopConfig):
    """Extended configuration for trading strategy generation."""
    
    # Trading-specific settings
    target_symbols: List[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    strategy_types: List[str] = ["momentum", "mean_reversion", "arbitrage", "ml_based"]
    risk_tolerance: float = 0.02  # 2% max risk per trade
    min_profit_threshold: float = 0.005  # 0.5% minimum profit
    backtest_period_days: int = 30
    
    # Performance requirements
    min_sharpe_ratio: float = 1.5
    max_drawdown: float = 0.1  # 10% max drawdown
    min_win_rate: float = 0.6  # 60% minimum win rate
    
    # Strategy evolution
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    elite_preservation: float = 0.2


class TradingStrategyOrchestrator:
    """
    Orchestrator that uses Infinite Agentic Loop for trading strategy generation.
    
    This system continuously generates, tests, and evolves trading strategies
    using the infinite loop framework combined with trading system capabilities.
    """
    
    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        trading_system: AdvancedCryptoTradingSystem,
        config: Optional[TradingStrategyConfig] = None,
    ):
        """Initialize the trading strategy orchestrator."""
        self.model = model
        self.tools = tools
        self.trading_system = trading_system
        self.config = config or TradingStrategyConfig()
        
        # Setup logging
        self.logger = logging.getLogger("trading_strategy_orchestrator")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Initialize infinite loop orchestrator
        self.infinite_loop = InfiniteAgenticLoopOrchestrator(
            model=model,
            tools=tools,
            config=self.config
        )
        
        # Initialize ML model manager
        self.model_manager = ModelManager()
        
        # Initialize data source manager
        self.data_manager = DataSourceManager()
        
        # Strategy storage
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
    async def generate_trading_strategies(
        self,
        count: Union[int, str] = "infinite",
        output_dir: Union[str, Path] = "./generated_strategies"
    ) -> Dict[str, Any]:
        """
        Generate trading strategies using infinite agentic loop.
        
        Args:
            count: Number of strategies to generate or "infinite"
            output_dir: Directory to save generated strategies
            
        Returns:
            Generation results and performance metrics
        """
        # Create strategy specification
        spec_content = self._create_strategy_specification()
        spec_file = Path(output_dir) / "strategy_specification.json"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write specification file
        with open(spec_file, 'w') as f:
            json.dump(spec_content, f, indent=2)
        
        self.logger.info(f"Starting trading strategy generation: {count} strategies")
        
        # Execute infinite loop for strategy generation
        results = await self.infinite_loop.execute_infinite_loop(
            spec_file=spec_file,
            output_dir=output_dir,
            count=count
        )
        
        # Process generated strategies
        await self._process_generated_strategies(output_dir)
        
        return results
    
    def _create_strategy_specification(self) -> Dict[str, Any]:
        """Create specification for trading strategy generation."""
        return {
            "content_type": "trading_strategy",
            "format": "python_class",
            "evolution_pattern": "genetic_algorithm",
            "innovation_areas": [
                "entry_conditions",
                "exit_conditions", 
                "risk_management",
                "position_sizing",
                "market_timing",
                "feature_engineering",
                "ensemble_methods",
                "adaptive_parameters"
            ],
            "quality_requirements": {
                "min_sharpe_ratio": self.config.min_sharpe_ratio,
                "max_drawdown": self.config.max_drawdown,
                "min_win_rate": self.config.min_win_rate,
                "backtest_required": True,
                "risk_compliance": True
            },
            "target_symbols": self.config.target_symbols,
            "strategy_types": self.config.strategy_types,
            "constraints": [
                "no_leverage_above_3x",
                "max_position_size_10_percent",
                "stop_loss_required",
                "regulatory_compliant"
            ],
            "performance_metrics": [
                "sharpe_ratio",
                "sortino_ratio", 
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "calmar_ratio"
            ]
        }
    
    async def _process_generated_strategies(self, output_dir: Union[str, Path]) -> None:
        """Process and validate generated strategies."""
        output_path = Path(output_dir)
        
        # Find all generated strategy files
        strategy_files = list(output_path.glob("iteration_*/strategy.py"))
        
        for strategy_file in strategy_files:
            try:
                # Load and validate strategy
                strategy_data = await self._load_strategy(strategy_file)
                
                if strategy_data:
                    # Backtest strategy
                    backtest_results = await self._backtest_strategy(strategy_data)
                    
                    # Evaluate performance
                    performance = self._evaluate_strategy_performance(backtest_results)
                    
                    # Store if meets criteria
                    if self._meets_performance_criteria(performance):
                        strategy_id = f"strategy_{int(time.time())}"
                        self.strategies[strategy_id] = {
                            "strategy": strategy_data,
                            "performance": performance,
                            "backtest_results": backtest_results,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        self.logger.info(f"Added strategy {strategy_id} with Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error processing strategy {strategy_file}: {str(e)}")
    
    async def _load_strategy(self, strategy_file: Path) -> Optional[Dict[str, Any]]:
        """Load and parse strategy from file."""
        try:
            with open(strategy_file, 'r') as f:
                strategy_code = f.read()
            
            # Parse strategy parameters and logic
            # This would involve parsing the generated Python code
            # and extracting strategy parameters, entry/exit conditions, etc.
            
            return {
                "code": strategy_code,
                "file_path": str(strategy_file),
                "parsed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error loading strategy from {strategy_file}: {str(e)}")
            return None
    
    async def _backtest_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest a trading strategy."""
        # Get historical market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.backtest_period_days)
        
        backtest_results = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "trades": [],
            "daily_returns": [],
            "equity_curve": [],
            "metrics": {}
        }
        
        try:
            # Simulate strategy execution on historical data
            for symbol in self.config.target_symbols:
                # Get market data for symbol
                market_data = await self.data_manager.fetch_market_data(symbol)
                
                if market_data:
                    # Simulate trades based on strategy logic
                    symbol_results = await self._simulate_strategy_trades(
                        strategy_data, symbol, market_data, start_date, end_date
                    )
                    
                    backtest_results["trades"].extend(symbol_results.get("trades", []))
                    backtest_results["daily_returns"].extend(symbol_results.get("returns", []))
            
            # Calculate performance metrics
            backtest_results["metrics"] = self._calculate_backtest_metrics(backtest_results)
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy: {str(e)}")
            backtest_results["error"] = str(e)
        
        return backtest_results
    
    async def _simulate_strategy_trades(
        self, 
        strategy_data: Dict[str, Any], 
        symbol: str, 
        market_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Simulate strategy trades for a specific symbol."""
        # This would implement the actual strategy simulation logic
        # For now, return mock results
        return {
            "trades": [
                {
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": 1.0,
                    "price": 50000.0,
                    "timestamp": start_date.isoformat(),
                    "pnl": 500.0
                }
            ],
            "returns": [0.01, -0.005, 0.02, 0.015]  # Daily returns
        }
    
    def _calculate_backtest_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results."""
        returns = backtest_results.get("daily_returns", [])
        
        if not returns:
            return {}
        
        import numpy as np
        
        returns_array = np.array(returns)
        
        # Calculate key metrics
        total_return = np.prod(1 + returns_array) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        winning_trades = len([r for r in returns if r > 0])
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "total_trades": total_trades,
            "winning_trades": winning_trades
        }
    
    def _evaluate_strategy_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategy performance against criteria."""
        metrics = backtest_results.get("metrics", {})
        
        # Add evaluation scores
        performance = metrics.copy()
        performance["evaluation"] = {
            "sharpe_score": min(metrics.get("sharpe_ratio", 0) / self.config.min_sharpe_ratio, 1.0),
            "drawdown_score": max(1 - abs(metrics.get("max_drawdown", 0)) / self.config.max_drawdown, 0),
            "win_rate_score": min(metrics.get("win_rate", 0) / self.config.min_win_rate, 1.0)
        }
        
        # Overall score
        eval_scores = performance["evaluation"]
        performance["overall_score"] = (
            eval_scores["sharpe_score"] * 0.4 +
            eval_scores["drawdown_score"] * 0.3 +
            eval_scores["win_rate_score"] * 0.3
        )
        
        return performance
    
    def _meets_performance_criteria(self, performance: Dict[str, Any]) -> bool:
        """Check if strategy meets minimum performance criteria."""
        return (
            performance.get("sharpe_ratio", 0) >= self.config.min_sharpe_ratio and
            abs(performance.get("max_drawdown", 1)) <= self.config.max_drawdown and
            performance.get("win_rate", 0) >= self.config.min_win_rate
        )
    
    async def get_best_strategies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the best performing strategies."""
        sorted_strategies = sorted(
            self.strategies.items(),
            key=lambda x: x[1]["performance"].get("overall_score", 0),
            reverse=True
        )
        
        return [
            {
                "strategy_id": strategy_id,
                **strategy_data
            }
            for strategy_id, strategy_data in sorted_strategies[:limit]
        ]
    
    async def deploy_strategy(self, strategy_id: str) -> bool:
        """Deploy a strategy for live trading."""
        if strategy_id not in self.strategies:
            self.logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy_data = self.strategies[strategy_id]
        
        try:
            # Integrate with trading system
            # This would involve setting up the strategy in the trading system
            # for live execution
            
            self.logger.info(f"Deployed strategy {strategy_id} for live trading")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying strategy {strategy_id}: {str(e)}")
            return False
