"""
Trading Strategies Framework

Multi-strategy execution framework for institutional trading.
Implements sophisticated algorithmic trading strategies including:
- Momentum strategies (RSI, MACD, Moving Average Crossover)
- Mean reversion strategies (Bollinger Bands, Z-Score)
- Arbitrage strategies (Pairs Trading, Statistical Arbitrage)
- Machine Learning strategies (Random Forest, LSTM)
"""

from .arbitrage_strategies import PairsTradingStrategy, StatisticalArbitrageStrategy
from .backtesting import BacktestingEngine
from .base_strategy import EnhancedBaseStrategy, StrategySignal, StrategyState
from .mean_reversion_strategies import BollingerBandsStrategy, ZScoreStrategy
from .ml_strategies import LSTMStrategy, RandomForestStrategy
from .momentum_strategies import MACDStrategy, MovingAverageCrossoverStrategy, RSIStrategy
from .strategy_manager import StrategyManager
from .technical_indicators import TechnicalIndicators

__all__ = [
    "EnhancedBaseStrategy",
    "StrategySignal",
    "StrategyState",
    "TechnicalIndicators",
    "RSIStrategy",
    "MACDStrategy",
    "MovingAverageCrossoverStrategy",
    "BollingerBandsStrategy",
    "ZScoreStrategy",
    "PairsTradingStrategy",
    "StatisticalArbitrageStrategy",
    "RandomForestStrategy",
    "LSTMStrategy",
    "StrategyManager",
    "BacktestingEngine",
]
