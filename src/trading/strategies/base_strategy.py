"""
Enhanced Base Strategy Framework

Provides a comprehensive base class for all algorithmic trading strategies
with advanced features including signal generation, risk management,
performance tracking, and backtesting capabilities.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.base_models import BaseStrategy, MarketData
from ..core.enums import OrderSide, OrderType, StrategyType


class StrategySignal(Enum):
    """Enhanced strategy signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class StrategyState(Enum):
    """Strategy execution states."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    BACKTESTING = "backtesting"


@dataclass
class StrategyPosition:
    """Represents a strategy position."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    entry_time: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def update_metrics(self, trade_pnl: Decimal) -> None:
        """Update metrics with new trade."""
        self.total_trades += 1
        self.total_pnl += trade_pnl

        if trade_pnl > 0:
            self.winning_trades += 1
            self.avg_win = (
                self.avg_win * (self.winning_trades - 1) + trade_pnl
            ) / self.winning_trades
        else:
            self.losing_trades += 1
            self.avg_loss = (
                self.avg_loss * (self.losing_trades - 1) + abs(trade_pnl)
            ) / self.losing_trades

        # Update win rate
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Update profit factor
        total_wins = self.avg_win * self.winning_trades
        total_losses = self.avg_loss * self.losing_trades
        self.profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")


@dataclass
class StrategySignalData:
    """Strategy signal with metadata."""

    signal: StrategySignal
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    price: Decimal
    volume: Optional[Decimal] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedBaseStrategy(BaseStrategy):
    """Enhanced base strategy class with advanced features."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        strategy_type: StrategyType,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy_id, name, strategy_type, parameters)

        self.symbols = symbols
        self.timeframe = timeframe
        self.risk_parameters = risk_parameters or {}

        # Strategy state
        self.state = StrategyState.INACTIVE
        self.positions: Dict[str, StrategyPosition] = {}
        self.metrics = StrategyMetrics()

        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.signals_history: List[StrategySignalData] = []

        # Risk management
        self.max_position_size = self.risk_parameters.get("max_position_size", Decimal("1000"))
        self.max_drawdown_limit = self.risk_parameters.get("max_drawdown_limit", Decimal("0.1"))
        self.stop_loss_pct = self.risk_parameters.get("stop_loss_pct", 0.02)
        self.take_profit_pct = self.risk_parameters.get("take_profit_pct", 0.04)

        # Logging
        self.logger = logging.getLogger(f"Strategy.{self.name}")

    @abstractmethod
    async def generate_signal(
        self, symbol: str, market_data: MarketData
    ) -> Optional[StrategySignalData]:
        """Generate trading signal for a symbol.

        Args:
            symbol: Trading symbol
            market_data: Current market data

        Returns:
            Strategy signal data or None
        """
        pass

    @abstractmethod
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size for a signal.

        Args:
            symbol: Trading symbol
            signal: Strategy signal data

        Returns:
            Position size
        """
        pass

    async def start(self) -> None:
        """Start the strategy."""
        self.state = StrategyState.ACTIVE
        self.is_active = True
        self.logger.info(f"Strategy {self.name} started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self.state = StrategyState.INACTIVE
        self.is_active = False
        self.logger.info(f"Strategy {self.name} stopped")

    async def pause(self) -> None:
        """Pause the strategy."""
        self.state = StrategyState.PAUSED
        self.logger.info(f"Strategy {self.name} paused")

    async def resume(self) -> None:
        """Resume the strategy."""
        self.state = StrategyState.ACTIVE
        self.logger.info(f"Strategy {self.name} resumed")

    async def update_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Update market data for a symbol."""
        self.market_data[symbol] = data

    async def process_signal(
        self, symbol: str, signal: StrategySignalData
    ) -> Optional[Dict[str, Any]]:
        """Process a trading signal and generate orders.

        Args:
            symbol: Trading symbol
            signal: Strategy signal data

        Returns:
            Order data or None
        """
        if self.state != StrategyState.ACTIVE:
            return None

        # Check risk limits
        if not await self._check_risk_limits(symbol, signal):
            return None

        # Calculate position size
        position_size = await self.calculate_position_size(symbol, signal)

        if position_size <= 0:
            return None

        # Generate order based on signal
        order_data = await self._generate_order(symbol, signal, position_size)

        # Store signal
        self.signals_history.append(signal)

        return order_data

    async def _check_risk_limits(self, symbol: str, signal: StrategySignalData) -> bool:
        """Check if signal passes risk limits."""
        # Check maximum drawdown
        if self.metrics.max_drawdown > self.max_drawdown_limit:
            self.logger.warning(f"Maximum drawdown exceeded: {self.metrics.max_drawdown}")
            return False

        # Check position limits
        current_position = self.positions.get(symbol)
        if current_position and abs(current_position.quantity) >= self.max_position_size:
            self.logger.warning(f"Position size limit reached for {symbol}")
            return False

        return True

    async def _generate_order(
        self, symbol: str, signal: StrategySignalData, position_size: Decimal
    ) -> Dict[str, Any]:
        """Generate order data from signal."""
        side = (
            OrderSide.BUY
            if signal.signal
            in [StrategySignal.STRONG_BUY, StrategySignal.BUY, StrategySignal.WEAK_BUY]
            else OrderSide.SELL
        )

        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None

        if side == OrderSide.BUY:
            stop_loss = signal.price * (1 - Decimal(str(self.stop_loss_pct)))
            take_profit = signal.price * (1 + Decimal(str(self.take_profit_pct)))
        else:
            stop_loss = signal.price * (1 + Decimal(str(self.stop_loss_pct)))
            take_profit = signal.price * (1 - Decimal(str(self.take_profit_pct)))

        return {
            "symbol": symbol,
            "side": side,
            "order_type": OrderType.MARKET,
            "quantity": position_size,
            "price": signal.price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategy_id": self.strategy_id,
            "signal_strength": signal.strength,
            "signal_confidence": signal.confidence,
            "timestamp": signal.timestamp,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "state": self.state.value,
            "total_trades": self.metrics.total_trades,
            "winning_trades": self.metrics.winning_trades,
            "losing_trades": self.metrics.losing_trades,
            "win_rate": self.metrics.win_rate,
            "total_pnl": float(self.metrics.total_pnl),
            "max_drawdown": float(self.metrics.max_drawdown),
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "profit_factor": self.metrics.profit_factor,
            "avg_win": float(self.metrics.avg_win),
            "avg_loss": float(self.metrics.avg_loss),
            "active_positions": len(self.positions),
            "symbols": self.symbols,
            "timeframe": self.timeframe,
        }
