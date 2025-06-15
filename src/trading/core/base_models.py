"""
Base models for the institutional trading system.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .enums import (
    AssetClass,
    Currency,
    Exchange,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskLevel,
    StrategyType,
    TimeInForce,
)


@dataclass
class BaseOrder:
    """Base order model for all order types."""

    # Core order fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None

    # Order management
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.DAY
    exchange: Optional[Exchange] = None
    currency: Currency = Currency.USD

    # Execution details
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    account_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to be filled."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity

    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value of the order."""
        if self.price is not None:
            return self.quantity * self.price
        return None


@dataclass
class BasePosition:
    """Base position model for portfolio management."""

    # Core position fields
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    market_price: Optional[Decimal] = None

    # Position details
    side: OrderSide = OrderSide.BUY  # LONG or SHORT
    asset_class: AssetClass = AssetClass.EQUITY
    exchange: Optional[Exchange] = None
    currency: Currency = Currency.USD

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")

    # Risk metrics
    risk_level: RiskLevel = RiskLevel.MEDIUM
    var_contribution: Optional[Decimal] = None
    beta: Optional[float] = None

    # Timestamps
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    account_id: Optional[str] = None

    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate current market value."""
        if self.market_price is not None:
            return abs(self.quantity) * self.market_price
        return None

    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis."""
        return abs(self.quantity) * self.average_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.quantity == 0


@dataclass
class BaseTrade:
    """Base trade model for execution tracking."""

    # Core trade fields
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")

    # Trade details
    exchange: Optional[Exchange] = None
    currency: Currency = Currency.USD
    commission: Decimal = Decimal("0")

    # Execution details
    execution_id: Optional[str] = None
    counterparty: Optional[str] = None
    settlement_date: Optional[datetime] = None

    # Timestamps
    executed_at: datetime = field(default_factory=datetime.utcnow)
    reported_at: Optional[datetime] = None

    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    account_id: Optional[str] = None

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.quantity * self.price

    @property
    def gross_value(self) -> Decimal:
        """Calculate gross value (including commission)."""
        return self.notional_value + self.commission


class BaseStrategy(ABC):
    """Base strategy class for all trading strategies."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        strategy_type: StrategyType,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.strategy_type = strategy_type
        self.parameters = parameters or {}
        self.is_active = False
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Performance tracking
        self.total_pnl = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk metrics
        self.max_drawdown = Decimal("0")
        self.sharpe_ratio: Optional[float] = None
        self.var_limit = Decimal("0")

    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on market data."""
        pass

    @abstractmethod
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trading signal before execution."""
        pass

    @abstractmethod
    async def calculate_position_size(self, signal: Dict[str, Any]) -> Decimal:
        """Calculate position size for a trading signal."""
        pass

    def update_performance(self, trade: BaseTrade) -> None:
        """Update strategy performance metrics."""
        self.total_trades += 1
        self.updated_at = datetime.utcnow()

        # Calculate P&L (simplified)
        if trade.side == OrderSide.BUY:
            # For buy trades, we'll need the corresponding sell to calculate P&L
            pass
        else:
            # For sell trades, calculate against average cost
            pass

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> Optional[float]:
        """Calculate profit factor (gross profit / gross loss)."""
        # Implementation depends on detailed P&L tracking
        return None


@dataclass
class MarketData:
    """Market data model for real-time and historical data."""

    symbol: str
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    open: Optional[Decimal] = None
    close: Optional[Decimal] = None

    # Level 2 data
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None

    # Metadata
    exchange: Optional[Exchange] = None
    data_source: Optional[str] = None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return float(self.spread / self.mid_price * 10000)
        return None
