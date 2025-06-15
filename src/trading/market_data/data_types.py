"""
Market data types and models for institutional trading.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Union

from ..core.enums import Exchange


class MarketDataType(Enum):
    """Types of market data."""

    TICK = "TICK"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    DEPTH = "DEPTH"
    OHLCV = "OHLCV"
    NEWS = "NEWS"
    FUNDAMENTAL = "FUNDAMENTAL"
    ORDERBOOK_L1 = "ORDERBOOK_L1"
    ORDERBOOK_L2 = "ORDERBOOK_L2"
    ORDERBOOK_L3 = "ORDERBOOK_L3"


class FeedStatus(Enum):
    """Market data feed status."""

    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    STALE = "STALE"


class TradeCondition(Enum):
    """Trade execution conditions."""

    REGULAR = "REGULAR"
    OPENING = "OPENING"
    CLOSING = "CLOSING"
    HALT = "HALT"
    CROSS = "CROSS"
    BLOCK = "BLOCK"
    SWEEP = "SWEEP"


@dataclass
class Tick:
    """Basic tick data structure."""

    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    exchange: Exchange
    tick_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Metadata
    sequence_number: Optional[int] = None
    feed_timestamp: Optional[datetime] = None
    latency_us: Optional[int] = None  # Microseconds

    def __post_init__(self):
        """Calculate latency if feed timestamp is available."""
        if self.feed_timestamp and not self.latency_us:
            delta = self.timestamp - self.feed_timestamp
            self.latency_us = int(delta.total_seconds() * 1_000_000)


@dataclass
class Quote:
    """Bid/Ask quote data."""

    symbol: str
    timestamp: datetime
    bid_price: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    exchange: Exchange = Exchange.NYSE
    quote_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Quote metadata
    bid_count: Optional[int] = None  # Number of orders at bid
    ask_count: Optional[int] = None  # Number of orders at ask
    sequence_number: Optional[int] = None
    feed_timestamp: Optional[datetime] = None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return float(self.spread / self.mid_price * 10000)
        return None


@dataclass
class Trade:
    """Trade execution data."""

    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    exchange: Exchange
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Trade details
    condition: TradeCondition = TradeCondition.REGULAR
    buyer_initiated: Optional[bool] = None  # True if buyer initiated
    sequence_number: Optional[int] = None
    feed_timestamp: Optional[datetime] = None

    # Trade classification
    aggressive_side: Optional[str] = None  # "BUY" or "SELL"
    trade_type: Optional[str] = None  # "MARKET", "LIMIT", etc.

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value."""
        return self.price * self.size


@dataclass
class OrderBookLevel:
    """Single level in order book."""

    price: Decimal
    size: Decimal
    count: int = 1  # Number of orders at this level

    def __post_init__(self):
        """Validate level data."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size < 0:
            raise ValueError("Size cannot be negative")
        if self.count < 0:
            raise ValueError("Count cannot be negative")


@dataclass
class OrderBook:
    """Level 2 order book data."""

    symbol: str
    timestamp: datetime
    exchange: Exchange
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    sequence_number: Optional[int] = None

    def __post_init__(self):
        """Sort bids and asks."""
        # Sort bids descending (highest first)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        # Sort asks ascending (lowest first)
        self.asks.sort(key=lambda x: x.price)

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    def get_depth(self, side: str, levels: int = 5) -> List[OrderBookLevel]:
        """Get market depth for specified side."""
        if side.upper() == "BID":
            return self.bids[:levels]
        elif side.upper() == "ASK":
            return self.asks[:levels]
        else:
            raise ValueError("Side must be 'BID' or 'ASK'")

    def get_total_size(self, side: str, levels: int = 5) -> Decimal:
        """Get total size for specified side and levels."""
        depth = self.get_depth(side, levels)
        return sum(level.size for level in depth)

    def get_weighted_price(self, side: str, levels: int = 5) -> Optional[Decimal]:
        """Get size-weighted average price."""
        depth = self.get_depth(side, levels)
        if not depth:
            return None

        total_value = sum(level.price * level.size for level in depth)
        total_size = sum(level.size for level in depth)

        if total_size > 0:
            return total_value / total_size
        return None


@dataclass
class OHLCV:
    """OHLCV bar data."""

    symbol: str
    timestamp: datetime
    timeframe: str  # "1m", "5m", "1h", "1d", etc.
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    exchange: Exchange

    # Additional metrics
    vwap: Optional[Decimal] = None  # Volume weighted average price
    trade_count: Optional[int] = None

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def price_range(self) -> Decimal:
        """Calculate price range (high - low)."""
        return self.high - self.low

    @property
    def body_size(self) -> Decimal:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open


@dataclass
class MarketDataSnapshot:
    """Complete market data snapshot for a symbol."""

    symbol: str
    timestamp: datetime
    exchange: Exchange

    # Latest data
    last_trade: Optional[Trade] = None
    last_quote: Optional[Quote] = None
    order_book: Optional[OrderBook] = None

    # Daily statistics
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    vwap: Optional[Decimal] = None

    # Derived metrics
    change: Optional[Decimal] = None
    change_percent: Optional[float] = None

    @property
    def current_price(self) -> Optional[Decimal]:
        """Get current price from last trade or mid quote."""
        if self.last_trade:
            return self.last_trade.price
        elif self.last_quote and self.last_quote.mid_price:
            return self.last_quote.mid_price
        return None

    def update_daily_stats(self) -> None:
        """Update daily statistics."""
        current = self.current_price
        if current and self.open_price:
            self.change = current - self.open_price
            self.change_percent = float(self.change / self.open_price * 100)


@dataclass
class FeedMetrics:
    """Market data feed performance metrics."""

    feed_name: str
    symbol: str
    timestamp: datetime

    # Latency metrics (microseconds)
    min_latency_us: int = 0
    max_latency_us: int = 0
    avg_latency_us: int = 0
    p99_latency_us: int = 0

    # Throughput metrics
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0

    # Quality metrics
    total_messages: int = 0
    dropped_messages: int = 0
    out_of_order_messages: int = 0
    duplicate_messages: int = 0

    # Connection metrics
    connection_uptime_seconds: float = 0.0
    reconnection_count: int = 0
    last_reconnection: Optional[datetime] = None

    @property
    def drop_rate(self) -> float:
        """Calculate message drop rate."""
        if self.total_messages > 0:
            return self.dropped_messages / self.total_messages
        return 0.0

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        if self.total_messages == 0:
            return 0.0

        # Factors: drop rate, out of order rate, latency
        drop_penalty = self.drop_rate
        ooo_penalty = self.out_of_order_messages / self.total_messages
        latency_penalty = min(1.0, self.avg_latency_us / 10000)  # Normalize to 10ms

        return max(0.0, 1.0 - drop_penalty - ooo_penalty - latency_penalty)


# Type aliases for convenience
TickData = Union[Tick, Quote, Trade]
MarketDataMessage = Union[Tick, Quote, Trade, OrderBook, OHLCV]

# Constants
DEFAULT_BOOK_DEPTH = 10
MAX_BOOK_DEPTH = 100
TICK_SIZE_PRECISION = 8  # Decimal places for tick sizes
