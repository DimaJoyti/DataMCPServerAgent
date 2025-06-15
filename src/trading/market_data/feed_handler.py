"""
High-performance market data feed handler for institutional trading.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.enums import Exchange
from .data_types import (
    FeedMetrics,
    FeedStatus,
    MarketDataMessage,
    MarketDataType,
    OrderBook,
    Quote,
    Tick,
    Trade,
)


class BaseFeedHandler(ABC):
    """Base class for market data feed handlers."""

    def __init__(
        self, name: str, exchange: Exchange, symbols: List[str], data_types: List[MarketDataType]
    ):
        self.name = name
        self.exchange = exchange
        self.symbols = set(symbols)
        self.data_types = set(data_types)

        # Status and metrics
        self.status = FeedStatus.DISCONNECTED
        self.logger = logging.getLogger(f"FeedHandler.{name}")
        self.metrics: Dict[str, FeedMetrics] = {}

        # Message handling
        self.message_handlers: Dict[MarketDataType, List[Callable]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=100000)

        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.last_message_time: Optional[datetime] = None
        self.message_count = 0
        self.error_count = 0

        # Latency tracking
        self.latency_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0
        self.is_running = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the market data feed."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the market data feed."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[MarketDataType]) -> bool:
        """Subscribe to market data for specified symbols and types."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[MarketDataType]) -> bool:
        """Unsubscribe from market data."""
        pass

    async def start(self) -> None:
        """Start the feed handler."""
        self.logger.info(f"Starting feed handler: {self.name}")
        self.is_running = True
        self.start_time = datetime.utcnow()

        # Start background tasks
        asyncio.create_task(self._connection_manager())
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._metrics_updater())

        self.logger.info(f"Feed handler started: {self.name}")

    async def stop(self) -> None:
        """Stop the feed handler."""
        self.logger.info(f"Stopping feed handler: {self.name}")
        self.is_running = False

        await self.disconnect()

        self.logger.info(f"Feed handler stopped: {self.name}")

    def add_message_handler(
        self, data_type: MarketDataType, handler: Callable[[MarketDataMessage], None]
    ) -> None:
        """Add a message handler for specific data type."""
        self.message_handlers[data_type].append(handler)
        self.logger.debug(f"Added handler for {data_type}")

    def remove_message_handler(
        self, data_type: MarketDataType, handler: Callable[[MarketDataMessage], None]
    ) -> None:
        """Remove a message handler."""
        if handler in self.message_handlers[data_type]:
            self.message_handlers[data_type].remove(handler)
            self.logger.debug(f"Removed handler for {data_type}")

    async def _connection_manager(self) -> None:
        """Manage connection and reconnection logic."""
        while self.is_running:
            try:
                if self.status == FeedStatus.DISCONNECTED:
                    self.logger.info(f"Attempting to connect to {self.name}")

                    if await self.connect():
                        self.status = FeedStatus.CONNECTED
                        self.reconnect_attempts = 0
                        self.logger.info(f"Connected to {self.name}")

                        # Subscribe to symbols
                        if self.symbols and self.data_types:
                            await self.subscribe(list(self.symbols), list(self.data_types))
                    else:
                        self.status = FeedStatus.ERROR
                        self.reconnect_attempts += 1

                        if self.reconnect_attempts >= self.max_reconnect_attempts:
                            self.logger.error(f"Max reconnection attempts reached for {self.name}")
                            break

                        self.logger.warning(
                            f"Connection failed for {self.name}, attempt {self.reconnect_attempts}"
                        )
                        await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)

                elif self.status == FeedStatus.CONNECTED:
                    # Check for stale data
                    if self.last_message_time:
                        time_since_last = (
                            datetime.utcnow() - self.last_message_time
                        ).total_seconds()
                        if time_since_last > 30:  # 30 seconds timeout
                            self.logger.warning(f"Stale data detected for {self.name}")
                            self.status = FeedStatus.STALE

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error in connection manager for {self.name}: {str(e)}")
                self.status = FeedStatus.ERROR
                await asyncio.sleep(5)

    async def _message_processor(self) -> None:
        """Process incoming messages from the queue."""
        while self.is_running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                # Update metrics
                self.message_count += 1
                self.last_message_time = datetime.utcnow()

                # Determine message type
                message_type = self._get_message_type(message)

                # Update latency metrics
                if hasattr(message, "latency_us") and message.latency_us:
                    self.latency_samples[message.symbol].append(message.latency_us)

                # Call handlers
                handlers = self.message_handlers.get(message_type, [])
                for handler in handlers:
                    try:
                        (
                            await handler(message)
                            if asyncio.iscoroutinefunction(handler)
                            else handler(message)
                        )
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {str(e)}")
                        self.error_count += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.error_count += 1

    async def _metrics_updater(self) -> None:
        """Update feed metrics periodically."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                for symbol in self.symbols:
                    self._update_symbol_metrics(symbol)

            except Exception as e:
                self.logger.error(f"Error updating metrics: {str(e)}")

    def _update_symbol_metrics(self, symbol: str) -> None:
        """Update metrics for a specific symbol."""
        now = datetime.utcnow()

        # Calculate latency metrics
        latency_data = list(self.latency_samples[symbol])

        if latency_data:
            min_latency = min(latency_data)
            max_latency = max(latency_data)
            avg_latency = sum(latency_data) // len(latency_data)
            p99_latency = sorted(latency_data)[int(len(latency_data) * 0.99)]
        else:
            min_latency = max_latency = avg_latency = p99_latency = 0

        # Calculate throughput
        uptime = (now - self.start_time).total_seconds() if self.start_time else 1
        messages_per_second = self.message_count / uptime

        # Update metrics
        self.metrics[symbol] = FeedMetrics(
            feed_name=self.name,
            symbol=symbol,
            timestamp=now,
            min_latency_us=min_latency,
            max_latency_us=max_latency,
            avg_latency_us=avg_latency,
            p99_latency_us=p99_latency,
            messages_per_second=messages_per_second,
            total_messages=self.message_count,
            connection_uptime_seconds=uptime,
            reconnection_count=self.reconnect_attempts,
        )

    def _get_message_type(self, message: MarketDataMessage) -> MarketDataType:
        """Determine the type of market data message."""
        if isinstance(message, Tick):
            return MarketDataType.TICK
        elif isinstance(message, Quote):
            return MarketDataType.QUOTE
        elif isinstance(message, Trade):
            return MarketDataType.TRADE
        elif isinstance(message, OrderBook):
            return MarketDataType.ORDERBOOK_L2
        else:
            return MarketDataType.TICK  # Default

    async def queue_message(self, message: MarketDataMessage) -> None:
        """Queue a message for processing."""
        try:
            await self.message_queue.put_nowait(message)
        except asyncio.QueueFull:
            self.logger.warning(f"Message queue full for {self.name}, dropping message")
            self.error_count += 1

    def get_metrics(self, symbol: Optional[str] = None) -> Dict[str, FeedMetrics]:
        """Get feed metrics."""
        if symbol:
            return {symbol: self.metrics.get(symbol)} if symbol in self.metrics else {}
        return self.metrics.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get feed status information."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0

        return {
            "name": self.name,
            "exchange": self.exchange.value,
            "status": self.status.value,
            "symbols": list(self.symbols),
            "data_types": [dt.value for dt in self.data_types],
            "uptime_seconds": uptime,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "reconnect_attempts": self.reconnect_attempts,
            "queue_size": self.message_queue.qsize(),
            "last_message_time": (
                self.last_message_time.isoformat() if self.last_message_time else None
            ),
        }


class MockFeedHandler(BaseFeedHandler):
    """Mock feed handler for testing and demonstration."""

    def __init__(self, name: str, exchange: Exchange, symbols: List[str]):
        super().__init__(
            name=name,
            exchange=exchange,
            symbols=symbols,
            data_types=[MarketDataType.TICK, MarketDataType.QUOTE, MarketDataType.TRADE],
        )
        self.simulation_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Mock connection."""
        await asyncio.sleep(0.1)  # Simulate connection time
        return True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        if self.simulation_task:
            self.simulation_task.cancel()
        await asyncio.sleep(0.1)

    async def subscribe(self, symbols: List[str], data_types: List[MarketDataType]) -> bool:
        """Mock subscription."""
        self.symbols.update(symbols)
        self.data_types.update(data_types)

        # Start data simulation
        self.simulation_task = asyncio.create_task(self._simulate_data())

        return True

    async def unsubscribe(self, symbols: List[str], data_types: List[MarketDataType]) -> bool:
        """Mock unsubscription."""
        for symbol in symbols:
            self.symbols.discard(symbol)

        for data_type in data_types:
            self.data_types.discard(data_type)

        return True

    async def _simulate_data(self) -> None:
        """Simulate market data."""
        import random
        from decimal import Decimal

        base_prices = {symbol: Decimal("100.00") for symbol in self.symbols}

        while self.is_running and self.status == FeedStatus.CONNECTED:
            try:
                for symbol in self.symbols:
                    # Simulate price movement
                    change = Decimal(str(random.uniform(-0.1, 0.1)))
                    base_prices[symbol] += change
                    price = max(Decimal("1.00"), base_prices[symbol])

                    # Generate tick
                    if MarketDataType.TICK in self.data_types:
                        tick = Tick(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            price=price,
                            size=Decimal(str(random.randint(100, 1000))),
                            exchange=self.exchange,
                            latency_us=random.randint(100, 1000),
                        )
                        await self.queue_message(tick)

                    # Generate quote
                    if MarketDataType.QUOTE in self.data_types:
                        spread = Decimal("0.01")
                        quote = Quote(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            bid_price=price - spread / 2,
                            bid_size=Decimal(str(random.randint(500, 2000))),
                            ask_price=price + spread / 2,
                            ask_size=Decimal(str(random.randint(500, 2000))),
                            exchange=self.exchange,
                        )
                        await self.queue_message(quote)

                    # Generate trade
                    if MarketDataType.TRADE in self.data_types and random.random() < 0.3:
                        trade = Trade(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            price=price,
                            size=Decimal(str(random.randint(100, 500))),
                            exchange=self.exchange,
                        )
                        await self.queue_message(trade)

                # Simulate realistic feed frequency
                await asyncio.sleep(0.01)  # 100 messages per second

            except Exception as e:
                self.logger.error(f"Error in data simulation: {str(e)}")
                await asyncio.sleep(1)
