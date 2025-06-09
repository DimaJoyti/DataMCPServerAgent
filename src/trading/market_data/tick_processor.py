"""
High-performance tick data processor for institutional trading.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set

from .data_types import (
    MarketDataMessage, MarketDataSnapshot, MarketDataType,
    OHLCV, Quote, Tick, Trade, OrderBook
)
from ..core.enums import Exchange


class TickProcessor:
    """
    High-performance tick data processor.
    
    Features:
    - Real-time tick aggregation
    - OHLCV bar generation
    - Market data normalization
    - Latency optimization
    - Data quality monitoring
    """
    
    def __init__(
        self,
        name: str = "TickProcessor",
        max_symbols: int = 10000,
        tick_buffer_size: int = 100000
    ):
        self.name = name
        self.max_symbols = max_symbols
        self.tick_buffer_size = tick_buffer_size
        
        self.logger = logging.getLogger(f"TickProcessor.{name}")
        self.is_running = False
        
        # Data storage
        self.snapshots: Dict[str, MarketDataSnapshot] = {}
        self.tick_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=tick_buffer_size))
        self.ohlcv_data: Dict[str, Dict[str, OHLCV]] = defaultdict(dict)  # symbol -> timeframe -> OHLCV
        
        # Processing queues
        self.tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000000)
        self.quote_queue: asyncio.Queue = asyncio.Queue(maxsize=100000)
        self.trade_queue: asyncio.Queue = asyncio.Queue(maxsize=100000)
        
        # Event handlers
        self.tick_handlers: List[Callable] = []
        self.bar_handlers: List[Callable] = []
        self.snapshot_handlers: List[Callable] = []
        
        # Performance metrics
        self.processed_ticks = 0
        self.processed_quotes = 0
        self.processed_trades = 0
        self.processing_latency_us = deque(maxlen=1000)
        
        # Bar generation settings
        self.bar_timeframes = ["1m", "5m", "15m", "1h", "1d"]
        self.bar_intervals = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1)
        }
        
        # Data quality tracking
        self.quality_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_messages": 0,
            "out_of_sequence": 0,
            "duplicate_messages": 0,
            "stale_messages": 0,
            "last_sequence": 0
        })
    
    async def start(self) -> None:
        """Start the tick processor."""
        self.logger.info(f"Starting tick processor: {self.name}")
        self.is_running = True
        
        # Start processing tasks
        asyncio.create_task(self._process_ticks())
        asyncio.create_task(self._process_quotes())
        asyncio.create_task(self._process_trades())
        asyncio.create_task(self._generate_bars())
        asyncio.create_task(self._cleanup_old_data())
        
        self.logger.info(f"Tick processor started: {self.name}")
    
    async def stop(self) -> None:
        """Stop the tick processor."""
        self.logger.info(f"Stopping tick processor: {self.name}")
        self.is_running = False
        self.logger.info(f"Tick processor stopped: {self.name}")
    
    async def process_message(self, message: MarketDataMessage) -> None:
        """Process incoming market data message."""
        start_time = datetime.utcnow()
        
        try:
            if isinstance(message, Tick):
                await self.tick_queue.put(message)
            elif isinstance(message, Quote):
                await self.quote_queue.put(message)
            elif isinstance(message, Trade):
                await self.trade_queue.put(message)
            elif isinstance(message, OrderBook):
                await self._process_order_book(message)
            
            # Track processing latency
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1_000_000
            self.processing_latency_us.append(processing_time)
            
        except asyncio.QueueFull:
            self.logger.warning(f"Queue full, dropping message for {message.symbol}")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    async def _process_ticks(self) -> None:
        """Process tick data."""
        while self.is_running:
            try:
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                
                # Update snapshot
                await self._update_snapshot_from_tick(tick)
                
                # Store in buffer
                self.tick_buffers[tick.symbol].append(tick)
                
                # Update metrics
                self.processed_ticks += 1
                self._update_quality_metrics(tick)
                
                # Trigger handlers
                for handler in self.tick_handlers:
                    try:
                        await handler(tick) if asyncio.iscoroutinefunction(handler) else handler(tick)
                    except Exception as e:
                        self.logger.error(f"Error in tick handler: {str(e)}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing tick: {str(e)}")
    
    async def _process_quotes(self) -> None:
        """Process quote data."""
        while self.is_running:
            try:
                quote = await asyncio.wait_for(self.quote_queue.get(), timeout=1.0)
                
                # Update snapshot
                await self._update_snapshot_from_quote(quote)
                
                # Update metrics
                self.processed_quotes += 1
                self._update_quality_metrics(quote)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing quote: {str(e)}")
    
    async def _process_trades(self) -> None:
        """Process trade data."""
        while self.is_running:
            try:
                trade = await asyncio.wait_for(self.trade_queue.get(), timeout=1.0)
                
                # Update snapshot
                await self._update_snapshot_from_trade(trade)
                
                # Update OHLCV data
                await self._update_ohlcv_from_trade(trade)
                
                # Update metrics
                self.processed_trades += 1
                self._update_quality_metrics(trade)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing trade: {str(e)}")
    
    async def _update_snapshot_from_tick(self, tick: Tick) -> None:
        """Update market data snapshot from tick."""
        symbol = tick.symbol
        
        if symbol not in self.snapshots:
            self.snapshots[symbol] = MarketDataSnapshot(
                symbol=symbol,
                timestamp=tick.timestamp,
                exchange=tick.exchange
            )
        
        snapshot = self.snapshots[symbol]
        snapshot.timestamp = tick.timestamp
        
        # Update daily stats
        if snapshot.open_price is None:
            snapshot.open_price = tick.price
        
        if snapshot.high_price is None or tick.price > snapshot.high_price:
            snapshot.high_price = tick.price
        
        if snapshot.low_price is None or tick.price < snapshot.low_price:
            snapshot.low_price = tick.price
        
        # Update derived metrics
        snapshot.update_daily_stats()
    
    async def _update_snapshot_from_quote(self, quote: Quote) -> None:
        """Update market data snapshot from quote."""
        symbol = quote.symbol
        
        if symbol not in self.snapshots:
            self.snapshots[symbol] = MarketDataSnapshot(
                symbol=symbol,
                timestamp=quote.timestamp,
                exchange=quote.exchange
            )
        
        snapshot = self.snapshots[symbol]
        snapshot.last_quote = quote
        snapshot.timestamp = quote.timestamp
    
    async def _update_snapshot_from_trade(self, trade: Trade) -> None:
        """Update market data snapshot from trade."""
        symbol = trade.symbol
        
        if symbol not in self.snapshots:
            self.snapshots[symbol] = MarketDataSnapshot(
                symbol=symbol,
                timestamp=trade.timestamp,
                exchange=trade.exchange
            )
        
        snapshot = self.snapshots[symbol]
        snapshot.last_trade = trade
        snapshot.timestamp = trade.timestamp
        
        # Update volume
        if snapshot.volume is None:
            snapshot.volume = trade.size
        else:
            snapshot.volume += trade.size
    
    async def _process_order_book(self, order_book: OrderBook) -> None:
        """Process order book data."""
        symbol = order_book.symbol
        
        if symbol not in self.snapshots:
            self.snapshots[symbol] = MarketDataSnapshot(
                symbol=symbol,
                timestamp=order_book.timestamp,
                exchange=order_book.exchange
            )
        
        snapshot = self.snapshots[symbol]
        snapshot.order_book = order_book
        snapshot.timestamp = order_book.timestamp
    
    async def _update_ohlcv_from_trade(self, trade: Trade) -> None:
        """Update OHLCV data from trade."""
        symbol = trade.symbol
        
        for timeframe in self.bar_timeframes:
            # Get current bar timestamp
            bar_timestamp = self._get_bar_timestamp(trade.timestamp, timeframe)
            
            if timeframe not in self.ohlcv_data[symbol]:
                # Create new bar
                self.ohlcv_data[symbol][timeframe] = OHLCV(
                    symbol=symbol,
                    timestamp=bar_timestamp,
                    timeframe=timeframe,
                    open=trade.price,
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size,
                    exchange=trade.exchange,
                    trade_count=1
                )
            else:
                bar = self.ohlcv_data[symbol][timeframe]
                
                # Check if we need a new bar
                if bar_timestamp > bar.timestamp:
                    # Trigger bar completion event
                    for handler in self.bar_handlers:
                        try:
                            await handler(bar) if asyncio.iscoroutinefunction(handler) else handler(bar)
                        except Exception as e:
                            self.logger.error(f"Error in bar handler: {str(e)}")
                    
                    # Create new bar
                    self.ohlcv_data[symbol][timeframe] = OHLCV(
                        symbol=symbol,
                        timestamp=bar_timestamp,
                        timeframe=timeframe,
                        open=trade.price,
                        high=trade.price,
                        low=trade.price,
                        close=trade.price,
                        volume=trade.size,
                        exchange=trade.exchange,
                        trade_count=1
                    )
                else:
                    # Update existing bar
                    bar.high = max(bar.high, trade.price)
                    bar.low = min(bar.low, trade.price)
                    bar.close = trade.price
                    bar.volume += trade.size
                    bar.trade_count = (bar.trade_count or 0) + 1
    
    def _get_bar_timestamp(self, timestamp: datetime, timeframe: str) -> datetime:
        """Get normalized bar timestamp for timeframe."""
        if timeframe == "1m":
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == "5m":
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == "15m":
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == "1h":
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == "1d":
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp
    
    async def _generate_bars(self) -> None:
        """Generate periodic bar updates."""
        while self.is_running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                now = datetime.utcnow()
                
                # Check for completed bars
                for symbol in list(self.ohlcv_data.keys()):
                    for timeframe in list(self.ohlcv_data[symbol].keys()):
                        bar = self.ohlcv_data[symbol][timeframe]
                        interval = self.bar_intervals[timeframe]
                        
                        # Check if bar should be completed
                        if now >= bar.timestamp + interval:
                            # Trigger bar completion
                            for handler in self.bar_handlers:
                                try:
                                    await handler(bar) if asyncio.iscoroutinefunction(handler) else handler(bar)
                                except Exception as e:
                                    self.logger.error(f"Error in bar handler: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error generating bars: {str(e)}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean up old tick buffers
                for symbol in list(self.tick_buffers.keys()):
                    buffer = self.tick_buffers[symbol]
                    # Remove old ticks
                    while buffer and buffer[0].timestamp < cutoff_time:
                        buffer.popleft()
                
                self.logger.debug("Completed data cleanup")
                
            except Exception as e:
                self.logger.error(f"Error in data cleanup: {str(e)}")
    
    def _update_quality_metrics(self, message: MarketDataMessage) -> None:
        """Update data quality metrics."""
        symbol = message.symbol
        metrics = self.quality_metrics[symbol]
        
        metrics["total_messages"] += 1
        
        # Check sequence numbers
        if hasattr(message, 'sequence_number') and message.sequence_number:
            if message.sequence_number <= metrics["last_sequence"]:
                if message.sequence_number == metrics["last_sequence"]:
                    metrics["duplicate_messages"] += 1
                else:
                    metrics["out_of_sequence"] += 1
            metrics["last_sequence"] = max(metrics["last_sequence"], message.sequence_number)
        
        # Check for stale data
        if hasattr(message, 'feed_timestamp') and message.feed_timestamp:
            age = (message.timestamp - message.feed_timestamp).total_seconds()
            if age > 1.0:  # More than 1 second old
                metrics["stale_messages"] += 1
    
    def get_snapshot(self, symbol: str) -> Optional[MarketDataSnapshot]:
        """Get current market data snapshot for symbol."""
        return self.snapshots.get(symbol)
    
    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[OHLCV]:
        """Get latest OHLCV bar for symbol and timeframe."""
        return self.ohlcv_data.get(symbol, {}).get(timeframe)
    
    def get_tick_history(self, symbol: str, count: int = 100) -> List[Tick]:
        """Get recent tick history for symbol."""
        buffer = self.tick_buffers.get(symbol, deque())
        return list(buffer)[-count:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_latency = sum(self.processing_latency_us) / len(self.processing_latency_us) if self.processing_latency_us else 0
        
        return {
            "processed_ticks": self.processed_ticks,
            "processed_quotes": self.processed_quotes,
            "processed_trades": self.processed_trades,
            "average_processing_latency_us": avg_latency,
            "active_symbols": len(self.snapshots),
            "tick_queue_size": self.tick_queue.qsize(),
            "quote_queue_size": self.quote_queue.qsize(),
            "trade_queue_size": self.trade_queue.qsize()
        }
    
    def get_quality_metrics(self, symbol: Optional[str] = None) -> Dict[str, Dict]:
        """Get data quality metrics."""
        if symbol:
            return {symbol: self.quality_metrics.get(symbol, {})}
        return dict(self.quality_metrics)
    
    def add_tick_handler(self, handler: Callable) -> None:
        """Add tick event handler."""
        self.tick_handlers.append(handler)
    
    def add_bar_handler(self, handler: Callable) -> None:
        """Add bar completion handler."""
        self.bar_handlers.append(handler)
    
    def add_snapshot_handler(self, handler: Callable) -> None:
        """Add snapshot update handler."""
        self.snapshot_handlers.append(handler)
