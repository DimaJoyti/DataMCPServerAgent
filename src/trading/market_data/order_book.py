"""
High-performance order book management for institutional trading.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .data_types import OrderBook, OrderBookLevel, Quote, Trade
from ..core.enums import Exchange


class OrderBookManager:
    """
    High-performance order book manager.
    
    Features:
    - Real-time Level 2 order book reconstruction
    - Best bid/offer (BBO) tracking
    - Market depth analysis
    - Order book imbalance detection
    - Liquidity metrics calculation
    """
    
    def __init__(
        self,
        name: str = "OrderBookManager",
        max_depth: int = 100,
        update_frequency_ms: int = 10
    ):
        self.name = name
        self.max_depth = max_depth
        self.update_frequency_ms = update_frequency_ms
        
        self.logger = logging.getLogger(f"OrderBookManager.{name}")
        self.is_running = False
        
        # Order book storage
        self.order_books: Dict[str, OrderBook] = {}
        self.book_snapshots: Dict[str, List[OrderBook]] = defaultdict(list)
        
        # BBO tracking
        self.bbo_history: Dict[str, List[Tuple[datetime, Decimal, Decimal]]] = defaultdict(list)
        
        # Imbalance tracking
        self.imbalance_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Performance metrics
        self.update_count = 0
        self.reconstruction_errors = 0
        
        # Event handlers
        self.book_update_handlers: List[callable] = []
        self.bbo_change_handlers: List[callable] = []
        self.imbalance_handlers: List[callable] = []
    
    async def start(self) -> None:
        """Start the order book manager."""
        self.logger.info(f"Starting order book manager: {self.name}")
        self.is_running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_books())
        asyncio.create_task(self._calculate_metrics())
        
        self.logger.info(f"Order book manager started: {self.name}")
    
    async def stop(self) -> None:
        """Stop the order book manager."""
        self.logger.info(f"Stopping order book manager: {self.name}")
        self.is_running = False
        self.logger.info(f"Order book manager stopped: {self.name}")
    
    async def update_book(self, order_book: OrderBook) -> None:
        """Update order book with new data."""
        try:
            symbol = order_book.symbol
            
            # Store previous BBO for comparison
            previous_book = self.order_books.get(symbol)
            previous_bbo = None
            if previous_book and previous_book.best_bid and previous_book.best_ask:
                previous_bbo = (previous_book.best_bid.price, previous_book.best_ask.price)
            
            # Update order book
            self.order_books[symbol] = order_book
            self.update_count += 1
            
            # Track BBO changes
            if order_book.best_bid and order_book.best_ask:
                current_bbo = (order_book.best_bid.price, order_book.best_ask.price)
                
                # Store BBO history
                self.bbo_history[symbol].append((order_book.timestamp, current_bbo[0], current_bbo[1]))
                
                # Keep only recent history
                if len(self.bbo_history[symbol]) > 1000:
                    self.bbo_history[symbol] = self.bbo_history[symbol][-1000:]
                
                # Trigger BBO change handlers if BBO changed
                if previous_bbo != current_bbo:
                    for handler in self.bbo_change_handlers:
                        try:
                            await handler(symbol, current_bbo, previous_bbo)
                        except Exception as e:
                            self.logger.error(f"Error in BBO change handler: {str(e)}")
            
            # Calculate and track imbalance
            imbalance = self._calculate_imbalance(order_book)
            if imbalance is not None:
                self.imbalance_history[symbol].append((order_book.timestamp, imbalance))
                
                # Keep only recent history
                if len(self.imbalance_history[symbol]) > 1000:
                    self.imbalance_history[symbol] = self.imbalance_history[symbol][-1000:]
                
                # Trigger imbalance handlers for significant imbalances
                if abs(imbalance) > 0.3:  # 30% imbalance threshold
                    for handler in self.imbalance_handlers:
                        try:
                            await handler(symbol, imbalance)
                        except Exception as e:
                            self.logger.error(f"Error in imbalance handler: {str(e)}")
            
            # Store snapshot
            self.book_snapshots[symbol].append(order_book)
            if len(self.book_snapshots[symbol]) > 100:
                self.book_snapshots[symbol] = self.book_snapshots[symbol][-100:]
            
            # Trigger update handlers
            for handler in self.book_update_handlers:
                try:
                    await handler(order_book)
                except Exception as e:
                    self.logger.error(f"Error in book update handler: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error updating order book for {order_book.symbol}: {str(e)}")
            self.reconstruction_errors += 1
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for symbol."""
        return self.order_books.get(symbol)
    
    def get_best_bid_ask(self, symbol: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid and ask prices."""
        book = self.order_books.get(symbol)
        if book and book.best_bid and book.best_ask:
            return (book.best_bid.price, book.best_ask.price)
        return None
    
    def get_market_depth(self, symbol: str, levels: int = 5) -> Optional[Dict[str, List[OrderBookLevel]]]:
        """Get market depth for specified levels."""
        book = self.order_books.get(symbol)
        if not book:
            return None
        
        return {
            "bids": book.get_depth("BID", levels),
            "asks": book.get_depth("ASK", levels)
        }
    
    def get_spread(self, symbol: str) -> Optional[Decimal]:
        """Get current spread for symbol."""
        book = self.order_books.get(symbol)
        return book.spread if book else None
    
    def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mid price for symbol."""
        book = self.order_books.get(symbol)
        return book.mid_price if book else None
    
    def get_liquidity_at_price(self, symbol: str, price: Decimal, side: str) -> Decimal:
        """Get available liquidity at or better than specified price."""
        book = self.order_books.get(symbol)
        if not book:
            return Decimal('0')
        
        total_liquidity = Decimal('0')
        
        if side.upper() == "BID":
            for level in book.bids:
                if level.price >= price:
                    total_liquidity += level.size
                else:
                    break
        elif side.upper() == "ASK":
            for level in book.asks:
                if level.price <= price:
                    total_liquidity += level.size
                else:
                    break
        
        return total_liquidity
    
    def get_price_for_quantity(self, symbol: str, quantity: Decimal, side: str) -> Optional[Decimal]:
        """Get average price for executing specified quantity."""
        book = self.order_books.get(symbol)
        if not book:
            return None
        
        remaining_quantity = quantity
        total_cost = Decimal('0')
        
        levels = book.bids if side.upper() == "SELL" else book.asks
        
        for level in levels:
            if remaining_quantity <= 0:
                break
            
            available_quantity = min(level.size, remaining_quantity)
            total_cost += available_quantity * level.price
            remaining_quantity -= available_quantity
        
        if remaining_quantity > 0:
            # Not enough liquidity
            return None
        
        return total_cost / quantity
    
    def _calculate_imbalance(self, order_book: OrderBook) -> Optional[float]:
        """Calculate order book imbalance."""
        if not order_book.bids or not order_book.asks:
            return None
        
        # Calculate imbalance using top 5 levels
        bid_volume = sum(level.size for level in order_book.bids[:5])
        ask_volume = sum(level.size for level in order_book.asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Imbalance: positive = more bids, negative = more asks
        return float((bid_volume - ask_volume) / total_volume)
    
    def get_imbalance(self, symbol: str) -> Optional[float]:
        """Get current order book imbalance."""
        book = self.order_books.get(symbol)
        return self._calculate_imbalance(book) if book else None
    
    def get_bbo_history(self, symbol: str, count: int = 100) -> List[Tuple[datetime, Decimal, Decimal]]:
        """Get BBO history for symbol."""
        history = self.bbo_history.get(symbol, [])
        return history[-count:]
    
    def get_imbalance_history(self, symbol: str, count: int = 100) -> List[Tuple[datetime, float]]:
        """Get imbalance history for symbol."""
        history = self.imbalance_history.get(symbol, [])
        return history[-count:]
    
    def calculate_market_impact(self, symbol: str, quantity: Decimal, side: str) -> Optional[Dict[str, Any]]:
        """Calculate estimated market impact for order."""
        book = self.order_books.get(symbol)
        if not book:
            return None
        
        # Get current mid price
        mid_price = book.mid_price
        if not mid_price:
            return None
        
        # Calculate execution price
        execution_price = self.get_price_for_quantity(symbol, quantity, side)
        if not execution_price:
            return None
        
        # Calculate impact
        if side.upper() == "BUY":
            impact = execution_price - mid_price
        else:
            impact = mid_price - execution_price
        
        impact_bps = float(impact / mid_price * 10000) if mid_price > 0 else 0.0
        
        return {
            "mid_price": mid_price,
            "execution_price": execution_price,
            "impact_absolute": impact,
            "impact_bps": impact_bps,
            "quantity": quantity,
            "side": side
        }
    
    def get_liquidity_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive liquidity metrics."""
        book = self.order_books.get(symbol)
        if not book:
            return None
        
        # Calculate metrics for different levels
        levels = [1, 5, 10]
        metrics = {}
        
        for level_count in levels:
            bid_depth = book.get_depth("BID", level_count)
            ask_depth = book.get_depth("ASK", level_count)
            
            bid_volume = sum(level.size for level in bid_depth)
            ask_volume = sum(level.size for level in ask_depth)
            
            metrics[f"bid_volume_L{level_count}"] = float(bid_volume)
            metrics[f"ask_volume_L{level_count}"] = float(ask_volume)
            metrics[f"total_volume_L{level_count}"] = float(bid_volume + ask_volume)
        
        # Add spread metrics
        if book.spread:
            metrics["spread_absolute"] = float(book.spread)
            metrics["spread_bps"] = book.best_bid.price and book.best_ask.price and float(book.spread / book.mid_price * 10000) if book.mid_price else 0
        
        # Add imbalance
        metrics["imbalance"] = self._calculate_imbalance(book)
        
        return metrics
    
    async def _monitor_books(self) -> None:
        """Monitor order book health and performance."""
        while self.is_running:
            try:
                await asyncio.sleep(self.update_frequency_ms / 1000)
                
                # Check for stale books
                now = datetime.utcnow()
                stale_threshold = 5  # 5 seconds
                
                for symbol, book in self.order_books.items():
                    age = (now - book.timestamp).total_seconds()
                    if age > stale_threshold:
                        self.logger.warning(f"Stale order book for {symbol}: {age:.1f}s old")
                
            except Exception as e:
                self.logger.error(f"Error monitoring order books: {str(e)}")
    
    async def _calculate_metrics(self) -> None:
        """Calculate and update performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Log performance metrics
                self.logger.debug(
                    f"Order book metrics - Updates: {self.update_count}, "
                    f"Errors: {self.reconstruction_errors}, "
                    f"Active books: {len(self.order_books)}"
                )
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order book manager statistics."""
        return {
            "active_books": len(self.order_books),
            "total_updates": self.update_count,
            "reconstruction_errors": self.reconstruction_errors,
            "error_rate": self.reconstruction_errors / max(1, self.update_count),
            "symbols": list(self.order_books.keys())
        }
    
    def add_book_update_handler(self, handler: callable) -> None:
        """Add order book update handler."""
        self.book_update_handlers.append(handler)
    
    def add_bbo_change_handler(self, handler: callable) -> None:
        """Add BBO change handler."""
        self.bbo_change_handlers.append(handler)
    
    def add_imbalance_handler(self, handler: callable) -> None:
        """Add imbalance handler."""
        self.imbalance_handlers.append(handler)
