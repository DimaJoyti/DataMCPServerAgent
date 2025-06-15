"""
Fill management system for institutional trading.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.base_models import BaseTrade


class FillManager:
    """
    Fill management system for tracking and processing trade executions.

    Features:
    - Real-time fill processing
    - Fill aggregation and allocation
    - Average price calculation
    - Commission tracking
    - Fill reporting and analytics
    """

    def __init__(self):
        self.logger = logging.getLogger("FillManager")
        self.is_active = False

        # Fill storage
        self.fills: Dict[str, BaseTrade] = {}
        self.fills_by_order: Dict[str, List[str]] = defaultdict(list)
        self.fills_by_symbol: Dict[str, List[str]] = defaultdict(list)

        # Fill processing queue
        self.fill_queue: asyncio.Queue = asyncio.Queue()

        # Performance tracking
        self.total_fills_processed = 0
        self.total_volume_traded = Decimal("0")
        self.total_commission_paid = Decimal("0")

        # Event handlers
        self.fill_event_handlers: List[callable] = []

    async def start(self) -> None:
        """Start the fill manager."""
        self.logger.info("Starting Fill Manager")
        self.is_active = True

        # Start fill processing task
        asyncio.create_task(self._process_fills())

        self.logger.info("Fill Manager started")

    async def stop(self) -> None:
        """Stop the fill manager."""
        self.logger.info("Stopping Fill Manager")
        self.is_active = False
        self.logger.info("Fill Manager stopped")

    async def process_fill(self, fill: BaseTrade) -> None:
        """
        Process a new fill.

        Args:
            fill: Trade fill to process
        """
        await self.fill_queue.put(fill)

    async def _process_fills(self) -> None:
        """Process fills from the queue."""
        while self.is_active:
            try:
                # Wait for fill with timeout
                fill = await asyncio.wait_for(self.fill_queue.get(), timeout=1.0)
                await self._handle_fill(fill)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing fill: {str(e)}")

    async def _handle_fill(self, fill: BaseTrade) -> None:
        """Handle a single fill."""
        try:
            # Store fill
            self.fills[fill.trade_id] = fill
            self.fills_by_order[fill.order_id].append(fill.trade_id)
            self.fills_by_symbol[fill.symbol].append(fill.trade_id)

            # Update order status
            await self._update_order_from_fill(fill)

            # Update performance metrics
            self.total_fills_processed += 1
            self.total_volume_traded += fill.quantity
            self.total_commission_paid += fill.commission

            # Trigger events
            await self._trigger_fill_event("FILL_RECEIVED", fill)

            self.logger.info(
                f"Fill processed: {fill.trade_id} - {fill.symbol} "
                f"{fill.quantity} @ {fill.price}"
            )

        except Exception as e:
            self.logger.error(f"Error handling fill {fill.trade_id}: {str(e)}")

    async def _update_order_from_fill(self, fill: BaseTrade) -> None:
        """Update order status based on fill."""
        # This would integrate with the OMS to update order status
        # For now, we'll just log the update
        self.logger.debug(f"Order {fill.order_id} filled: {fill.quantity} @ {fill.price}")

    def get_fills_for_order(self, order_id: str) -> List[BaseTrade]:
        """Get all fills for an order."""
        fill_ids = self.fills_by_order.get(order_id, [])
        return [self.fills[fill_id] for fill_id in fill_ids if fill_id in self.fills]

    def get_fills_for_symbol(self, symbol: str) -> List[BaseTrade]:
        """Get all fills for a symbol."""
        fill_ids = self.fills_by_symbol.get(symbol, [])
        return [self.fills[fill_id] for fill_id in fill_ids if fill_id in self.fills]

    def calculate_average_price(self, order_id: str) -> Optional[Decimal]:
        """Calculate average fill price for an order."""
        fills = self.get_fills_for_order(order_id)

        if not fills:
            return None

        total_value = Decimal("0")
        total_quantity = Decimal("0")

        for fill in fills:
            total_value += fill.quantity * fill.price
            total_quantity += fill.quantity

        if total_quantity > 0:
            return total_value / total_quantity

        return None

    def calculate_total_quantity(self, order_id: str) -> Decimal:
        """Calculate total filled quantity for an order."""
        fills = self.get_fills_for_order(order_id)
        return sum(fill.quantity for fill in fills)

    def calculate_total_commission(self, order_id: str) -> Decimal:
        """Calculate total commission for an order."""
        fills = self.get_fills_for_order(order_id)
        return sum(fill.commission for fill in fills)

    def get_fill_statistics(self) -> Dict[str, Any]:
        """Get fill processing statistics."""
        return {
            "total_fills_processed": self.total_fills_processed,
            "total_volume_traded": float(self.total_volume_traded),
            "total_commission_paid": float(self.total_commission_paid),
            "average_fill_size": float(
                self.total_volume_traded / max(1, self.total_fills_processed)
            ),
            "symbols_traded": len(self.fills_by_symbol),
            "orders_with_fills": len(self.fills_by_order),
        }

    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a specific symbol."""
        fills = self.get_fills_for_symbol(symbol)

        if not fills:
            return {
                "symbol": symbol,
                "total_fills": 0,
                "total_volume": 0.0,
                "total_commission": 0.0,
                "average_price": 0.0,
                "price_range": {"min": 0.0, "max": 0.0},
            }

        total_volume = sum(fill.quantity for fill in fills)
        total_value = sum(fill.quantity * fill.price for fill in fills)
        total_commission = sum(fill.commission for fill in fills)

        prices = [fill.price for fill in fills]

        return {
            "symbol": symbol,
            "total_fills": len(fills),
            "total_volume": float(total_volume),
            "total_commission": float(total_commission),
            "average_price": float(total_value / total_volume) if total_volume > 0 else 0.0,
            "price_range": {"min": float(min(prices)), "max": float(max(prices))},
        }

    def get_order_fill_summary(self, order_id: str) -> Dict[str, Any]:
        """Get fill summary for an order."""
        fills = self.get_fills_for_order(order_id)

        if not fills:
            return {
                "order_id": order_id,
                "total_fills": 0,
                "total_quantity": 0.0,
                "average_price": 0.0,
                "total_commission": 0.0,
                "first_fill_time": None,
                "last_fill_time": None,
            }

        total_quantity = sum(fill.quantity for fill in fills)
        total_value = sum(fill.quantity * fill.price for fill in fills)
        total_commission = sum(fill.commission for fill in fills)

        fill_times = [fill.executed_at for fill in fills]

        return {
            "order_id": order_id,
            "total_fills": len(fills),
            "total_quantity": float(total_quantity),
            "average_price": float(total_value / total_quantity) if total_quantity > 0 else 0.0,
            "total_commission": float(total_commission),
            "first_fill_time": min(fill_times).isoformat(),
            "last_fill_time": max(fill_times).isoformat(),
        }

    async def generate_fill_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive fill report."""

        # Filter fills based on criteria
        filtered_fills = []

        for fill in self.fills.values():
            # Time filter
            if start_time and fill.executed_at < start_time:
                continue
            if end_time and fill.executed_at > end_time:
                continue

            # Symbol filter
            if symbol and fill.symbol != symbol:
                continue

            filtered_fills.append(fill)

        if not filtered_fills:
            return {
                "report_period": {
                    "start": start_time.isoformat() if start_time else None,
                    "end": end_time.isoformat() if end_time else None,
                },
                "symbol_filter": symbol,
                "summary": {"total_fills": 0, "total_volume": 0.0, "total_commission": 0.0},
                "by_symbol": {},
                "by_side": {"BUY": 0, "SELL": 0},
            }

        # Calculate summary statistics
        total_volume = sum(fill.quantity for fill in filtered_fills)
        total_value = sum(fill.quantity * fill.price for fill in filtered_fills)
        total_commission = sum(fill.commission for fill in filtered_fills)

        # Group by symbol
        by_symbol = defaultdict(lambda: {"volume": Decimal("0"), "value": Decimal("0"), "count": 0})
        for fill in filtered_fills:
            by_symbol[fill.symbol]["volume"] += fill.quantity
            by_symbol[fill.symbol]["value"] += fill.quantity * fill.price
            by_symbol[fill.symbol]["count"] += 1

        # Group by side
        by_side = defaultdict(int)
        for fill in filtered_fills:
            by_side[fill.side.value] += 1

        return {
            "report_period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "symbol_filter": symbol,
            "summary": {
                "total_fills": len(filtered_fills),
                "total_volume": float(total_volume),
                "total_value": float(total_value),
                "total_commission": float(total_commission),
                "average_fill_size": float(total_volume / len(filtered_fills)),
                "average_price": float(total_value / total_volume) if total_volume > 0 else 0.0,
            },
            "by_symbol": {
                symbol: {
                    "volume": float(data["volume"]),
                    "value": float(data["value"]),
                    "count": data["count"],
                    "average_price": (
                        float(data["value"] / data["volume"]) if data["volume"] > 0 else 0.0
                    ),
                }
                for symbol, data in by_symbol.items()
            },
            "by_side": dict(by_side),
        }

    async def _trigger_fill_event(self, event_type: str, fill: BaseTrade) -> None:
        """Trigger fill event handlers."""
        for handler in self.fill_event_handlers:
            try:
                await handler(event_type, fill)
            except Exception as e:
                self.logger.error(f"Error in fill event handler: {str(e)}")

    def add_fill_event_handler(self, handler: callable) -> None:
        """Add fill event handler."""
        self.fill_event_handlers.append(handler)

    def remove_fill_event_handler(self, handler: callable) -> None:
        """Remove fill event handler."""
        if handler in self.fill_event_handlers:
            self.fill_event_handlers.remove(handler)
