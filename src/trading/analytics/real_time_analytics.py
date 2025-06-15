"""
Real-time analytics engine for institutional trading.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.base_models import BaseOrder, BasePosition, BaseTrade
from ..core.enums import OrderSide
from ..market_data.data_types import MarketDataSnapshot


class RealTimeAnalytics:
    """
    Real-time analytics engine for institutional trading.

    Features:
    - Real-time P&L calculation
    - Position monitoring
    - Performance attribution
    - Risk metrics calculation
    - Trade analytics
    """

    def __init__(self, name: str = "RealTimeAnalytics", calculation_frequency_ms: int = 100):
        self.name = name
        self.calculation_frequency_ms = calculation_frequency_ms

        self.logger = logging.getLogger(f"RealTimeAnalytics.{name}")
        self.is_running = False

        # Data storage
        self.positions: Dict[str, BasePosition] = {}
        self.trades: Dict[str, BaseTrade] = {}
        self.orders: Dict[str, BaseOrder] = {}
        self.market_data: Dict[str, MarketDataSnapshot] = {}

        # P&L tracking
        self.realized_pnl: Dict[str, Decimal] = defaultdict(Decimal)
        self.unrealized_pnl: Dict[str, Decimal] = defaultdict(Decimal)
        self.total_pnl: Dict[str, Decimal] = defaultdict(Decimal)

        # Performance metrics
        self.pnl_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.returns_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Portfolio metrics
        self.portfolio_value = Decimal("1000000")  # $1M default
        self.portfolio_pnl = Decimal("0")
        self.portfolio_returns: deque = deque(maxlen=1000)

        # Risk metrics
        self.var_confidence = 0.95
        self.var_lookback_days = 252

        # Event handlers
        self.pnl_update_handlers: List[callable] = []
        self.risk_alert_handlers: List[callable] = []

        # Performance tracking
        self.calculation_count = 0
        self.calculation_latency_us: deque = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the real-time analytics engine."""
        self.logger.info(f"Starting real-time analytics: {self.name}")
        self.is_running = True

        # Start calculation tasks
        asyncio.create_task(self._calculate_pnl())
        asyncio.create_task(self._calculate_risk_metrics())
        asyncio.create_task(self._monitor_performance())

        self.logger.info(f"Real-time analytics started: {self.name}")

    async def stop(self) -> None:
        """Stop the real-time analytics engine."""
        self.logger.info(f"Stopping real-time analytics: {self.name}")
        self.is_running = False
        self.logger.info(f"Real-time analytics stopped: {self.name}")

    async def update_position(self, position: BasePosition) -> None:
        """Update position data."""
        self.positions[position.symbol] = position
        await self._recalculate_pnl(position.symbol)

    async def update_trade(self, trade: BaseTrade) -> None:
        """Update trade data."""
        self.trades[trade.trade_id] = trade

        # Update position from trade
        await self._update_position_from_trade(trade)

        # Recalculate P&L
        await self._recalculate_pnl(trade.symbol)

    async def update_order(self, order: BaseOrder) -> None:
        """Update order data."""
        self.orders[order.order_id] = order

    async def update_market_data(self, snapshot: MarketDataSnapshot) -> None:
        """Update market data snapshot."""
        self.market_data[snapshot.symbol] = snapshot

        # Recalculate unrealized P&L
        await self._recalculate_pnl(snapshot.symbol)

    async def _update_position_from_trade(self, trade: BaseTrade) -> None:
        """Update position from trade execution."""
        symbol = trade.symbol

        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = BasePosition(
                symbol=symbol,
                quantity=Decimal("0"),
                average_price=Decimal("0"),
                market_price=trade.price,
            )

        position = self.positions[symbol]

        # Calculate new position
        if trade.side == OrderSide.BUY:
            new_quantity = position.quantity + trade.quantity
        else:
            new_quantity = position.quantity - trade.quantity

        # Update average price
        if new_quantity != 0:
            if (position.quantity >= 0 and trade.side == OrderSide.BUY) or (
                position.quantity <= 0 and trade.side == OrderSide.SELL
            ):
                # Adding to position
                total_cost = (position.quantity * position.average_price) + (
                    trade.quantity * trade.price
                )
                position.average_price = total_cost / new_quantity
            # If reducing position, keep same average price

        position.quantity = new_quantity
        position.market_price = trade.price
        position.updated_at = datetime.utcnow()

        # Update realized P&L if position was reduced
        if abs(new_quantity) < abs(position.quantity):
            closed_quantity = abs(position.quantity) - abs(new_quantity)
            if trade.side == OrderSide.SELL and position.quantity > 0:
                realized_gain = closed_quantity * (trade.price - position.average_price)
            elif trade.side == OrderSide.BUY and position.quantity < 0:
                realized_gain = closed_quantity * (position.average_price - trade.price)
            else:
                realized_gain = Decimal("0")

            self.realized_pnl[symbol] += realized_gain
            position.realized_pnl += realized_gain

    async def _recalculate_pnl(self, symbol: str) -> None:
        """Recalculate P&L for a symbol."""
        start_time = datetime.utcnow()

        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Get current market price
            market_data = self.market_data.get(symbol)
            if market_data:
                current_price = market_data.current_price
                if current_price:
                    position.market_price = current_price

            # Calculate unrealized P&L
            if position.market_price and position.quantity != 0:
                unrealized = position.quantity * (position.market_price - position.average_price)
                self.unrealized_pnl[symbol] = unrealized
                position.unrealized_pnl = unrealized
            else:
                self.unrealized_pnl[symbol] = Decimal("0")
                position.unrealized_pnl = Decimal("0")

            # Calculate total P&L
            total = self.realized_pnl[symbol] + self.unrealized_pnl[symbol]
            self.total_pnl[symbol] = total

            # Store P&L history
            timestamp = datetime.utcnow()
            self.pnl_history[symbol].append((timestamp, float(total)))

            # Calculate returns
            if len(self.pnl_history[symbol]) > 1:
                previous_pnl = self.pnl_history[symbol][-2][1]
                if previous_pnl != 0:
                    return_pct = (float(total) - previous_pnl) / abs(previous_pnl)
                    self.returns_history[symbol].append(return_pct)

            # Update portfolio P&L
            self.portfolio_pnl = sum(self.total_pnl.values())

            # Trigger P&L update handlers
            for handler in self.pnl_update_handlers:
                try:
                    await handler(
                        symbol, total, self.realized_pnl[symbol], self.unrealized_pnl[symbol]
                    )
                except Exception as e:
                    self.logger.error(f"Error in P&L update handler: {str(e)}")

            # Track calculation latency
            calculation_time = (datetime.utcnow() - start_time).total_seconds() * 1_000_000
            self.calculation_latency_us.append(calculation_time)
            self.calculation_count += 1

        except Exception as e:
            self.logger.error(f"Error recalculating P&L for {symbol}: {str(e)}")

    async def _calculate_pnl(self) -> None:
        """Continuously calculate P&L for all positions."""
        while self.is_running:
            try:
                # Recalculate P&L for all positions
                for symbol in list(self.positions.keys()):
                    await self._recalculate_pnl(symbol)

                await asyncio.sleep(self.calculation_frequency_ms / 1000)

            except Exception as e:
                self.logger.error(f"Error in P&L calculation loop: {str(e)}")
                await asyncio.sleep(1)

    async def _calculate_risk_metrics(self) -> None:
        """Calculate risk metrics periodically."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Calculate every 10 seconds

                # Calculate portfolio VaR
                portfolio_var = self._calculate_portfolio_var()

                # Check risk limits
                if (
                    portfolio_var and abs(portfolio_var) > float(self.portfolio_value) * 0.02
                ):  # 2% VaR limit
                    for handler in self.risk_alert_handlers:
                        try:
                            await handler(
                                "VAR_LIMIT_EXCEEDED",
                                {
                                    "var": portfolio_var,
                                    "limit": float(self.portfolio_value) * 0.02,
                                    "portfolio_value": float(self.portfolio_value),
                                },
                            )
                        except Exception as e:
                            self.logger.error(f"Error in risk alert handler: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error calculating risk metrics: {str(e)}")

    async def _monitor_performance(self) -> None:
        """Monitor analytics performance."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                avg_latency = (
                    sum(self.calculation_latency_us) / len(self.calculation_latency_us)
                    if self.calculation_latency_us
                    else 0
                )

                self.logger.debug(
                    f"Analytics performance - Calculations: {self.calculation_count}, "
                    f"Avg latency: {avg_latency:.1f}μs, "
                    f"Active positions: {len(self.positions)}"
                )

            except Exception as e:
                self.logger.error(f"Error monitoring performance: {str(e)}")

    def _calculate_portfolio_var(self) -> Optional[float]:
        """Calculate portfolio Value at Risk."""
        if len(self.portfolio_returns) < 30:  # Need at least 30 observations
            return None

        returns = list(self.portfolio_returns)
        returns.sort()

        # Calculate VaR at specified confidence level
        var_index = int((1 - self.var_confidence) * len(returns))
        var_return = returns[var_index]

        return var_return * float(self.portfolio_value)

    def get_position_pnl(self, symbol: str) -> Dict[str, Decimal]:
        """Get P&L breakdown for a position."""
        return {
            "realized": self.realized_pnl.get(symbol, Decimal("0")),
            "unrealized": self.unrealized_pnl.get(symbol, Decimal("0")),
            "total": self.total_pnl.get(symbol, Decimal("0")),
        }

    def get_portfolio_pnl(self) -> Dict[str, Any]:
        """Get portfolio P&L summary."""
        total_realized = sum(self.realized_pnl.values())
        total_unrealized = sum(self.unrealized_pnl.values())

        return {
            "realized": float(total_realized),
            "unrealized": float(total_unrealized),
            "total": float(self.portfolio_pnl),
            "portfolio_value": float(self.portfolio_value),
            "return_pct": (
                float(self.portfolio_pnl / self.portfolio_value * 100)
                if self.portfolio_value > 0
                else 0.0
            ),
        }

    def get_position_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive position metrics."""
        position = self.positions.get(symbol)
        if not position:
            return None

        pnl = self.get_position_pnl(symbol)
        returns = list(self.returns_history[symbol])

        metrics = {
            "symbol": symbol,
            "quantity": float(position.quantity),
            "average_price": float(position.average_price),
            "market_price": float(position.market_price) if position.market_price else None,
            "market_value": float(position.market_value) if position.market_value else None,
            "pnl": {
                "realized": float(pnl["realized"]),
                "unrealized": float(pnl["unrealized"]),
                "total": float(pnl["total"]),
            },
        }

        # Add return statistics if available
        if returns:
            import statistics

            metrics["returns"] = {
                "count": len(returns),
                "mean": statistics.mean(returns),
                "std": statistics.stdev(returns) if len(returns) > 1 else 0.0,
                "min": min(returns),
                "max": max(returns),
            }

        return metrics

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        portfolio_pnl = self.get_portfolio_pnl()
        returns = list(self.portfolio_returns)

        metrics = {
            "portfolio_value": portfolio_pnl["portfolio_value"],
            "pnl": portfolio_pnl,
            "positions": {"count": len(self.positions), "symbols": list(self.positions.keys())},
            "trades": {"count": len(self.trades)},
        }

        # Add return statistics
        if returns:
            import statistics

            metrics["returns"] = {
                "count": len(returns),
                "mean": statistics.mean(returns),
                "std": statistics.stdev(returns) if len(returns) > 1 else 0.0,
                "sharpe": (
                    statistics.mean(returns) / statistics.stdev(returns) * (252**0.5)
                    if len(returns) > 1 and statistics.stdev(returns) > 0
                    else 0.0
                ),
            }

            # Calculate VaR
            var = self._calculate_portfolio_var()
            if var:
                metrics["var"] = {
                    "confidence": self.var_confidence,
                    "value": var,
                    "percentage": var / portfolio_pnl["portfolio_value"] * 100,
                }

        return metrics

    def get_analytics_performance(self) -> Dict[str, Any]:
        """Get analytics engine performance metrics."""
        avg_latency = (
            sum(self.calculation_latency_us) / len(self.calculation_latency_us)
            if self.calculation_latency_us
            else 0
        )

        return {
            "calculation_count": self.calculation_count,
            "average_latency_us": avg_latency,
            "calculations_per_second": self.calculation_count
            / max(1, self.calculation_frequency_ms / 1000),
            "active_positions": len(self.positions),
            "active_symbols": len(set(self.positions.keys()) | set(self.market_data.keys())),
        }

    def add_pnl_update_handler(self, handler: callable) -> None:
        """Add P&L update handler."""
        self.pnl_update_handlers.append(handler)

    def add_risk_alert_handler(self, handler: callable) -> None:
        """Add risk alert handler."""
        self.risk_alert_handlers.append(handler)
