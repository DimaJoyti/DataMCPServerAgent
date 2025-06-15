"""
Trading execution module for the Fetch.ai Advanced Crypto Trading System.

This module provides functionality for executing trades on cryptocurrency exchanges
using the CCXT library. It includes order management, position tracking, and
execution strategies.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import ccxt
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv
from uagents import Context, Model

from .base_agent import BaseAgent, BaseAgentState
from .risk_agent import RiskManagementAgent
from .trading_system import AdvancedCryptoTradingSystem, TradeRecommendation, TradingSignal

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Types of orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order statuses."""

    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"


class Order(Model):
    """Model for an order."""

    id: Optional[str] = None
    exchange: str
    symbol: str
    type: OrderType
    side: OrderSide
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: str
    params: Dict[str, Any] = {}


class Position(Model):
    """Model for a position."""

    symbol: str
    side: OrderSide
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str
    orders: List[Order] = []


class ExecutionStrategy(str, Enum):
    """Execution strategies."""

    SIMPLE = "simple"  # Execute immediately at market price
    TWAP = "twap"  # Time-Weighted Average Price
    ICEBERG = "iceberg"  # Split large orders into smaller ones
    SMART = "smart"  # Adaptive strategy based on market conditions


class TradingExecutionState(BaseAgentState):
    """State model for the Trading Execution."""

    orders: List[Order] = []
    positions: List[Position] = []
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SIMPLE
    max_slippage: float = 0.01  # 1% maximum slippage
    dry_run: bool = True  # Default to dry run mode for safety
    auto_execute: bool = False  # Whether to automatically execute recommendations


class TradingExecution(BaseAgent):
    """Trading execution for cryptocurrency exchanges."""

    def __init__(
        self,
        name: str = "trading_execution",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        trading_system: Optional[AdvancedCryptoTradingSystem] = None,
        risk_agent: Optional[RiskManagementAgent] = None,
        dry_run: bool = True,
    ):
        """Initialize the Trading Execution.

        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
            exchange_id: ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            trading_system: Trading system to execute recommendations from
            risk_agent: Risk management agent for position sizing
            dry_run: Whether to run in dry run mode (no real trades)
        """
        super().__init__(name, seed, port, endpoint, logger)

        # Initialize state
        self.state = TradingExecutionState()
        self.state.dry_run = dry_run

        # Initialize exchange
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret

        # Initialize synchronous exchange for market data
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        # Initialize asynchronous exchange for trading
        exchange_async_class = getattr(ccxt_async, exchange_id)
        self.exchange_async = exchange_async_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        # Store trading system and risk agent
        self.trading_system = trading_system
        self.risk_agent = risk_agent

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register handlers for the agent."""

        @self.agent.on_interval(period=60.0)
        async def check_recommendations(ctx: Context):
            """Check for new trading recommendations."""
            if not self.trading_system:
                ctx.logger.warning("No trading system available")
                return

            if not self.state.auto_execute:
                ctx.logger.info("Auto-execution is disabled")
                return

            # Get recent recommendations
            recommendations = self.trading_system.state.recent_recommendations

            if not recommendations:
                ctx.logger.info("No recent recommendations to execute")
                return

            # Execute the most recent recommendation
            latest_rec = recommendations[-1]

            # Check if recommendation is recent (within last 5 minutes)
            rec_time = datetime.fromisoformat(latest_rec.timestamp)
            if datetime.now() - rec_time > timedelta(minutes=5):
                ctx.logger.info("Latest recommendation is too old")
                return

            # Execute recommendation
            await self.execute_recommendation(latest_rec)

        @self.agent.on_interval(period=300.0)
        async def update_positions(ctx: Context):
            """Update positions with current market prices."""
            if not self.state.positions:
                return

            for i, position in enumerate(self.state.positions):
                try:
                    # Get current price
                    ticker = await self.exchange_async.fetch_ticker(position.symbol)
                    current_price = ticker["last"]

                    # Update position
                    self.state.positions[i].current_price = current_price

                    # Calculate unrealized PnL
                    if position.side == OrderSide.BUY:
                        unrealized_pnl = (current_price - position.entry_price) * position.amount
                    else:  # SELL
                        unrealized_pnl = (position.entry_price - current_price) * position.amount

                    self.state.positions[i].unrealized_pnl = unrealized_pnl

                    # Check stop loss and take profit
                    if (
                        position.stop_loss
                        and position.side == OrderSide.BUY
                        and current_price <= position.stop_loss
                    ):
                        ctx.logger.info(f"Stop loss triggered for {position.symbol}")
                        await self.close_position(position.symbol)

                    if (
                        position.stop_loss
                        and position.side == OrderSide.SELL
                        and current_price >= position.stop_loss
                    ):
                        ctx.logger.info(f"Stop loss triggered for {position.symbol}")
                        await self.close_position(position.symbol)

                    if (
                        position.take_profit
                        and position.side == OrderSide.BUY
                        and current_price >= position.take_profit
                    ):
                        ctx.logger.info(f"Take profit triggered for {position.symbol}")
                        await self.close_position(position.symbol)

                    if (
                        position.take_profit
                        and position.side == OrderSide.SELL
                        and current_price <= position.take_profit
                    ):
                        ctx.logger.info(f"Take profit triggered for {position.symbol}")
                        await self.close_position(position.symbol)

                except Exception as e:
                    ctx.logger.error(f"Error updating position for {position.symbol}: {str(e)}")

    async def execute_recommendation(self, recommendation: TradeRecommendation) -> Optional[Order]:
        """Execute a trading recommendation.

        Args:
            recommendation: Trading recommendation to execute

        Returns:
            Executed order or None if execution failed
        """
        self.logger.info(
            f"Executing recommendation for {recommendation.symbol}: {recommendation.signal}"
        )

        try:
            # Determine order side
            if recommendation.signal == TradingSignal.BUY:
                side = OrderSide.BUY
            elif recommendation.signal == TradingSignal.SELL:
                side = OrderSide.SELL
            else:
                self.logger.info(f"No action for {recommendation.signal} signal")
                return None

            # Get position size
            position_size = recommendation.position_size

            if not position_size and self.risk_agent:
                # Get position size from risk agent
                risk_assessment = await self._get_risk_assessment(
                    recommendation.symbol, recommendation.entry_price
                )
                position_size = risk_assessment.get("position_size")

            if not position_size:
                # Default to 1% of available balance
                balance = await self.get_balance()
                position_size = balance * 0.01 / recommendation.entry_price

            # Create order
            order = Order(
                exchange=self.exchange_id,
                symbol=recommendation.symbol,
                type=OrderType.MARKET,
                side=side,
                amount=position_size,
                price=recommendation.entry_price,
                stop_price=recommendation.stop_loss,
                take_profit_price=recommendation.take_profit,
                status=OrderStatus.PENDING,
                timestamp=datetime.now().isoformat(),
            )

            # Execute order
            executed_order = await self._execute_order(order)

            if executed_order:
                # Update state
                self.state.orders.append(executed_order)

                # Create or update position
                await self._update_position(executed_order)

                self.logger.info(f"Successfully executed order for {recommendation.symbol}")
                return executed_order
            else:
                self.logger.error(f"Failed to execute order for {recommendation.symbol}")
                return None

        except Exception as e:
            self.logger.error(
                f"Error executing recommendation for {recommendation.symbol}: {str(e)}"
            )
            return None

    async def _execute_order(self, order: Order) -> Optional[Order]:
        """Execute an order.

        Args:
            order: Order to execute

        Returns:
            Executed order or None if execution failed
        """
        if self.state.dry_run:
            self.logger.info(
                f"DRY RUN: Would execute {order.side} {order.amount} {order.symbol} at {order.price}"
            )

            # Simulate order execution
            order.id = f"dry-run-{int(time.time())}"
            order.status = OrderStatus.CLOSED
            order.filled = order.amount
            order.cost = order.amount * order.price
            order.fee = order.cost * 0.001  # Simulate 0.1% fee

            return order

        try:
            # Execute order on exchange
            if order.type == OrderType.MARKET:
                result = await self.exchange_async.create_order(
                    symbol=order.symbol, type="market", side=order.side, amount=order.amount
                )
            elif order.type == OrderType.LIMIT:
                result = await self.exchange_async.create_order(
                    symbol=order.symbol,
                    type="limit",
                    side=order.side,
                    amount=order.amount,
                    price=order.price,
                )
            else:
                self.logger.error(f"Unsupported order type: {order.type}")
                return None

            # Update order with result
            order.id = result["id"]
            order.status = OrderStatus.CLOSED if result["status"] == "closed" else OrderStatus.OPEN
            order.filled = result["filled"]
            order.cost = result["cost"] if "cost" in result else order.filled * order.price
            order.fee = (
                result["fee"]["cost"] if "fee" in result and "cost" in result["fee"] else 0.0
            )

            # Create stop loss and take profit orders if needed
            if order.stop_price:
                await self._create_stop_loss(order)

            if order.take_profit_price:
                await self._create_take_profit(order)

            return order

        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None

    async def _create_stop_loss(self, order: Order) -> Optional[Order]:
        """Create a stop loss order.

        Args:
            order: Parent order

        Returns:
            Stop loss order or None if creation failed
        """
        if self.state.dry_run:
            self.logger.info(f"DRY RUN: Would create stop loss at {order.stop_price}")

            # Simulate stop loss order
            stop_order = Order(
                id=f"dry-run-sl-{int(time.time())}",
                exchange=self.exchange_id,
                symbol=order.symbol,
                type=OrderType.STOP_LOSS,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.stop_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now().isoformat(),
            )

            self.state.orders.append(stop_order)
            return stop_order

        try:
            # Create stop loss order on exchange
            result = await self.exchange_async.create_order(
                symbol=order.symbol,
                type="stop_loss",
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.stop_price,
            )

            # Create order object
            stop_order = Order(
                id=result["id"],
                exchange=self.exchange_id,
                symbol=order.symbol,
                type=OrderType.STOP_LOSS,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.stop_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now().isoformat(),
            )

            self.state.orders.append(stop_order)
            return stop_order

        except Exception as e:
            self.logger.error(f"Error creating stop loss: {str(e)}")
            return None

    async def _create_take_profit(self, order: Order) -> Optional[Order]:
        """Create a take profit order.

        Args:
            order: Parent order

        Returns:
            Take profit order or None if creation failed
        """
        if self.state.dry_run:
            self.logger.info(f"DRY RUN: Would create take profit at {order.take_profit_price}")

            # Simulate take profit order
            tp_order = Order(
                id=f"dry-run-tp-{int(time.time())}",
                exchange=self.exchange_id,
                symbol=order.symbol,
                type=OrderType.TAKE_PROFIT,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.take_profit_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now().isoformat(),
            )

            self.state.orders.append(tp_order)
            return tp_order

        try:
            # Create take profit order on exchange
            result = await self.exchange_async.create_order(
                symbol=order.symbol,
                type="take_profit",
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.take_profit_price,
            )

            # Create order object
            tp_order = Order(
                id=result["id"],
                exchange=self.exchange_id,
                symbol=order.symbol,
                type=OrderType.TAKE_PROFIT,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.amount,
                price=order.take_profit_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now().isoformat(),
            )

            self.state.orders.append(tp_order)
            return tp_order

        except Exception as e:
            self.logger.error(f"Error creating take profit: {str(e)}")
            return None

    async def _update_position(self, order: Order):
        """Update position based on executed order.

        Args:
            order: Executed order
        """
        # Find existing position
        position = next((p for p in self.state.positions if p.symbol == order.symbol), None)

        if position:
            # Update existing position
            if order.side == position.side:
                # Increase position
                new_amount = position.amount + order.filled
                new_entry_price = (
                    position.entry_price * position.amount + order.price * order.filled
                ) / new_amount

                position.amount = new_amount
                position.entry_price = new_entry_price
                position.orders.append(order)

            else:
                # Decrease position
                if order.filled >= position.amount:
                    # Close position
                    realized_pnl = 0
                    if position.side == OrderSide.BUY:
                        realized_pnl = (order.price - position.entry_price) * position.amount
                    else:
                        realized_pnl = (position.entry_price - order.price) * position.amount

                    position.realized_pnl += realized_pnl
                    position.amount = 0

                    # Remove position
                    self.state.positions = [
                        p for p in self.state.positions if p.symbol != order.symbol
                    ]

                else:
                    # Partially close position
                    realized_pnl = 0
                    if position.side == OrderSide.BUY:
                        realized_pnl = (order.price - position.entry_price) * order.filled
                    else:
                        realized_pnl = (position.entry_price - order.price) * order.filled

                    position.realized_pnl += realized_pnl
                    position.amount -= order.filled
                    position.orders.append(order)
        else:
            # Create new position
            position = Position(
                symbol=order.symbol,
                side=order.side,
                amount=order.filled,
                entry_price=order.price,
                current_price=order.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                stop_loss=order.stop_price,
                take_profit=order.take_profit_price,
                timestamp=datetime.now().isoformat(),
                orders=[order],
            )

            self.state.positions.append(position)

    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close a position.

        Args:
            symbol: Symbol of the position to close

        Returns:
            Order used to close the position or None if closing failed
        """
        # Find position
        position = next((p for p in self.state.positions if p.symbol == symbol), None)

        if not position:
            self.logger.warning(f"No position found for {symbol}")
            return None

        # Create order to close position
        order = Order(
            exchange=self.exchange_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
            amount=position.amount,
            status=OrderStatus.PENDING,
            timestamp=datetime.now().isoformat(),
        )

        # Execute order
        executed_order = await self._execute_order(order)

        if executed_order:
            # Update position
            await self._update_position(executed_order)

            self.logger.info(f"Successfully closed position for {symbol}")
            return executed_order
        else:
            self.logger.error(f"Failed to close position for {symbol}")
            return None

    async def get_balance(self) -> float:
        """Get available balance.

        Returns:
            Available balance
        """
        if self.state.dry_run:
            return 10000.0  # Mock balance for dry run

        try:
            # Get balance from exchange
            balance = await self.exchange_async.fetch_balance()

            # Get USD balance
            usd_balance = balance["total"]["USD"] if "USD" in balance["total"] else 0.0
            usdt_balance = balance["total"]["USDT"] if "USDT" in balance["total"] else 0.0

            return usd_balance + usdt_balance

        except Exception as e:
            self.logger.error(f"Error getting balance: {str(e)}")
            return 0.0

    async def _get_risk_assessment(self, symbol: str, entry_price: float) -> Dict[str, Any]:
        """Get risk assessment from risk agent.

        Args:
            symbol: Trading symbol
            entry_price: Entry price

        Returns:
            Risk assessment
        """
        if not self.risk_agent:
            return {"position_size": None}

        try:
            # Get account balance
            balance = await self.get_balance()

            # Get risk assessment
            assessment = await self.risk_agent._assess_risk(symbol, entry_price, balance)

            return {
                "position_size": assessment.position_sizing.recommended_position_size,
                "stop_loss": assessment.stop_loss.stop_loss_price,
                "take_profit": assessment.stop_loss.take_profit_price,
                "risk_level": assessment.overall_risk_level,
            }

        except Exception as e:
            self.logger.error(f"Error getting risk assessment: {str(e)}")
            return {"position_size": None}

    def set_execution_strategy(self, strategy: ExecutionStrategy):
        """Set the execution strategy.

        Args:
            strategy: Execution strategy to use
        """
        self.state.execution_strategy = strategy
        self.logger.info(f"Set execution strategy to {strategy}")

    def set_auto_execute(self, auto_execute: bool):
        """Set whether to automatically execute recommendations.

        Args:
            auto_execute: Whether to automatically execute recommendations
        """
        self.state.auto_execute = auto_execute
        self.logger.info(f"Set auto-execute to {auto_execute}")

    def set_dry_run(self, dry_run: bool):
        """Set whether to run in dry run mode.

        Args:
            dry_run: Whether to run in dry run mode
        """
        self.state.dry_run = dry_run
        self.logger.info(f"Set dry run to {dry_run}")

    async def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history.

        Args:
            symbol: Symbol to filter by (optional)

        Returns:
            List of orders
        """
        if symbol:
            return [o for o in self.state.orders if o.symbol == symbol]
        else:
            return self.state.orders

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions.

        Args:
            symbol: Symbol to filter by (optional)

        Returns:
            List of positions
        """
        if symbol:
            return [p for p in self.state.positions if p.symbol == symbol]
        else:
            return self.state.positions


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Get API credentials from environment variables
    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="execution_trading_system",
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
    )

    # Create risk agent
    risk_agent = RiskManagementAgent()

    # Create trading execution
    execution = TradingExecution(
        name="trading_execution",
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        trading_system=trading_system,
        risk_agent=risk_agent,
        dry_run=True,  # Start in dry run mode for safety
    )

    # Set auto-execute to true
    execution.set_auto_execute(True)

    # Start agents
    await asyncio.gather(
        trading_system.start_all_agents(), risk_agent.run_async(), execution.run_async()
    )


if __name__ == "__main__":
    asyncio.run(main())
