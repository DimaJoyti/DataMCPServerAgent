"""
Backtesting Framework for Trading Strategies

Comprehensive backtesting engine for evaluating algorithmic trading strategies
with detailed performance metrics and risk analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core.base_models import MarketData
from ..core.enums import OrderSide
from .base_strategy import EnhancedBaseStrategy, StrategySignal, StrategySignalData, StrategyState


@dataclass
class BacktestTrade:
    """Represents a completed trade in backtesting."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: OrderSide
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_percentage: Decimal
    commission: Decimal = Decimal("0")
    strategy_id: str = ""
    signal_strength: float = 0.0
    signal_confidence: float = 0.0


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics."""

    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # PnL metrics
    total_pnl: Decimal = Decimal("0")
    total_pnl_percentage: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")

    # Risk metrics
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_percentage: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Advanced metrics
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    expectancy: Decimal = Decimal("0")

    # Time-based metrics
    avg_trade_duration: timedelta = timedelta()
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Exposure metrics
    market_exposure: float = 0.0  # Percentage of time in market

    def calculate_metrics(self, trades: List[BacktestTrade], initial_capital: Decimal) -> None:
        """Calculate all metrics from trade list."""
        if not trades:
            return

        self.total_trades = len(trades)

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # PnL calculations
        self.total_pnl = sum(t.pnl for t in trades)
        self.total_pnl_percentage = (self.total_pnl / initial_capital) * 100

        if winning_trades:
            self.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            self.largest_win = max(t.pnl for t in winning_trades)

        if losing_trades:
            self.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            self.largest_loss = min(t.pnl for t in losing_trades)

        # Risk metrics
        self._calculate_drawdown(trades, initial_capital)
        self._calculate_ratios(trades, initial_capital)

        # Advanced metrics
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        self.profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")

        self.expectancy = (self.avg_win * self.win_rate) + (self.avg_loss * (1 - self.win_rate))

        # Time-based metrics
        durations = [(t.exit_time - t.entry_time) for t in trades]
        self.avg_trade_duration = sum(durations, timedelta()) / len(durations)

        self._calculate_consecutive_trades(trades)

    def _calculate_drawdown(self, trades: List[BacktestTrade], initial_capital: Decimal) -> None:
        """Calculate maximum drawdown."""
        if not trades:
            return

        # Calculate equity curve
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)

        # Calculate drawdown
        peak = equity[0]
        max_dd = Decimal("0")
        max_dd_pct = Decimal("0")

        for value in equity[1:]:
            if value > peak:
                peak = value

            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else Decimal("0")

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct

        self.max_drawdown = max_dd
        self.max_drawdown_percentage = max_dd_pct

    def _calculate_ratios(self, trades: List[BacktestTrade], initial_capital: Decimal) -> None:
        """Calculate Sharpe, Sortino, and Calmar ratios."""
        if not trades:
            return

        # Calculate daily returns
        daily_returns = []
        current_capital = initial_capital

        for trade in trades:
            daily_return = float(trade.pnl / current_capital)
            daily_returns.append(daily_return)
            current_capital += trade.pnl

        if not daily_returns:
            return

        returns_array = np.array(daily_returns)

        # Sharpe Ratio (assuming risk-free rate of 0)
        if np.std(returns_array) > 0:
            self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)

        # Sortino Ratio (downside deviation)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                self.sortino_ratio = np.mean(returns_array) / downside_std * np.sqrt(252)

        # Calmar Ratio
        annual_return = float(self.total_pnl_percentage) / 100
        if self.max_drawdown_percentage > 0:
            self.calmar_ratio = annual_return / float(self.max_drawdown_percentage) * 100

    def _calculate_consecutive_trades(self, trades: List[BacktestTrade]) -> None:
        """Calculate maximum consecutive wins and losses."""
        if not trades:
            return

        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        self.max_consecutive_wins = max_wins
        self.max_consecutive_losses = max_losses


class BacktestingEngine:
    """Comprehensive backtesting engine for trading strategies."""

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.0005,  # 0.05% slippage
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate

        # Backtest state
        self.current_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.equity_curve: List[Tuple[datetime, Decimal]] = []

        # Performance tracking
        self.metrics = BacktestMetrics()

    async def run_backtest(
        self,
        strategy: EnhancedBaseStrategy,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestMetrics:
        """Run comprehensive backtest for a strategy."""
        try:
            # Reset backtest state
            self._reset_backtest()

            # Set strategy to backtesting mode
            strategy.state = StrategyState.BACKTESTING

            # Get all timestamps across all symbols
            all_timestamps = set()
            for symbol, df in historical_data.items():
                if "timestamp" in df.columns:
                    timestamps = pd.to_datetime(df["timestamp"])
                    all_timestamps.update(
                        timestamps[(timestamps >= start_date) & (timestamps <= end_date)]
                    )

            sorted_timestamps = sorted(all_timestamps)

            # Process each timestamp
            for timestamp in sorted_timestamps:
                await self._process_timestamp(strategy, historical_data, timestamp)

            # Close any remaining open positions
            await self._close_all_positions(historical_data, sorted_timestamps[-1])

            # Calculate final metrics
            self.metrics.calculate_metrics(self.trades, self.initial_capital)

            return self.metrics

        except Exception as e:
            print(f"Error running backtest: {e}")
            return self.metrics

    def _reset_backtest(self) -> None:
        """Reset backtest state."""
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_positions = {}
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.metrics = BacktestMetrics()

    async def _process_timestamp(
        self,
        strategy: EnhancedBaseStrategy,
        historical_data: Dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> None:
        """Process a single timestamp in the backtest."""
        try:
            # Update strategy with current market data
            for symbol in strategy.symbols:
                if symbol in historical_data:
                    df = historical_data[symbol]

                    # Get data up to current timestamp
                    if "timestamp" in df.columns:
                        current_data = df[pd.to_datetime(df["timestamp"]) <= timestamp]
                    else:
                        current_data = (
                            df.iloc[: df.index.get_loc(timestamp) + 1]
                            if timestamp in df.index
                            else df
                        )

                    if len(current_data) > 0:
                        await strategy.update_market_data(symbol, current_data)

                        # Create market data object for current timestamp
                        latest_row = current_data.iloc[-1]
                        market_data = MarketData(
                            symbol=symbol,
                            price=Decimal(str(latest_row["close"])),
                            volume=Decimal(str(latest_row.get("volume", 0))),
                            timestamp=timestamp,
                            open_price=Decimal(str(latest_row.get("open", latest_row["close"]))),
                            high_price=Decimal(str(latest_row.get("high", latest_row["close"]))),
                            low_price=Decimal(str(latest_row.get("low", latest_row["close"]))),
                        )

                        # Generate and process signals
                        signal = await strategy.generate_signal(symbol, market_data)
                        if signal:
                            await self._process_signal(
                                strategy, symbol, signal, market_data, timestamp
                            )

            # Update equity curve
            current_equity = self._calculate_current_equity(historical_data, timestamp)
            self.equity_curve.append((timestamp, current_equity))

        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")

    async def _process_signal(
        self,
        strategy: EnhancedBaseStrategy,
        symbol: str,
        signal: StrategySignalData,
        market_data: MarketData,
        timestamp: datetime,
    ) -> None:
        """Process a trading signal."""
        try:
            # Check if we have an open position
            position_key = f"{strategy.strategy_id}_{symbol}"

            if position_key in self.open_positions:
                # Check for exit signals
                if self._should_exit_position(signal, self.open_positions[position_key]):
                    await self._close_position(position_key, market_data, timestamp, signal)
            else:
                # Check for entry signals
                if signal.signal not in [StrategySignal.HOLD]:
                    await self._open_position(strategy, symbol, signal, market_data, timestamp)

        except Exception as e:
            print(f"Error processing signal for {symbol}: {e}")

    def _should_exit_position(self, signal: StrategySignalData, position: Dict[str, Any]) -> bool:
        """Determine if position should be closed based on signal."""
        position_side = position["side"]

        # Exit on opposite signals
        if position_side == OrderSide.BUY:
            return signal.signal in [
                StrategySignal.SELL,
                StrategySignal.STRONG_SELL,
                StrategySignal.WEAK_SELL,
            ]
        else:
            return signal.signal in [
                StrategySignal.BUY,
                StrategySignal.STRONG_BUY,
                StrategySignal.WEAK_BUY,
            ]

    async def _open_position(
        self,
        strategy: EnhancedBaseStrategy,
        symbol: str,
        signal: StrategySignalData,
        market_data: MarketData,
        timestamp: datetime,
    ) -> None:
        """Open a new position."""
        try:
            # Determine position side
            if signal.signal in [
                StrategySignal.BUY,
                StrategySignal.STRONG_BUY,
                StrategySignal.WEAK_BUY,
            ]:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # Calculate position size
            position_size = await strategy.calculate_position_size(symbol, signal)

            # Apply slippage
            entry_price = market_data.price
            if side == OrderSide.BUY:
                entry_price *= 1 + Decimal(str(self.slippage_rate))
            else:
                entry_price *= 1 - Decimal(str(self.slippage_rate))

            # Calculate required capital
            required_capital = position_size * entry_price
            commission = required_capital * Decimal(str(self.commission_rate))
            total_required = required_capital + commission

            # Check if we have enough capital
            if total_required <= self.current_capital:
                position_key = f"{strategy.strategy_id}_{symbol}"

                self.open_positions[position_key] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": position_size,
                    "entry_price": entry_price,
                    "entry_time": timestamp,
                    "strategy_id": strategy.strategy_id,
                    "signal_strength": signal.strength,
                    "signal_confidence": signal.confidence,
                    "commission": commission,
                }

                # Update capital
                self.current_capital -= total_required

        except Exception as e:
            print(f"Error opening position for {symbol}: {e}")

    async def _close_position(
        self,
        position_key: str,
        market_data: MarketData,
        timestamp: datetime,
        signal: StrategySignalData,
    ) -> None:
        """Close an existing position."""
        try:
            position = self.open_positions[position_key]

            # Apply slippage
            exit_price = market_data.price
            if position["side"] == OrderSide.BUY:
                exit_price *= 1 - Decimal(str(self.slippage_rate))
            else:
                exit_price *= 1 + Decimal(str(self.slippage_rate))

            # Calculate PnL
            if position["side"] == OrderSide.BUY:
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
            else:
                pnl = (position["entry_price"] - exit_price) * position["quantity"]

            # Calculate commission
            exit_commission = position["quantity"] * exit_price * Decimal(str(self.commission_rate))
            total_commission = position["commission"] + exit_commission

            # Net PnL after commissions
            net_pnl = pnl - total_commission
            pnl_percentage = (net_pnl / (position["entry_price"] * position["quantity"])) * 100

            # Create trade record
            trade = BacktestTrade(
                entry_time=position["entry_time"],
                exit_time=timestamp,
                symbol=position["symbol"],
                side=position["side"],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                quantity=position["quantity"],
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                commission=total_commission,
                strategy_id=position["strategy_id"],
                signal_strength=position["signal_strength"],
                signal_confidence=position["signal_confidence"],
            )

            self.trades.append(trade)

            # Update capital
            proceeds = position["quantity"] * exit_price - exit_commission
            self.current_capital += proceeds

            # Remove position
            del self.open_positions[position_key]

        except Exception as e:
            print(f"Error closing position {position_key}: {e}")

    async def _close_all_positions(
        self, historical_data: Dict[str, pd.DataFrame], final_timestamp: datetime
    ) -> None:
        """Close all remaining open positions at the end of backtest."""
        for position_key in list(self.open_positions.keys()):
            position = self.open_positions[position_key]
            symbol = position["symbol"]

            if symbol in historical_data:
                df = historical_data[symbol]
                latest_row = df.iloc[-1]

                market_data = MarketData(
                    symbol=symbol,
                    price=Decimal(str(latest_row["close"])),
                    volume=Decimal(str(latest_row.get("volume", 0))),
                    timestamp=final_timestamp,
                )

                # Create dummy exit signal
                exit_signal = StrategySignalData(
                    signal=StrategySignal.HOLD,
                    strength=0.5,
                    confidence=0.5,
                    timestamp=final_timestamp,
                    price=market_data.price,
                )

                await self._close_position(position_key, market_data, final_timestamp, exit_signal)

    def _calculate_current_equity(
        self, historical_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> Decimal:
        """Calculate current equity including open positions."""
        equity = self.current_capital

        for position in self.open_positions.values():
            symbol = position["symbol"]
            if symbol in historical_data:
                df = historical_data[symbol]

                # Get current price
                if "timestamp" in df.columns:
                    current_data = df[pd.to_datetime(df["timestamp"]) <= timestamp]
                else:
                    current_data = (
                        df.iloc[: df.index.get_loc(timestamp) + 1] if timestamp in df.index else df
                    )

                if len(current_data) > 0:
                    current_price = Decimal(str(current_data.iloc[-1]["close"]))

                    # Calculate unrealized PnL
                    if position["side"] == OrderSide.BUY:
                        unrealized_pnl = (current_price - position["entry_price"]) * position[
                            "quantity"
                        ]
                    else:
                        unrealized_pnl = (position["entry_price"] - current_price) * position[
                            "quantity"
                        ]

                    equity += unrealized_pnl

        return equity

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        return {
            "summary": {
                "initial_capital": float(self.initial_capital),
                "final_capital": float(self.current_capital),
                "total_pnl": float(self.metrics.total_pnl),
                "total_pnl_percentage": float(self.metrics.total_pnl_percentage),
                "total_trades": self.metrics.total_trades,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown": float(self.metrics.max_drawdown),
                "max_drawdown_percentage": float(self.metrics.max_drawdown_percentage),
            },
            "detailed_metrics": {
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
                "avg_win": float(self.metrics.avg_win),
                "avg_loss": float(self.metrics.avg_loss),
                "largest_win": float(self.metrics.largest_win),
                "largest_loss": float(self.metrics.largest_loss),
                "sortino_ratio": self.metrics.sortino_ratio,
                "calmar_ratio": self.metrics.calmar_ratio,
                "expectancy": float(self.metrics.expectancy),
                "avg_trade_duration": str(self.metrics.avg_trade_duration),
                "max_consecutive_wins": self.metrics.max_consecutive_wins,
                "max_consecutive_losses": self.metrics.max_consecutive_losses,
            },
            "trades": [
                {
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "quantity": float(trade.quantity),
                    "pnl": float(trade.pnl),
                    "pnl_percentage": float(trade.pnl_percentage),
                    "commission": float(trade.commission),
                }
                for trade in self.trades
            ],
            "equity_curve": [
                {"timestamp": timestamp.isoformat(), "equity": float(equity)}
                for timestamp, equity in self.equity_curve
            ],
        }
