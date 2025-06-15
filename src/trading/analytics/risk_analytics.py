"""
Real-time risk analytics for institutional trading.
"""

import asyncio
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_models import BasePosition
from ..market_data.data_types import MarketDataSnapshot


class RiskAnalytics:
    """
    Real-time risk analytics engine.

    Features:
    - Value at Risk (VaR) calculation
    - Position risk monitoring
    - Concentration risk analysis
    - Correlation analysis
    - Stress testing
    - Risk limit monitoring
    """

    def __init__(
        self,
        name: str = "RiskAnalytics",
        var_confidence_levels: List[float] = [0.95, 0.99],
        lookback_days: int = 252,
        calculation_frequency_seconds: int = 30,
    ):
        self.name = name
        self.var_confidence_levels = var_confidence_levels
        self.lookback_days = lookback_days
        self.calculation_frequency = calculation_frequency_seconds

        self.logger = logging.getLogger(f"RiskAnalytics.{name}")
        self.is_running = False

        # Data storage
        self.positions: Dict[str, BasePosition] = {}
        self.market_data: Dict[str, MarketDataSnapshot] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_days))
        self.returns_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_days))

        # Portfolio data
        self.portfolio_value = Decimal("1000000")  # $1M default
        self.portfolio_returns: deque = deque(maxlen=lookback_days)
        self.portfolio_var_history: deque = deque(maxlen=100)

        # Risk metrics
        self.var_metrics: Dict[str, Dict] = {}
        self.concentration_metrics: Dict[str, Any] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

        # Risk limits
        self.position_limits: Dict[str, Decimal] = {}
        self.sector_limits: Dict[str, Decimal] = {}
        self.var_limit = Decimal("50000")  # $50k VaR limit
        self.concentration_limit = 0.25  # 25% max concentration

        # Alert handlers
        self.risk_alert_handlers: List[callable] = []

        # Performance tracking
        self.calculation_count = 0
        self.calculation_latency_us: deque = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the risk analytics engine."""
        self.logger.info(f"Starting risk analytics: {self.name}")
        self.is_running = True

        # Start calculation tasks
        asyncio.create_task(self._calculate_var())
        asyncio.create_task(self._monitor_concentration())
        asyncio.create_task(self._calculate_correlations())
        asyncio.create_task(self._monitor_limits())

        self.logger.info(f"Risk analytics started: {self.name}")

    async def stop(self) -> None:
        """Stop the risk analytics engine."""
        self.logger.info(f"Stopping risk analytics: {self.name}")
        self.is_running = False
        self.logger.info(f"Risk analytics stopped: {self.name}")

    async def update_position(self, position: BasePosition) -> None:
        """Update position data."""
        self.positions[position.symbol] = position

    async def update_market_data(self, snapshot: MarketDataSnapshot) -> None:
        """Update market data and calculate returns."""
        symbol = snapshot.symbol
        self.market_data[symbol] = snapshot

        current_price = snapshot.current_price
        if current_price:
            # Store price history
            self.price_history[symbol].append((datetime.utcnow(), float(current_price)))

            # Calculate returns if we have previous price
            if len(self.price_history[symbol]) > 1:
                prev_price = self.price_history[symbol][-2][1]
                if prev_price > 0:
                    return_pct = (float(current_price) - prev_price) / prev_price
                    self.returns_history[symbol].append(return_pct)

    async def _calculate_var(self) -> None:
        """Calculate Value at Risk metrics."""
        while self.is_running:
            try:
                start_time = datetime.utcnow()

                # Calculate individual position VaR
                for symbol in self.positions.keys():
                    await self._calculate_position_var(symbol)

                # Calculate portfolio VaR
                await self._calculate_portfolio_var()

                # Track calculation latency
                calculation_time = (datetime.utcnow() - start_time).total_seconds() * 1_000_000
                self.calculation_latency_us.append(calculation_time)
                self.calculation_count += 1

                await asyncio.sleep(self.calculation_frequency)

            except Exception as e:
                self.logger.error(f"Error calculating VaR: {str(e)}")
                await asyncio.sleep(self.calculation_frequency)

    async def _calculate_position_var(self, symbol: str) -> None:
        """Calculate VaR for individual position."""
        position = self.positions.get(symbol)
        returns = list(self.returns_history[symbol])

        if not position or len(returns) < 30:
            return

        position_value = position.market_value or Decimal("0")
        if position_value == 0:
            return

        var_metrics = {
            "symbol": symbol,
            "position_value": float(position_value),
            "timestamp": datetime.utcnow(),
        }

        # Calculate VaR for each confidence level
        for confidence in self.var_confidence_levels:
            var_return = self._calculate_var_return(returns, confidence)
            var_amount = float(position_value) * abs(var_return)

            var_metrics[f"var_{int(confidence*100)}"] = {
                "return": var_return,
                "amount": var_amount,
                "percentage": var_return * 100,
            }

        # Calculate volatility
        if len(returns) > 1:
            volatility = statistics.stdev(returns)
            annualized_vol = volatility * (252**0.5)  # Annualize

            var_metrics["volatility"] = {"daily": volatility, "annualized": annualized_vol}

        self.var_metrics[symbol] = var_metrics

    async def _calculate_portfolio_var(self) -> None:
        """Calculate portfolio-level VaR."""
        if not self.portfolio_returns or len(self.portfolio_returns) < 30:
            return

        returns = list(self.portfolio_returns)

        portfolio_var = {
            "portfolio_value": float(self.portfolio_value),
            "timestamp": datetime.utcnow(),
        }

        # Calculate VaR for each confidence level
        for confidence in self.var_confidence_levels:
            var_return = self._calculate_var_return(returns, confidence)
            var_amount = float(self.portfolio_value) * abs(var_return)

            portfolio_var[f"var_{int(confidence*100)}"] = {
                "return": var_return,
                "amount": var_amount,
                "percentage": var_return * 100,
            }

        # Calculate portfolio volatility
        if len(returns) > 1:
            volatility = statistics.stdev(returns)
            annualized_vol = volatility * (252**0.5)

            portfolio_var["volatility"] = {"daily": volatility, "annualized": annualized_vol}

        self.var_metrics["PORTFOLIO"] = portfolio_var
        self.portfolio_var_history.append(
            (datetime.utcnow(), portfolio_var.get("var_95", {}).get("amount", 0))
        )

    def _calculate_var_return(self, returns: List[float], confidence: float) -> float:
        """Calculate VaR return for given confidence level."""
        if len(returns) < 10:
            return 0.0

        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))

        return sorted_returns[var_index]

    async def _monitor_concentration(self) -> None:
        """Monitor concentration risk."""
        while self.is_running:
            try:
                await self._calculate_concentration_metrics()
                await asyncio.sleep(self.calculation_frequency)

            except Exception as e:
                self.logger.error(f"Error monitoring concentration: {str(e)}")
                await asyncio.sleep(self.calculation_frequency)

    async def _calculate_concentration_metrics(self) -> None:
        """Calculate concentration risk metrics."""
        if not self.positions:
            return

        # Calculate position concentrations
        total_portfolio_value = sum(
            pos.market_value or Decimal("0") for pos in self.positions.values()
        )

        if total_portfolio_value == 0:
            return

        position_concentrations = {}
        sector_concentrations = defaultdict(Decimal)

        for symbol, position in self.positions.items():
            position_value = position.market_value or Decimal("0")
            concentration = float(position_value / total_portfolio_value)
            position_concentrations[symbol] = concentration

            # Group by sector (simplified - would use actual sector data)
            sector = self._get_sector(symbol)
            sector_concentrations[sector] += position_value

        # Calculate sector concentrations
        sector_concentrations_pct = {
            sector: float(value / total_portfolio_value)
            for sector, value in sector_concentrations.items()
        }

        # Find largest concentrations
        largest_position = max(position_concentrations.values()) if position_concentrations else 0
        largest_sector = max(sector_concentrations_pct.values()) if sector_concentrations_pct else 0

        self.concentration_metrics = {
            "timestamp": datetime.utcnow(),
            "total_portfolio_value": float(total_portfolio_value),
            "position_concentrations": position_concentrations,
            "sector_concentrations": sector_concentrations_pct,
            "largest_position_pct": largest_position,
            "largest_sector_pct": largest_sector,
            "herfindahl_index": sum(c**2 for c in position_concentrations.values()),
        }

        # Check concentration limits
        if largest_position > self.concentration_limit:
            await self._trigger_risk_alert(
                "CONCENTRATION_LIMIT_EXCEEDED",
                {
                    "type": "position",
                    "concentration": largest_position,
                    "limit": self.concentration_limit,
                },
            )

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified implementation)."""
        # This would integrate with actual sector classification
        tech_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        finance_symbols = ["JPM", "BAC", "WFC", "GS", "MS"]

        if symbol in tech_symbols:
            return "Technology"
        elif symbol in finance_symbols:
            return "Finance"
        else:
            return "Other"

    async def _calculate_correlations(self) -> None:
        """Calculate correlation matrix."""
        while self.is_running:
            try:
                await asyncio.sleep(self.calculation_frequency * 2)  # Less frequent

                symbols = list(self.returns_history.keys())

                # Calculate pairwise correlations
                for i, symbol1 in enumerate(symbols):
                    for symbol2 in symbols[i + 1 :]:
                        correlation = self._calculate_correlation(symbol1, symbol2)
                        if correlation is not None:
                            self.correlation_matrix[(symbol1, symbol2)] = correlation
                            self.correlation_matrix[(symbol2, symbol1)] = correlation

            except Exception as e:
                self.logger.error(f"Error calculating correlations: {str(e)}")

    def _calculate_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation between two symbols."""
        returns1 = list(self.returns_history[symbol1])
        returns2 = list(self.returns_history[symbol2])

        if len(returns1) < 30 or len(returns2) < 30:
            return None

        # Align returns by taking minimum length
        min_length = min(len(returns1), len(returns2))
        returns1 = returns1[-min_length:]
        returns2 = returns2[-min_length:]

        try:
            correlation = statistics.correlation(returns1, returns2)
            return correlation
        except:
            return None

    async def _monitor_limits(self) -> None:
        """Monitor risk limits and trigger alerts."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check VaR limits
                portfolio_var = self.var_metrics.get("PORTFOLIO", {})
                var_95 = portfolio_var.get("var_95", {}).get("amount", 0)

                if var_95 > float(self.var_limit):
                    await self._trigger_risk_alert(
                        "VAR_LIMIT_EXCEEDED",
                        {
                            "var_amount": var_95,
                            "var_limit": float(self.var_limit),
                            "confidence": 95,
                        },
                    )

                # Check position limits
                for symbol, position in self.positions.items():
                    if symbol in self.position_limits:
                        position_size = abs(position.quantity)
                        limit = self.position_limits[symbol]

                        if position_size > limit:
                            await self._trigger_risk_alert(
                                "POSITION_LIMIT_EXCEEDED",
                                {
                                    "symbol": symbol,
                                    "position_size": float(position_size),
                                    "limit": float(limit),
                                },
                            )

            except Exception as e:
                self.logger.error(f"Error monitoring limits: {str(e)}")

    async def _trigger_risk_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger risk alert handlers."""
        alert = {"type": alert_type, "timestamp": datetime.utcnow(), "data": data}

        self.logger.warning(f"Risk alert: {alert_type} - {data}")

        for handler in self.risk_alert_handlers:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                self.logger.error(f"Error in risk alert handler: {str(e)}")

    def get_var_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get VaR metrics."""
        if symbol:
            return self.var_metrics.get(symbol, {})
        return self.var_metrics.copy()

    def get_concentration_metrics(self) -> Dict[str, Any]:
        """Get concentration risk metrics."""
        return self.concentration_metrics.copy()

    def get_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get correlation matrix."""
        return self.correlation_matrix.copy()

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        portfolio_var = self.var_metrics.get("PORTFOLIO", {})

        return {
            "timestamp": datetime.utcnow(),
            "portfolio_value": float(self.portfolio_value),
            "var_metrics": portfolio_var,
            "concentration_metrics": self.concentration_metrics,
            "active_positions": len(self.positions),
            "risk_limits": {
                "var_limit": float(self.var_limit),
                "concentration_limit": self.concentration_limit,
                "position_limits": {k: float(v) for k, v in self.position_limits.items()},
            },
        }

    def set_risk_limits(
        self,
        var_limit: Optional[Decimal] = None,
        concentration_limit: Optional[float] = None,
        position_limits: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Update risk limits."""
        if var_limit is not None:
            self.var_limit = var_limit

        if concentration_limit is not None:
            self.concentration_limit = concentration_limit

        if position_limits is not None:
            self.position_limits.update(position_limits)

    def add_risk_alert_handler(self, handler: callable) -> None:
        """Add risk alert handler."""
        self.risk_alert_handlers.append(handler)

    def get_analytics_performance(self) -> Dict[str, Any]:
        """Get analytics performance metrics."""
        avg_latency = (
            sum(self.calculation_latency_us) / len(self.calculation_latency_us)
            if self.calculation_latency_us
            else 0
        )

        return {
            "calculation_count": self.calculation_count,
            "average_latency_us": avg_latency,
            "active_positions": len(self.positions),
            "symbols_tracked": len(self.returns_history),
            "correlation_pairs": len(self.correlation_matrix),
        }
