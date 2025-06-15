"""
Arbitrage Trading Strategies

Implements arbitrage-based algorithmic trading strategies including:
- Pairs Trading Strategy
- Statistical Arbitrage Strategy
- Cross-Exchange Arbitrage Strategy
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..core.base_models import MarketData
from ..core.enums import OrderSide, StrategyType
from .base_strategy import EnhancedBaseStrategy, StrategySignal, StrategySignalData


class PairsTradingStrategy(EnhancedBaseStrategy):
    """Pairs trading strategy based on statistical relationships between assets."""

    def __init__(
        self,
        strategy_id: str,
        symbol_pairs: List[Tuple[str, str]],  # List of (symbol1, symbol2) pairs
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None,
    ):
        default_params = {
            "lookback_period": 60,
            "entry_threshold": 2.0,  # Z-score threshold for entry
            "exit_threshold": 0.5,  # Z-score threshold for exit
            "stop_loss_threshold": 3.5,  # Z-score threshold for stop loss
            "min_correlation": 0.7,  # Minimum correlation for pair validity
            "cointegration_pvalue": 0.05,  # P-value threshold for cointegration
            "half_life_max": 30,  # Maximum half-life for mean reversion
            "volume_filter": True,
        }

        if parameters:
            default_params.update(parameters)

        # Extract all symbols from pairs
        all_symbols = list(set([symbol for pair in symbol_pairs for symbol in pair]))

        super().__init__(
            strategy_id=strategy_id,
            name="Pairs Trading Strategy",
            strategy_type=StrategyType.ARBITRAGE,
            symbols=all_symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters,
        )

        self.symbol_pairs = symbol_pairs
        self.pair_relationships: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.active_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    async def generate_signal(
        self, symbol: str, market_data: MarketData
    ) -> Optional[StrategySignalData]:
        """Generate pairs trading signal."""
        try:
            signals = []

            # Check all pairs involving this symbol
            for pair in self.symbol_pairs:
                if symbol in pair:
                    signal = await self._generate_pair_signal(pair, symbol, market_data)
                    if signal:
                        signals.append(signal)

            # Return the strongest signal
            if signals:
                return max(signals, key=lambda s: s.strength * s.confidence)

            return None

        except Exception as e:
            self.logger.error(f"Error generating pairs trading signal for {symbol}: {e}")
            return None

    async def _generate_pair_signal(
        self, pair: Tuple[str, str], symbol: str, market_data: MarketData
    ) -> Optional[StrategySignalData]:
        """Generate signal for a specific pair."""
        try:
            symbol1, symbol2 = pair
            other_symbol = symbol2 if symbol == symbol1 else symbol1

            # Get historical data for both symbols
            df1 = self.market_data.get(symbol1)
            df2 = self.market_data.get(symbol2)

            if df1 is None or df2 is None or len(df1) < self.parameters["lookback_period"]:
                return None

            # Align data by timestamp
            df1_aligned = df1.set_index("timestamp") if "timestamp" in df1.columns else df1
            df2_aligned = df2.set_index("timestamp") if "timestamp" in df2.columns else df2

            # Get common time range
            common_index = df1_aligned.index.intersection(df2_aligned.index)
            if len(common_index) < self.parameters["lookback_period"]:
                return None

            price1 = df1_aligned.loc[common_index, "close"]
            price2 = df2_aligned.loc[common_index, "close"]

            # Update pair relationship
            await self._update_pair_relationship(pair, price1, price2)

            relationship = self.pair_relationships.get(pair)
            if not relationship or not relationship["is_valid"]:
                return None

            # Calculate current spread
            current_price1 = price1.iloc[-1]
            current_price2 = price2.iloc[-1]

            # Calculate spread using the relationship
            if relationship["hedge_ratio"]:
                spread = current_price1 - relationship["hedge_ratio"] * current_price2
            else:
                spread = np.log(current_price1) - np.log(current_price2)

            # Calculate Z-score
            spread_series = self._calculate_spread_series(price1, price2, relationship)
            spread_mean = spread_series.mean()
            spread_std = spread_series.std()

            if spread_std == 0:
                return None

            z_score = (spread - spread_mean) / spread_std

            # Generate signals
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0

            # Entry signals
            if abs(z_score) >= self.parameters["entry_threshold"]:
                if z_score > 0:  # Spread too high - short symbol1, long symbol2
                    if symbol == symbol1:
                        signal = StrategySignal.SELL
                    else:
                        signal = StrategySignal.BUY
                else:  # Spread too low - long symbol1, short symbol2
                    if symbol == symbol1:
                        signal = StrategySignal.BUY
                    else:
                        signal = StrategySignal.SELL

                strength = min(abs(z_score) / self.parameters["entry_threshold"], 1.0)
                confidence = relationship["confidence"]

                # Strong signal for extreme Z-scores
                if abs(z_score) >= self.parameters["entry_threshold"] * 1.5:
                    if signal == StrategySignal.BUY:
                        signal = StrategySignal.STRONG_BUY
                    else:
                        signal = StrategySignal.STRONG_SELL
                    confidence = min(confidence * 1.2, 0.95)

            # Exit signals for existing positions
            elif pair in self.active_pairs and abs(z_score) <= self.parameters["exit_threshold"]:
                active_position = self.active_pairs[pair]
                if active_position["symbol"] == symbol:
                    # Exit signal - reverse the original position
                    if active_position["side"] == OrderSide.BUY:
                        signal = StrategySignal.WEAK_SELL
                    else:
                        signal = StrategySignal.WEAK_BUY

                    strength = 0.5
                    confidence = 0.7

            # Stop loss signals
            elif abs(z_score) >= self.parameters["stop_loss_threshold"]:
                if pair in self.active_pairs:
                    active_position = self.active_pairs[pair]
                    if active_position["symbol"] == symbol:
                        # Emergency exit
                        if active_position["side"] == OrderSide.BUY:
                            signal = StrategySignal.STRONG_SELL
                        else:
                            signal = StrategySignal.STRONG_BUY

                        strength = 1.0
                        confidence = 0.9

            if signal == StrategySignal.HOLD:
                return None

            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    "z_score": z_score,
                    "spread": spread,
                    "spread_mean": spread_mean,
                    "spread_std": spread_std,
                    "hedge_ratio": relationship["hedge_ratio"],
                    "correlation": relationship["correlation"],
                    "half_life": relationship.get("half_life", 0),
                },
                metadata={
                    "strategy": "Pairs_Trading",
                    "pair": f"{symbol1}_{symbol2}",
                    "other_symbol": other_symbol,
                    "timeframe": self.timeframe,
                },
            )

        except Exception as e:
            self.logger.error(f"Error generating pair signal for {pair}: {e}")
            return None

    async def _update_pair_relationship(
        self, pair: Tuple[str, str], price1: pd.Series, price2: pd.Series
    ) -> None:
        """Update statistical relationship between a pair."""
        try:
            # Calculate correlation
            correlation = price1.corr(price2)

            # Calculate hedge ratio using linear regression
            X = price2.values.reshape(-1, 1)
            y = price1.values

            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]

            # Calculate cointegration (simplified)
            residuals = y - reg.predict(X)

            # ADF test would be more appropriate here
            # For now, use a simple stationarity check
            adf_pvalue = 0.01 if np.std(residuals) < np.std(price1) * 0.5 else 0.1

            # Calculate half-life of mean reversion
            half_life = self._calculate_half_life(residuals)

            # Determine if pair is valid for trading
            is_valid = (
                abs(correlation) >= self.parameters["min_correlation"]
                and adf_pvalue <= self.parameters["cointegration_pvalue"]
                and half_life <= self.parameters["half_life_max"]
            )

            # Calculate confidence based on statistical measures
            confidence = min(
                abs(correlation) * 0.4
                + (1 - adf_pvalue) * 0.3
                + max(
                    0,
                    (self.parameters["half_life_max"] - half_life)
                    / self.parameters["half_life_max"],
                )
                * 0.3,
                0.95,
            )

            self.pair_relationships[pair] = {
                "correlation": correlation,
                "hedge_ratio": hedge_ratio,
                "adf_pvalue": adf_pvalue,
                "half_life": half_life,
                "is_valid": is_valid,
                "confidence": confidence,
                "last_updated": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error updating pair relationship for {pair}: {e}")

    def _calculate_spread_series(
        self, price1: pd.Series, price2: pd.Series, relationship: Dict[str, Any]
    ) -> pd.Series:
        """Calculate spread series for the pair."""
        if relationship["hedge_ratio"]:
            return price1 - relationship["hedge_ratio"] * price2
        else:
            return np.log(price1) - np.log(price2)

    def _calculate_half_life(self, residuals: np.ndarray) -> float:
        """Calculate half-life of mean reversion."""
        try:
            # Simple half-life calculation
            residuals_lagged = residuals[:-1]
            residuals_diff = np.diff(residuals)

            if len(residuals_lagged) == 0:
                return float("inf")

            # Linear regression: residuals_diff = alpha + beta * residuals_lagged
            X = residuals_lagged.reshape(-1, 1)
            y = residuals_diff

            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]

            if beta >= 0:
                return float("inf")  # No mean reversion

            half_life = -np.log(2) / beta
            return max(half_life, 1.0)  # Minimum 1 period

        except:
            return float("inf")

    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size for pairs trading."""
        base_size = self.max_position_size * Decimal(str(signal.strength))

        # Adjust based on Z-score magnitude
        z_score = abs(signal.indicators.get("z_score", 0))
        z_factor = min(z_score / 2.0, 1.5)  # Cap at 1.5x

        # Adjust based on correlation strength
        correlation = abs(signal.indicators.get("correlation", 0.5))
        corr_factor = correlation  # Higher correlation = larger position

        # Adjust based on half-life (faster mean reversion = larger position)
        half_life = signal.indicators.get("half_life", 30)
        half_life_factor = max(0.5, (30 - half_life) / 30)

        adjusted_size = (
            base_size
            * Decimal(str(z_factor))
            * Decimal(str(corr_factor))
            * Decimal(str(half_life_factor))
        )

        return min(adjusted_size, self.max_position_size)


class StatisticalArbitrageStrategy(EnhancedBaseStrategy):
    """Statistical arbitrage strategy using multiple assets."""

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None,
    ):
        default_params = {
            "lookback_period": 100,
            "min_assets": 5,
            "entry_threshold": 1.5,
            "exit_threshold": 0.3,
            "correlation_threshold": 0.6,
            "rebalance_frequency": 24,  # hours
            "max_positions": 10,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(
            strategy_id=strategy_id,
            name="Statistical Arbitrage Strategy",
            strategy_type=StrategyType.ARBITRAGE,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters,
        )

        self.portfolio_weights: Dict[str, float] = {}
        self.expected_returns: Dict[str, float] = {}
        self.last_rebalance = datetime.now()

    async def generate_signal(
        self, symbol: str, market_data: MarketData
    ) -> Optional[StrategySignalData]:
        """Generate statistical arbitrage signal."""
        try:
            # Check if rebalancing is needed
            if (datetime.now() - self.last_rebalance).total_seconds() > self.parameters[
                "rebalance_frequency"
            ] * 3600:
                await self._rebalance_portfolio()
                self.last_rebalance = datetime.now()

            if symbol not in self.portfolio_weights:
                return None

            # Get historical data
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters["lookback_period"]:
                return None

            # Calculate expected vs actual returns
            returns = df["close"].pct_change().dropna()
            current_return = returns.iloc[-1]
            expected_return = self.expected_returns.get(symbol, 0)

            # Calculate Z-score of return deviation
            return_std = returns.rolling(window=20).std().iloc[-1]
            if return_std == 0:
                return None

            return_z_score = (current_return - expected_return) / return_std

            # Generate signals based on statistical deviation
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0

            if abs(return_z_score) >= self.parameters["entry_threshold"]:
                if return_z_score > 0:  # Asset outperforming - potential sell
                    signal = StrategySignal.SELL
                else:  # Asset underperforming - potential buy
                    signal = StrategySignal.BUY

                strength = min(abs(return_z_score) / self.parameters["entry_threshold"], 1.0)
                confidence = 0.7

            elif abs(return_z_score) <= self.parameters["exit_threshold"]:
                # Mean reversion - exit signal
                if return_z_score > 0:
                    signal = StrategySignal.WEAK_BUY
                else:
                    signal = StrategySignal.WEAK_SELL

                strength = 0.3
                confidence = 0.5

            if signal == StrategySignal.HOLD:
                return None

            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    "return_z_score": return_z_score,
                    "current_return": current_return,
                    "expected_return": expected_return,
                    "return_std": return_std,
                    "portfolio_weight": self.portfolio_weights.get(symbol, 0),
                },
                metadata={"strategy": "Statistical_Arbitrage", "timeframe": self.timeframe},
            )

        except Exception as e:
            self.logger.error(f"Error generating statistical arbitrage signal for {symbol}: {e}")
            return None

    async def _rebalance_portfolio(self) -> None:
        """Rebalance portfolio weights based on statistical relationships."""
        try:
            # Get return data for all symbols
            returns_data = {}
            for symbol in self.symbols:
                df = self.market_data.get(symbol)
                if df is not None and len(df) >= self.parameters["lookback_period"]:
                    returns = df["close"].pct_change().dropna()
                    returns_data[symbol] = returns.tail(self.parameters["lookback_period"])

            if len(returns_data) < self.parameters["min_assets"]:
                return

            # Create returns matrix
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 20:  # Minimum data points
                return

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()

            # Simple equal-weight portfolio (can be enhanced with optimization)
            n_assets = len(returns_data)
            equal_weight = 1.0 / n_assets

            # Update weights and expected returns
            for symbol in returns_data.keys():
                self.portfolio_weights[symbol] = equal_weight
                self.expected_returns[symbol] = returns_data[symbol].mean()

            self.logger.info(f"Portfolio rebalanced with {n_assets} assets")

        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")

    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size for statistical arbitrage."""
        base_size = self.max_position_size * Decimal(str(signal.strength))

        # Adjust based on portfolio weight
        portfolio_weight = signal.indicators.get("portfolio_weight", 0.1)
        weight_factor = portfolio_weight * 2  # Scale up the weight influence

        # Adjust based on Z-score magnitude
        z_score = abs(signal.indicators.get("return_z_score", 0))
        z_factor = min(z_score, 2.0)  # Cap at 2x

        adjusted_size = base_size * Decimal(str(weight_factor)) * Decimal(str(z_factor))

        return min(adjusted_size, self.max_position_size)
