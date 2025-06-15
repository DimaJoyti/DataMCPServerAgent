"""
Feature engineering for machine learning in trading.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..market_data.data_types import OHLCV, Quote, Trade


class FeatureEngineer:
    """
    Advanced feature engineering for trading ML models.

    Features:
    - Technical indicators
    - Statistical features
    - Market microstructure features
    - Cross-asset features
    - Alternative data features
    """

    def __init__(
        self,
        name: str = "FeatureEngineer",
        lookback_periods: List[int] = [5, 10, 20, 50, 100],
        update_frequency_seconds: int = 1,
    ):
        self.name = name
        self.lookback_periods = lookback_periods
        self.update_frequency = update_frequency_seconds

        self.logger = logging.getLogger(f"FeatureEngineer.{name}")
        self.is_running = False

        # Feature cache
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.raw_data_cache: Dict[str, deque] = {}

        # Performance tracking
        self.feature_calculation_count = 0
        self.calculation_latency_us: deque = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the feature engineer."""
        self.logger.info(f"Starting feature engineer: {self.name}")
        self.is_running = True
        self.logger.info(f"Feature engineer started: {self.name}")

    async def stop(self) -> None:
        """Stop the feature engineer."""
        self.logger.info(f"Stopping feature engineer: {self.name}")
        self.is_running = False
        self.logger.info(f"Feature engineer stopped: {self.name}")

    async def extract_features(
        self, symbol: str, market_data: List[Dict], feature_types: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Extract features from market data.

        Args:
            symbol: Trading symbol
            market_data: List of market data points
            feature_types: Types of features to extract

        Returns:
            DataFrame with features
        """
        start_time = datetime.utcnow()

        try:
            if len(market_data) < max(self.lookback_periods):
                return None

            # Convert to DataFrame
            df = self._prepare_dataframe(market_data)
            if df.empty:
                return None

            # Extract different types of features
            features = pd.DataFrame(index=df.index)

            if not feature_types or "technical" in feature_types:
                technical_features = self._extract_technical_features(df)
                features = pd.concat([features, technical_features], axis=1)

            if not feature_types or "statistical" in feature_types:
                statistical_features = self._extract_statistical_features(df)
                features = pd.concat([features, statistical_features], axis=1)

            if not feature_types or "microstructure" in feature_types:
                microstructure_features = self._extract_microstructure_features(df)
                features = pd.concat([features, microstructure_features], axis=1)

            if not feature_types or "momentum" in feature_types:
                momentum_features = self._extract_momentum_features(df)
                features = pd.concat([features, momentum_features], axis=1)

            if not feature_types or "volatility" in feature_types:
                volatility_features = self._extract_volatility_features(df)
                features = pd.concat([features, volatility_features], axis=1)

            # Remove NaN values
            features = features.fillna(method="ffill").fillna(0)

            # Cache features
            self.feature_cache[symbol] = features

            # Track performance
            calculation_time = (datetime.utcnow() - start_time).total_seconds() * 1_000_000
            self.calculation_latency_us.append(calculation_time)
            self.feature_calculation_count += 1

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features for {symbol}: {str(e)}")
            return None

    def _prepare_dataframe(self, market_data: List[Dict]) -> pd.DataFrame:
        """Prepare DataFrame from market data."""
        try:
            rows = []

            for item in market_data:
                data = item["data"]
                row = {"timestamp": item["timestamp"], "type": item["type"]}

                if isinstance(data, (Quote, Trade)):
                    if hasattr(data, "price"):
                        row["price"] = float(data.price)
                    if hasattr(data, "size"):
                        row["size"] = float(data.size)
                    if hasattr(data, "bid_price") and data.bid_price:
                        row["bid"] = float(data.bid_price)
                    if hasattr(data, "ask_price") and data.ask_price:
                        row["ask"] = float(data.ask_price)
                    if hasattr(data, "bid_size") and data.bid_size:
                        row["bid_size"] = float(data.bid_size)
                    if hasattr(data, "ask_size") and data.ask_size:
                        row["ask_size"] = float(data.ask_size)

                elif isinstance(data, OHLCV):
                    row.update(
                        {
                            "open": float(data.open),
                            "high": float(data.high),
                            "low": float(data.low),
                            "close": float(data.close),
                            "volume": float(data.volume),
                        }
                    )

                rows.append(row)

            df = pd.DataFrame(rows)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            return df

        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()

    def _extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features."""
        features = pd.DataFrame(index=df.index)

        try:
            # Use price column (could be from trades or close prices)
            price_col = "price" if "price" in df.columns else "close"
            if price_col not in df.columns:
                return features

            prices = df[price_col]

            # Moving averages
            for period in self.lookback_periods:
                if len(prices) >= period:
                    ma = prices.rolling(window=period).mean()
                    features[f"ma_{period}"] = ma
                    features[f"price_ma_ratio_{period}"] = prices / ma
                    features[f"ma_slope_{period}"] = ma.diff(5)

            # Exponential moving averages
            for period in [12, 26, 50]:
                if len(prices) >= period:
                    ema = prices.ewm(span=period).mean()
                    features[f"ema_{period}"] = ema
                    features[f"price_ema_ratio_{period}"] = prices / ema

            # MACD
            if len(prices) >= 26:
                ema12 = prices.ewm(span=12).mean()
                ema26 = prices.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                features["macd"] = macd
                features["macd_signal"] = signal
                features["macd_histogram"] = macd - signal

            # RSI
            if len(prices) >= 14:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            for period in [20, 50]:
                if len(prices) >= period:
                    ma = prices.rolling(window=period).mean()
                    std = prices.rolling(window=period).std()
                    features[f"bb_upper_{period}"] = ma + (2 * std)
                    features[f"bb_lower_{period}"] = ma - (2 * std)
                    features[f"bb_position_{period}"] = (prices - ma) / (2 * std)

            # Price momentum
            for period in [1, 5, 10, 20]:
                if len(prices) >= period:
                    features[f"momentum_{period}"] = prices.pct_change(period)

        except Exception as e:
            self.logger.error(f"Error extracting technical features: {str(e)}")

        return features

    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features."""
        features = pd.DataFrame(index=df.index)

        try:
            price_col = "price" if "price" in df.columns else "close"
            if price_col not in df.columns:
                return features

            prices = df[price_col]

            # Rolling statistics
            for period in self.lookback_periods:
                if len(prices) >= period:
                    rolling_prices = prices.rolling(window=period)

                    features[f"std_{period}"] = rolling_prices.std()
                    features[f"var_{period}"] = rolling_prices.var()
                    features[f"skew_{period}"] = rolling_prices.skew()
                    features[f"kurt_{period}"] = rolling_prices.kurt()
                    features[f"min_{period}"] = rolling_prices.min()
                    features[f"max_{period}"] = rolling_prices.max()
                    features[f"median_{period}"] = rolling_prices.median()
                    features[f"quantile_25_{period}"] = rolling_prices.quantile(0.25)
                    features[f"quantile_75_{period}"] = rolling_prices.quantile(0.75)

            # Returns-based features
            returns = prices.pct_change()
            for period in [5, 10, 20]:
                if len(returns) >= period:
                    rolling_returns = returns.rolling(window=period)

                    features[f"return_mean_{period}"] = rolling_returns.mean()
                    features[f"return_std_{period}"] = rolling_returns.std()
                    features[f"return_skew_{period}"] = rolling_returns.skew()
                    features[f"sharpe_{period}"] = rolling_returns.mean() / rolling_returns.std()

            # Autocorrelation
            for lag in [1, 5, 10]:
                if len(returns) >= lag + 20:
                    features[f"autocorr_{lag}"] = returns.rolling(window=20).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                    )

        except Exception as e:
            self.logger.error(f"Error extracting statistical features: {str(e)}")

        return features

    def _extract_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract market microstructure features."""
        features = pd.DataFrame(index=df.index)

        try:
            # Bid-ask spread features
            if "bid" in df.columns and "ask" in df.columns:
                spread = df["ask"] - df["bid"]
                mid_price = (df["bid"] + df["ask"]) / 2

                features["spread"] = spread
                features["spread_bps"] = (spread / mid_price) * 10000
                features["mid_price"] = mid_price

                # Rolling spread statistics
                for period in [10, 20, 50]:
                    if len(spread) >= period:
                        features[f"spread_mean_{period}"] = spread.rolling(window=period).mean()
                        features[f"spread_std_{period}"] = spread.rolling(window=period).std()

            # Order book imbalance
            if "bid_size" in df.columns and "ask_size" in df.columns:
                total_size = df["bid_size"] + df["ask_size"]
                imbalance = (df["bid_size"] - df["ask_size"]) / total_size
                features["order_imbalance"] = imbalance

                for period in [10, 20]:
                    if len(imbalance) >= period:
                        features[f"imbalance_mean_{period}"] = imbalance.rolling(
                            window=period
                        ).mean()

            # Volume features
            if "size" in df.columns:
                volume = df["size"]

                for period in [10, 20, 50]:
                    if len(volume) >= period:
                        features[f"volume_mean_{period}"] = volume.rolling(window=period).mean()
                        features[f"volume_std_{period}"] = volume.rolling(window=period).std()
                        features[f"volume_ratio_{period}"] = (
                            volume / volume.rolling(window=period).mean()
                        )

        except Exception as e:
            self.logger.error(f"Error extracting microstructure features: {str(e)}")

        return features

    def _extract_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum-based features."""
        features = pd.DataFrame(index=df.index)

        try:
            price_col = "price" if "price" in df.columns else "close"
            if price_col not in df.columns:
                return features

            prices = df[price_col]

            # Rate of change
            for period in [5, 10, 20]:
                if len(prices) >= period:
                    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
                    features[f"roc_{period}"] = roc

            # Momentum oscillator
            for period in [10, 20]:
                if len(prices) >= period:
                    momentum = prices - prices.shift(period)
                    features[f"momentum_osc_{period}"] = momentum

            # Williams %R
            for period in [14, 20]:
                if len(prices) >= period and "high" in df.columns and "low" in df.columns:
                    highest_high = df["high"].rolling(window=period).max()
                    lowest_low = df["low"].rolling(window=period).min()
                    williams_r = ((highest_high - prices) / (highest_high - lowest_low)) * -100
                    features[f"williams_r_{period}"] = williams_r

        except Exception as e:
            self.logger.error(f"Error extracting momentum features: {str(e)}")

        return features

    def _extract_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-based features."""
        features = pd.DataFrame(index=df.index)

        try:
            price_col = "price" if "price" in df.columns else "close"
            if price_col not in df.columns:
                return features

            prices = df[price_col]
            returns = prices.pct_change()

            # Realized volatility
            for period in [10, 20, 50]:
                if len(returns) >= period:
                    realized_vol = returns.rolling(window=period).std() * np.sqrt(252)
                    features[f"realized_vol_{period}"] = realized_vol

            # GARCH-like features
            if len(returns) >= 20:
                # Simple volatility clustering measure
                vol_proxy = returns.abs()
                for period in [5, 10]:
                    features[f"vol_clustering_{period}"] = vol_proxy.rolling(window=period).mean()

            # True Range (if OHLC data available)
            if all(col in df.columns for col in ["high", "low", "close"]):
                prev_close = df["close"].shift(1)
                tr1 = df["high"] - df["low"]
                tr2 = abs(df["high"] - prev_close)
                tr3 = abs(df["low"] - prev_close)
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                features["true_range"] = true_range

                # Average True Range
                for period in [14, 20]:
                    if len(true_range) >= period:
                        atr = true_range.rolling(window=period).mean()
                        features[f"atr_{period}"] = atr
                        features[f"atr_ratio_{period}"] = true_range / atr

        except Exception as e:
            self.logger.error(f"Error extracting volatility features: {str(e)}")

        return features

    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using correlation."""
        try:
            correlations = features.corrwith(target).abs().sort_values(ascending=False)
            return correlations.to_dict()
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def get_cached_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached features for a symbol."""
        return self.feature_cache.get(symbol)

    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature engineering statistics."""
        avg_latency = (
            sum(self.calculation_latency_us) / len(self.calculation_latency_us)
            if self.calculation_latency_us
            else 0
        )

        return {
            "feature_calculations": self.feature_calculation_count,
            "average_latency_us": avg_latency,
            "cached_symbols": len(self.feature_cache),
            "lookback_periods": self.lookback_periods,
        }
