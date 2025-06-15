"""
ML data pipeline for trading applications.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..market_data.data_types import OHLCV


class MLDataPipeline:
    """
    ML data pipeline for preparing training and inference data.

    Features:
    - Data collection and aggregation
    - Feature preparation
    - Target variable creation
    - Data validation
    - Real-time data streaming
    """

    def __init__(
        self,
        name: str = "MLDataPipeline",
        buffer_size: int = 10000,
        update_frequency_seconds: int = 60,
    ):
        self.name = name
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency_seconds

        self.logger = logging.getLogger(f"MLDataPipeline.{name}")
        self.is_running = False

        # Data buffers
        self.market_data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.feature_buffer: Dict[str, pd.DataFrame] = {}
        self.target_buffer: Dict[str, pd.Series] = {}

        # Pipeline configuration
        self.target_horizons = [5, 15, 30, 60]  # minutes
        self.feature_lag = 1  # minutes

        # Performance tracking
        self.pipeline_runs = 0
        self.last_update: Dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the ML data pipeline."""
        self.logger.info(f"Starting ML data pipeline: {self.name}")
        self.is_running = True

        # Start background tasks
        asyncio.create_task(self._update_pipeline())

        self.logger.info(f"ML data pipeline started: {self.name}")

    async def stop(self) -> None:
        """Stop the ML data pipeline."""
        self.logger.info(f"Stopping ML data pipeline: {self.name}")
        self.is_running = False
        self.logger.info(f"ML data pipeline stopped: {self.name}")

    async def add_market_data(self, symbol: str, data: Any) -> None:
        """Add market data to the pipeline."""
        self.market_data_buffer[symbol].append(
            {
                "timestamp": data.timestamp if hasattr(data, "timestamp") else datetime.utcnow(),
                "data": data,
                "type": type(data).__name__,
            }
        )

    async def _update_pipeline(self) -> None:
        """Update the data pipeline periodically."""
        while self.is_running:
            try:
                for symbol in list(self.market_data_buffer.keys()):
                    await self._process_symbol_data(symbol)

                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                self.logger.error(f"Error in pipeline update: {str(e)}")
                await asyncio.sleep(self.update_frequency)

    async def _process_symbol_data(self, symbol: str) -> None:
        """Process data for a specific symbol."""
        try:
            buffer = self.market_data_buffer[symbol]
            if len(buffer) < 100:  # Need minimum data
                return

            # Convert to DataFrame
            df = self._buffer_to_dataframe(buffer)
            if df.empty:
                return

            # Create features
            features = await self._create_features(df)
            if features is not None and not features.empty:
                self.feature_buffer[symbol] = features

            # Create targets
            targets = await self._create_targets(df)
            if targets is not None and not targets.empty:
                self.target_buffer[symbol] = targets

            self.last_update[symbol] = datetime.utcnow()
            self.pipeline_runs += 1

        except Exception as e:
            self.logger.error(f"Error processing data for {symbol}: {str(e)}")

    def _buffer_to_dataframe(self, buffer: deque) -> pd.DataFrame:
        """Convert buffer data to DataFrame."""
        try:
            rows = []

            for item in buffer:
                data = item["data"]
                row = {"timestamp": item["timestamp"], "type": item["type"]}

                # Extract relevant fields based on data type
                if hasattr(data, "price"):
                    row["price"] = float(data.price)
                if hasattr(data, "size"):
                    row["size"] = float(data.size)
                if hasattr(data, "bid_price") and data.bid_price:
                    row["bid"] = float(data.bid_price)
                if hasattr(data, "ask_price") and data.ask_price:
                    row["ask"] = float(data.ask_price)
                if hasattr(data, "volume"):
                    row["volume"] = float(data.volume)

                # OHLCV data
                if isinstance(data, OHLCV):
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

                # Forward fill missing values
                df = df.fillna(method="ffill")

            return df

        except Exception as e:
            self.logger.error(f"Error converting buffer to DataFrame: {str(e)}")
            return pd.DataFrame()

    async def _create_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create features from market data."""
        try:
            features = pd.DataFrame(index=df.index)

            # Price-based features
            if "price" in df.columns:
                price = df["price"]

                # Returns
                features["return_1m"] = price.pct_change()
                features["return_5m"] = price.pct_change(5)
                features["return_15m"] = price.pct_change(15)

                # Moving averages
                features["ma_5"] = price.rolling(5).mean()
                features["ma_20"] = price.rolling(20).mean()
                features["ma_ratio"] = price / features["ma_20"]

                # Volatility
                features["volatility_5m"] = price.pct_change().rolling(5).std()
                features["volatility_20m"] = price.pct_change().rolling(20).std()

                # Momentum
                features["momentum_5m"] = price - price.shift(5)
                features["momentum_20m"] = price - price.shift(20)

            # Spread features
            if "bid" in df.columns and "ask" in df.columns:
                spread = df["ask"] - df["bid"]
                mid_price = (df["bid"] + df["ask"]) / 2

                features["spread"] = spread
                features["spread_bps"] = (spread / mid_price) * 10000
                features["mid_price"] = mid_price

            # Volume features
            if "volume" in df.columns:
                volume = df["volume"]
                features["volume"] = volume
                features["volume_ma"] = volume.rolling(20).mean()
                features["volume_ratio"] = volume / features["volume_ma"]

            # Technical indicators
            if "close" in df.columns:
                close = df["close"]

                # RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                features["rsi"] = 100 - (100 / (1 + rs))

                # MACD
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                features["macd"] = ema12 - ema26
                features["macd_signal"] = features["macd"].ewm(span=9).mean()

            # Remove NaN values
            features = features.fillna(method="ffill").fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return None

    async def _create_targets(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Create target variables for prediction."""
        try:
            # Use price or close for target
            price_col = "price" if "price" in df.columns else "close"
            if price_col not in df.columns:
                return None

            price = df[price_col]
            targets = pd.DataFrame(index=df.index)

            # Create targets for different horizons
            for horizon in self.target_horizons:
                # Future return
                future_price = price.shift(-horizon)
                future_return = (future_price - price) / price
                targets[f"return_{horizon}m"] = future_return

                # Direction (classification target)
                targets[f"direction_{horizon}m"] = (future_return > 0).astype(int)

            # Return the main target (5-minute return)
            return targets["return_5m"]

        except Exception as e:
            self.logger.error(f"Error creating targets: {str(e)}")
            return None

    def get_training_data(
        self, symbol: str, min_samples: int = 100
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Get training data for a symbol."""
        try:
            if symbol not in self.feature_buffer or symbol not in self.target_buffer:
                return None

            features = self.feature_buffer[symbol]
            targets = self.target_buffer[symbol]

            # Align features and targets
            aligned_data = pd.concat([features, targets], axis=1, join="inner")
            aligned_data = aligned_data.dropna()

            if len(aligned_data) < min_samples:
                return None

            X = aligned_data.iloc[:, :-1]  # All columns except last (target)
            y = aligned_data.iloc[:, -1]  # Last column (target)

            return X, y

        except Exception as e:
            self.logger.error(f"Error getting training data for {symbol}: {str(e)}")
            return None

    def get_latest_features(self, symbol: str) -> Optional[pd.Series]:
        """Get latest features for real-time prediction."""
        try:
            if symbol not in self.feature_buffer:
                return None

            features = self.feature_buffer[symbol]
            if features.empty:
                return None

            return features.iloc[-1]

        except Exception as e:
            self.logger.error(f"Error getting latest features for {symbol}: {str(e)}")
            return None

    def validate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Validate data quality for a symbol."""
        try:
            quality_report = {
                "symbol": symbol,
                "timestamp": datetime.utcnow(),
                "data_available": False,
                "features_available": False,
                "targets_available": False,
                "data_points": 0,
                "feature_count": 0,
                "missing_values": 0,
                "data_freshness_minutes": None,
            }

            # Check data availability
            if symbol in self.market_data_buffer:
                buffer = self.market_data_buffer[symbol]
                quality_report["data_available"] = len(buffer) > 0
                quality_report["data_points"] = len(buffer)

                if buffer:
                    last_data_time = buffer[-1]["timestamp"]
                    freshness = (datetime.utcnow() - last_data_time).total_seconds() / 60
                    quality_report["data_freshness_minutes"] = freshness

            # Check features
            if symbol in self.feature_buffer:
                features = self.feature_buffer[symbol]
                quality_report["features_available"] = not features.empty
                quality_report["feature_count"] = len(features.columns) if not features.empty else 0
                quality_report["missing_values"] = (
                    features.isnull().sum().sum() if not features.empty else 0
                )

            # Check targets
            if symbol in self.target_buffer:
                targets = self.target_buffer[symbol]
                quality_report["targets_available"] = not targets.empty

            return quality_report

        except Exception as e:
            self.logger.error(f"Error validating data quality for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pipeline_runs": self.pipeline_runs,
            "symbols_tracked": len(self.market_data_buffer),
            "symbols_with_features": len(self.feature_buffer),
            "symbols_with_targets": len(self.target_buffer),
            "total_data_points": sum(len(buffer) for buffer in self.market_data_buffer.values()),
            "last_updates": {symbol: time.isoformat() for symbol, time in self.last_update.items()},
            "target_horizons": self.target_horizons,
            "feature_lag_minutes": self.feature_lag,
        }
