"""
Core Machine Learning Engine for Institutional Trading.
"""

import asyncio
import logging
import pickle
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..market_data.data_types import OHLCV, MarketDataSnapshot, Quote, Trade
from .feature_engineering import FeatureEngineer
from .model_manager import ModelManager


class MLEngine:
    """
    Core Machine Learning Engine for institutional trading.

    Features:
    - Real-time ML inference
    - Model lifecycle management
    - Feature engineering pipeline
    - Performance monitoring
    - A/B testing framework
    - Model explainability
    """

    def __init__(
        self,
        name: str = "InstitutionalMLEngine",
        model_cache_size: int = 100,
        inference_timeout_ms: int = 10,
        feature_window_size: int = 1000,
    ):
        self.name = name
        self.model_cache_size = model_cache_size
        self.inference_timeout_ms = inference_timeout_ms
        self.feature_window_size = feature_window_size

        self.logger = logging.getLogger(f"MLEngine.{name}")
        self.is_running = False

        # Core components
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()

        # Data storage
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=feature_window_size))
        self.predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.features: Dict[str, pd.DataFrame] = {}

        # Model registry
        self.active_models: Dict[str, Dict] = {}  # model_id -> model_info
        self.model_performance: Dict[str, Dict] = defaultdict(dict)

        # Inference tracking
        self.inference_count = 0
        self.inference_latency_us: deque = deque(maxlen=1000)
        self.prediction_accuracy: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # A/B testing
        self.ab_tests: Dict[str, Dict] = {}
        self.test_results: Dict[str, Dict] = defaultdict(dict)

        # Event handlers
        self.prediction_handlers: List[callable] = []
        self.model_update_handlers: List[callable] = []

        # Configuration
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    async def start(self) -> None:
        """Start the ML engine."""
        self.logger.info(f"Starting ML engine: {self.name}")
        self.is_running = True

        # Start components
        await self.feature_engineer.start()
        await self.model_manager.start()

        # Start background tasks
        asyncio.create_task(self._monitor_model_performance())
        asyncio.create_task(self._update_features())
        asyncio.create_task(self._cleanup_old_data())

        self.logger.info(f"ML engine started: {self.name}")

    async def stop(self) -> None:
        """Stop the ML engine."""
        self.logger.info(f"Stopping ML engine: {self.name}")
        self.is_running = False

        # Stop components
        await self.feature_engineer.stop()
        await self.model_manager.stop()

        self.logger.info(f"ML engine stopped: {self.name}")

    async def update_market_data(
        self, symbol: str, data: Union[Quote, Trade, OHLCV, MarketDataSnapshot]
    ) -> None:
        """Update market data for feature engineering."""
        self.market_data[symbol].append(
            {"timestamp": data.timestamp, "data": data, "type": type(data).__name__}
        )

        # Trigger feature update
        await self._update_symbol_features(symbol)

    async def predict(
        self,
        model_id: str,
        symbol: str,
        prediction_type: str = "price_direction",
        horizon_minutes: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate ML prediction for a symbol.

        Args:
            model_id: ID of the model to use
            symbol: Trading symbol
            prediction_type: Type of prediction (price_direction, volatility, etc.)
            horizon_minutes: Prediction horizon in minutes

        Returns:
            Prediction result with confidence and metadata
        """
        start_time = time.time()

        try:
            # Check if model is active
            if model_id not in self.active_models:
                self.logger.warning(f"Model {model_id} not found in active models")
                return None

            model_info = self.active_models[model_id]
            model = model_info["model"]

            # Get features for symbol
            features = await self._get_features_for_prediction(
                symbol, model_info["feature_columns"]
            )
            if features is None:
                self.logger.warning(f"No features available for {symbol}")
                return None

            # Make prediction
            if hasattr(model, "predict_proba"):
                # Classification model
                probabilities = model.predict_proba(features.values.reshape(1, -1))[0]
                prediction = model.classes_[np.argmax(probabilities)]
                confidence = float(np.max(probabilities))
            else:
                # Regression model
                prediction = float(model.predict(features.values.reshape(1, -1))[0])
                confidence = 0.8  # Default confidence for regression

            # Create prediction result
            result = {
                "model_id": model_id,
                "symbol": symbol,
                "prediction_type": prediction_type,
                "prediction": prediction,
                "confidence": confidence,
                "horizon_minutes": horizon_minutes,
                "timestamp": datetime.utcnow(),
                "features_used": features.index.tolist(),
                "model_version": model_info.get("version", "1.0"),
            }

            # Store prediction
            self.predictions[symbol].append(result)

            # Track inference latency
            inference_time = (time.time() - start_time) * 1_000_000  # microseconds
            self.inference_latency_us.append(inference_time)
            self.inference_count += 1

            # Trigger prediction handlers
            for handler in self.prediction_handlers:
                try:
                    (
                        await handler(result)
                        if asyncio.iscoroutinefunction(handler)
                        else handler(result)
                    )
                except Exception as e:
                    self.logger.error(f"Error in prediction handler: {str(e)}")

            self.logger.debug(
                f"Generated prediction for {symbol}: {prediction} (confidence: {confidence:.2f})"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            return None

    async def register_model(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        feature_columns: List[str],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Register a new model with the engine."""
        try:
            model_info = {
                "model": model,
                "model_type": model_type,
                "feature_columns": feature_columns,
                "metadata": metadata or {},
                "registered_at": datetime.utcnow(),
                "version": metadata.get("version", "1.0"),
                "performance": {},
            }

            self.active_models[model_id] = model_info

            # Save model to disk
            model_path = self.models_dir / f"{model_id}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model_info, f)

            self.logger.info(f"Registered model: {model_id} ({model_type})")

            # Trigger model update handlers
            for handler in self.model_update_handlers:
                try:
                    await handler(model_id, "registered", model_info)
                except Exception as e:
                    self.logger.error(f"Error in model update handler: {str(e)}")

            return True

        except Exception as e:
            self.logger.error(f"Error registering model {model_id}: {str(e)}")
            return False

    async def load_model(self, model_id: str) -> bool:
        """Load a model from disk."""
        try:
            model_path = self.models_dir / f"{model_id}.pkl"
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False

            with open(model_path, "rb") as f:
                model_info = pickle.load(f)

            self.active_models[model_id] = model_info
            self.logger.info(f"Loaded model: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            return False

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id in self.active_models:
            del self.active_models[model_id]
            self.logger.info(f"Unloaded model: {model_id}")
            return True
        return False

    async def start_ab_test(
        self,
        test_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        duration_hours: int = 24,
    ) -> bool:
        """Start an A/B test between two models."""
        try:
            test_config = {
                "test_id": test_id,
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": traffic_split,
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow() + timedelta(hours=duration_hours),
                "results": {"model_a": [], "model_b": []},
            }

            self.ab_tests[test_id] = test_config
            self.logger.info(f"Started A/B test: {test_id} ({model_a} vs {model_b})")
            return True

        except Exception as e:
            self.logger.error(f"Error starting A/B test {test_id}: {str(e)}")
            return False

    async def _update_symbol_features(self, symbol: str) -> None:
        """Update features for a symbol."""
        try:
            market_data_list = list(self.market_data[symbol])
            if len(market_data_list) < 10:  # Need minimum data
                return

            # Extract features using feature engineer
            features = await self.feature_engineer.extract_features(symbol, market_data_list)
            if features is not None:
                self.features[symbol] = features

        except Exception as e:
            self.logger.error(f"Error updating features for {symbol}: {str(e)}")

    async def _get_features_for_prediction(
        self, symbol: str, feature_columns: List[str]
    ) -> Optional[pd.Series]:
        """Get features for prediction."""
        if symbol not in self.features:
            return None

        symbol_features = self.features[symbol]
        if symbol_features.empty:
            return None

        # Get latest features
        latest_features = symbol_features.iloc[-1]

        # Select required columns
        try:
            return latest_features[feature_columns]
        except KeyError:
            # Some features missing, return None
            return None

    async def _update_features(self) -> None:
        """Continuously update features for all symbols."""
        while self.is_running:
            try:
                for symbol in list(self.market_data.keys()):
                    await self._update_symbol_features(symbol)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                self.logger.error(f"Error in feature update loop: {str(e)}")
                await asyncio.sleep(5)

    async def _monitor_model_performance(self) -> None:
        """Monitor model performance and accuracy."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute

                for model_id in list(self.active_models.keys()):
                    await self._calculate_model_performance(model_id)

            except Exception as e:
                self.logger.error(f"Error monitoring model performance: {str(e)}")

    async def _calculate_model_performance(self, model_id: str) -> None:
        """Calculate performance metrics for a model."""
        try:
            # This would implement actual performance calculation
            # based on prediction accuracy, Sharpe ratio, etc.

            model_info = self.active_models.get(model_id)
            if not model_info:
                return

            # Placeholder performance calculation
            performance = {
                "accuracy": 0.65,  # Would calculate from actual predictions
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
                "total_predictions": self.inference_count,
                "avg_latency_us": (
                    sum(self.inference_latency_us) / len(self.inference_latency_us)
                    if self.inference_latency_us
                    else 0
                ),
            }

            self.model_performance[model_id] = performance

        except Exception as e:
            self.logger.error(f"Error calculating performance for {model_id}: {str(e)}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                # Clean up old predictions
                for symbol in list(self.predictions.keys()):
                    predictions = self.predictions[symbol]
                    while predictions and predictions[0]["timestamp"] < cutoff_time:
                        predictions.popleft()

                self.logger.debug("Completed ML data cleanup")

            except Exception as e:
                self.logger.error(f"Error in ML data cleanup: {str(e)}")

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a model."""
        model_info = self.active_models.get(model_id)
        if not model_info:
            return None

        # Return safe copy without the actual model object
        return {
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "feature_columns": model_info["feature_columns"],
            "metadata": model_info["metadata"],
            "registered_at": model_info["registered_at"],
            "version": model_info["version"],
            "performance": self.model_performance.get(model_id, {}),
        }

    def get_predictions(self, symbol: str, count: int = 10) -> List[Dict]:
        """Get recent predictions for a symbol."""
        predictions = list(self.predictions[symbol])
        return predictions[-count:]

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get ML engine statistics."""
        avg_latency = (
            sum(self.inference_latency_us) / len(self.inference_latency_us)
            if self.inference_latency_us
            else 0
        )

        return {
            "active_models": len(self.active_models),
            "total_predictions": self.inference_count,
            "average_latency_us": avg_latency,
            "symbols_tracked": len(self.market_data),
            "ab_tests_active": len(self.ab_tests),
            "features_available": len(self.features),
        }

    def add_prediction_handler(self, handler: callable) -> None:
        """Add prediction event handler."""
        self.prediction_handlers.append(handler)

    def add_model_update_handler(self, handler: callable) -> None:
        """Add model update event handler."""
        self.model_update_handlers.append(handler)
