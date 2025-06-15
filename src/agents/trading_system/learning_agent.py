"""
Learning Optimization Agent for the Fetch.ai Advanced Crypto Trading System.

This agent continuously improves system performance through machine learning.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from uagents import Context, Model

from .base_agent import BaseAgent, BaseAgentState


class ModelType(str, Enum):
    """Types of machine learning models."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SUPPORT_VECTOR_MACHINE = "support_vector_machine"


class PredictionTarget(str, Enum):
    """Prediction targets."""

    PRICE_DIRECTION = "price_direction"
    VOLATILITY = "volatility"
    TRADING_VOLUME = "trading_volume"
    SENTIMENT_IMPACT = "sentiment_impact"


class TrainingResult(Model):
    """Model for training results."""

    model_type: ModelType
    target: PredictionTarget
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    sample_size: int
    timestamp: str


class Prediction(Model):
    """Model for a prediction."""

    symbol: str
    target: PredictionTarget
    prediction: str
    confidence: float
    model_type: ModelType
    features_used: List[str]
    timestamp: str


class PerformanceMetric(Model):
    """Model for a performance metric."""

    name: str
    value: float
    timestamp: str


class SystemImprovement(Model):
    """Model for a system improvement."""

    component: str
    improvement: str
    expected_impact: str
    confidence: float
    timestamp: str


class LearningAgentState(BaseAgentState):
    """State model for the Learning Optimization Agent."""

    models: Dict[str, Any] = {}  # Model name -> model object
    training_results: List[TrainingResult] = []
    recent_predictions: List[Prediction] = []
    performance_metrics: List[PerformanceMetric] = []
    system_improvements: List[SystemImprovement] = []
    symbols_to_track: List[str] = ["BTC/USD", "ETH/USD"]
    training_interval: int = 86400  # 24 hours in seconds
    prediction_interval: int = 3600  # 1 hour in seconds


class LearningOptimizationAgent(BaseAgent):
    """Agent for continuously improving system performance through machine learning."""

    def __init__(
        self,
        name: str = "learning_agent",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the Learning Optimization Agent.

        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
        """
        super().__init__(name, seed, port, endpoint, logger)

        # Initialize agent state
        self.state = LearningAgentState()

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register handlers for the agent."""

        @self.agent.on_interval(period=self.state.training_interval)
        async def train_models(ctx: Context):
            """Train machine learning models."""
            ctx.logger.info("Training machine learning models")

            # Train models for each symbol and target
            for symbol in self.state.symbols_to_track:
                for target in PredictionTarget:
                    await self._train_model(ctx, symbol, target)

        @self.agent.on_interval(period=self.state.prediction_interval)
        async def make_predictions(ctx: Context):
            """Make predictions using trained models."""
            ctx.logger.info("Making predictions")

            # Make predictions for each symbol and target
            for symbol in self.state.symbols_to_track:
                for target in PredictionTarget:
                    await self._make_prediction(ctx, symbol, target)

        @self.agent.on_interval(period=self.state.training_interval * 7)
        async def suggest_improvements(ctx: Context):
            """Suggest system improvements based on performance metrics."""
            ctx.logger.info("Suggesting system improvements")

            # Analyze performance and suggest improvements
            improvements = await self._suggest_improvements(ctx)

            # Update state
            self.state.system_improvements.extend(improvements)

            # Broadcast improvements to other agents
            # Implementation depends on the communication protocol

    async def _train_model(self, ctx: Context, symbol: str, target: PredictionTarget):
        """Train a machine learning model.

        Args:
            ctx: Agent context
            symbol: Trading symbol
            target: Prediction target
        """
        ctx.logger.info(f"Training model for {symbol} - {target}")

        try:
            # Fetch training data
            X_train, y_train = await self._fetch_training_data(symbol, target)

            if len(X_train) == 0 or len(y_train) == 0:
                ctx.logger.warning(f"No training data for {symbol} - {target}")
                return

            # Create and train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate model
            accuracy = model.score(X_train, y_train)

            # For simplicity, we'll use accuracy for all metrics
            # In a real system, we would calculate these properly
            precision = accuracy
            recall = accuracy
            f1_score = accuracy

            # Save model
            model_name = f"{symbol}_{target.value}_{ModelType.RANDOM_FOREST.value}"
            self.state.models[model_name] = model

            # Save training result
            result = TrainingResult(
                model_type=ModelType.RANDOM_FOREST,
                target=target,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                training_time=training_time,
                sample_size=len(X_train),
                timestamp=datetime.now().isoformat(),
            )

            self.state.training_results.append(result)
            if len(self.state.training_results) > 100:
                self.state.training_results.pop(0)

            ctx.logger.info(
                f"Trained model for {symbol} - {target}: "
                f"Accuracy: {accuracy:.4f}, Sample size: {len(X_train)}"
            )

            # Save performance metric
            self.state.performance_metrics.append(
                PerformanceMetric(
                    name=f"{symbol}_{target.value}_accuracy",
                    value=accuracy,
                    timestamp=datetime.now().isoformat(),
                )
            )

        except Exception as e:
            ctx.logger.error(f"Error training model for {symbol} - {target}: {str(e)}")

    async def _make_prediction(self, ctx: Context, symbol: str, target: PredictionTarget):
        """Make a prediction using a trained model.

        Args:
            ctx: Agent context
            symbol: Trading symbol
            target: Prediction target
        """
        model_name = f"{symbol}_{target.value}_{ModelType.RANDOM_FOREST.value}"

        if model_name not in self.state.models:
            ctx.logger.warning(f"No trained model for {symbol} - {target}")
            return

        try:
            # Fetch features for prediction
            features = await self._fetch_prediction_features(symbol, target)

            if not features:
                ctx.logger.warning(f"No features for prediction for {symbol} - {target}")
                return

            # Make prediction
            model = self.state.models[model_name]
            prediction_proba = model.predict_proba([features])[0]
            prediction_class = model.predict([features])[0]
            confidence = max(prediction_proba)

            # Convert prediction to string
            if target == PredictionTarget.PRICE_DIRECTION:
                prediction_str = "up" if prediction_class == 1 else "down"
            elif target == PredictionTarget.VOLATILITY:
                prediction_str = "high" if prediction_class == 1 else "low"
            elif target == PredictionTarget.TRADING_VOLUME:
                prediction_str = "high" if prediction_class == 1 else "low"
            else:  # SENTIMENT_IMPACT
                prediction_str = "positive" if prediction_class == 1 else "negative"

            # Create prediction
            prediction = Prediction(
                symbol=symbol,
                target=target,
                prediction=prediction_str,
                confidence=confidence,
                model_type=ModelType.RANDOM_FOREST,
                features_used=["feature1", "feature2", "feature3"],  # Placeholder
                timestamp=datetime.now().isoformat(),
            )

            # Update state
            self.state.recent_predictions.append(prediction)
            if len(self.state.recent_predictions) > 100:
                self.state.recent_predictions.pop(0)

            ctx.logger.info(
                f"Prediction for {symbol} - {target}: "
                f"{prediction_str.upper()} (confidence: {confidence:.4f})"
            )

            # Broadcast prediction to other agents
            # Implementation depends on the communication protocol

        except Exception as e:
            ctx.logger.error(f"Error making prediction for {symbol} - {target}: {str(e)}")

    async def _suggest_improvements(self, ctx: Context) -> List[SystemImprovement]:
        """Suggest system improvements based on performance metrics.

        Args:
            ctx: Agent context

        Returns:
            List of system improvements
        """
        improvements = []

        # Analyze model performance
        if self.state.training_results:
            # Find models with low accuracy
            low_accuracy_models = [
                result for result in self.state.training_results if result.accuracy < 0.6
            ]

            for result in low_accuracy_models:
                # Suggest improvement
                improvement = SystemImprovement(
                    component=f"Model for {result.target.value}",
                    improvement="Increase training data size or try different model architecture",
                    expected_impact="Improved prediction accuracy",
                    confidence=0.7,
                    timestamp=datetime.now().isoformat(),
                )

                improvements.append(improvement)

        # Suggest general improvements
        improvements.append(
            SystemImprovement(
                component="Data Collection",
                improvement="Add more data sources for sentiment analysis",
                expected_impact="More accurate sentiment predictions",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
            )
        )

        improvements.append(
            SystemImprovement(
                component="Technical Analysis",
                improvement="Add more advanced indicators like Ichimoku Cloud",
                expected_impact="Better trend identification",
                confidence=0.6,
                timestamp=datetime.now().isoformat(),
            )
        )

        return improvements

    async def _fetch_training_data(
        self, symbol: str, target: PredictionTarget
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch training data for a symbol and target.

        This is a mock implementation. In a real system, this would fetch
        actual historical data and prepare features and labels.

        Args:
            symbol: Trading symbol
            target: Prediction target

        Returns:
            Tuple of (features, labels)
        """
        # Mock data for demonstration

        # Generate random features and labels
        sample_size = 100
        features = np.random.rand(sample_size, 10)  # 10 features

        if target == PredictionTarget.PRICE_DIRECTION:
            # Binary classification: up (1) or down (0)
            labels = np.random.randint(0, 2, size=sample_size)
        elif target == PredictionTarget.VOLATILITY:
            # Binary classification: high (1) or low (0)
            labels = np.random.randint(0, 2, size=sample_size)
        elif target == PredictionTarget.TRADING_VOLUME:
            # Binary classification: high (1) or low (0)
            labels = np.random.randint(0, 2, size=sample_size)
        else:  # SENTIMENT_IMPACT
            # Binary classification: positive (1) or negative (0)
            labels = np.random.randint(0, 2, size=sample_size)

        return features, labels

    async def _fetch_prediction_features(
        self, symbol: str, target: PredictionTarget
    ) -> Optional[List[float]]:
        """Fetch features for prediction.

        This is a mock implementation. In a real system, this would fetch
        actual current data and prepare features.

        Args:
            symbol: Trading symbol
            target: Prediction target

        Returns:
            List of features or None if not available
        """
        # Mock data for demonstration
        import random

        # Generate random features
        features = [random.random() for _ in range(10)]  # 10 features

        return features
