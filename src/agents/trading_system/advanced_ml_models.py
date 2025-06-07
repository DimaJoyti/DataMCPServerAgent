"""
Advanced machine learning models for the Fetch.ai Advanced Crypto Trading System.

This module provides more sophisticated machine learning models for prediction
and continuous improvement.
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from uagents import Agent, Context, Model, Protocol

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Types of machine learning models."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

class PredictionTarget(str, Enum):
    """Prediction targets."""

    PRICE_DIRECTION = "price_direction"
    VOLATILITY = "volatility"
    TRADING_VOLUME = "trading_volume"
    SENTIMENT_IMPACT = "sentiment_impact"
    OPTIMAL_ENTRY = "optimal_entry"
    OPTIMAL_EXIT = "optimal_exit"

class ModelConfig(Model):
    """Model for machine learning model configuration."""

    model_type: ModelType
    target: PredictionTarget
    hyperparameters: Dict[str, Any] = {}
    feature_engineering: Dict[str, Any] = {}
    preprocessing: Dict[str, Any] = {}

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

class AdvancedMLModel:
    """Base class for advanced machine learning models."""

    def __init__(
        self,
        model_type: ModelType,
        target: PredictionTarget,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the advanced ML model.

        Args:
            model_type: Type of model
            target: Prediction target
            config: Model configuration
            logger: Logger instance
        """
        self.model_type = model_type
        self.target = target
        self.config = config or ModelConfig(
            model_type=model_type,
            target=target
        )
        self.logger = logger or logging.getLogger(f"{model_type}_{target}")

        # Initialize model
        self.model = self._create_model()

        # Initialize preprocessors
        self.feature_scaler = StandardScaler()
        self.target_scaler = None  # Only used for regression

    def _create_model(self) -> Any:
        """Create the machine learning model.

        Returns:
            Machine learning model
        """
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=self.config.hyperparameters.get("n_estimators", 100),
                max_depth=self.config.hyperparameters.get("max_depth", None),
                min_samples_split=self.config.hyperparameters.get("min_samples_split", 2),
                random_state=42
            )
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=self.config.hyperparameters.get("n_estimators", 100),
                learning_rate=self.config.hyperparameters.get("learning_rate", 0.1),
                max_depth=self.config.hyperparameters.get("max_depth", 3),
                random_state=42
            )
        elif self.model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(
                hidden_layer_sizes=self.config.hyperparameters.get("hidden_layer_sizes", (100,)),
                activation=self.config.hyperparameters.get("activation", "relu"),
                solver=self.config.hyperparameters.get("solver", "adam"),
                alpha=self.config.hyperparameters.get("alpha", 0.0001),
                learning_rate=self.config.hyperparameters.get("learning_rate", "constant"),
                max_iter=self.config.hyperparameters.get("max_iter", 200),
                random_state=42
            )
        elif self.model_type == ModelType.ENSEMBLE:
            # Create an ensemble of models
            models = []

            # Add Random Forest
            models.append(RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ))

            # Add Gradient Boosting
            models.append(GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))

            # Add Neural Network
            models.append(MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate="constant",
                max_iter=200,
                random_state=42
            ))

            return models
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train the model.

        Args:
            X: Features
            y: Labels

        Returns:
            Training result
        """
        start_time = datetime.now()

        # Preprocess data
        X_scaled = self.feature_scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        if self.model_type == ModelType.ENSEMBLE:
            # Train each model in the ensemble
            for model in self.model:
                model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        # Evaluate model
        if self.model_type == ModelType.ENSEMBLE:
            # Make predictions with each model
            predictions = []
            for model in self.model:
                predictions.append(model.predict(X_test))

            # Use majority voting
            y_pred = np.zeros(len(y_test))
            for i in range(len(y_test)):
                votes = [predictions[j][i] for j in range(len(self.model))]
                y_pred[i] = max(set(votes), key=votes.count)
        else:
            y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()

        # Create result
        result = TrainingResult(
            model_type=self.model_type,
            target=self.target,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            sample_size=len(X),
            timestamp=datetime.now().isoformat()
        )

        self.logger.info(
            f"Trained {self.model_type} model for {self.target}: "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
            f"Training time: {training_time:.2f}s"
        )

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Preprocess data
        X_scaled = self.feature_scaler.transform(X)

        # Make predictions
        if self.model_type == ModelType.ENSEMBLE:
            # Make predictions with each model
            predictions = []
            for model in self.model:
                predictions.append(model.predict(X_scaled))

            # Use majority voting
            y_pred = np.zeros(len(X))
            for i in range(len(X)):
                votes = [predictions[j][i] for j in range(len(self.model))]
                y_pred[i] = max(set(votes), key=votes.count)

            return y_pred
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions.

        Args:
            X: Features

        Returns:
            Probability predictions
        """
        # Preprocess data
        X_scaled = self.feature_scaler.transform(X)

        # Make predictions
        if self.model_type == ModelType.ENSEMBLE:
            # Make predictions with each model
            probas = []
            for model in self.model:
                probas.append(model.predict_proba(X_scaled))

            # Average probabilities
            y_proba = np.zeros(probas[0].shape)
            for proba in probas:
                y_proba += proba
            y_proba /= len(probas)

            return y_proba
        else:
            return self.model.predict_proba(X_scaled)

    def save(self, path: str):
        """Save the model to a file.

        Args:
            path: Path to save to
        """
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "config": self.config.dict()
            }, f)

    @classmethod
    def load(cls, path: str) -> "AdvancedMLModel":
        """Load a model from a file.

        Args:
            path: Path to load from

        Returns:
            Loaded model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create config
        config = ModelConfig(**data["config"])

        # Create model
        model = cls(
            model_type=config.model_type,
            target=config.target,
            config=config
        )

        # Load model and preprocessors
        model.model = data["model"]
        model.feature_scaler = data["feature_scaler"]
        model.target_scaler = data["target_scaler"]

        return model

class ModelManager:
    """Manager for advanced machine learning models."""

    def __init__(
        self,
        models_dir: str = "models",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the model manager.

        Args:
            models_dir: Directory to store models
            logger: Logger instance
        """
        self.models_dir = models_dir
        self.logger = logger or logging.getLogger("model_manager")

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Initialize models
        self.models = {}

    def get_model(
        self,
        model_type: ModelType,
        target: PredictionTarget,
        config: Optional[ModelConfig] = None
    ) -> AdvancedMLModel:
        """Get a model.

        Args:
            model_type: Type of model
            target: Prediction target
            config: Model configuration

        Returns:
            Model
        """
        model_key = f"{model_type}_{target}"

        if model_key not in self.models:
            # Try to load model from file
            model_path = os.path.join(self.models_dir, f"{model_key}.pkl")
            if os.path.exists(model_path):
                self.logger.info(f"Loading model from {model_path}")
                self.models[model_key] = AdvancedMLModel.load(model_path)
            else:
                # Create new model
                self.logger.info(f"Creating new {model_type} model for {target}")
                self.models[model_key] = AdvancedMLModel(
                    model_type=model_type,
                    target=target,
                    config=config
                )

        return self.models[model_key]

    def save_model(
        self,
        model_type: ModelType,
        target: PredictionTarget
    ):
        """Save a model to a file.

        Args:
            model_type: Type of model
            target: Prediction target
        """
        model_key = f"{model_type}_{target}"

        if model_key in self.models:
            model_path = os.path.join(self.models_dir, f"{model_key}.pkl")
            self.logger.info(f"Saving model to {model_path}")
            self.models[model_key].save(model_path)
        else:
            self.logger.warning(f"Model {model_key} not found")

    def save_all_models(self):
        """Save all models to files."""
        for model_key, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_key}.pkl")
            self.logger.info(f"Saving model to {model_path}")
            model.save(model_path)

# Example usage
if __name__ == "__main__":
    # Create model manager
    manager = ModelManager()

    # Create model
    model = manager.get_model(
        model_type=ModelType.ENSEMBLE,
        target=PredictionTarget.PRICE_DIRECTION
    )

    # Generate random data for demonstration
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, size=1000)

    # Train model
    result = model.train(X, y)
    print(f"Training result: {result}")

    # Make predictions
    X_new = np.random.rand(10, 10)
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")

    # Save model
    manager.save_model(
        model_type=ModelType.ENSEMBLE,
        target=PredictionTarget.PRICE_DIRECTION
    )
