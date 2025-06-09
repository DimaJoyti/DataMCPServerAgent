"""
Advanced price prediction models for trading.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Simplified ML implementations
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self.feature_importances_ = np.abs(self.coef_) / np.sum(np.abs(self.coef_))
        return self

    def predict(self, X):
        X = np.array(X)
        return self.intercept_ + np.dot(X, self.coef_)

class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1
        return (X - self.mean_) / self.scale_

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


class PricePredictionModel:
    """
    Advanced price prediction model using multiple algorithms.
    
    Features:
    - Multiple model ensemble
    - Time series forecasting
    - Feature importance analysis
    - Model validation
    - Real-time prediction
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        prediction_horizon: int = 5,  # minutes
        lookback_window: int = 100
    ):
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.lookback_window = lookback_window
        
        self.logger = logging.getLogger(f"PricePredictionModel.{model_type}")
        
        # Models
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Performance tracking
        self.training_history = []
        self.prediction_history = []
        
        # Initialize models based on type
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize prediction models."""
        # Use simplified linear regression models
        if self.model_type == "ensemble":
            self.models = {
                'linear_1': SimpleLinearRegression(),
                'linear_2': SimpleLinearRegression(),
                'linear_3': SimpleLinearRegression()
            }
        else:
            self.models = {
                'linear': SimpleLinearRegression()
            }

        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = SimpleStandardScaler()
    
    def prepare_training_data(
        self,
        price_data: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for price prediction.
        
        Args:
            price_data: Historical price series
            features: Feature matrix
            
        Returns:
            Tuple of (X, y) for training
        """
        try:
            # Align features and prices
            aligned_data = pd.concat([features, price_data], axis=1, join='inner')
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < self.lookback_window + self.prediction_horizon:
                raise ValueError("Insufficient data for training")
            
            X_list = []
            y_list = []
            
            # Create sliding windows
            for i in range(self.lookback_window, len(aligned_data) - self.prediction_horizon):
                # Features for current window
                feature_window = aligned_data.iloc[i-self.lookback_window:i, :-1].values.flatten()
                X_list.append(feature_window)
                
                # Target: price change over prediction horizon
                current_price = aligned_data.iloc[i, -1]
                future_price = aligned_data.iloc[i + self.prediction_horizon, -1]
                price_change = (future_price - current_price) / current_price
                y_list.append(price_change)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train(
        self,
        price_data: pd.Series,
        features: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the price prediction model.
        
        Args:
            price_data: Historical price series
            features: Feature matrix
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        try:
            self.logger.info("Starting price prediction model training")
            
            # Prepare data
            X, y = self.prepare_training_data(price_data, features)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            training_results = {}
            
            # Train each model
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name} model")
                
                # Scale features
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Validate model
                y_pred_train = model.predict(X_train_scaled)
                y_pred_val = model.predict(X_val_scaled)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                val_metrics = self._calculate_metrics(y_val, y_pred_val)
                
                training_results[model_name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_importance': self._get_feature_importance(model, features.columns)
                }
                
                self.logger.info(
                    f"{model_name} - Train R²: {train_metrics['r2']:.3f}, "
                    f"Val R²: {val_metrics['r2']:.3f}"
                )
            
            self.is_trained = True
            
            # Store training history
            training_record = {
                'timestamp': datetime.utcnow(),
                'samples': len(X),
                'features': X.shape[1],
                'results': training_results
            }
            self.training_history.append(training_record)
            
            self.logger.info("Price prediction model training completed")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training price prediction model: {str(e)}")
            raise
    
    def predict(
        self,
        current_features: pd.Series,
        price_history: pd.Series
    ) -> Dict[str, Any]:
        """
        Generate price prediction.
        
        Args:
            current_features: Current feature values
            price_history: Recent price history
            
        Returns:
            Prediction results
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare feature vector (flatten recent features)
            feature_vector = current_features.values.flatten().reshape(1, -1)
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Make prediction
                pred = model.predict(feature_vector_scaled)[0]
                predictions[model_name] = pred
                
                # Calculate confidence (simplified)
                if hasattr(model, 'predict_proba'):
                    # For models with probability estimates
                    confidence = 0.8  # Placeholder
                else:
                    # For regression models, use feature importance
                    confidence = 0.7  # Placeholder
                
                confidences[model_name] = confidence
            
            # Ensemble prediction (weighted average)
            if len(predictions) > 1:
                weights = np.array(list(confidences.values()))
                weights = weights / weights.sum()
                
                ensemble_pred = np.average(list(predictions.values()), weights=weights)
                ensemble_confidence = np.average(list(confidences.values()), weights=weights)
                
                predictions['ensemble'] = ensemble_pred
                confidences['ensemble'] = ensemble_confidence
            
            # Convert to price prediction
            current_price = price_history.iloc[-1] if len(price_history) > 0 else 100.0
            
            result = {
                'timestamp': datetime.utcnow(),
                'current_price': current_price,
                'predicted_change_pct': predictions.get('ensemble', list(predictions.values())[0]),
                'predicted_price': current_price * (1 + predictions.get('ensemble', list(predictions.values())[0])),
                'confidence': confidences.get('ensemble', list(confidences.values())[0]),
                'horizon_minutes': self.prediction_horizon,
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }
            
            # Store prediction
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making price prediction: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                
                # Handle flattened features
                if len(importances) != len(feature_names):
                    # Features were flattened, group by original feature names
                    features_per_window = len(feature_names)
                    grouped_importance = {}
                    
                    for i, feature_name in enumerate(feature_names):
                        # Sum importance across time windows
                        total_importance = 0
                        for j in range(0, len(importances), features_per_window):
                            if j + i < len(importances):
                                total_importance += importances[j + i]
                        grouped_importance[feature_name] = total_importance
                    
                    return grouped_importance
                else:
                    return dict(zip(feature_names, importances))
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_)
                return dict(zip(feature_names, coefficients))
            
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def evaluate_predictions(self, actual_prices: pd.Series) -> Dict[str, Any]:
        """Evaluate prediction accuracy against actual prices."""
        try:
            if not self.prediction_history:
                return {}
            
            # Match predictions with actual outcomes
            evaluation_results = []
            
            for pred in self.prediction_history:
                pred_time = pred['timestamp']
                target_time = pred_time + timedelta(minutes=self.prediction_horizon)
                
                # Find actual price at target time
                actual_price_at_target = self._get_price_at_time(actual_prices, target_time)
                
                if actual_price_at_target is not None:
                    actual_change = (actual_price_at_target - pred['current_price']) / pred['current_price']
                    predicted_change = pred['predicted_change_pct']
                    
                    evaluation_results.append({
                        'predicted_change': predicted_change,
                        'actual_change': actual_change,
                        'error': abs(predicted_change - actual_change),
                        'direction_correct': (predicted_change * actual_change) > 0
                    })
            
            if not evaluation_results:
                return {}
            
            # Calculate aggregate metrics
            errors = [r['error'] for r in evaluation_results]
            direction_accuracy = sum(r['direction_correct'] for r in evaluation_results) / len(evaluation_results)
            
            return {
                'total_predictions': len(evaluation_results),
                'mean_absolute_error': np.mean(errors),
                'direction_accuracy': direction_accuracy,
                'rmse': np.sqrt(np.mean([e**2 for e in errors]))
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            return {}
    
    def _get_price_at_time(self, prices: pd.Series, target_time: datetime) -> Optional[float]:
        """Get price at specific time (with interpolation if needed)."""
        try:
            # Find closest price to target time
            time_diffs = abs(prices.index - target_time)
            closest_idx = time_diffs.idxmin()
            
            # Only use if within reasonable time window (e.g., 2 minutes)
            if time_diffs[closest_idx] <= timedelta(minutes=2):
                return prices[closest_idx]
            
            return None
            
        except Exception:
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and statistics."""
        return {
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'lookback_window': self.lookback_window,
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'training_history_count': len(self.training_history),
            'prediction_history_count': len(self.prediction_history),
            'last_training': self.training_history[-1]['timestamp'] if self.training_history else None
        }
