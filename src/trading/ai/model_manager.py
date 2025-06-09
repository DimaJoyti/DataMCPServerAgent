"""
Model management and lifecycle for trading ML models.
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Simplified ML implementations (no external dependencies)
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        return self

    def predict(self, X):
        X = np.array(X)
        return self.intercept_ + np.dot(X, self.coef_)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero
        return (X - self.mean_) / self.scale_

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


class ModelManager:
    """
    Model lifecycle management for trading ML models.
    
    Features:
    - Model training and validation
    - Model versioning and registry
    - Performance monitoring
    - Automated retraining
    - Model deployment
    """
    
    def __init__(
        self,
        name: str = "ModelManager",
        models_dir: str = "models",
        retrain_frequency_hours: int = 24
    ):
        self.name = name
        self.models_dir = Path(models_dir)
        self.retrain_frequency = timedelta(hours=retrain_frequency_hours)
        
        self.logger = logging.getLogger(f"ModelManager.{name}")
        self.is_running = False
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry: Dict[str, Dict] = {}
        self.model_performance: Dict[str, Dict] = {}
        
        # Training data
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.scalers: Dict[str, SimpleStandardScaler] = {}
        
        # Performance tracking
        self.training_count = 0
        self.last_training_time: Dict[str, datetime] = {}
    
    async def start(self) -> None:
        """Start the model manager."""
        self.logger.info(f"Starting model manager: {self.name}")
        self.is_running = True
        
        # Load existing models
        await self._load_existing_models()
        
        # Start background tasks
        asyncio.create_task(self._monitor_model_performance())
        asyncio.create_task(self._automated_retraining())
        
        self.logger.info(f"Model manager started: {self.name}")
    
    async def stop(self) -> None:
        """Stop the model manager."""
        self.logger.info(f"Stopping model manager: {self.name}")
        self.is_running = False
        self.logger.info(f"Model manager stopped: {self.name}")
    
    async def train_model(
        self,
        model_id: str,
        model_type: str,
        features: pd.DataFrame,
        target: pd.Series,
        model_params: Optional[Dict] = None
    ) -> bool:
        """
        Train a new model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (classification, regression)
            features: Feature matrix
            target: Target variable
            model_params: Model hyperparameters
            
        Returns:
            True if training successful
        """
        try:
            self.logger.info(f"Training model: {model_id} ({model_type})")
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = SimpleStandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create model
            model = self._create_model(model_type, model_params or {})
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test_scaled, y_test)

            performance = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'mean_absolute_error': np.mean(np.abs(y_test - y_pred))
            }
            
            # Store model info
            model_info = {
                'model': model,
                'scaler': scaler,
                'model_type': model_type,
                'feature_columns': features.columns.tolist(),
                'trained_at': datetime.utcnow(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'performance': performance,
                'version': '1.0'
            }
            
            self.model_registry[model_id] = model_info
            self.scalers[model_id] = scaler
            self.last_training_time[model_id] = datetime.utcnow()
            
            # Save model to disk
            await self._save_model(model_id, model_info)
            
            self.training_count += 1
            self.logger.info(f"Model {model_id} trained successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {str(e)}")
            return False
    
    def _create_model(self, model_type: str, params: Dict) -> Any:
        """Create a model instance."""
        # Use simplified linear regression for both classification and regression
        return SimpleLinearRegression()
    
    async def _save_model(self, model_id: str, model_info: Dict) -> None:
        """Save model to disk."""
        try:
            model_path = self.models_dir / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            self.logger.debug(f"Saved model {model_id} to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {str(e)}")
    
    async def _load_existing_models(self) -> None:
        """Load existing models from disk."""
        try:
            for model_file in self.models_dir.glob("*.pkl"):
                model_id = model_file.stem
                
                try:
                    with open(model_file, 'rb') as f:
                        model_info = pickle.load(f)
                    
                    self.model_registry[model_id] = model_info
                    if 'scaler' in model_info:
                        self.scalers[model_id] = model_info['scaler']
                    
                    self.logger.info(f"Loaded model: {model_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_id}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error loading existing models: {str(e)}")
    
    async def _monitor_model_performance(self) -> None:
        """Monitor model performance over time."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for model_id in list(self.model_registry.keys()):
                    await self._update_model_performance(model_id)
                
            except Exception as e:
                self.logger.error(f"Error monitoring model performance: {str(e)}")
    
    async def _update_model_performance(self, model_id: str) -> None:
        """Update performance metrics for a model."""
        try:
            # This would implement real-time performance tracking
            # based on actual predictions vs outcomes
            
            model_info = self.model_registry.get(model_id)
            if not model_info:
                return
            
            # Placeholder performance update
            current_performance = {
                'timestamp': datetime.utcnow(),
                'prediction_count': 100,  # Would track actual predictions
                'accuracy': 0.65,  # Would calculate from real data
                'drift_score': 0.1,  # Model drift detection
                'latency_ms': 2.5
            }
            
            self.model_performance[model_id] = current_performance
            
        except Exception as e:
            self.logger.error(f"Error updating performance for {model_id}: {str(e)}")
    
    async def _automated_retraining(self) -> None:
        """Automated model retraining."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                for model_id in list(self.model_registry.keys()):
                    last_training = self.last_training_time.get(model_id)
                    
                    if last_training and datetime.utcnow() - last_training > self.retrain_frequency:
                        # Check if retraining is needed
                        performance = self.model_performance.get(model_id, {})
                        drift_score = performance.get('drift_score', 0)
                        
                        if drift_score > 0.2:  # Significant drift detected
                            self.logger.info(f"Scheduling retraining for {model_id} due to drift")
                            # Would trigger retraining with new data
                
            except Exception as e:
                self.logger.error(f"Error in automated retraining: {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a trained model."""
        model_info = self.model_registry.get(model_id)
        return model_info['model'] if model_info else None
    
    def get_scaler(self, model_id: str) -> Optional[SimpleStandardScaler]:
        """Get the scaler for a model."""
        return self.scalers.get(model_id)
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information."""
        model_info = self.model_registry.get(model_id)
        if not model_info:
            return None
        
        # Return safe copy without the actual model object
        return {
            'model_id': model_id,
            'model_type': model_info['model_type'],
            'feature_columns': model_info['feature_columns'],
            'trained_at': model_info['trained_at'],
            'training_samples': model_info['training_samples'],
            'test_samples': model_info['test_samples'],
            'performance': model_info['performance'],
            'version': model_info['version'],
            'current_performance': self.model_performance.get(model_id, {})
        }
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_registry.keys())
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        try:
            # Remove from registry
            if model_id in self.model_registry:
                del self.model_registry[model_id]
            
            if model_id in self.scalers:
                del self.scalers[model_id]
            
            if model_id in self.last_training_time:
                del self.last_training_time[model_id]
            
            # Remove file
            model_path = self.models_dir / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()
            
            self.logger.info(f"Deleted model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        return {
            'total_models': len(self.model_registry),
            'training_count': self.training_count,
            'models_with_performance': len(self.model_performance),
            'models_dir': str(self.models_dir),
            'retrain_frequency_hours': self.retrain_frequency.total_seconds() / 3600
        }
