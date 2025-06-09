"""
Machine Learning Trading Strategies

Implements ML-based algorithmic trading strategies including:
- Random Forest Strategy
- LSTM Neural Network Strategy
- Support Vector Machine Strategy
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .base_strategy import (
    EnhancedBaseStrategy, StrategySignal, StrategySignalData, StrategyState
)
from .technical_indicators import TechnicalIndicators
from ..core.base_models import MarketData
from ..core.enums import StrategyType


class RandomForestStrategy(EnhancedBaseStrategy):
    """Random Forest-based trading strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        if not ML_AVAILABLE:
            raise ImportError("Machine learning libraries not available. Install scikit-learn and tensorflow.")
        
        default_params = {
            'lookback_period': 100,
            'feature_period': 20,
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'retrain_frequency': 168,  # hours (1 week)
            'prediction_threshold': 0.6,
            'feature_importance_threshold': 0.01
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="Random Forest Strategy",
            strategy_type=StrategyType.ML,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
        
        self.models: Dict[str, RandomForestClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.last_training: Dict[str, datetime] = {}
        self.prediction_accuracy: Dict[str, float] = {}
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate ML-based trading signal using Random Forest."""
        try:
            # Check if model needs training/retraining
            if (symbol not in self.models or 
                symbol not in self.last_training or
                (datetime.now() - self.last_training[symbol]).total_seconds() > 
                self.parameters['retrain_frequency'] * 3600):
                
                await self._train_model(symbol)
            
            if symbol not in self.models:
                return None
            
            # Prepare features
            features = await self._prepare_features(symbol)
            if features is None:
                return None
            
            # Make prediction
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
            
            # Convert prediction to signal
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Get prediction probabilities
            if len(prediction_proba) >= 3:  # [sell, hold, buy]
                sell_prob = prediction_proba[0]
                hold_prob = prediction_proba[1]
                buy_prob = prediction_proba[2]
                
                max_prob = max(prediction_proba)
                
                if max_prob >= self.parameters['prediction_threshold']:
                    if prediction == 2:  # Buy
                        signal = StrategySignal.BUY
                        strength = buy_prob
                        confidence = buy_prob
                        
                        if buy_prob >= 0.8:
                            signal = StrategySignal.STRONG_BUY
                    
                    elif prediction == 0:  # Sell
                        signal = StrategySignal.SELL
                        strength = sell_prob
                        confidence = sell_prob
                        
                        if sell_prob >= 0.8:
                            signal = StrategySignal.STRONG_SELL
                    
                    # Weak signals for lower confidence
                    elif max_prob >= 0.5:
                        if prediction == 2:
                            signal = StrategySignal.WEAK_BUY
                            strength = buy_prob * 0.7
                            confidence = buy_prob * 0.7
                        elif prediction == 0:
                            signal = StrategySignal.WEAK_SELL
                            strength = sell_prob * 0.7
                            confidence = sell_prob * 0.7
            
            # Adjust confidence based on model accuracy
            model_accuracy = self.prediction_accuracy.get(symbol, 0.5)
            confidence *= model_accuracy
            
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
                    'prediction': int(prediction),
                    'buy_probability': prediction_proba[2] if len(prediction_proba) >= 3 else 0,
                    'sell_probability': prediction_proba[0] if len(prediction_proba) >= 3 else 0,
                    'hold_probability': prediction_proba[1] if len(prediction_proba) >= 3 else 0,
                    'model_accuracy': model_accuracy,
                    'feature_count': len(features)
                },
                metadata={
                    'strategy': 'Random_Forest',
                    'timeframe': self.timeframe,
                    'model_type': 'RandomForestClassifier',
                    'n_estimators': self.parameters['n_estimators']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating Random Forest signal for {symbol}: {e}")
            return None
    
    async def _train_model(self, symbol: str) -> None:
        """Train Random Forest model for a symbol."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['lookback_period'] + 50:
                return
            
            # Prepare training data
            X, y = await self._prepare_training_data(symbol, df)
            if X is None or len(X) < 50:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=self.parameters['n_estimators'],
                max_depth=self.parameters['max_depth'],
                min_samples_split=self.parameters['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and metrics
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.prediction_accuracy[symbol] = accuracy
            self.last_training[symbol] = datetime.now()
            
            # Store feature importance
            feature_names = self._get_feature_names()
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            self.feature_importance[symbol] = importance_dict
            
            self.logger.info(f"Trained Random Forest model for {symbol} with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model for {symbol}: {e}")
    
    async def _prepare_training_data(self, symbol: str, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with features and labels."""
        try:
            # Calculate technical indicators
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            
            # Create features
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'z_score'
            ]
            
            # Add price-based features
            df_with_indicators['price_change'] = df_with_indicators['close'].pct_change()
            df_with_indicators['volume_change'] = df_with_indicators['volume'].pct_change()
            df_with_indicators['high_low_ratio'] = df_with_indicators['high'] / df_with_indicators['low']
            
            feature_columns.extend(['price_change', 'volume_change', 'high_low_ratio'])
            
            # Add rolling statistics
            for period in [5, 10, 20]:
                df_with_indicators[f'return_{period}d'] = df_with_indicators['close'].pct_change(period)
                df_with_indicators[f'volatility_{period}d'] = df_with_indicators['close'].pct_change().rolling(period).std()
                feature_columns.extend([f'return_{period}d', f'volatility_{period}d'])
            
            # Create labels (future price direction)
            future_periods = 5  # Predict 5 periods ahead
            df_with_indicators['future_return'] = df_with_indicators['close'].shift(-future_periods) / df_with_indicators['close'] - 1
            
            # Convert to classification labels
            def create_labels(future_return):
                if future_return > 0.02:  # 2% gain
                    return 2  # Buy
                elif future_return < -0.02:  # 2% loss
                    return 0  # Sell
                else:
                    return 1  # Hold
            
            df_with_indicators['label'] = df_with_indicators['future_return'].apply(create_labels)
            
            # Remove rows with NaN values
            df_clean = df_with_indicators[feature_columns + ['label']].dropna()
            
            if len(df_clean) < 50:
                return None, None
            
            X = df_clean[feature_columns].values
            y = df_clean['label'].values
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for {symbol}: {e}")
            return None, None
    
    async def _prepare_features(self, symbol: str) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < 50:
                return None
            
            # Calculate indicators
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            
            # Get the same features used in training
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'z_score'
            ]
            
            # Add price-based features
            df_with_indicators['price_change'] = df_with_indicators['close'].pct_change()
            df_with_indicators['volume_change'] = df_with_indicators['volume'].pct_change()
            df_with_indicators['high_low_ratio'] = df_with_indicators['high'] / df_with_indicators['low']
            
            feature_columns.extend(['price_change', 'volume_change', 'high_low_ratio'])
            
            # Add rolling statistics
            for period in [5, 10, 20]:
                df_with_indicators[f'return_{period}d'] = df_with_indicators['close'].pct_change(period)
                df_with_indicators[f'volatility_{period}d'] = df_with_indicators['close'].pct_change().rolling(period).std()
                feature_columns.extend([f'return_{period}d', f'volatility_{period}d'])
            
            # Get latest features
            latest_features = df_with_indicators[feature_columns].iloc[-1].values
            
            # Check for NaN values
            if np.isnan(latest_features).any():
                return None
            
            return latest_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance tracking."""
        feature_names = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'z_score',
            'price_change', 'volume_change', 'high_low_ratio'
        ]
        
        for period in [5, 10, 20]:
            feature_names.extend([f'return_{period}d', f'volatility_{period}d'])
        
        return feature_names
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on ML prediction confidence."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on prediction confidence
        confidence_factor = signal.confidence
        
        # Adjust based on model accuracy
        model_accuracy = signal.indicators.get('model_accuracy', 0.5)
        accuracy_factor = model_accuracy
        
        # Adjust based on prediction probability
        if signal.signal in [StrategySignal.BUY, StrategySignal.STRONG_BUY]:
            prob_factor = signal.indicators.get('buy_probability', 0.5)
        else:
            prob_factor = signal.indicators.get('sell_probability', 0.5)
        
        adjusted_size = (base_size * 
                        Decimal(str(confidence_factor)) * 
                        Decimal(str(accuracy_factor)) * 
                        Decimal(str(prob_factor)))
        
        return min(adjusted_size, self.max_position_size)


class LSTMStrategy(EnhancedBaseStrategy):
    """LSTM Neural Network-based trading strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        if not ML_AVAILABLE:
            raise ImportError("TensorFlow not available. Install tensorflow.")
        
        default_params = {
            'sequence_length': 60,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'retrain_frequency': 336,  # hours (2 weeks)
            'prediction_threshold': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="LSTM Strategy",
            strategy_type=StrategyType.ML,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
        
        self.models: Dict[str, tf.keras.Model] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.last_training: Dict[str, datetime] = {}
        self.model_loss: Dict[str, float] = {}
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate LSTM-based trading signal."""
        try:
            # Check if model needs training/retraining
            if (symbol not in self.models or 
                symbol not in self.last_training or
                (datetime.now() - self.last_training[symbol]).total_seconds() > 
                self.parameters['retrain_frequency'] * 3600):
                
                await self._train_lstm_model(symbol)
            
            if symbol not in self.models:
                return None
            
            # Prepare sequence data
            sequence = await self._prepare_sequence(symbol)
            if sequence is None:
                return None
            
            # Make prediction
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            sequence_scaled = scaler.transform(sequence)
            sequence_reshaped = sequence_scaled.reshape(1, self.parameters['sequence_length'], -1)
            
            prediction = model.predict(sequence_reshaped, verbose=0)[0][0]
            
            # Convert prediction to signal
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Prediction is expected price change
            if abs(prediction) >= self.parameters['prediction_threshold']:
                if prediction > 0:
                    signal = StrategySignal.BUY
                    if prediction > 0.02:  # 2% predicted gain
                        signal = StrategySignal.STRONG_BUY
                else:
                    signal = StrategySignal.SELL
                    if prediction < -0.02:  # 2% predicted loss
                        signal = StrategySignal.STRONG_SELL
                
                strength = min(abs(prediction) * 10, 1.0)  # Scale prediction to strength
                confidence = strength * 0.8  # Conservative confidence
            
            # Weak signals for smaller predictions
            elif abs(prediction) >= 0.005:  # 0.5% threshold
                if prediction > 0:
                    signal = StrategySignal.WEAK_BUY
                else:
                    signal = StrategySignal.WEAK_SELL
                
                strength = min(abs(prediction) * 20, 0.5)
                confidence = strength * 0.6
            
            # Adjust confidence based on model performance
            model_loss = self.model_loss.get(symbol, 1.0)
            confidence *= max(0.3, 1.0 - model_loss)  # Lower loss = higher confidence
            
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
                    'prediction': prediction,
                    'model_loss': model_loss,
                    'sequence_length': self.parameters['sequence_length']
                },
                metadata={
                    'strategy': 'LSTM',
                    'timeframe': self.timeframe,
                    'model_type': 'LSTM_Neural_Network'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating LSTM signal for {symbol}: {e}")
            return None
    
    async def _train_lstm_model(self, symbol: str) -> None:
        """Train LSTM model for a symbol."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['sequence_length'] + 100:
                return
            
            # Prepare training data
            X, y = await self._prepare_lstm_training_data(symbol, df)
            if X is None or len(X) < 50:
                return
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(self.parameters['lstm_units'], return_sequences=True, 
                     input_shape=(self.parameters['sequence_length'], X.shape[2])),
                Dropout(self.parameters['dropout_rate']),
                LSTM(self.parameters['lstm_units'], return_sequences=False),
                Dropout(self.parameters['dropout_rate']),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=self.parameters['batch_size'],
                epochs=self.parameters['epochs'],
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Store model and metrics
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_loss[symbol] = min(history.history['val_loss'])
            self.last_training[symbol] = datetime.now()
            
            self.logger.info(f"Trained LSTM model for {symbol} with validation loss: {self.model_loss[symbol]:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model for {symbol}: {e}")
    
    async def _prepare_lstm_training_data(self, symbol: str, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare LSTM training data with sequences."""
        try:
            # Calculate features
            df_features = TechnicalIndicators.calculate_all_indicators(df)
            
            # Select features for LSTM
            feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower']
            
            # Add price changes
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['volume_change'] = df_features['volume'].pct_change()
            feature_columns.extend(['price_change', 'volume_change'])
            
            # Clean data
            df_clean = df_features[feature_columns].dropna()
            
            if len(df_clean) < self.parameters['sequence_length'] + 10:
                return None, None
            
            # Create sequences
            X, y = [], []
            for i in range(self.parameters['sequence_length'], len(df_clean) - 1):
                # Features sequence
                X.append(df_clean.iloc[i-self.parameters['sequence_length']:i].values)
                
                # Target: next period price change
                current_price = df_clean['close'].iloc[i]
                next_price = df_clean['close'].iloc[i + 1]
                price_change = (next_price - current_price) / current_price
                y.append(price_change)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing LSTM training data for {symbol}: {e}")
            return None, None
    
    async def _prepare_sequence(self, symbol: str) -> Optional[np.ndarray]:
        """Prepare sequence for LSTM prediction."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['sequence_length'] + 10:
                return None
            
            # Calculate features
            df_features = TechnicalIndicators.calculate_all_indicators(df)
            
            # Select same features as training
            feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower']
            
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['volume_change'] = df_features['volume'].pct_change()
            feature_columns.extend(['price_change', 'volume_change'])
            
            # Get latest sequence
            df_clean = df_features[feature_columns].dropna()
            
            if len(df_clean) < self.parameters['sequence_length']:
                return None
            
            sequence = df_clean.tail(self.parameters['sequence_length']).values
            
            return sequence
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on LSTM prediction confidence."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on prediction magnitude
        prediction = abs(signal.indicators.get('prediction', 0))
        prediction_factor = min(prediction * 20, 1.5)  # Scale prediction impact
        
        # Adjust based on model performance (lower loss = higher confidence)
        model_loss = signal.indicators.get('model_loss', 1.0)
        loss_factor = max(0.5, 1.0 - model_loss)
        
        adjusted_size = (base_size * 
                        Decimal(str(prediction_factor)) * 
                        Decimal(str(loss_factor)))
        
        return min(adjusted_size, self.max_position_size)
