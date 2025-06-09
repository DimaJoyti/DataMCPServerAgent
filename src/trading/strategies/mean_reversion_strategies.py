"""
Mean Reversion Trading Strategies

Implements mean reversion algorithmic trading strategies including:
- Bollinger Bands Strategy
- Z-Score Strategy
- RSI Mean Reversion Strategy
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_strategy import (
    EnhancedBaseStrategy, StrategySignal, StrategySignalData, StrategyState
)
from .technical_indicators import TechnicalIndicators
from ..core.base_models import MarketData
from ..core.enums import StrategyType


class BollingerBandsStrategy(EnhancedBaseStrategy):
    """Bollinger Bands mean reversion strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'oversold_threshold': 0.1,  # Distance from lower band
            'overbought_threshold': 0.1,  # Distance from upper band
            'mean_reversion_threshold': 0.5,  # Distance from middle band for exit
            'volume_confirmation': True,
            'rsi_filter': True,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="Bollinger Bands Mean Reversion Strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate Bollinger Bands mean reversion signal."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['bb_period'] + 20:
                return None
            
            # Calculate Bollinger Bands
            bb_data = TechnicalIndicators.bollinger_bands(
                df['close'], 
                self.parameters['bb_period'], 
                self.parameters['bb_std_dev']
            )
            
            upper_band = bb_data['upper']
            middle_band = bb_data['middle']
            lower_band = bb_data['lower']
            
            current_price = df['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Calculate position relative to bands
            band_width = current_upper - current_lower
            upper_distance = (current_upper - current_price) / band_width
            lower_distance = (current_price - current_lower) / band_width
            middle_distance = abs(current_price - current_middle) / band_width
            
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # RSI filter if enabled
            rsi_confirmation = True
            current_rsi = None
            if self.parameters['rsi_filter']:
                rsi = TechnicalIndicators.rsi(df['close'])
                current_rsi = rsi.iloc[-1]
                
                # Only trade when RSI confirms oversold/overbought
                if lower_distance <= self.parameters['oversold_threshold']:
                    rsi_confirmation = current_rsi <= self.parameters['rsi_oversold']
                elif upper_distance <= self.parameters['overbought_threshold']:
                    rsi_confirmation = current_rsi >= self.parameters['rsi_overbought']
            
            # Volume confirmation
            volume_confirmation = True
            if self.parameters['volume_confirmation'] and market_data.volume:
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                volume_confirmation = market_data.volume > avg_volume * 0.8
            
            # Generate signals
            if (lower_distance <= self.parameters['oversold_threshold'] and 
                rsi_confirmation and volume_confirmation):
                # Price near lower band - potential buy
                signal = StrategySignal.BUY
                strength = 1.0 - lower_distance  # Closer to band = stronger signal
                confidence = 0.8
                
                # Very close to lower band
                if lower_distance <= 0.05:
                    signal = StrategySignal.STRONG_BUY
                    confidence = 0.9
            
            elif (upper_distance <= self.parameters['overbought_threshold'] and 
                  rsi_confirmation and volume_confirmation):
                # Price near upper band - potential sell
                signal = StrategySignal.SELL
                strength = 1.0 - upper_distance
                confidence = 0.8
                
                # Very close to upper band
                if upper_distance <= 0.05:
                    signal = StrategySignal.STRONG_SELL
                    confidence = 0.9
            
            # Mean reversion exit signals
            elif middle_distance <= self.parameters['mean_reversion_threshold']:
                # Price near middle band - potential exit/weak counter-trend
                if current_price > current_middle:
                    signal = StrategySignal.WEAK_SELL
                    strength = 0.3
                    confidence = 0.4
                else:
                    signal = StrategySignal.WEAK_BUY
                    strength = 0.3
                    confidence = 0.4
            
            # Band squeeze detection (low volatility)
            band_squeeze = band_width < df['close'].rolling(window=20).std().iloc[-1] * 1.5
            if band_squeeze:
                confidence *= 0.7  # Reduce confidence during low volatility
            
            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'upper_distance': upper_distance,
                    'lower_distance': lower_distance,
                    'middle_distance': middle_distance,
                    'band_width': band_width,
                    'band_squeeze': band_squeeze,
                    'rsi': current_rsi
                },
                metadata={
                    'strategy': 'Bollinger_Bands',
                    'timeframe': self.timeframe,
                    'bb_period': self.parameters['bb_period'],
                    'bb_std_dev': self.parameters['bb_std_dev']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands signal for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on distance from bands."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on distance from bands (closer = larger position)
        if signal.signal in [StrategySignal.BUY, StrategySignal.STRONG_BUY]:
            distance_factor = 1.0 - signal.indicators.get('lower_distance', 0.5)
        elif signal.signal in [StrategySignal.SELL, StrategySignal.STRONG_SELL]:
            distance_factor = 1.0 - signal.indicators.get('upper_distance', 0.5)
        else:
            distance_factor = 0.5
        
        adjusted_size = base_size * Decimal(str(distance_factor + 0.5))  # Ensure minimum 0.5x
        
        return min(adjusted_size, self.max_position_size)


class ZScoreStrategy(EnhancedBaseStrategy):
    """Z-Score based mean reversion strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            'lookback_period': 20,
            'entry_threshold': 2.0,  # Z-score threshold for entry
            'exit_threshold': 0.5,   # Z-score threshold for exit
            'extreme_threshold': 3.0,  # Extreme Z-score for strong signals
            'min_volatility': 0.01,  # Minimum volatility for trading
            'volume_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="Z-Score Mean Reversion Strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate Z-Score mean reversion signal."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['lookback_period'] + 10:
                return None
            
            # Calculate Z-Score
            z_score = TechnicalIndicators.z_score(df['close'], self.parameters['lookback_period'])
            current_z = z_score.iloc[-1]
            prev_z = z_score.iloc[-2]
            
            # Calculate rolling volatility
            volatility = df['close'].pct_change().rolling(window=self.parameters['lookback_period']).std().iloc[-1]
            
            # Skip if volatility is too low
            if volatility < self.parameters['min_volatility']:
                return None
            
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Volume filter
            volume_ok = True
            if self.parameters['volume_filter'] and market_data.volume:
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                volume_ok = market_data.volume > avg_volume * 0.5
            
            if not volume_ok:
                return None
            
            # Extreme oversold (strong buy signal)
            if current_z <= -self.parameters['extreme_threshold']:
                signal = StrategySignal.STRONG_BUY
                strength = min(abs(current_z) / self.parameters['extreme_threshold'], 1.0)
                confidence = 0.9
            
            # Extreme overbought (strong sell signal)
            elif current_z >= self.parameters['extreme_threshold']:
                signal = StrategySignal.STRONG_SELL
                strength = min(abs(current_z) / self.parameters['extreme_threshold'], 1.0)
                confidence = 0.9
            
            # Regular oversold (buy signal)
            elif current_z <= -self.parameters['entry_threshold']:
                signal = StrategySignal.BUY
                strength = min(abs(current_z) / self.parameters['entry_threshold'], 1.0)
                confidence = 0.7
            
            # Regular overbought (sell signal)
            elif current_z >= self.parameters['entry_threshold']:
                signal = StrategySignal.SELL
                strength = min(abs(current_z) / self.parameters['entry_threshold'], 1.0)
                confidence = 0.7
            
            # Mean reversion (exit signals)
            elif abs(current_z) <= self.parameters['exit_threshold']:
                if prev_z > self.parameters['exit_threshold']:
                    signal = StrategySignal.WEAK_SELL
                    strength = 0.3
                    confidence = 0.5
                elif prev_z < -self.parameters['exit_threshold']:
                    signal = StrategySignal.WEAK_BUY
                    strength = 0.3
                    confidence = 0.5
            
            # Z-Score momentum (trend continuation vs reversal)
            z_momentum = current_z - prev_z
            if abs(z_momentum) > 0.5:  # Strong momentum
                if signal in [StrategySignal.BUY, StrategySignal.STRONG_BUY] and z_momentum > 0:
                    confidence *= 0.8  # Reduce confidence if momentum against mean reversion
                elif signal in [StrategySignal.SELL, StrategySignal.STRONG_SELL] and z_momentum < 0:
                    confidence *= 0.8
            
            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'z_momentum': z_momentum,
                    'volatility': volatility,
                    'rolling_mean': df['close'].rolling(window=self.parameters['lookback_period']).mean().iloc[-1],
                    'rolling_std': df['close'].rolling(window=self.parameters['lookback_period']).std().iloc[-1]
                },
                metadata={
                    'strategy': 'Z_Score',
                    'timeframe': self.timeframe,
                    'lookback_period': self.parameters['lookback_period'],
                    'entry_threshold': self.parameters['entry_threshold'],
                    'exit_threshold': self.parameters['exit_threshold']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating Z-Score signal for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on Z-Score magnitude."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on Z-Score magnitude (higher absolute Z-Score = larger position)
        z_score = abs(signal.indicators.get('z_score', 0))
        z_factor = min(z_score / 2.0, 1.5)  # Cap at 1.5x
        
        # Adjust based on volatility (higher volatility = smaller position)
        volatility = signal.indicators.get('volatility', 0.02)
        vol_factor = max(0.02 / volatility, 0.5)  # Minimum 0.5x
        
        adjusted_size = base_size * Decimal(str(z_factor)) * Decimal(str(vol_factor))
        
        return min(adjusted_size, self.max_position_size)


class RSIMeanReversionStrategy(EnhancedBaseStrategy):
    """RSI-based mean reversion strategy with divergence detection."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            'rsi_period': 14,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'oversold': 30,
            'overbought': 70,
            'mean_level': 50,
            'divergence_lookback': 10,
            'price_change_threshold': 0.02
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="RSI Mean Reversion Strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate RSI mean reversion signal with divergence detection."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['rsi_period'] + self.parameters['divergence_lookback'] + 10:
                return None
            
            # Calculate RSI
            rsi = TechnicalIndicators.rsi(df['close'], self.parameters['rsi_period'])
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            current_price = df['close'].iloc[-1]
            
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Detect divergences
            bullish_divergence = self._detect_bullish_divergence(df, rsi)
            bearish_divergence = self._detect_bearish_divergence(df, rsi)
            
            # Extreme levels with mean reversion bias
            if current_rsi <= self.parameters['extreme_oversold']:
                signal = StrategySignal.STRONG_BUY
                strength = (self.parameters['extreme_oversold'] - current_rsi) / self.parameters['extreme_oversold']
                confidence = 0.9
                
                if bullish_divergence:
                    confidence = 0.95  # Higher confidence with divergence
            
            elif current_rsi >= self.parameters['extreme_overbought']:
                signal = StrategySignal.STRONG_SELL
                strength = (current_rsi - self.parameters['extreme_overbought']) / (100 - self.parameters['extreme_overbought'])
                confidence = 0.9
                
                if bearish_divergence:
                    confidence = 0.95
            
            # Regular oversold/overbought levels
            elif current_rsi <= self.parameters['oversold'] and prev_rsi > current_rsi:
                signal = StrategySignal.BUY
                strength = (self.parameters['oversold'] - current_rsi) / self.parameters['oversold']
                confidence = 0.7
                
                if bullish_divergence:
                    confidence = 0.8
            
            elif current_rsi >= self.parameters['overbought'] and prev_rsi < current_rsi:
                signal = StrategySignal.SELL
                strength = (current_rsi - self.parameters['overbought']) / (100 - self.parameters['overbought'])
                confidence = 0.7
                
                if bearish_divergence:
                    confidence = 0.8
            
            # Mean reversion to 50 level
            elif abs(current_rsi - self.parameters['mean_level']) < 5:
                if prev_rsi > self.parameters['overbought']:
                    signal = StrategySignal.WEAK_SELL
                    strength = 0.3
                    confidence = 0.4
                elif prev_rsi < self.parameters['oversold']:
                    signal = StrategySignal.WEAK_BUY
                    strength = 0.3
                    confidence = 0.4
            
            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    'rsi': current_rsi,
                    'prev_rsi': prev_rsi,
                    'bullish_divergence': bullish_divergence,
                    'bearish_divergence': bearish_divergence,
                    'distance_from_mean': abs(current_rsi - self.parameters['mean_level'])
                },
                metadata={
                    'strategy': 'RSI_Mean_Reversion',
                    'timeframe': self.timeframe,
                    'rsi_period': self.parameters['rsi_period']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating RSI mean reversion signal for {symbol}: {e}")
            return None
    
    def _detect_bullish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Detect bullish divergence (price makes lower low, RSI makes higher low)."""
        try:
            lookback = self.parameters['divergence_lookback']
            
            # Find recent lows in price and RSI
            price_recent = df['close'].iloc[-lookback:].min()
            price_prev_low = df['close'].iloc[-lookback*2:-lookback].min()
            
            rsi_recent = rsi.iloc[-lookback:].min()
            rsi_prev_low = rsi.iloc[-lookback*2:-lookback].min()
            
            # Bullish divergence: price lower low, RSI higher low
            return (price_recent < price_prev_low and rsi_recent > rsi_prev_low and 
                    rsi_recent < self.parameters['oversold'])
        except:
            return False
    
    def _detect_bearish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Detect bearish divergence (price makes higher high, RSI makes lower high)."""
        try:
            lookback = self.parameters['divergence_lookback']
            
            # Find recent highs in price and RSI
            price_recent = df['close'].iloc[-lookback:].max()
            price_prev_high = df['close'].iloc[-lookback*2:-lookback].max()
            
            rsi_recent = rsi.iloc[-lookback:].max()
            rsi_prev_high = rsi.iloc[-lookback*2:-lookback].max()
            
            # Bearish divergence: price higher high, RSI lower high
            return (price_recent > price_prev_high and rsi_recent < rsi_prev_high and 
                    rsi_recent > self.parameters['overbought'])
        except:
            return False
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on RSI distance from mean and divergence."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on distance from RSI mean (50)
        distance_from_mean = signal.indicators.get('distance_from_mean', 25)
        distance_factor = min(distance_from_mean / 25, 1.5)  # Cap at 1.5x
        
        # Bonus for divergence
        divergence_bonus = 1.0
        if signal.indicators.get('bullish_divergence') or signal.indicators.get('bearish_divergence'):
            divergence_bonus = 1.3
        
        adjusted_size = base_size * Decimal(str(distance_factor)) * Decimal(str(divergence_bonus))
        
        return min(adjusted_size, self.max_position_size)
