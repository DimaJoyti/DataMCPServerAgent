"""
Momentum Trading Strategies

Implements various momentum-based algorithmic trading strategies including:
- RSI Strategy
- MACD Strategy  
- Moving Average Crossover Strategy
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


class RSIStrategy(EnhancedBaseStrategy):
    """RSI-based momentum strategy."""
    
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
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'min_volume': 1000
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="RSI Momentum Strategy",
            strategy_type=StrategyType.MOMENTUM,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate RSI-based trading signal."""
        try:
            # Get historical data
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['rsi_period'] + 10:
                return None
            
            # Calculate RSI
            rsi = TechnicalIndicators.rsi(df['close'], self.parameters['rsi_period'])
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            # Calculate signal strength and confidence
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Strong signals
            if current_rsi <= self.parameters['extreme_oversold']:
                signal = StrategySignal.STRONG_BUY
                strength = 1.0
                confidence = 0.9
            elif current_rsi >= self.parameters['extreme_overbought']:
                signal = StrategySignal.STRONG_SELL
                strength = 1.0
                confidence = 0.9
            
            # Regular signals
            elif current_rsi <= self.parameters['oversold_threshold'] and prev_rsi > current_rsi:
                signal = StrategySignal.BUY
                strength = 0.7
                confidence = 0.7
            elif current_rsi >= self.parameters['overbought_threshold'] and prev_rsi < current_rsi:
                signal = StrategySignal.SELL
                strength = 0.7
                confidence = 0.7
            
            # Weak signals (RSI divergence)
            elif (current_rsi > prev_rsi and 
                  current_rsi < self.parameters['oversold_threshold'] + 10):
                signal = StrategySignal.WEAK_BUY
                strength = 0.4
                confidence = 0.5
            elif (current_rsi < prev_rsi and 
                  current_rsi > self.parameters['overbought_threshold'] - 10):
                signal = StrategySignal.WEAK_SELL
                strength = 0.4
                confidence = 0.5
            
            # Volume confirmation
            if market_data.volume and market_data.volume < self.parameters['min_volume']:
                confidence *= 0.7  # Reduce confidence for low volume
            
            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    'rsi': current_rsi,
                    'rsi_prev': prev_rsi
                },
                metadata={
                    'strategy': 'RSI',
                    'timeframe': self.timeframe,
                    'oversold_threshold': self.parameters['oversold_threshold'],
                    'overbought_threshold': self.parameters['overbought_threshold']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating RSI signal for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on signal strength and risk parameters."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust for confidence
        adjusted_size = base_size * Decimal(str(signal.confidence))
        
        # Risk-based adjustment
        if signal.signal in [StrategySignal.STRONG_BUY, StrategySignal.STRONG_SELL]:
            adjusted_size *= Decimal('1.2')  # Increase for strong signals
        elif signal.signal in [StrategySignal.WEAK_BUY, StrategySignal.WEAK_SELL]:
            adjusted_size *= Decimal('0.5')  # Decrease for weak signals
        
        return min(adjusted_size, self.max_position_size)


class MACDStrategy(EnhancedBaseStrategy):
    """MACD-based momentum strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'min_histogram_threshold': 0.001,
            'divergence_lookback': 5
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="MACD Momentum Strategy",
            strategy_type=StrategyType.MOMENTUM,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate MACD-based trading signal."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < self.parameters['slow_period'] + 20:
                return None
            
            # Calculate MACD
            macd_data = TechnicalIndicators.macd(
                df['close'],
                self.parameters['fast_period'],
                self.parameters['slow_period'],
                self.parameters['signal_period']
            )
            
            macd_line = macd_data['macd']
            signal_line = macd_data['signal']
            histogram = macd_data['histogram']
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]
            
            # Generate signals
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # MACD line crosses above signal line (bullish)
            if (current_macd > current_signal and 
                macd_line.iloc[-2] <= signal_line.iloc[-2]):
                
                if current_histogram > self.parameters['min_histogram_threshold']:
                    signal = StrategySignal.BUY
                    strength = min(abs(current_histogram) * 100, 1.0)
                    confidence = 0.8
                else:
                    signal = StrategySignal.WEAK_BUY
                    strength = 0.4
                    confidence = 0.5
            
            # MACD line crosses below signal line (bearish)
            elif (current_macd < current_signal and 
                  macd_line.iloc[-2] >= signal_line.iloc[-2]):
                
                if abs(current_histogram) > self.parameters['min_histogram_threshold']:
                    signal = StrategySignal.SELL
                    strength = min(abs(current_histogram) * 100, 1.0)
                    confidence = 0.8
                else:
                    signal = StrategySignal.WEAK_SELL
                    strength = 0.4
                    confidence = 0.5
            
            # Histogram momentum
            elif current_histogram > prev_histogram > 0:
                signal = StrategySignal.WEAK_BUY
                strength = 0.3
                confidence = 0.4
            elif current_histogram < prev_histogram < 0:
                signal = StrategySignal.WEAK_SELL
                strength = 0.3
                confidence = 0.4
            
            # Zero line crossover (stronger signal)
            if current_macd > 0 and macd_line.iloc[-2] <= 0:
                if signal == StrategySignal.BUY:
                    signal = StrategySignal.STRONG_BUY
                    confidence = 0.9
                elif signal in [StrategySignal.WEAK_BUY, StrategySignal.HOLD]:
                    signal = StrategySignal.BUY
                    strength = 0.7
                    confidence = 0.7
            elif current_macd < 0 and macd_line.iloc[-2] >= 0:
                if signal == StrategySignal.SELL:
                    signal = StrategySignal.STRONG_SELL
                    confidence = 0.9
                elif signal in [StrategySignal.WEAK_SELL, StrategySignal.HOLD]:
                    signal = StrategySignal.SELL
                    strength = 0.7
                    confidence = 0.7
            
            return StrategySignalData(
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                price=market_data.price,
                volume=market_data.volume,
                indicators={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_histogram,
                    'prev_histogram': prev_histogram
                },
                metadata={
                    'strategy': 'MACD',
                    'timeframe': self.timeframe,
                    'crossover': current_macd > current_signal
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating MACD signal for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on MACD signal strength."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on histogram magnitude
        histogram = signal.indicators.get('histogram', 0)
        histogram_factor = min(abs(histogram) * 50, 1.5)  # Cap at 1.5x
        
        adjusted_size = base_size * Decimal(str(histogram_factor))
        
        return min(adjusted_size, self.max_position_size)


class MovingAverageCrossoverStrategy(EnhancedBaseStrategy):
    """Moving Average Crossover momentum strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'ma_type': 'ema',  # 'sma' or 'ema'
            'min_separation': 0.005,  # Minimum % separation for valid signal
            'trend_confirmation_period': 200
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="Moving Average Crossover Strategy",
            strategy_type=StrategyType.MOMENTUM,
            symbols=symbols,
            timeframe=timeframe,
            parameters=default_params,
            risk_parameters=risk_parameters
        )
    
    async def generate_signal(self, symbol: str, market_data: MarketData) -> Optional[StrategySignalData]:
        """Generate Moving Average crossover signal."""
        try:
            df = self.market_data.get(symbol)
            if df is None or len(df) < max(self.parameters['slow_ma_period'], 
                                          self.parameters.get('trend_confirmation_period', 200)) + 10:
                return None
            
            # Calculate moving averages
            if self.parameters['ma_type'] == 'ema':
                fast_ma = TechnicalIndicators.ema(df['close'], self.parameters['fast_ma_period'])
                slow_ma = TechnicalIndicators.ema(df['close'], self.parameters['slow_ma_period'])
            else:
                fast_ma = TechnicalIndicators.sma(df['close'], self.parameters['fast_ma_period'])
                slow_ma = TechnicalIndicators.sma(df['close'], self.parameters['slow_ma_period'])
            
            # Trend confirmation MA
            trend_ma = None
            if 'trend_confirmation_period' in self.parameters:
                trend_ma = TechnicalIndicators.sma(df['close'], self.parameters['trend_confirmation_period'])
            
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            current_price = df['close'].iloc[-1]
            
            # Calculate separation percentage
            separation = abs(current_fast - current_slow) / current_slow
            
            signal = StrategySignal.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Golden Cross (fast MA crosses above slow MA)
            if (current_fast > current_slow and prev_fast <= prev_slow and 
                separation >= self.parameters['min_separation']):
                
                signal = StrategySignal.BUY
                strength = min(separation * 20, 1.0)  # Scale separation to strength
                confidence = 0.7
                
                # Trend confirmation
                if trend_ma is not None and current_price > trend_ma.iloc[-1]:
                    signal = StrategySignal.STRONG_BUY
                    confidence = 0.9
            
            # Death Cross (fast MA crosses below slow MA)
            elif (current_fast < current_slow and prev_fast >= prev_slow and 
                  separation >= self.parameters['min_separation']):
                
                signal = StrategySignal.SELL
                strength = min(separation * 20, 1.0)
                confidence = 0.7
                
                # Trend confirmation
                if trend_ma is not None and current_price < trend_ma.iloc[-1]:
                    signal = StrategySignal.STRONG_SELL
                    confidence = 0.9
            
            # Momentum signals (no crossover but increasing separation)
            elif current_fast > current_slow:
                if current_fast - current_slow > prev_fast - prev_slow:
                    signal = StrategySignal.WEAK_BUY
                    strength = 0.3
                    confidence = 0.4
            elif current_fast < current_slow:
                if current_slow - current_fast > prev_slow - prev_fast:
                    signal = StrategySignal.WEAK_SELL
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
                    'fast_ma': current_fast,
                    'slow_ma': current_slow,
                    'separation': separation,
                    'trend_ma': trend_ma.iloc[-1] if trend_ma is not None else None
                },
                metadata={
                    'strategy': 'MA_Crossover',
                    'timeframe': self.timeframe,
                    'ma_type': self.parameters['ma_type'],
                    'fast_period': self.parameters['fast_ma_period'],
                    'slow_period': self.parameters['slow_ma_period']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating MA crossover signal for {symbol}: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData) -> Decimal:
        """Calculate position size based on MA separation and trend strength."""
        base_size = self.max_position_size * Decimal(str(signal.strength))
        
        # Adjust based on MA separation (higher separation = stronger signal)
        separation = signal.indicators.get('separation', 0)
        separation_factor = min(separation * 100, 2.0)  # Cap at 2x
        
        adjusted_size = base_size * Decimal(str(separation_factor))
        
        return min(adjusted_size, self.max_position_size)
