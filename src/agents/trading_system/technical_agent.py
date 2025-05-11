"""
Technical Analysis Agent for the Fetch.ai Advanced Crypto Trading System.

This agent performs multi-timeframe analysis with primary and secondary indicators.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import ccxt
import numpy as np
import pandas as pd
import ta
from uagents import Agent, Context, Model, Protocol

from .base_agent import BaseAgent, BaseAgentState

class Timeframe(str, Enum):
    """Trading timeframes."""
    
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class IndicatorType(str, Enum):
    """Types of technical indicators."""
    
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"

class Indicator(Model):
    """Model for a technical indicator."""
    
    name: str
    type: IndicatorType
    value: float
    signal: str  # "buy", "sell", or "neutral"
    timeframe: Timeframe
    timestamp: str

class TechnicalAnalysisResult(Model):
    """Model for technical analysis results."""
    
    symbol: str
    timestamp: str
    timeframes: List[Timeframe]
    primary_indicators: List[Indicator] = []
    secondary_indicators: List[Indicator] = []
    overall_signal: str = "neutral"  # "buy", "sell", or "neutral"
    confidence: float = 0.0  # 0.0 to 1.0

class TechnicalAgentState(BaseAgentState):
    """State model for the Technical Analysis Agent."""
    
    symbols_to_track: List[str] = ["BTC/USD", "ETH/USD"]
    timeframes: List[Timeframe] = [Timeframe.HOUR_1, Timeframe.DAY_1]
    primary_indicator_types: List[str] = ["rsi", "macd", "bollinger"]
    secondary_indicator_types: List[str] = ["volume", "atr", "adx"]
    analysis_interval: int = 3600  # seconds
    recent_analyses: List[TechnicalAnalysisResult] = []

class TechnicalAnalysisAgent(BaseAgent):
    """Agent for performing technical analysis on cryptocurrency markets."""
    
    def __init__(
        self,
        name: str = "technical_agent",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """Initialize the Technical Analysis Agent.
        
        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
            exchange_id: ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
        """
        super().__init__(name, seed, port, endpoint, logger)
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Initialize agent state
        self.state = TechnicalAgentState()
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for the agent."""
        
        @self.agent.on_interval(period=self.state.analysis_interval)
        async def analyze_markets(ctx: Context):
            """Analyze markets for tracked symbols."""
            for symbol in self.state.symbols_to_track:
                for timeframe in self.state.timeframes:
                    await self._analyze_market(ctx, symbol, timeframe)
    
    async def _analyze_market(self, ctx: Context, symbol: str, timeframe: Timeframe):
        """Analyze a market for a specific symbol and timeframe.
        
        Args:
            ctx: Agent context
            symbol: Trading symbol to analyze
            timeframe: Timeframe to analyze
        """
        ctx.logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
        
        try:
            # Fetch OHLCV data
            ohlcv = await self._fetch_ohlcv(symbol, timeframe)
            if not ohlcv:
                ctx.logger.warning(f"No OHLCV data for {symbol} on {timeframe}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate primary indicators
            primary_indicators = self._calculate_primary_indicators(df, timeframe)
            
            # Calculate secondary indicators
            secondary_indicators = self._calculate_secondary_indicators(df, timeframe)
            
            # Determine overall signal
            overall_signal, confidence = self._determine_overall_signal(
                primary_indicators, secondary_indicators
            )
            
            # Create result
            result = TechnicalAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                timeframes=[timeframe],
                primary_indicators=primary_indicators,
                secondary_indicators=secondary_indicators,
                overall_signal=overall_signal,
                confidence=confidence
            )
            
            # Update state
            self.state.recent_analyses.append(result)
            if len(self.state.recent_analyses) > 20:
                self.state.recent_analyses.pop(0)
            
            ctx.logger.info(
                f"Technical analysis for {symbol} on {timeframe}: "
                f"{overall_signal.upper()} (confidence: {confidence:.2f})"
            )
            
            # Broadcast result to other agents
            # Implementation depends on the communication protocol
            
        except Exception as e:
            ctx.logger.error(f"Error analyzing {symbol} on {timeframe}: {str(e)}")
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: Timeframe) -> List[List[float]]:
        """Fetch OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            OHLCV data
        """
        try:
            # Convert timeframe to exchange format if needed
            tf = timeframe.value
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=100)
            return ohlcv
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            return []
    
    def _calculate_primary_indicators(self, df: pd.DataFrame, timeframe: Timeframe) -> List[Indicator]:
        """Calculate primary technical indicators.
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe
            
        Returns:
            List of indicators
        """
        indicators = []
        timestamp = datetime.now().isoformat()
        
        # RSI
        if "rsi" in self.state.primary_indicator_types:
            rsi = ta.momentum.RSIIndicator(df['close']).rsi()
            last_rsi = rsi.iloc[-1]
            
            signal = "neutral"
            if last_rsi < 30:
                signal = "buy"
            elif last_rsi > 70:
                signal = "sell"
            
            indicators.append(Indicator(
                name="RSI",
                type=IndicatorType.MOMENTUM,
                value=float(last_rsi),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        # MACD
        if "macd" in self.state.primary_indicator_types:
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            
            signal = "neutral"
            if macd_line > signal_line:
                signal = "buy"
            elif macd_line < signal_line:
                signal = "sell"
            
            indicators.append(Indicator(
                name="MACD",
                type=IndicatorType.TREND,
                value=float(macd_line - signal_line),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        # Bollinger Bands
        if "bollinger" in self.state.primary_indicator_types:
            bollinger = ta.volatility.BollingerBands(df['close'])
            upper = bollinger.bollinger_hband().iloc[-1]
            lower = bollinger.bollinger_lband().iloc[-1]
            current = df['close'].iloc[-1]
            
            signal = "neutral"
            if current < lower:
                signal = "buy"
            elif current > upper:
                signal = "sell"
            
            # Calculate percent from middle band
            middle = bollinger.bollinger_mavg().iloc[-1]
            percent = (current - middle) / (upper - middle) if upper != middle else 0
            
            indicators.append(Indicator(
                name="Bollinger Bands",
                type=IndicatorType.VOLATILITY,
                value=float(percent),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        return indicators
    
    def _calculate_secondary_indicators(self, df: pd.DataFrame, timeframe: Timeframe) -> List[Indicator]:
        """Calculate secondary technical indicators.
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe
            
        Returns:
            List of indicators
        """
        indicators = []
        timestamp = datetime.now().isoformat()
        
        # Volume
        if "volume" in self.state.secondary_indicator_types:
            volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            signal = "neutral"
            if volume > avg_volume * 1.5:
                # High volume could confirm a trend
                if df['close'].iloc[-1] > df['close'].iloc[-2]:
                    signal = "buy"
                else:
                    signal = "sell"
            
            indicators.append(Indicator(
                name="Volume",
                type=IndicatorType.VOLUME,
                value=float(volume / avg_volume),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        # ATR (Average True Range)
        if "atr" in self.state.secondary_indicator_types:
            atr = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range().iloc[-1]
            
            # ATR doesn't give buy/sell signals directly
            # It's used to measure volatility
            signal = "neutral"
            
            indicators.append(Indicator(
                name="ATR",
                type=IndicatorType.VOLATILITY,
                value=float(atr),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        # ADX (Average Directional Index)
        if "adx" in self.state.secondary_indicator_types:
            adx = ta.trend.ADXIndicator(
                df['high'], df['low'], df['close']
            ).adx().iloc[-1]
            
            signal = "neutral"
            if adx > 25:
                # Strong trend, but need +DI and -DI to determine direction
                # This is simplified; a real implementation would check +DI and -DI
                if df['close'].iloc[-1] > df['close'].iloc[-5]:
                    signal = "buy"
                else:
                    signal = "sell"
            
            indicators.append(Indicator(
                name="ADX",
                type=IndicatorType.TREND,
                value=float(adx),
                signal=signal,
                timeframe=timeframe,
                timestamp=timestamp
            ))
        
        return indicators
    
    def _determine_overall_signal(
        self, 
        primary_indicators: List[Indicator], 
        secondary_indicators: List[Indicator]
    ) -> tuple[str, float]:
        """Determine overall signal from indicators.
        
        Args:
            primary_indicators: Primary indicators
            secondary_indicators: Secondary indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        # Count buy and sell signals from primary indicators
        buy_count = sum(1 for ind in primary_indicators if ind.signal == "buy")
        sell_count = sum(1 for ind in primary_indicators if ind.signal == "sell")
        
        # Weight primary indicators more heavily
        primary_weight = 0.7
        secondary_weight = 0.3
        
        # Calculate primary signal
        primary_total = len(primary_indicators)
        if primary_total == 0:
            primary_signal = "neutral"
            primary_confidence = 0.0
        else:
            if buy_count > sell_count:
                primary_signal = "buy"
                primary_confidence = buy_count / primary_total
            elif sell_count > buy_count:
                primary_signal = "sell"
                primary_confidence = sell_count / primary_total
            else:
                primary_signal = "neutral"
                primary_confidence = 0.5
        
        # Count buy and sell signals from secondary indicators
        buy_count = sum(1 for ind in secondary_indicators if ind.signal == "buy")
        sell_count = sum(1 for ind in secondary_indicators if ind.signal == "sell")
        
        # Calculate secondary signal
        secondary_total = len(secondary_indicators)
        if secondary_total == 0:
            secondary_signal = "neutral"
            secondary_confidence = 0.0
        else:
            if buy_count > sell_count:
                secondary_signal = "buy"
                secondary_confidence = buy_count / secondary_total
            elif sell_count > buy_count:
                secondary_signal = "sell"
                secondary_confidence = sell_count / secondary_total
            else:
                secondary_signal = "neutral"
                secondary_confidence = 0.5
        
        # Combine signals
        if primary_signal == secondary_signal:
            overall_signal = primary_signal
            confidence = primary_weight * primary_confidence + secondary_weight * secondary_confidence
        elif primary_signal == "neutral":
            overall_signal = secondary_signal
            confidence = secondary_confidence * secondary_weight
        elif secondary_signal == "neutral":
            overall_signal = primary_signal
            confidence = primary_confidence * primary_weight
        else:
            # Conflicting signals
            if primary_confidence * primary_weight > secondary_confidence * secondary_weight:
                overall_signal = primary_signal
                confidence = primary_confidence * primary_weight - secondary_confidence * secondary_weight
            else:
                overall_signal = secondary_signal
                confidence = secondary_confidence * secondary_weight - primary_confidence * primary_weight
        
        return overall_signal, confidence
