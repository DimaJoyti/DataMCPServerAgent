"""
Macro-Correlation Agent for the Fetch.ai Advanced Crypto Trading System.

This agent analyzes relationships between crypto and traditional markets.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from uagents import Context, Model

from .base_agent import BaseAgent, BaseAgentState


class MarketType(str, Enum):
    """Types of markets."""

    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"


class CorrelationStrength(str, Enum):
    """Correlation strength."""

    STRONG_POSITIVE = "strong_positive"  # 0.7 to 1.0
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    WEAK_POSITIVE = "weak_positive"  # 0.0 to 0.3
    WEAK_NEGATIVE = "weak_negative"  # -0.3 to 0.0
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    STRONG_NEGATIVE = "strong_negative"  # -1.0 to -0.7


class MarketData(Model):
    """Model for market data."""

    symbol: str
    market_type: MarketType
    price: float
    timestamp: str
    change_24h: float
    volume_24h: float


class CorrelationPair(Model):
    """Model for a correlation between two markets."""

    crypto_symbol: str
    traditional_symbol: str
    traditional_market_type: MarketType
    correlation_coefficient: float
    correlation_strength: CorrelationStrength
    timeframe: str  # e.g., "1d", "7d", "30d", "90d"
    sample_size: int
    timestamp: str


class MacroEvent(Model):
    """Model for a macroeconomic event."""

    name: str
    description: str
    event_time: str
    impact: str  # "high", "medium", "low"
    affected_markets: List[MarketType]
    expected_crypto_impact: str  # "positive", "negative", "neutral"


class MacroAnalysis(Model):
    """Model for a macro analysis result."""

    crypto_symbol: str
    correlations: List[CorrelationPair]
    upcoming_events: List[MacroEvent]
    overall_macro_sentiment: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    timestamp: str


class MacroAgentState(BaseAgentState):
    """State model for the Macro-Correlation Agent."""

    crypto_symbols: List[str] = ["BTC/USD", "ETH/USD"]
    traditional_symbols: Dict[str, MarketType] = {
        "SPY": MarketType.INDEX,  # S&P 500
        "QQQ": MarketType.INDEX,  # NASDAQ
        "GLD": MarketType.COMMODITY,  # Gold
        "SLV": MarketType.COMMODITY,  # Silver
        "DXY": MarketType.INDEX,  # US Dollar Index
        "TNX": MarketType.BOND,  # 10-Year Treasury Yield
    }
    timeframes: List[str] = ["1d", "7d", "30d", "90d"]
    recent_analyses: List[MacroAnalysis] = []
    market_data: Dict[str, List[MarketData]] = {}
    correlations: List[CorrelationPair] = []
    upcoming_events: List[MacroEvent] = []
    analysis_interval: int = 86400  # 24 hours in seconds


class MacroCorrelationAgent(BaseAgent):
    """Agent for analyzing macro correlations between crypto and traditional markets."""

    def __init__(
        self,
        name: str = "macro_agent",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the Macro-Correlation Agent.

        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
        """
        super().__init__(name, seed, port, endpoint, logger)

        # Initialize agent state
        self.state = MacroAgentState()

        # Initialize upcoming events
        self._initialize_upcoming_events()

        # Register handlers
        self._register_handlers()

    def _initialize_upcoming_events(self):
        """Initialize upcoming macroeconomic events."""
        # These are example events
        self.state.upcoming_events = [
            MacroEvent(
                name="Federal Reserve Interest Rate Decision",
                description="The Federal Reserve announces its decision on interest rates",
                event_time=(datetime.now() + timedelta(days=15)).isoformat(),
                impact="high",
                affected_markets=[MarketType.STOCK, MarketType.BOND, MarketType.FOREX],
                expected_crypto_impact="negative",
            ),
            MacroEvent(
                name="US CPI Data Release",
                description="Release of Consumer Price Index data for the US",
                event_time=(datetime.now() + timedelta(days=7)).isoformat(),
                impact="medium",
                affected_markets=[MarketType.STOCK, MarketType.BOND],
                expected_crypto_impact="neutral",
            ),
            MacroEvent(
                name="ECB Monetary Policy Statement",
                description="European Central Bank releases its monetary policy statement",
                event_time=(datetime.now() + timedelta(days=21)).isoformat(),
                impact="medium",
                affected_markets=[MarketType.FOREX, MarketType.BOND],
                expected_crypto_impact="neutral",
            ),
        ]

    def _register_handlers(self):
        """Register handlers for the agent."""

        @self.agent.on_interval(period=self.state.analysis_interval)
        async def analyze_correlations(ctx: Context):
            """Analyze correlations between crypto and traditional markets."""
            ctx.logger.info("Analyzing macro correlations")

            # Update market data
            await self._update_market_data(ctx)

            # Calculate correlations
            await self._calculate_correlations(ctx)

            # Perform macro analysis for each crypto symbol
            for symbol in self.state.crypto_symbols:
                analysis = await self._perform_macro_analysis(ctx, symbol)

                # Update state
                self.state.recent_analyses.append(analysis)
                if len(self.state.recent_analyses) > 10:
                    self.state.recent_analyses.pop(0)

                ctx.logger.info(
                    f"Macro analysis for {symbol}: "
                    f"{analysis.overall_macro_sentiment.upper()} "
                    f"(confidence: {analysis.confidence:.2f})"
                )

                # Broadcast analysis to other agents
                # Implementation depends on the communication protocol

    async def _update_market_data(self, ctx: Context):
        """Update market data for all symbols.

        Args:
            ctx: Agent context
        """
        # Update crypto market data
        for symbol in self.state.crypto_symbols:
            data = await self._fetch_market_data(symbol, MarketType.CRYPTO)

            if symbol not in self.state.market_data:
                self.state.market_data[symbol] = []

            self.state.market_data[symbol].append(data)

            # Keep only recent data (last 90 days)
            if len(self.state.market_data[symbol]) > 90:
                self.state.market_data[symbol] = self.state.market_data[symbol][-90:]

        # Update traditional market data
        for symbol, market_type in self.state.traditional_symbols.items():
            data = await self._fetch_market_data(symbol, market_type)

            if symbol not in self.state.market_data:
                self.state.market_data[symbol] = []

            self.state.market_data[symbol].append(data)

            # Keep only recent data (last 90 days)
            if len(self.state.market_data[symbol]) > 90:
                self.state.market_data[symbol] = self.state.market_data[symbol][-90:]

    async def _fetch_market_data(self, symbol: str, market_type: MarketType) -> MarketData:
        """Fetch market data for a symbol.

        This is a mock implementation. In a real system, this would call
        external APIs to fetch market data.

        Args:
            symbol: Market symbol
            market_type: Type of market

        Returns:
            Market data
        """
        # Mock data for demonstration
        import random

        price = 100.0 + random.uniform(-10.0, 10.0)
        change_24h = random.uniform(-5.0, 5.0)
        volume_24h = 1000000.0 + random.uniform(-100000.0, 100000.0)

        return MarketData(
            symbol=symbol,
            market_type=market_type,
            price=price,
            timestamp=datetime.now().isoformat(),
            change_24h=change_24h,
            volume_24h=volume_24h,
        )

    async def _calculate_correlations(self, ctx: Context):
        """Calculate correlations between crypto and traditional markets.

        Args:
            ctx: Agent context
        """
        # Clear previous correlations
        self.state.correlations = []

        # Calculate correlations for each crypto symbol and timeframe
        for crypto_symbol in self.state.crypto_symbols:
            for traditional_symbol, market_type in self.state.traditional_symbols.items():
                for timeframe in self.state.timeframes:
                    correlation = await self._calculate_correlation(
                        crypto_symbol, traditional_symbol, market_type, timeframe
                    )

                    if correlation:
                        self.state.correlations.append(correlation)

    async def _calculate_correlation(
        self, crypto_symbol: str, traditional_symbol: str, market_type: MarketType, timeframe: str
    ) -> Optional[CorrelationPair]:
        """Calculate correlation between a crypto and traditional market.

        Args:
            crypto_symbol: Crypto symbol
            traditional_symbol: Traditional market symbol
            market_type: Type of traditional market
            timeframe: Timeframe for correlation

        Returns:
            Correlation pair or None if not enough data
        """
        # Get market data
        if (
            crypto_symbol not in self.state.market_data
            or traditional_symbol not in self.state.market_data
        ):
            return None

        crypto_data = self.state.market_data[crypto_symbol]
        traditional_data = self.state.market_data[traditional_symbol]

        # Determine sample size based on timeframe
        if timeframe == "1d":
            sample_size = 1
        elif timeframe == "7d":
            sample_size = 7
        elif timeframe == "30d":
            sample_size = 30
        else:  # "90d"
            sample_size = 90

        # Ensure we have enough data
        if len(crypto_data) < sample_size or len(traditional_data) < sample_size:
            return None

        # Get recent data
        crypto_prices = [data.price for data in crypto_data[-sample_size:]]
        traditional_prices = [data.price for data in traditional_data[-sample_size:]]

        # Calculate correlation coefficient
        correlation = np.corrcoef(crypto_prices, traditional_prices)[0, 1]

        # Determine correlation strength
        if correlation >= 0.7:
            strength = CorrelationStrength.STRONG_POSITIVE
        elif correlation >= 0.3:
            strength = CorrelationStrength.MODERATE_POSITIVE
        elif correlation >= 0:
            strength = CorrelationStrength.WEAK_POSITIVE
        elif correlation >= -0.3:
            strength = CorrelationStrength.WEAK_NEGATIVE
        elif correlation >= -0.7:
            strength = CorrelationStrength.MODERATE_NEGATIVE
        else:
            strength = CorrelationStrength.STRONG_NEGATIVE

        return CorrelationPair(
            crypto_symbol=crypto_symbol,
            traditional_symbol=traditional_symbol,
            traditional_market_type=market_type,
            correlation_coefficient=correlation,
            correlation_strength=strength,
            timeframe=timeframe,
            sample_size=sample_size,
            timestamp=datetime.now().isoformat(),
        )

    async def _perform_macro_analysis(self, ctx: Context, crypto_symbol: str) -> MacroAnalysis:
        """Perform macro analysis for a crypto symbol.

        Args:
            ctx: Agent context
            crypto_symbol: Crypto symbol

        Returns:
            Macro analysis
        """
        # Get correlations for this crypto symbol
        correlations = [c for c in self.state.correlations if c.crypto_symbol == crypto_symbol]

        # Get upcoming events
        upcoming_events = self.state.upcoming_events

        # Determine overall macro sentiment
        sentiment, confidence = self._determine_macro_sentiment(correlations, upcoming_events)

        return MacroAnalysis(
            crypto_symbol=crypto_symbol,
            correlations=correlations,
            upcoming_events=upcoming_events,
            overall_macro_sentiment=sentiment,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )

    def _determine_macro_sentiment(
        self, correlations: List[CorrelationPair], upcoming_events: List[MacroEvent]
    ) -> tuple[str, float]:
        """Determine overall macro sentiment.

        Args:
            correlations: List of correlations
            upcoming_events: List of upcoming events

        Returns:
            Tuple of (sentiment, confidence)
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated analysis

        # Count positive and negative correlations
        positive_count = sum(
            1
            for c in correlations
            if c.correlation_strength
            in [CorrelationStrength.STRONG_POSITIVE, CorrelationStrength.MODERATE_POSITIVE]
        )

        negative_count = sum(
            1
            for c in correlations
            if c.correlation_strength
            in [CorrelationStrength.STRONG_NEGATIVE, CorrelationStrength.MODERATE_NEGATIVE]
        )

        # Count positive and negative event impacts
        positive_events = sum(1 for e in upcoming_events if e.expected_crypto_impact == "positive")

        negative_events = sum(1 for e in upcoming_events if e.expected_crypto_impact == "negative")

        # Calculate overall sentiment
        correlation_score = positive_count - negative_count
        event_score = positive_events - negative_events

        total_score = correlation_score + event_score

        # Determine sentiment
        if total_score > 1:
            sentiment = "bullish"
        elif total_score < -1:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # Calculate confidence
        total_correlations = len(correlations)
        total_events = len(upcoming_events)

        if total_correlations + total_events == 0:
            confidence = 0.5  # Default confidence
        else:
            # Confidence based on the strength of the signal
            confidence = min(0.5 + abs(total_score) * 0.1, 0.9)

        return sentiment, confidence
