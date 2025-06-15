"""
Advanced Crypto Trading System for the Fetch.ai platform.

This module integrates all specialized agents into a cohesive trading system.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import ccxt
from uagents import Context, Model

from .base_agent import BaseAgent, BaseAgentState
from .learning_agent import LearningOptimizationAgent
from .macro_agent import MacroCorrelationAgent
from .regulatory_agent import RegulatoryComplianceAgent
from .risk_agent import RiskManagementAgent
from .sentiment_agent import SentimentIntelligenceAgent
from .technical_agent import TechnicalAnalysisAgent


class TradingSignal(str, Enum):
    """Trading signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(str, Enum):
    """Signal strength levels."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TradingSignalSource(str, Enum):
    """Sources of trading signals."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    RISK = "risk"
    REGULATORY = "regulatory"
    LEARNING = "learning"


class TradeRecommendation(Model):
    """Model for a trade recommendation."""

    symbol: str
    signal: TradingSignal
    strength: SignalStrength
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    timeframe: str
    sources: List[TradingSignalSource]
    confidence: float
    reasoning: str
    timestamp: str


class TradingSystemState(BaseAgentState):
    """State model for the Advanced Crypto Trading System."""

    symbols_to_track: List[str] = ["BTC/USD", "ETH/USD"]
    recent_recommendations: List[TradeRecommendation] = []
    agent_addresses: Dict[str, str] = {}
    analysis_interval: int = 3600  # 1 hour in seconds


class AdvancedCryptoTradingSystem(BaseAgent):
    """Advanced Crypto Trading System integrating all specialized agents."""

    def __init__(
        self,
        name: str = "trading_system",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """Initialize the Advanced Crypto Trading System.

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
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        # Initialize agent state
        self.state = TradingSystemState()

        # Initialize specialized agents
        self.sentiment_agent = SentimentIntelligenceAgent()
        self.technical_agent = TechnicalAnalysisAgent(
            exchange_id=exchange_id, api_key=api_key, api_secret=api_secret
        )
        self.risk_agent = RiskManagementAgent()
        self.regulatory_agent = RegulatoryComplianceAgent()
        self.macro_agent = MacroCorrelationAgent()
        self.learning_agent = LearningOptimizationAgent()

        # Store agent addresses
        self.state.agent_addresses = {
            "sentiment": self.sentiment_agent.get_address(),
            "technical": self.technical_agent.get_address(),
            "risk": self.risk_agent.get_address(),
            "regulatory": self.regulatory_agent.get_address(),
            "macro": self.macro_agent.get_address(),
            "learning": self.learning_agent.get_address(),
        }

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register handlers for the agent."""

        @self.agent.on_interval(period=self.state.analysis_interval)
        async def generate_recommendations(ctx: Context):
            """Generate trading recommendations."""
            ctx.logger.info("Generating trading recommendations")

            for symbol in self.state.symbols_to_track:
                recommendation = await self._generate_recommendation(ctx, symbol)

                if recommendation:
                    # Update state
                    self.state.recent_recommendations.append(recommendation)
                    if len(self.state.recent_recommendations) > 20:
                        self.state.recent_recommendations.pop(0)

                    ctx.logger.info(
                        f"Recommendation for {symbol}: "
                        f"{recommendation.signal.upper()} "
                        f"({recommendation.strength}) "
                        f"with {recommendation.confidence:.2f} confidence"
                    )

    async def _generate_recommendation(
        self, ctx: Context, symbol: str
    ) -> Optional[TradeRecommendation]:
        """Generate a trading recommendation for a symbol.

        Args:
            ctx: Agent context
            symbol: Trading symbol

        Returns:
            Trade recommendation or None if not available
        """
        try:
            # Collect signals from all agents
            technical_signal = await self._get_technical_signal(symbol)
            sentiment_signal = await self._get_sentiment_signal(symbol)
            macro_signal = await self._get_macro_signal(symbol)
            risk_assessment = await self._get_risk_assessment(symbol)
            regulatory_check = await self._get_regulatory_check(symbol)
            learning_insight = await self._get_learning_insight(symbol)

            # Combine signals
            combined_signal, strength, confidence, sources = self._combine_signals(
                technical_signal,
                sentiment_signal,
                macro_signal,
                risk_assessment,
                regulatory_check,
                learning_insight,
            )

            # If no clear signal, return None
            if combined_signal == TradingSignal.HOLD and strength == SignalStrength.WEAK:
                return None

            # Get current market price
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # Calculate entry, stop loss, and take profit
            entry_price = current_price

            # For simplicity, we'll use fixed percentages
            # In a real system, these would be calculated based on volatility and risk
            stop_loss = (
                entry_price * 0.95 if combined_signal == TradingSignal.BUY else entry_price * 1.05
            )
            take_profit = (
                entry_price * 1.1 if combined_signal == TradingSignal.BUY else entry_price * 0.9
            )

            # Calculate position size (if risk assessment available)
            position_size = risk_assessment.get(
                "position_size", 0.1
            )  # Default to 10% of available balance

            # Generate reasoning
            reasoning = self._generate_reasoning(
                symbol,
                combined_signal,
                technical_signal,
                sentiment_signal,
                macro_signal,
                risk_assessment,
                regulatory_check,
                learning_insight,
            )

            return TradeRecommendation(
                symbol=symbol,
                signal=combined_signal,
                strength=strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timeframe="1h",
                sources=sources,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            ctx.logger.error(f"Error generating recommendation for {symbol}: {str(e)}")
            return None

    async def _get_technical_signal(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis signal.

        This is a mock implementation. In a real system, this would
        communicate with the Technical Analysis Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Technical analysis signal
        """
        # Mock data for demonstration
        import random

        signals = [TradingSignal.BUY, TradingSignal.SELL, TradingSignal.HOLD]
        strengths = [SignalStrength.STRONG, SignalStrength.MODERATE, SignalStrength.WEAK]

        return {
            "signal": random.choice(signals),
            "strength": random.choice(strengths),
            "confidence": random.uniform(0.5, 0.9),
            "indicators": ["RSI", "MACD", "Bollinger Bands"],
        }

    async def _get_sentiment_signal(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis signal.

        This is a mock implementation. In a real system, this would
        communicate with the Sentiment Intelligence Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Sentiment analysis signal
        """
        # Mock data for demonstration
        import random

        signals = [TradingSignal.BUY, TradingSignal.SELL, TradingSignal.HOLD]
        strengths = [SignalStrength.STRONG, SignalStrength.MODERATE, SignalStrength.WEAK]

        return {
            "signal": random.choice(signals),
            "strength": random.choice(strengths),
            "confidence": random.uniform(0.5, 0.9),
            "sentiment_score": random.uniform(-1.0, 1.0),
        }

    async def _get_macro_signal(self, symbol: str) -> Dict[str, Any]:
        """Get macro correlation signal.

        This is a mock implementation. In a real system, this would
        communicate with the Macro-Correlation Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Macro correlation signal
        """
        # Mock data for demonstration
        import random

        signals = [TradingSignal.BUY, TradingSignal.SELL, TradingSignal.HOLD]
        strengths = [SignalStrength.STRONG, SignalStrength.MODERATE, SignalStrength.WEAK]

        return {
            "signal": random.choice(signals),
            "strength": random.choice(strengths),
            "confidence": random.uniform(0.5, 0.9),
            "macro_sentiment": random.choice(["bullish", "bearish", "neutral"]),
        }

    async def _get_risk_assessment(self, symbol: str) -> Dict[str, Any]:
        """Get risk assessment.

        This is a mock implementation. In a real system, this would
        communicate with the Risk Management Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Risk assessment
        """
        # Mock data for demonstration
        import random

        return {
            "risk_level": random.choice(["low", "medium", "high"]),
            "position_size": random.uniform(0.05, 0.2),
            "stop_loss_percentage": random.uniform(0.03, 0.07),
            "confidence": random.uniform(0.5, 0.9),
        }

    async def _get_regulatory_check(self, symbol: str) -> Dict[str, Any]:
        """Get regulatory compliance check.

        This is a mock implementation. In a real system, this would
        communicate with the Regulatory Compliance Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Regulatory compliance check
        """
        # Mock data for demonstration
        import random

        return {
            "status": random.choice(["compliant", "needs_review", "non_compliant"]),
            "issues": [],
            "confidence": random.uniform(0.8, 1.0),
        }

    async def _get_learning_insight(self, symbol: str) -> Dict[str, Any]:
        """Get learning insight.

        This is a mock implementation. In a real system, this would
        communicate with the Learning Optimization Agent.

        Args:
            symbol: Trading symbol

        Returns:
            Learning insight
        """
        # Mock data for demonstration
        import random

        signals = [TradingSignal.BUY, TradingSignal.SELL, TradingSignal.HOLD]

        return {
            "signal": random.choice(signals),
            "prediction": random.choice(["up", "down", "sideways"]),
            "confidence": random.uniform(0.5, 0.9),
            "features_used": ["price", "volume", "sentiment"],
        }

    def _combine_signals(
        self,
        technical_signal: Dict[str, Any],
        sentiment_signal: Dict[str, Any],
        macro_signal: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        regulatory_check: Dict[str, Any],
        learning_insight: Dict[str, Any],
    ) -> tuple[TradingSignal, SignalStrength, float, List[TradingSignalSource]]:
        """Combine signals from all agents.

        Args:
            technical_signal: Technical analysis signal
            sentiment_signal: Sentiment analysis signal
            macro_signal: Macro correlation signal
            risk_assessment: Risk assessment
            regulatory_check: Regulatory compliance check
            learning_insight: Learning insight

        Returns:
            Tuple of (signal, strength, confidence, sources)
        """
        # Check regulatory compliance first
        if regulatory_check["status"] == "non_compliant":
            return TradingSignal.HOLD, SignalStrength.STRONG, 1.0, [TradingSignalSource.REGULATORY]

        # Check risk level
        if risk_assessment["risk_level"] == "high":
            # If high risk, be more conservative
            risk_multiplier = 0.5
        elif risk_assessment["risk_level"] == "medium":
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0

        # Assign weights to each signal source
        weights = {"technical": 0.3, "sentiment": 0.2, "macro": 0.2, "learning": 0.3}

        # Calculate weighted scores for BUY and SELL
        buy_score = 0.0
        sell_score = 0.0

        # Technical signal
        if technical_signal["signal"] == TradingSignal.BUY:
            buy_score += weights["technical"] * technical_signal["confidence"]
        elif technical_signal["signal"] == TradingSignal.SELL:
            sell_score += weights["technical"] * technical_signal["confidence"]

        # Sentiment signal
        if sentiment_signal["signal"] == TradingSignal.BUY:
            buy_score += weights["sentiment"] * sentiment_signal["confidence"]
        elif sentiment_signal["signal"] == TradingSignal.SELL:
            sell_score += weights["sentiment"] * sentiment_signal["confidence"]

        # Macro signal
        if macro_signal["signal"] == TradingSignal.BUY:
            buy_score += weights["macro"] * macro_signal["confidence"]
        elif macro_signal["signal"] == TradingSignal.SELL:
            sell_score += weights["macro"] * macro_signal["confidence"]

        # Learning insight
        if learning_insight["signal"] == TradingSignal.BUY:
            buy_score += weights["learning"] * learning_insight["confidence"]
        elif learning_insight["signal"] == TradingSignal.SELL:
            sell_score += weights["learning"] * learning_insight["confidence"]

        # Apply risk multiplier
        buy_score *= risk_multiplier
        sell_score *= risk_multiplier

        # Determine signal
        if buy_score > sell_score and buy_score > 0.3:
            signal = TradingSignal.BUY
            score = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            signal = TradingSignal.SELL
            score = sell_score
        else:
            signal = TradingSignal.HOLD
            score = max(buy_score, sell_score)

        # Determine strength
        if score > 0.7:
            strength = SignalStrength.STRONG
        elif score > 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Determine sources
        sources = []
        if technical_signal["signal"] == signal:
            sources.append(TradingSignalSource.TECHNICAL)
        if sentiment_signal["signal"] == signal:
            sources.append(TradingSignalSource.SENTIMENT)
        if macro_signal["signal"] == signal:
            sources.append(TradingSignalSource.MACRO)
        if learning_insight["signal"] == signal:
            sources.append(TradingSignalSource.LEARNING)

        # Add risk source if it influenced the decision
        if risk_multiplier < 1.0:
            sources.append(TradingSignalSource.RISK)

        # Add regulatory source if it influenced the decision
        if regulatory_check["status"] == "needs_review":
            sources.append(TradingSignalSource.REGULATORY)

        return signal, strength, score, sources

    def _generate_reasoning(
        self,
        symbol: str,
        signal: TradingSignal,
        technical_signal: Dict[str, Any],
        sentiment_signal: Dict[str, Any],
        macro_signal: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        regulatory_check: Dict[str, Any],
        learning_insight: Dict[str, Any],
    ) -> str:
        """Generate reasoning for a trading recommendation.

        Args:
            symbol: Trading symbol
            signal: Trading signal
            technical_signal: Technical analysis signal
            sentiment_signal: Sentiment analysis signal
            macro_signal: Macro correlation signal
            risk_assessment: Risk assessment
            regulatory_check: Regulatory compliance check
            learning_insight: Learning insight

        Returns:
            Reasoning
        """
        reasons = []

        if signal == TradingSignal.BUY:
            if technical_signal["signal"] == TradingSignal.BUY:
                reasons.append(
                    f"Technical indicators ({', '.join(technical_signal['indicators'])}) suggest bullish momentum"
                )

            if sentiment_signal["signal"] == TradingSignal.BUY:
                reasons.append(
                    f"Positive sentiment with score {sentiment_signal['sentiment_score']:.2f}"
                )

            if macro_signal["signal"] == TradingSignal.BUY:
                reasons.append(
                    f"Favorable macro conditions with {macro_signal['macro_sentiment']} outlook"
                )

            if learning_insight["signal"] == TradingSignal.BUY:
                reasons.append(f"ML model predicts price movement {learning_insight['prediction']}")

        elif signal == TradingSignal.SELL:
            if technical_signal["signal"] == TradingSignal.SELL:
                reasons.append(
                    f"Technical indicators ({', '.join(technical_signal['indicators'])}) suggest bearish momentum"
                )

            if sentiment_signal["signal"] == TradingSignal.SELL:
                reasons.append(
                    f"Negative sentiment with score {sentiment_signal['sentiment_score']:.2f}"
                )

            if macro_signal["signal"] == TradingSignal.SELL:
                reasons.append(
                    f"Unfavorable macro conditions with {macro_signal['macro_sentiment']} outlook"
                )

            if learning_insight["signal"] == TradingSignal.SELL:
                reasons.append(f"ML model predicts price movement {learning_insight['prediction']}")

        else:  # HOLD
            reasons.append("Mixed signals suggest holding current position")

            if risk_assessment["risk_level"] == "high":
                reasons.append("High risk level suggests caution")

            if regulatory_check["status"] == "needs_review":
                reasons.append("Regulatory concerns require review")

        return ". ".join(reasons)

    async def start_all_agents(self):
        """Start all specialized agents."""
        # Start agents asynchronously
        await asyncio.gather(
            self.sentiment_agent.run_async(),
            self.technical_agent.run_async(),
            self.risk_agent.run_async(),
            self.regulatory_agent.run_async(),
            self.macro_agent.run_async(),
            self.learning_agent.run_async(),
            self.agent.async_run(),
        )

    def run_all(self):
        """Run all agents in the system."""
        asyncio.run(self.start_all_agents())
