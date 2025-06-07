"""
Sentiment Intelligence Agent for the Fetch.ai Advanced Crypto Trading System.

This agent analyzes news and social media with VADER sentiment analysis,
incorporating source credibility weighting.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from uagents import Agent, Context, Model, Protocol
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base_agent import BaseAgent, BaseAgentState

class NewsSource(Model):
    """Model for a news source."""

    name: str
    url: str
    credibility_score: float = 1.0  # 0.0 to 1.0
    category: str = "general"

class SentimentData(Model):
    """Model for sentiment data."""

    source: str
    title: str
    url: str
    content: str
    timestamp: str
    sentiment_score: float = 0.0  # -1.0 to 1.0
    weighted_score: float = 0.0  # Adjusted by source credibility

class SentimentAnalysisResult(Model):
    """Model for sentiment analysis results."""

    symbol: str
    overall_sentiment: float = 0.0  # -1.0 to 1.0
    data_points: int = 0
    timestamp: str
    sources: List[str] = []
    detailed_scores: Dict[str, float] = {}

class SentimentAgentState(BaseAgentState):
    """State model for the Sentiment Intelligence Agent."""

    sources: List[NewsSource] = []
    recent_analyses: List[SentimentAnalysisResult] = []
    symbols_to_track: List[str] = ["BTC/USD", "ETH/USD"]
    analysis_interval: int = 3600  # seconds

class SentimentIntelligenceAgent(BaseAgent):
    """Agent for analyzing news and social media sentiment."""

    def __init__(
        self,
        name: str = "sentiment_agent",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Sentiment Intelligence Agent.

        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
        """
        super().__init__(name, seed, port, endpoint, logger)

        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

        # Initialize agent state
        self.state = SentimentAgentState()

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register handlers for the agent."""

        @self.agent.on_interval(period=self.state.analysis_interval)
        async def analyze_sentiment(ctx: Context):
            """Analyze sentiment for tracked symbols."""
            for symbol in self.state.symbols_to_track:
                await self._analyze_symbol_sentiment(ctx, symbol)

        @self.agent.on_message(model=NewsSource)
        async def handle_new_source(ctx: Context, sender: str, source: NewsSource):
            """Handle new source registration."""
            ctx.logger.info(f"Received new source from {sender}: {source.name}")

            # Add source if it doesn't exist
            if not any(s.name == source.name for s in self.state.sources):
                self.state.sources.append(source)
                ctx.logger.info(f"Added new source: {source.name}")
            else:
                # Update existing source
                for i, s in enumerate(self.state.sources):
                    if s.name == source.name:
                        self.state.sources[i] = source
                        ctx.logger.info(f"Updated source: {source.name}")

    async def _analyze_symbol_sentiment(self, ctx: Context, symbol: str):
        """Analyze sentiment for a specific symbol.

        Args:
            ctx: Agent context
            symbol: Trading symbol to analyze
        """
        ctx.logger.info(f"Analyzing sentiment for {symbol}")

        # Fetch news and social media data for the symbol
        # This would typically call external APIs or use web scraping
        news_data = await self._fetch_news_data(symbol)

        # Analyze sentiment for each piece of content
        sentiment_data = []
        for item in news_data:
            score = self._analyze_text_sentiment(item["content"])

            # Find source credibility
            source_credibility = 1.0
            for source in self.state.sources:
                if source.name == item["source"]:
                    source_credibility = source.credibility_score
                    break

            # Calculate weighted score
            weighted_score = score * source_credibility

            sentiment_data.append(SentimentData(
                source=item["source"],
                title=item["title"],
                url=item["url"],
                content=item["content"],
                timestamp=item["timestamp"],
                sentiment_score=score,
                weighted_score=weighted_score
            ))

        # Calculate overall sentiment
        if sentiment_data:
            overall_sentiment = sum(data.weighted_score for data in sentiment_data) / len(sentiment_data)

            # Create result
            result = SentimentAnalysisResult(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                data_points=len(sentiment_data),
                timestamp=datetime.now().isoformat(),
                sources=[data.source for data in sentiment_data],
                detailed_scores={data.source: data.weighted_score for data in sentiment_data}
            )

            # Update state
            self.state.recent_analyses.append(result)
            if len(self.state.recent_analyses) > 10:
                self.state.recent_analyses.pop(0)

            ctx.logger.info(f"Sentiment analysis for {symbol}: {overall_sentiment:.2f} ({len(sentiment_data)} data points)")

            # Broadcast result to other agents
            # Implementation depends on the communication protocol
        else:
            ctx.logger.warning(f"No sentiment data found for {symbol}")

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a text using VADER.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1.0 and 1.0
        """
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment["compound"]

    async def _fetch_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news data for a symbol.

        This is a mock implementation. In a real system, this would call
        external APIs or use web scraping.

        Args:
            symbol: Trading symbol

        Returns:
            List of news items
        """
        # Mock data for demonstration
        return [
            {
                "source": "CryptoNews",
                "title": f"{symbol} price surges after positive regulatory news",
                "url": "https://example.com/news/1",
                "content": f"The price of {symbol} has increased by 5% following positive regulatory developments.",
                "timestamp": datetime.now().isoformat()
            },
            {
                "source": "TradingView",
                "title": f"Technical analysis suggests {symbol} may continue uptrend",
                "url": "https://example.com/news/2",
                "content": f"Technical indicators are showing strong bullish signals for {symbol} in the short term.",
                "timestamp": datetime.now().isoformat()
            }
        ]
