"""
Additional data sources for the Fetch.ai Advanced Crypto Trading System.

This module provides integration with various data sources for market data,
news, social media, and other relevant information.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import requests
from dotenv import load_dotenv
from uagents import Agent, Context, Model, Protocol

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSourceType(str, Enum):
    """Types of data sources."""

    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC_CALENDAR = "economic_calendar"
    ON_CHAIN = "on_chain"
    ALTERNATIVE = "alternative"

class DataSource(Model):
    """Model for a data source."""

    name: str
    type: DataSourceType
    url: str
    api_key: Optional[str] = None
    active: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    last_request: Optional[str] = None

class NewsArticle(Model):
    """Model for a news article."""

    title: str
    url: str
    source: str
    content: str
    published_at: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None

class SocialMediaPost(Model):
    """Model for a social media post."""

    platform: str
    user: str
    content: str
    published_at: str
    likes: int = 0
    shares: int = 0
    comments: int = 0
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None

class EconomicEvent(Model):
    """Model for an economic event."""

    name: str
    country: str
    date: str
    time: Optional[str] = None
    impact: str  # "high", "medium", "low"
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

class OnChainMetric(Model):
    """Model for an on-chain metric."""

    name: str
    symbol: str
    value: float
    timestamp: str
    change_24h: Optional[float] = None
    change_7d: Optional[float] = None

class AlternativeDataPoint(Model):
    """Model for an alternative data point."""

    name: str
    value: Any
    timestamp: str
    source: str
    metadata: Dict[str, Any] = {}

class DataSourceManager:
    """Manager for data sources."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the data source manager.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("data_source_manager")

        # Initialize data sources
        self.data_sources = {}

        # Initialize data cache
        self.cache = {
            "news": [],
            "social_media": [],
            "economic_calendar": [],
            "on_chain": {},
            "alternative": {}
        }

        # Initialize default data sources
        self._initialize_default_data_sources()

    def _initialize_default_data_sources(self):
        """Initialize default data sources."""
        # CoinGecko for market data
        self.add_data_source(DataSource(
            name="CoinGecko",
            type=DataSourceType.MARKET_DATA,
            url="https://api.coingecko.com/api/v3",
            rate_limit=50
        ))

        # CryptoCompare for market data
        self.add_data_source(DataSource(
            name="CryptoCompare",
            type=DataSourceType.MARKET_DATA,
            url="https://min-api.cryptocompare.com/data",
            api_key=os.getenv("CRYPTOCOMPARE_API_KEY"),
            rate_limit=100
        ))

        # CryptoPanic for news
        self.add_data_source(DataSource(
            name="CryptoPanic",
            type=DataSourceType.NEWS,
            url="https://cryptopanic.com/api/v1",
            api_key=os.getenv("CRYPTOPANIC_API_KEY"),
            rate_limit=10
        ))

        # Lunarcrush for social media
        self.add_data_source(DataSource(
            name="LunarCrush",
            type=DataSourceType.SOCIAL_MEDIA,
            url="https://api.lunarcrush.com/v2",
            api_key=os.getenv("LUNARCRUSH_API_KEY"),
            rate_limit=10
        ))

        # ForexFactory for economic calendar
        self.add_data_source(DataSource(
            name="ForexFactory",
            type=DataSourceType.ECONOMIC_CALENDAR,
            url="https://forexfactory.com/calendar",
            rate_limit=1
        ))

        # Glassnode for on-chain metrics
        self.add_data_source(DataSource(
            name="Glassnode",
            type=DataSourceType.ON_CHAIN,
            url="https://api.glassnode.com/v1",
            api_key=os.getenv("GLASSNODE_API_KEY"),
            rate_limit=10
        ))

        # Alternative.me for Fear & Greed Index
        self.add_data_source(DataSource(
            name="Alternative.me",
            type=DataSourceType.ALTERNATIVE,
            url="https://api.alternative.me/fng",
            rate_limit=10
        ))

    def add_data_source(self, data_source: DataSource):
        """Add a data source.

        Args:
            data_source: Data source to add
        """
        key = f"{data_source.type}_{data_source.name}"
        self.data_sources[key] = data_source
        self.logger.info(f"Added data source: {data_source.name} ({data_source.type})")

    def get_data_source(self, type: DataSourceType, name: str) -> Optional[DataSource]:
        """Get a data source.

        Args:
            type: Type of data source
            name: Name of data source

        Returns:
            Data source or None if not found
        """
        key = f"{type}_{name}"
        return self.data_sources.get(key)

    def get_data_sources_by_type(self, type: DataSourceType) -> List[DataSource]:
        """Get data sources by type.

        Args:
            type: Type of data sources

        Returns:
            List of data sources
        """
        return [
            source for key, source in self.data_sources.items()
            if source.type == type and source.active
        ]

    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Market data
        """
        # Get market data sources
        sources = self.get_data_sources_by_type(DataSourceType.MARKET_DATA)

        if not sources:
            self.logger.warning("No market data sources available")
            return {}

        # Try each source
        for source in sources:
            try:
                if source.name == "CoinGecko":
                    data = await self._fetch_coingecko_market_data(source, symbol)
                    if data:
                        return data
                elif source.name == "CryptoCompare":
                    data = await self._fetch_cryptocompare_market_data(source, symbol)
                    if data:
                        return data
            except Exception as e:
                self.logger.error(f"Error fetching market data from {source.name}: {str(e)}")

        return {}

    async def _fetch_coingecko_market_data(self, source: DataSource, symbol: str) -> Dict[str, Any]:
        """Fetch market data from CoinGecko.

        Args:
            source: Data source
            symbol: Trading symbol

        Returns:
            Market data
        """
        # Convert symbol to CoinGecko format
        # Example: BTC/USD -> bitcoin
        coin_id = symbol.split("/")[0].lower()
        if coin_id == "btc":
            coin_id = "bitcoin"
        elif coin_id == "eth":
            coin_id = "ethereum"

        # Check rate limit
        if source.last_request:
            last_request_time = datetime.fromisoformat(source.last_request)
            elapsed = (datetime.now() - last_request_time).total_seconds()
            if elapsed < 60 / source.rate_limit:
                await asyncio.sleep(60 / source.rate_limit - elapsed)

        # Fetch data
        async with aiohttp.ClientSession() as session:
            url = f"{source.url}/coins/{coin_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update last request time
                    source.last_request = datetime.now().isoformat()

                    # Extract relevant data
                    market_data = data.get("market_data", {})
                    return {
                        "symbol": symbol,
                        "price": market_data.get("current_price", {}).get("usd"),
                        "market_cap": market_data.get("market_cap", {}).get("usd"),
                        "volume_24h": market_data.get("total_volume", {}).get("usd"),
                        "change_24h": market_data.get("price_change_percentage_24h"),
                        "change_7d": market_data.get("price_change_percentage_7d"),
                        "change_30d": market_data.get("price_change_percentage_30d"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "CoinGecko"
                    }
                else:
                    self.logger.error(f"Error fetching data from CoinGecko: Status {response.status}")
                    return {}

    async def _fetch_cryptocompare_market_data(self, source: DataSource, symbol: str) -> Dict[str, Any]:
        """Fetch market data from CryptoCompare.

        Args:
            source: Data source
            symbol: Trading symbol

        Returns:
            Market data
        """
        # Convert symbol to CryptoCompare format
        # Example: BTC/USD -> BTC
        coin = symbol.split("/")[0]
        currency = symbol.split("/")[1]

        # Check rate limit
        if source.last_request:
            last_request_time = datetime.fromisoformat(source.last_request)
            elapsed = (datetime.now() - last_request_time).total_seconds()
            if elapsed < 60 / source.rate_limit:
                await asyncio.sleep(60 / source.rate_limit - elapsed)

        # Fetch data
        async with aiohttp.ClientSession() as session:
            url = f"{source.url}/pricemultifull?fsyms={coin}&tsyms={currency}"
            headers = {}
            if source.api_key:
                headers["authorization"] = f"Apikey {source.api_key}"

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update last request time
                    source.last_request = datetime.now().isoformat()

                    # Extract relevant data
                    raw = data.get("RAW", {}).get(coin, {}).get(currency, {})
                    display = data.get("DISPLAY", {}).get(coin, {}).get(currency, {})

                    return {
                        "symbol": symbol,
                        "price": raw.get("PRICE"),
                        "market_cap": raw.get("MKTCAP"),
                        "volume_24h": raw.get("VOLUME24HOUR"),
                        "change_24h": raw.get("CHANGEPCT24HOUR"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "CryptoCompare"
                    }
                else:
                    self.logger.error(f"Error fetching data from CryptoCompare: Status {response.status}")
                    return {}

    async def fetch_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[NewsArticle]:
        """Fetch news articles.

        Args:
            symbol: Trading symbol (optional)
            limit: Maximum number of articles to fetch

        Returns:
            List of news articles
        """
        # Get news sources
        sources = self.get_data_sources_by_type(DataSourceType.NEWS)

        if not sources:
            self.logger.warning("No news sources available")
            return []

        # Try each source
        articles = []
        for source in sources:
            try:
                if source.name == "CryptoPanic":
                    new_articles = await self._fetch_cryptopanic_news(source, symbol, limit)
                    articles.extend(new_articles)

                    if len(articles) >= limit:
                        break
            except Exception as e:
                self.logger.error(f"Error fetching news from {source.name}: {str(e)}")

        # Update cache
        self.cache["news"] = articles

        return articles[:limit]

    async def _fetch_cryptopanic_news(self, source: DataSource, symbol: Optional[str], limit: int) -> List[NewsArticle]:
        """Fetch news from CryptoPanic.

        Args:
            source: Data source
            symbol: Trading symbol (optional)
            limit: Maximum number of articles to fetch

        Returns:
            List of news articles
        """
        # Check rate limit
        if source.last_request:
            last_request_time = datetime.fromisoformat(source.last_request)
            elapsed = (datetime.now() - last_request_time).total_seconds()
            if elapsed < 60 / source.rate_limit:
                await asyncio.sleep(60 / source.rate_limit - elapsed)

        # Fetch data
        async with aiohttp.ClientSession() as session:
            url = f"{source.url}/posts/?auth_token={source.api_key}&public=true&kind=news"

            if symbol:
                # Convert symbol to CryptoPanic format
                # Example: BTC/USD -> bitcoin
                currency = symbol.split("/")[0].lower()
                if currency == "btc":
                    currency = "bitcoin"
                elif currency == "eth":
                    currency = "ethereum"

                url += f"&currencies={currency}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update last request time
                    source.last_request = datetime.now().isoformat()

                    # Extract articles
                    articles = []
                    for result in data.get("results", [])[:limit]:
                        article = NewsArticle(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            source=result.get("source", {}).get("title", "CryptoPanic"),
                            content=result.get("title", ""),  # Use title as content since full content is not provided
                            published_at=result.get("published_at", datetime.now().isoformat()),
                            sentiment_score=None,  # Will be calculated later
                            relevance_score=None  # Will be calculated later
                        )
                        articles.append(article)

                    return articles
                else:
                    self.logger.error(f"Error fetching data from CryptoPanic: Status {response.status}")
                    return []

    async def fetch_social_media(self, symbol: str, limit: int = 10) -> List[SocialMediaPost]:
        """Fetch social media posts.

        Args:
            symbol: Trading symbol
            limit: Maximum number of posts to fetch

        Returns:
            List of social media posts
        """
        # Get social media sources
        sources = self.get_data_sources_by_type(DataSourceType.SOCIAL_MEDIA)

        if not sources:
            self.logger.warning("No social media sources available")
            return []

        # Try each source
        posts = []
        for source in sources:
            try:
                if source.name == "LunarCrush":
                    new_posts = await self._fetch_lunarcrush_social(source, symbol, limit)
                    posts.extend(new_posts)

                    if len(posts) >= limit:
                        break
            except Exception as e:
                self.logger.error(f"Error fetching social media from {source.name}: {str(e)}")

        # Update cache
        self.cache["social_media"] = posts

        return posts[:limit]

    async def _fetch_lunarcrush_social(self, source: DataSource, symbol: str, limit: int) -> List[SocialMediaPost]:
        """Fetch social media posts from LunarCrush.

        Args:
            source: Data source
            symbol: Trading symbol
            limit: Maximum number of posts to fetch

        Returns:
            List of social media posts
        """
        # Convert symbol to LunarCrush format
        # Example: BTC/USD -> BTC
        coin = symbol.split("/")[0]

        # Check rate limit
        if source.last_request:
            last_request_time = datetime.fromisoformat(source.last_request)
            elapsed = (datetime.now() - last_request_time).total_seconds()
            if elapsed < 60 / source.rate_limit:
                await asyncio.sleep(60 / source.rate_limit - elapsed)

        # Fetch data
        async with aiohttp.ClientSession() as session:
            url = f"{source.url}?data=feeds&key={source.api_key}&symbol={coin}&limit={limit}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update last request time
                    source.last_request = datetime.now().isoformat()

                    # Extract posts
                    posts = []
                    for item in data.get("data", [])[:limit]:
                        post = SocialMediaPost(
                            platform=item.get("type", "twitter"),
                            user=item.get("user_name", ""),
                            content=item.get("body", ""),
                            published_at=datetime.fromtimestamp(item.get("time", time.time())).isoformat(),
                            likes=item.get("likes", 0),
                            shares=item.get("retweets", 0),
                            comments=item.get("replies", 0),
                            sentiment_score=None,  # Will be calculated later
                            relevance_score=None  # Will be calculated later
                        )
                        posts.append(post)

                    return posts
                else:
                    self.logger.error(f"Error fetching data from LunarCrush: Status {response.status}")
                    return []

    async def fetch_fear_greed_index(self) -> Optional[AlternativeDataPoint]:
        """Fetch Fear & Greed Index.

        Returns:
            Fear & Greed Index data point or None if not available
        """
        # Get Alternative.me data source
        source = self.get_data_source(DataSourceType.ALTERNATIVE, "Alternative.me")

        if not source:
            self.logger.warning("Alternative.me data source not available")
            return None

        try:
            # Check rate limit
            if source.last_request:
                last_request_time = datetime.fromisoformat(source.last_request)
                elapsed = (datetime.now() - last_request_time).total_seconds()
                if elapsed < 60 / source.rate_limit:
                    await asyncio.sleep(60 / source.rate_limit - elapsed)

            # Fetch data
            async with aiohttp.ClientSession() as session:
                url = f"{source.url}/"

                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Update last request time
                        source.last_request = datetime.now().isoformat()

                        # Extract data
                        if data.get("data") and len(data["data"]) > 0:
                            item = data["data"][0]

                            data_point = AlternativeDataPoint(
                                name="Fear & Greed Index",
                                value=int(item.get("value", 0)),
                                timestamp=datetime.fromtimestamp(int(item.get("timestamp", time.time()))).isoformat(),
                                source="Alternative.me",
                                metadata={
                                    "classification": item.get("value_classification", ""),
                                    "previous_close": int(item.get("previous_close", 0)),
                                    "previous_1_week": int(item.get("previous_1_week", 0)),
                                    "previous_1_month": int(item.get("previous_1_month", 0))
                                }
                            )

                            # Update cache
                            self.cache["alternative"]["fear_greed_index"] = data_point

                            return data_point
                    else:
                        self.logger.error(f"Error fetching data from Alternative.me: Status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            return None

# Example usage
async def main():
    # Load environment variables
    load_dotenv()

    # Create data source manager
    manager = DataSourceManager()

    # Fetch market data
    market_data = await manager.fetch_market_data("BTC/USD")
    print(f"Market data: {market_data}")

    # Fetch news
    news = await manager.fetch_news("BTC/USD", limit=5)
    print(f"News: {news}")

    # Fetch social media
    social = await manager.fetch_social_media("BTC/USD", limit=5)
    print(f"Social media: {social}")

    # Fetch Fear & Greed Index
    fear_greed = await manager.fetch_fear_greed_index()
    print(f"Fear & Greed Index: {fear_greed}")

if __name__ == "__main__":
    asyncio.run(main())
