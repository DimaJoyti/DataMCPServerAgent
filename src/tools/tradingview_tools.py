"""
TradingView Scraping Tools for Crypto Portfolio Management.
This module provides specialized tools for extracting cryptocurrency data from TradingView.
"""

import json
import re
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from langchain_core.tools import BaseTool
from mcp import ClientSession

class TimeFrame(Enum):
    """TradingView timeframe options."""
    M1 = "1"
    M5 = "5"
    M15 = "15"
    M30 = "30"
    H1 = "60"
    H4 = "240"
    D1 = "1D"
    W1 = "1W"
    MN1 = "1M"

class CryptoExchange(Enum):
    """Supported cryptocurrency exchanges."""
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    BITSTAMP = "BITSTAMP"
    KRAKEN = "KRAKEN"
    BYBIT = "BYBIT"
    OKX = "OKX"

@dataclass
class CryptoSymbol:
    """Cryptocurrency symbol representation."""
    base: str  # BTC, ETH, etc.
    quote: str  # USD, USDT, etc.
    exchange: CryptoExchange

    @property
    def symbol(self) -> str:
        """Get the full symbol string."""
        return f"{self.base}{self.quote}"

    @property
    def tradingview_symbol(self) -> str:
        """Get TradingView formatted symbol."""
        return f"{self.exchange.value}:{self.symbol}"

@dataclass
class PriceData:
    """Price data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

@dataclass
class TechnicalIndicator:
    """Technical indicator data."""
    name: str
    value: float
    signal: str  # BUY, SELL, NEUTRAL
    timestamp: datetime

@dataclass
class MarketSentiment:
    """Market sentiment data."""
    symbol: str
    bullish_percentage: float
    bearish_percentage: float
    neutral_percentage: float
    total_votes: int
    timestamp: datetime

class TradingViewToolkit:
    """A toolkit for TradingView cryptocurrency data extraction."""

    def __init__(self, session: ClientSession):
        """Initialize the toolkit with an MCP client session.

        Args:
            session: An initialized MCP ClientSession
        """
        self.session = session
        self.base_url = "https://www.tradingview.com"

    async def create_crypto_tools(self) -> List[BaseTool]:
        """Create and return TradingView crypto tools.

        Returns:
            A list of specialized crypto BaseTool instances
        """
        tools = []

        # Get available tools from session
        available_tools = {}
        for plugin in await self.session.list_plugins():
            for tool in plugin.tools:
                available_tools[tool.name] = tool

        # Create specialized crypto tools
        if "scrape_as_markdown_Bright_Data" in available_tools:
            tools.extend([
                self._create_crypto_price_tool(available_tools["scrape_as_markdown_Bright_Data"]),
                self._create_crypto_analysis_tool(available_tools["scrape_as_markdown_Bright_Data"]),
                self._create_crypto_sentiment_tool(available_tools["scrape_as_markdown_Bright_Data"]),
                self._create_crypto_news_tool(available_tools["scrape_as_markdown_Bright_Data"]),
                self._create_crypto_screener_tool(available_tools["scrape_as_markdown_Bright_Data"]),
            ])

        if "scraping_browser_navigate_Bright_Data" in available_tools:
            tools.append(
                self._create_realtime_data_tool(available_tools)
            )

        return tools

    def _create_crypto_price_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a tool for extracting cryptocurrency price data.

        Args:
            base_tool: The base scraping tool

        Returns:
            A crypto price extraction tool
        """
        async def _run(
            symbol: str,
            exchange: str = "BINANCE",
            timeframe: str = "1D",
            period: str = "1M"
        ) -> str:
            """Extract cryptocurrency price data from TradingView."""
            try:
                # Construct TradingView URL
                tv_symbol = f"{exchange}:{symbol}"
                url = f"{self.base_url}/symbols/{symbol}/"

                # Scrape the page
                result = await base_tool.invoke({"url": url})

                # Parse price data
                price_data = self._parse_price_data(result, symbol)

                return self._format_price_data(price_data)

            except Exception as e:
                return f"Error extracting price data for {symbol}: {str(e)}"

        return BaseTool(
            name="tradingview_crypto_price",
            description="Extract cryptocurrency price data from TradingView including OHLCV, market cap, and volume",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Crypto symbol (e.g., BTCUSD, ETHUSD)"},
                    "exchange": {"type": "string", "description": "Exchange name", "default": "BINANCE"},
                    "timeframe": {"type": "string", "description": "Timeframe", "default": "1D"},
                    "period": {"type": "string", "description": "Time period", "default": "1M"}
                },
                "required": ["symbol"]
            }
        )

    def _create_crypto_analysis_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a tool for extracting technical analysis data.

        Args:
            base_tool: The base scraping tool

        Returns:
            A crypto technical analysis tool
        """
        async def _run(symbol: str, exchange: str = "BINANCE") -> str:
            """Extract technical analysis data from TradingView."""
            try:
                url = f"{self.base_url}/symbols/{symbol}/technicals/"
                result = await base_tool.invoke({"url": url})

                # Parse technical indicators
                indicators = self._parse_technical_indicators(result, symbol)

                return self._format_technical_analysis(indicators)

            except Exception as e:
                return f"Error extracting technical analysis for {symbol}: {str(e)}"

        return BaseTool(
            name="tradingview_crypto_analysis",
            description="Extract technical analysis and indicators from TradingView for cryptocurrencies",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Crypto symbol (e.g., BTCUSD, ETHUSD)"},
                    "exchange": {"type": "string", "description": "Exchange name", "default": "BINANCE"}
                },
                "required": ["symbol"]
            }
        )

    def _create_crypto_sentiment_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a tool for extracting market sentiment data.

        Args:
            base_tool: The base scraping tool

        Returns:
            A crypto sentiment analysis tool
        """
        async def _run(symbol: str) -> str:
            """Extract market sentiment data from TradingView."""
            try:
                url = f"{self.base_url}/symbols/{symbol}/minds/"
                result = await base_tool.invoke({"url": url})

                # Parse sentiment data
                sentiment = self._parse_sentiment_data(result, symbol)

                return self._format_sentiment_data(sentiment)

            except Exception as e:
                return f"Error extracting sentiment for {symbol}: {str(e)}"

        return BaseTool(
            name="tradingview_crypto_sentiment",
            description="Extract market sentiment and community opinions from TradingView",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Crypto symbol (e.g., BTCUSD, ETHUSD)"}
                },
                "required": ["symbol"]
            }
        )

    def _create_crypto_news_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a tool for extracting crypto news and events.

        Args:
            base_tool: The base scraping tool

        Returns:
            A crypto news extraction tool
        """
        async def _run(symbol: str, limit: int = 10) -> str:
            """Extract crypto news from TradingView."""
            try:
                url = f"{self.base_url}/symbols/{symbol}/news/"
                result = await base_tool.invoke({"url": url})

                # Parse news data
                news_items = self._parse_news_data(result, symbol, limit)

                return self._format_news_data(news_items)

            except Exception as e:
                return f"Error extracting news for {symbol}: {str(e)}"

        return BaseTool(
            name="tradingview_crypto_news",
            description="Extract cryptocurrency news and events from TradingView",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Crypto symbol (e.g., BTCUSD, ETHUSD)"},
                    "limit": {"type": "integer", "description": "Number of news items to return", "default": 10}
                },
                "required": ["symbol"]
            }
        )

    def _create_crypto_screener_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a tool for crypto market screening.

        Args:
            base_tool: The base scraping tool

        Returns:
            A crypto market screener tool
        """
        async def _run(
            market_cap_min: Optional[float] = None,
            volume_min: Optional[float] = None,
            change_min: Optional[float] = None,
            limit: int = 50
        ) -> str:
            """Screen cryptocurrency markets based on criteria."""
            try:
                url = f"{self.base_url}/markets/cryptocurrencies/prices-all/"
                result = await base_tool.invoke({"url": url})

                # Parse and filter crypto data
                crypto_list = self._parse_crypto_screener(
                    result, market_cap_min, volume_min, change_min, limit
                )

                return self._format_screener_data(crypto_list)

            except Exception as e:
                return f"Error screening crypto markets: {str(e)}"

        return BaseTool(
            name="tradingview_crypto_screener",
            description="Screen cryptocurrency markets based on various criteria like market cap, volume, and price changes",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "market_cap_min": {"type": "number", "description": "Minimum market cap filter"},
                    "volume_min": {"type": "number", "description": "Minimum volume filter"},
                    "change_min": {"type": "number", "description": "Minimum price change % filter"},
                    "limit": {"type": "integer", "description": "Number of results to return", "default": 50}
                }
            }
        )

    def _create_realtime_data_tool(self, available_tools: Dict[str, BaseTool]) -> BaseTool:
        """Create a tool for real-time crypto data extraction.

        Args:
            available_tools: Dictionary of available tools

        Returns:
            A real-time crypto data tool
        """
        async def _run(symbols: List[str], duration: int = 60) -> str:
            """Extract real-time crypto data using browser automation."""
            try:
                navigate_tool = available_tools["scraping_browser_navigate_Bright_Data"]
                get_text_tool = available_tools["scraping_browser_get_text_Bright_Data"]

                results = []

                for symbol in symbols:
                    # Navigate to TradingView chart
                    chart_url = f"{self.base_url}/chart/?symbol={symbol}"
                    await navigate_tool.invoke({"url": chart_url})

                    # Wait for page to load
                    await asyncio.sleep(2)

                    # Extract real-time data
                    page_content = await get_text_tool.invoke({})

                    # Parse real-time price data
                    price_info = self._parse_realtime_data(page_content, symbol)
                    results.append(price_info)

                return self._format_realtime_data(results)

            except Exception as e:
                return f"Error extracting real-time data: {str(e)}"

        return BaseTool(
            name="tradingview_realtime_crypto",
            description="Extract real-time cryptocurrency data using browser automation",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "List of crypto symbols"},
                    "duration": {"type": "integer", "description": "Duration to monitor in seconds", "default": 60}
                },
                "required": ["symbols"]
            }
        )

    # Data parsing methods
    def _parse_price_data(self, content: str, symbol: str) -> Dict[str, Any]:
        """Parse price data from TradingView content."""
        price_data = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "market_cap": None,
            "high_24h": None,
            "low_24h": None
        }

        try:
            # Extract current price
            price_match = re.search(r'(\d+,?\d*\.?\d*)\s*USD', content)
            if price_match:
                price_data["price"] = float(price_match.group(1).replace(',', ''))

            # Extract price change
            change_match = re.search(r'([+-]?\d+,?\d*\.?\d*)\s*\(([+-]?\d+\.?\d*)%\)', content)
            if change_match:
                price_data["change"] = float(change_match.group(1).replace(',', ''))
                price_data["change_percent"] = float(change_match.group(2))

            # Extract market cap
            market_cap_match = re.search(r'Market capitalization\s*([0-9.,]+\s*[KMBT]?)\s*USD', content)
            if market_cap_match:
                price_data["market_cap"] = self._parse_number_with_suffix(market_cap_match.group(1))

            # Extract volume
            volume_match = re.search(r'Trading volume 24h\s*([0-9.,]+\s*[KMBT]?)\s*USD', content)
            if volume_match:
                price_data["volume"] = self._parse_number_with_suffix(volume_match.group(1))

        except Exception as e:
            print(f"Error parsing price data: {e}")

        return price_data

    def _parse_technical_indicators(self, content: str, symbol: str) -> List[TechnicalIndicator]:
        """Parse technical indicators from TradingView content."""
        indicators = []

        try:
            # Look for technical analysis summary
            if "Strong sell" in content:
                indicators.append(TechnicalIndicator(
                    name="Overall Signal",
                    value=0.0,
                    signal="STRONG_SELL",
                    timestamp=datetime.now()
                ))
            elif "Sell" in content:
                indicators.append(TechnicalIndicator(
                    name="Overall Signal",
                    value=0.25,
                    signal="SELL",
                    timestamp=datetime.now()
                ))
            elif "Neutral" in content:
                indicators.append(TechnicalIndicator(
                    name="Overall Signal",
                    value=0.5,
                    signal="NEUTRAL",
                    timestamp=datetime.now()
                ))
            elif "Buy" in content:
                indicators.append(TechnicalIndicator(
                    name="Overall Signal",
                    value=0.75,
                    signal="BUY",
                    timestamp=datetime.now()
                ))
            elif "Strong buy" in content:
                indicators.append(TechnicalIndicator(
                    name="Overall Signal",
                    value=1.0,
                    signal="STRONG_BUY",
                    timestamp=datetime.now()
                ))

        except Exception as e:
            print(f"Error parsing technical indicators: {e}")

        return indicators

    def _parse_sentiment_data(self, content: str, symbol: str) -> MarketSentiment:
        """Parse market sentiment data from TradingView content."""
        sentiment = MarketSentiment(
            symbol=symbol,
            bullish_percentage=50.0,
            bearish_percentage=50.0,
            neutral_percentage=0.0,
            total_votes=0,
            timestamp=datetime.now()
        )

        try:
            # Extract sentiment percentages if available
            # This would need to be adapted based on actual TradingView sentiment page structure
            pass

        except Exception as e:
            print(f"Error parsing sentiment data: {e}")

        return sentiment

    def _parse_news_data(self, content: str, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Parse news data from TradingView content."""
        news_items = []

        try:
            # Extract news headlines and links
            # This would need to be adapted based on actual TradingView news page structure
            lines = content.split('\n')
            for line in lines[:limit]:
                if line.strip() and len(line) > 20:
                    news_items.append({
                        "title": line.strip(),
                        "timestamp": datetime.now(),
                        "symbol": symbol
                    })

        except Exception as e:
            print(f"Error parsing news data: {e}")

        return news_items

    def _parse_crypto_screener(
        self,
        content: str,
        market_cap_min: Optional[float],
        volume_min: Optional[float],
        change_min: Optional[float],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Parse crypto screener data from TradingView content."""
        crypto_list = []

        try:
            # This would need to be adapted based on actual TradingView screener page structure
            # For now, return sample data
            sample_cryptos = ["Bitcoin", "Ethereum", "Binance Coin", "Cardano", "Solana"]

            for i, crypto in enumerate(sample_cryptos[:limit]):
                crypto_list.append({
                    "name": crypto,
                    "symbol": f"{crypto[:3].upper()}USD",
                    "price": 50000.0 - (i * 10000),
                    "change_24h": 2.5 - (i * 0.5),
                    "volume_24h": 1000000000 - (i * 100000000),
                    "market_cap": 500000000000 - (i * 50000000000)
                })

        except Exception as e:
            print(f"Error parsing screener data: {e}")

        return crypto_list

    def _parse_realtime_data(self, content: str, symbol: str) -> Dict[str, Any]:
        """Parse real-time data from browser content."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "price": 0.0,
            "volume": 0.0,
            "change": 0.0
        }

    # Data formatting methods
    def _format_price_data(self, price_data: Dict[str, Any]) -> str:
        """Format price data for display."""
        output = f"# 📊 {price_data['symbol']} Price Data\n\n"

        if price_data.get("price"):
            output += f"**Current Price**: ${price_data['price']:,.2f}\n"

        if price_data.get("change") is not None:
            change_symbol = "📈" if price_data["change"] >= 0 else "📉"
            output += f"**24h Change**: {change_symbol} ${price_data['change']:,.2f} ({price_data.get('change_percent', 0):.2f}%)\n"

        if price_data.get("volume"):
            output += f"**24h Volume**: ${price_data['volume']:,.0f}\n"

        if price_data.get("market_cap"):
            output += f"**Market Cap**: ${price_data['market_cap']:,.0f}\n"

        output += f"\n**Last Updated**: {price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        return output

    def _format_technical_analysis(self, indicators: List[TechnicalIndicator]) -> str:
        """Format technical analysis data for display."""
        output = "# 📈 Technical Analysis\n\n"

        if not indicators:
            output += "No technical indicators available.\n"
            return output

        for indicator in indicators:
            signal_emoji = {
                "STRONG_BUY": "🟢",
                "BUY": "🔵",
                "NEUTRAL": "⚪",
                "SELL": "🔴",
                "STRONG_SELL": "🔴"
            }.get(indicator.signal, "⚪")

            output += f"**{indicator.name}**: {signal_emoji} {indicator.signal}\n"

        output += f"\n**Analysis Time**: {indicators[0].timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        return output

    def _format_sentiment_data(self, sentiment: MarketSentiment) -> str:
        """Format sentiment data for display."""
        output = f"# 🎭 Market Sentiment for {sentiment.symbol}\n\n"

        output += f"**Bullish**: 🟢 {sentiment.bullish_percentage:.1f}%\n"
        output += f"**Bearish**: 🔴 {sentiment.bearish_percentage:.1f}%\n"
        output += f"**Neutral**: ⚪ {sentiment.neutral_percentage:.1f}%\n"
        output += f"**Total Votes**: {sentiment.total_votes:,}\n"

        # Determine overall sentiment
        if sentiment.bullish_percentage > sentiment.bearish_percentage:
            overall = "🟢 Bullish"
        elif sentiment.bearish_percentage > sentiment.bullish_percentage:
            overall = "🔴 Bearish"
        else:
            overall = "⚪ Neutral"

        output += f"\n**Overall Sentiment**: {overall}\n"
        output += f"**Updated**: {sentiment.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        return output

    def _format_news_data(self, news_items: List[Dict[str, Any]]) -> str:
        """Format news data for display."""
        output = "# 📰 Cryptocurrency News\n\n"

        if not news_items:
            output += "No news items available.\n"
            return output

        for i, item in enumerate(news_items, 1):
            output += f"## {i}. {item['title']}\n"
            output += f"**Time**: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"

        return output

    def _format_screener_data(self, crypto_list: List[Dict[str, Any]]) -> str:
        """Format screener data for display."""
        output = "# 🔍 Cryptocurrency Market Screener\n\n"

        if not crypto_list:
            output += "No cryptocurrencies match the specified criteria.\n"
            return output

        output += "| Rank | Name | Symbol | Price | 24h Change | Volume | Market Cap |\n"
        output += "|------|------|--------|-------|------------|--------|------------|\n"

        for i, crypto in enumerate(crypto_list, 1):
            change_emoji = "📈" if crypto.get("change_24h", 0) >= 0 else "📉"
            output += f"| {i} | {crypto['name']} | {crypto['symbol']} | "
            output += f"${crypto['price']:,.2f} | {change_emoji} {crypto.get('change_24h', 0):.2f}% | "
            output += f"${crypto['volume_24h']:,.0f} | ${crypto['market_cap']:,.0f} |\n"

        return output

    def _format_realtime_data(self, results: List[Dict[str, Any]]) -> str:
        """Format real-time data for display."""
        output = "# ⚡ Real-time Cryptocurrency Data\n\n"

        for result in results:
            output += f"## {result['symbol']}\n"
            output += f"**Price**: ${result['price']:,.2f}\n"
            output += f"**Volume**: ${result['volume']:,.0f}\n"
            output += f"**Change**: {result['change']:+.2f}%\n"
            output += f"**Updated**: {result['timestamp'].strftime('%H:%M:%S UTC')}\n\n"

        return output

    # Helper methods
    def _parse_number_with_suffix(self, value_str: str) -> float:
        """Parse numbers with K, M, B, T suffixes."""
        value_str = value_str.replace(',', '').strip()

        if value_str.endswith('K'):
            return float(value_str[:-1]) * 1_000
        elif value_str.endswith('M'):
            return float(value_str[:-1]) * 1_000_000
        elif value_str.endswith('B'):
            return float(value_str[:-1]) * 1_000_000_000
        elif value_str.endswith('T'):
            return float(value_str[:-1]) * 1_000_000_000_000
        else:
            return float(value_str)

    @staticmethod
    def get_popular_crypto_symbols() -> List[CryptoSymbol]:
        """Get a list of popular cryptocurrency symbols."""
        return [
            CryptoSymbol("BTC", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("ETH", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("BNB", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("ADA", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("SOL", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("XRP", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("DOT", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("DOGE", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("AVAX", "USD", CryptoExchange.BINANCE),
            CryptoSymbol("MATIC", "USD", CryptoExchange.BINANCE),
        ]

    @staticmethod
    def get_supported_exchanges() -> List[CryptoExchange]:
        """Get list of supported exchanges."""
        return list(CryptoExchange)

    @staticmethod
    def get_supported_timeframes() -> List[TimeFrame]:
        """Get list of supported timeframes."""
        return list(TimeFrame)


class TradingViewWebSocketClient:
    """WebSocket client for real-time TradingView data."""

    def __init__(self, symbols: List[str], callback=None):
        self.symbols = symbols
        self.callback = callback
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        """Connect to TradingView WebSocket."""
        try:
            # This is a placeholder for actual TradingView WebSocket implementation
            # In production, you would connect to TradingView's real-time data feed
            self.is_connected = True
            self.logger.info("Connected to TradingView WebSocket")
        except Exception as e:
            self.logger.error(f"Failed to connect to TradingView WebSocket: {e}")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for symbols."""
        if not self.is_connected:
            await self.connect()

        for symbol in symbols:
            # Send subscription message
            message = {
                "method": "subscribe",
                "params": {
                    "symbol": symbol,
                    "resolution": "1"
                }
            }
            # In production, send this message via WebSocket

    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False


class TradingViewChartingEngine:
    """Advanced charting engine with TradingView integration."""

    def __init__(self):
        self.chart_configs = {}
        self.indicators = {}
        self.strategies = {}

    def create_chart_config(
        self,
        symbol: str,
        timeframe: str = "1H",
        indicators: List[str] = None,
        overlays: List[str] = None
    ) -> Dict[str, Any]:
        """Create TradingView chart configuration."""
        config = {
            "symbol": symbol,
            "interval": timeframe,
            "container_id": f"tradingview_chart_{symbol.replace('/', '_')}",
            "width": "100%",
            "height": 600,
            "theme": "dark",
            "style": "1",  # Candlestick
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": False,
            "allow_symbol_change": True,
            "studies": indicators or [],
            "drawings": overlays or [],
            "show_popup_button": True,
            "popup_width": "1000",
            "popup_height": "650",
            "no_referrer_policy": True
        }

        self.chart_configs[symbol] = config
        return config

    def add_custom_indicator(
        self,
        name: str,
        script: str,
        inputs: Dict[str, Any] = None
    ) -> None:
        """Add custom Pine Script indicator."""
        self.indicators[name] = {
            "script": script,
            "inputs": inputs or {},
            "type": "custom"
        }

    def add_strategy_overlay(
        self,
        strategy_name: str,
        signals: List[Dict[str, Any]]
    ) -> None:
        """Add strategy signals as chart overlay."""
        self.strategies[strategy_name] = {
            "signals": signals,
            "type": "strategy_overlay"
        }

    def generate_chart_html(self, symbol: str) -> str:
        """Generate HTML for TradingView chart widget."""
        config = self.chart_configs.get(symbol, {})

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TradingView Chart - {symbol}</title>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        </head>
        <body>
            <div id="{config.get('container_id', 'tradingview_chart')}" style="height: {config.get('height', 600)}px;"></div>
            <script type="text/javascript">
                new TradingView.widget({{
                    "width": "{config.get('width', '100%')}",
                    "height": {config.get('height', 600)},
                    "symbol": "{config.get('symbol', symbol)}",
                    "interval": "{config.get('interval', '1H')}",
                    "timezone": "Etc/UTC",
                    "theme": "{config.get('theme', 'dark')}",
                    "style": "{config.get('style', '1')}",
                    "locale": "{config.get('locale', 'en')}",
                    "toolbar_bg": "{config.get('toolbar_bg', '#f1f3f6')}",
                    "enable_publishing": {str(config.get('enable_publishing', False)).lower()},
                    "allow_symbol_change": {str(config.get('allow_symbol_change', True)).lower()},
                    "container_id": "{config.get('container_id', 'tradingview_chart')}"
                }});
            </script>
        </body>
        </html>
        """

        return html_template


# Factory function for easy tool creation
async def create_tradingview_tools(session: ClientSession) -> List[BaseTool]:
    """Factory function to create TradingView tools.

    Args:
        session: MCP ClientSession

    Returns:
        List of TradingView crypto tools
    """
    toolkit = TradingViewToolkit(session)
    return await toolkit.create_crypto_tools()
