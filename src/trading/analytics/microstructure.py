"""
Market microstructure analysis for institutional trading.
"""

import asyncio
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ..market_data.data_types import OrderBook, Quote, Trade


class MarketMicrostructureAnalyzer:
    """
    Market microstructure analyzer for institutional trading.

    Features:
    - Spread analysis and monitoring
    - Liquidity metrics calculation
    - Market impact estimation
    - Trade classification (aggressive vs passive)
    - Price discovery analysis
    - Market quality metrics
    """

    def __init__(
        self,
        name: str = "MicrostructureAnalyzer",
        analysis_window_minutes: int = 60,
        update_frequency_seconds: int = 10,
    ):
        self.name = name
        self.analysis_window = timedelta(minutes=analysis_window_minutes)
        self.update_frequency = update_frequency_seconds

        self.logger = logging.getLogger(f"MicrostructureAnalyzer.{name}")
        self.is_running = False

        # Data storage
        self.quotes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.order_books: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Spread metrics
        self.spread_metrics: Dict[str, Dict] = defaultdict(dict)
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Liquidity metrics
        self.liquidity_metrics: Dict[str, Dict] = defaultdict(dict)

        # Market impact models
        self.impact_models: Dict[str, Dict] = defaultdict(dict)

        # Trade classification
        self.trade_classification: Dict[str, Dict] = defaultdict(
            lambda: {"aggressive_buy": 0, "aggressive_sell": 0, "passive_buy": 0, "passive_sell": 0}
        )

        # Price discovery metrics
        self.price_discovery: Dict[str, Dict] = defaultdict(dict)

        # Performance tracking
        self.analysis_count = 0
        self.analysis_latency_us: deque = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the microstructure analyzer."""
        self.logger.info(f"Starting microstructure analyzer: {self.name}")
        self.is_running = True

        # Start analysis tasks
        asyncio.create_task(self._analyze_spreads())
        asyncio.create_task(self._analyze_liquidity())
        asyncio.create_task(self._analyze_market_impact())
        asyncio.create_task(self._classify_trades())
        asyncio.create_task(self._cleanup_old_data())

        self.logger.info(f"Microstructure analyzer started: {self.name}")

    async def stop(self) -> None:
        """Stop the microstructure analyzer."""
        self.logger.info(f"Stopping microstructure analyzer: {self.name}")
        self.is_running = False
        self.logger.info(f"Microstructure analyzer stopped: {self.name}")

    async def update_quote(self, quote: Quote) -> None:
        """Update with new quote data."""
        self.quotes[quote.symbol].append(quote)

    async def update_trade(self, trade: Trade) -> None:
        """Update with new trade data."""
        self.trades[trade.symbol].append(trade)

        # Classify trade immediately
        await self._classify_trade(trade)

    async def update_order_book(self, order_book: OrderBook) -> None:
        """Update with new order book data."""
        self.order_books[order_book.symbol].append(order_book)

    async def _analyze_spreads(self) -> None:
        """Analyze bid-ask spreads."""
        while self.is_running:
            try:
                start_time = datetime.utcnow()

                for symbol in list(self.quotes.keys()):
                    await self._calculate_spread_metrics(symbol)

                # Track analysis latency
                analysis_time = (datetime.utcnow() - start_time).total_seconds() * 1_000_000
                self.analysis_latency_us.append(analysis_time)
                self.analysis_count += 1

                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                self.logger.error(f"Error analyzing spreads: {str(e)}")
                await asyncio.sleep(self.update_frequency)

    async def _calculate_spread_metrics(self, symbol: str) -> None:
        """Calculate spread metrics for a symbol."""
        quotes = list(self.quotes[symbol])
        if len(quotes) < 10:
            return

        # Filter recent quotes
        cutoff_time = datetime.utcnow() - self.analysis_window
        recent_quotes = [q for q in quotes if q.timestamp >= cutoff_time]

        if not recent_quotes:
            return

        # Calculate spread statistics
        spreads = []
        spread_bps = []

        for quote in recent_quotes:
            if quote.spread is not None:
                spreads.append(float(quote.spread))

                if quote.spread_bps is not None:
                    spread_bps.append(quote.spread_bps)

        if spreads:
            metrics = {
                "timestamp": datetime.utcnow(),
                "count": len(spreads),
                "mean_spread": statistics.mean(spreads),
                "median_spread": statistics.median(spreads),
                "std_spread": statistics.stdev(spreads) if len(spreads) > 1 else 0.0,
                "min_spread": min(spreads),
                "max_spread": max(spreads),
                "p95_spread": sorted(spreads)[int(len(spreads) * 0.95)],
                "p99_spread": sorted(spreads)[int(len(spreads) * 0.99)],
            }

            if spread_bps:
                metrics.update(
                    {
                        "mean_spread_bps": statistics.mean(spread_bps),
                        "median_spread_bps": statistics.median(spread_bps),
                        "std_spread_bps": (
                            statistics.stdev(spread_bps) if len(spread_bps) > 1 else 0.0
                        ),
                    }
                )

            self.spread_metrics[symbol] = metrics
            self.spread_history[symbol].append((datetime.utcnow(), metrics["mean_spread"]))

    async def _analyze_liquidity(self) -> None:
        """Analyze market liquidity."""
        while self.is_running:
            try:
                for symbol in list(self.order_books.keys()):
                    await self._calculate_liquidity_metrics(symbol)

                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                self.logger.error(f"Error analyzing liquidity: {str(e)}")
                await asyncio.sleep(self.update_frequency)

    async def _calculate_liquidity_metrics(self, symbol: str) -> None:
        """Calculate liquidity metrics for a symbol."""
        books = list(self.order_books[symbol])
        if not books:
            return

        # Use most recent order book
        latest_book = books[-1]

        # Calculate depth metrics
        levels = [1, 5, 10]
        metrics = {"timestamp": datetime.utcnow()}

        for level in levels:
            bid_depth = latest_book.get_depth("BID", level)
            ask_depth = latest_book.get_depth("ASK", level)

            bid_volume = sum(l.size for l in bid_depth)
            ask_volume = sum(l.size for l in ask_depth)
            total_volume = bid_volume + ask_volume

            metrics[f"bid_volume_L{level}"] = float(bid_volume)
            metrics[f"ask_volume_L{level}"] = float(ask_volume)
            metrics[f"total_volume_L{level}"] = float(total_volume)

            # Calculate imbalance
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                metrics[f"imbalance_L{level}"] = float(imbalance)

        # Calculate weighted average prices
        if latest_book.bids and latest_book.asks:
            bid_wap = latest_book.get_weighted_price("BID", 5)
            ask_wap = latest_book.get_weighted_price("ASK", 5)

            if bid_wap and ask_wap:
                metrics["bid_wap"] = float(bid_wap)
                metrics["ask_wap"] = float(ask_wap)
                metrics["mid_wap"] = float((bid_wap + ask_wap) / 2)

        # Calculate resilience (how quickly liquidity replenishes)
        if len(books) >= 2:
            prev_book = books[-2]
            time_diff = (latest_book.timestamp - prev_book.timestamp).total_seconds()

            if time_diff > 0:
                # Compare top-of-book changes
                if (
                    latest_book.best_bid
                    and prev_book.best_bid
                    and latest_book.best_ask
                    and prev_book.best_ask
                ):

                    bid_change = abs(latest_book.best_bid.size - prev_book.best_bid.size)
                    ask_change = abs(latest_book.best_ask.size - prev_book.best_ask.size)

                    metrics["bid_resilience"] = float(bid_change / time_diff)
                    metrics["ask_resilience"] = float(ask_change / time_diff)

        self.liquidity_metrics[symbol] = metrics

    async def _analyze_market_impact(self) -> None:
        """Analyze market impact of trades."""
        while self.is_running:
            try:
                for symbol in list(self.trades.keys()):
                    await self._calculate_market_impact(symbol)

                await asyncio.sleep(self.update_frequency * 2)  # Less frequent

            except Exception as e:
                self.logger.error(f"Error analyzing market impact: {str(e)}")
                await asyncio.sleep(self.update_frequency)

    async def _calculate_market_impact(self, symbol: str) -> None:
        """Calculate market impact metrics for a symbol."""
        trades = list(self.trades[symbol])
        quotes = list(self.quotes[symbol])

        if len(trades) < 10 or len(quotes) < 10:
            return

        # Filter recent data
        cutoff_time = datetime.utcnow() - self.analysis_window
        recent_trades = [t for t in trades if t.timestamp >= cutoff_time]
        recent_quotes = [q for q in quotes if q.timestamp >= cutoff_time]

        if not recent_trades or not recent_quotes:
            return

        # Calculate temporary impact (immediate price movement)
        temporary_impacts = []
        permanent_impacts = []

        for trade in recent_trades:
            # Find quotes before and after trade
            pre_quotes = [q for q in recent_quotes if q.timestamp <= trade.timestamp]
            post_quotes = [q for q in recent_quotes if q.timestamp > trade.timestamp]

            if pre_quotes and post_quotes:
                pre_mid = pre_quotes[-1].mid_price
                post_mid = post_quotes[0].mid_price if len(post_quotes) > 0 else None

                # Find quote 1 minute after trade for permanent impact
                one_min_later = trade.timestamp + timedelta(minutes=1)
                later_quotes = [q for q in post_quotes if q.timestamp >= one_min_later]
                later_mid = later_quotes[0].mid_price if later_quotes else None

                if pre_mid and post_mid:
                    # Calculate impact in basis points
                    if trade.buyer_initiated:
                        temp_impact = (post_mid - pre_mid) / pre_mid * 10000
                    else:
                        temp_impact = (pre_mid - post_mid) / pre_mid * 10000

                    temporary_impacts.append(temp_impact)

                    if later_mid:
                        if trade.buyer_initiated:
                            perm_impact = (later_mid - pre_mid) / pre_mid * 10000
                        else:
                            perm_impact = (pre_mid - later_mid) / pre_mid * 10000

                        permanent_impacts.append(perm_impact)

        # Calculate impact statistics
        metrics = {"timestamp": datetime.utcnow()}

        if temporary_impacts:
            metrics.update(
                {
                    "temporary_impact_mean": statistics.mean(temporary_impacts),
                    "temporary_impact_median": statistics.median(temporary_impacts),
                    "temporary_impact_std": (
                        statistics.stdev(temporary_impacts) if len(temporary_impacts) > 1 else 0.0
                    ),
                }
            )

        if permanent_impacts:
            metrics.update(
                {
                    "permanent_impact_mean": statistics.mean(permanent_impacts),
                    "permanent_impact_median": statistics.median(permanent_impacts),
                    "permanent_impact_std": (
                        statistics.stdev(permanent_impacts) if len(permanent_impacts) > 1 else 0.0
                    ),
                }
            )

        # Calculate impact per unit volume
        if recent_trades and temporary_impacts:
            volumes = [float(t.size) for t in recent_trades]
            if volumes:
                avg_volume = statistics.mean(volumes)
                avg_temp_impact = statistics.mean(temporary_impacts)

                metrics["impact_per_volume"] = (
                    avg_temp_impact / avg_volume if avg_volume > 0 else 0.0
                )

        self.impact_models[symbol] = metrics

    async def _classify_trades(self) -> None:
        """Classify trades as aggressive or passive."""
        while self.is_running:
            try:
                await asyncio.sleep(self.update_frequency)

                # Classification happens in real-time in _classify_trade

            except Exception as e:
                self.logger.error(f"Error in trade classification: {str(e)}")

    async def _classify_trade(self, trade: Trade) -> None:
        """Classify a single trade."""
        symbol = trade.symbol

        # Find the most recent quote before the trade
        quotes = list(self.quotes[symbol])
        pre_quotes = [q for q in quotes if q.timestamp <= trade.timestamp]

        if not pre_quotes:
            return

        latest_quote = pre_quotes[-1]

        if not latest_quote.bid_price or not latest_quote.ask_price:
            return

        # Classify based on trade price relative to bid/ask
        mid_price = latest_quote.mid_price

        if trade.price >= latest_quote.ask_price:
            # Aggressive buy (trade at or above ask)
            self.trade_classification[symbol]["aggressive_buy"] += 1
            trade.aggressive_side = "BUY"
        elif trade.price <= latest_quote.bid_price:
            # Aggressive sell (trade at or below bid)
            self.trade_classification[symbol]["aggressive_sell"] += 1
            trade.aggressive_side = "SELL"
        elif mid_price and trade.price > mid_price:
            # Passive buy (trade above mid but below ask)
            self.trade_classification[symbol]["passive_buy"] += 1
        elif mid_price and trade.price < mid_price:
            # Passive sell (trade below mid but above bid)
            self.trade_classification[symbol]["passive_sell"] += 1

    async def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                cutoff_time = datetime.utcnow() - timedelta(hours=2)

                # Clean up old quotes
                for symbol in list(self.quotes.keys()):
                    quotes = self.quotes[symbol]
                    while quotes and quotes[0].timestamp < cutoff_time:
                        quotes.popleft()

                # Clean up old trades
                for symbol in list(self.trades.keys()):
                    trades = self.trades[symbol]
                    while trades and trades[0].timestamp < cutoff_time:
                        trades.popleft()

                self.logger.debug("Completed microstructure data cleanup")

            except Exception as e:
                self.logger.error(f"Error in data cleanup: {str(e)}")

    def get_spread_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get spread metrics for a symbol."""
        return self.spread_metrics.get(symbol)

    def get_liquidity_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get liquidity metrics for a symbol."""
        return self.liquidity_metrics.get(symbol)

    def get_market_impact_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market impact metrics for a symbol."""
        return self.impact_models.get(symbol)

    def get_trade_classification(self, symbol: str) -> Optional[Dict[str, int]]:
        """Get trade classification statistics for a symbol."""
        return self.trade_classification.get(symbol)

    def get_market_quality_score(self, symbol: str) -> Optional[float]:
        """Calculate overall market quality score (0-1)."""
        spread_metrics = self.spread_metrics.get(symbol)
        liquidity_metrics = self.liquidity_metrics.get(symbol)

        if not spread_metrics or not liquidity_metrics:
            return None

        # Factors: tight spreads, high liquidity, low imbalance
        spread_score = max(
            0, 1 - spread_metrics.get("mean_spread_bps", 100) / 100
        )  # Normalize to 100 bps
        liquidity_score = min(
            1, liquidity_metrics.get("total_volume_L5", 0) / 10000
        )  # Normalize to 10k shares
        imbalance_score = max(0, 1 - abs(liquidity_metrics.get("imbalance_L5", 0)))

        # Weighted average
        quality_score = spread_score * 0.4 + liquidity_score * 0.4 + imbalance_score * 0.2

        return quality_score

    def get_analyzer_performance(self) -> Dict[str, Any]:
        """Get analyzer performance metrics."""
        avg_latency = (
            sum(self.analysis_latency_us) / len(self.analysis_latency_us)
            if self.analysis_latency_us
            else 0
        )

        return {
            "analysis_count": self.analysis_count,
            "average_latency_us": avg_latency,
            "active_symbols": len(set(self.quotes.keys()) | set(self.trades.keys())),
            "total_quotes": sum(len(q) for q in self.quotes.values()),
            "total_trades": sum(len(t) for t in self.trades.values()),
        }
