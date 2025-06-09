"""
Smart Order Routing (SOR) system for institutional trading.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_models import BaseOrder, MarketData
from ..core.enums import Exchange, OrderSide, OrderStatus, OrderType


@dataclass
class VenueQuote:
    """Quote from a trading venue."""
    
    exchange: Exchange
    symbol: str
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    timestamp: datetime = None
    latency_ms: float = 0.0
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None


@dataclass
class RoutingDecision:
    """Smart routing decision."""
    
    primary_venue: Exchange
    backup_venues: List[Exchange]
    allocation: Dict[Exchange, Decimal]  # Quantity allocation per venue
    expected_price: Decimal
    expected_cost: Decimal
    routing_strategy: str
    confidence: float
    reasoning: str


class SmartOrderRouter:
    """
    Smart Order Routing system for optimal execution.
    
    Features:
    - Multi-venue price discovery
    - Latency-aware routing
    - Liquidity aggregation
    - Cost optimization
    - Real-time venue monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SmartOrderRouter")
        self.is_active = False
        
        # Venue configurations
        self.venues: Dict[Exchange, Dict] = {
            Exchange.NYSE: {
                "latency_ms": 0.5,
                "fee_rate": 0.0005,
                "min_order_size": Decimal('1'),
                "max_order_size": Decimal('1000000'),
                "reliability": 0.99,
                "market_share": 0.25
            },
            Exchange.NASDAQ: {
                "latency_ms": 0.3,
                "fee_rate": 0.0003,
                "min_order_size": Decimal('1'),
                "max_order_size": Decimal('1000000'),
                "reliability": 0.995,
                "market_share": 0.30
            },
            Exchange.BINANCE: {
                "latency_ms": 1.0,
                "fee_rate": 0.001,
                "min_order_size": Decimal('0.001'),
                "max_order_size": Decimal('100000'),
                "reliability": 0.98,
                "market_share": 0.40
            }
        }
        
        # Routing strategies
        self.routing_strategies = {
            "BEST_PRICE": self._route_best_price,
            "LOWEST_COST": self._route_lowest_cost,
            "FASTEST_EXECUTION": self._route_fastest,
            "LIQUIDITY_SEEKING": self._route_liquidity_seeking,
            "SMART_SPLIT": self._route_smart_split
        }
        
        # Performance tracking
        self.routing_stats = {
            "total_orders": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_latency_ms": 0.0,
            "cost_savings_bps": 0.0
        }
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[Exchange, VenueQuote]] = {}
        self.cache_ttl_seconds = 1.0
    
    async def start(self) -> None:
        """Start the smart order router."""
        self.logger.info("Starting Smart Order Router")
        self.is_active = True
        
        # Start market data collection
        asyncio.create_task(self._collect_market_data())
        
        # Start venue monitoring
        asyncio.create_task(self._monitor_venues())
        
        self.logger.info("Smart Order Router started")
    
    async def stop(self) -> None:
        """Stop the smart order router."""
        self.logger.info("Stopping Smart Order Router")
        self.is_active = False
        self.logger.info("Smart Order Router stopped")
    
    async def route_order(self, order: BaseOrder) -> RoutingDecision:
        """
        Route an order to optimal venue(s).
        
        Args:
            order: Order to route
            
        Returns:
            RoutingDecision with routing details
        """
        start_time = datetime.utcnow()
        
        try:
            # Get current market data
            venue_quotes = await self._get_venue_quotes(order.symbol)
            
            if not venue_quotes:
                raise Exception(f"No market data available for {order.symbol}")
            
            # Determine routing strategy
            strategy = self._select_routing_strategy(order, venue_quotes)
            
            # Execute routing strategy
            routing_decision = await self.routing_strategies[strategy](order, venue_quotes)
            
            # Update statistics
            self.routing_stats["total_orders"] += 1
            self.routing_stats["successful_routes"] += 1
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_latency_stats(latency)
            
            self.logger.info(
                f"Order routed: {order.order_id} -> {routing_decision.primary_venue} "
                f"(strategy: {strategy})"
            )
            
            return routing_decision
            
        except Exception as e:
            self.routing_stats["failed_routes"] += 1
            self.logger.error(f"Failed to route order {order.order_id}: {str(e)}")
            raise
    
    async def cancel_order(self, order: BaseOrder) -> bool:
        """Cancel order at venue."""
        # This would integrate with actual venue APIs
        await asyncio.sleep(0.001)  # Simulate cancellation latency
        return True
    
    async def modify_order(self, order: BaseOrder) -> bool:
        """Modify order at venue."""
        # This would integrate with actual venue APIs
        await asyncio.sleep(0.001)  # Simulate modification latency
        return True
    
    def _select_routing_strategy(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> str:
        """Select optimal routing strategy based on order characteristics."""
        
        # Large orders -> Smart split for liquidity
        if order.quantity > Decimal('10000'):
            return "SMART_SPLIT"
        
        # Market orders -> Fastest execution
        if order.order_type == OrderType.MARKET:
            return "FASTEST_EXECUTION"
        
        # Small orders -> Best price
        if order.quantity < Decimal('100'):
            return "BEST_PRICE"
        
        # Default to lowest cost
        return "LOWEST_COST"
    
    async def _route_best_price(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> RoutingDecision:
        """Route to venue with best price."""
        
        best_venue = None
        best_price = None
        
        for exchange, quote in venue_quotes.items():
            if order.side == OrderSide.BUY:
                price = quote.ask
            else:
                price = quote.bid
            
            if price is not None and (best_price is None or 
                (order.side == OrderSide.BUY and price < best_price) or
                (order.side == OrderSide.SELL and price > best_price)):
                best_price = price
                best_venue = exchange
        
        if best_venue is None:
            raise Exception("No suitable venue found")
        
        return RoutingDecision(
            primary_venue=best_venue,
            backup_venues=[v for v in venue_quotes.keys() if v != best_venue],
            allocation={best_venue: order.quantity},
            expected_price=best_price,
            expected_cost=self._calculate_execution_cost(order, best_venue, best_price),
            routing_strategy="BEST_PRICE",
            confidence=0.9,
            reasoning=f"Best price {best_price} at {best_venue}"
        )
    
    async def _route_lowest_cost(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> RoutingDecision:
        """Route to venue with lowest total cost (price + fees)."""
        
        best_venue = None
        lowest_cost = None
        best_price = None
        
        for exchange, quote in venue_quotes.items():
            if order.side == OrderSide.BUY:
                price = quote.ask
            else:
                price = quote.bid
            
            if price is not None:
                total_cost = self._calculate_execution_cost(order, exchange, price)
                
                if lowest_cost is None or total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_venue = exchange
                    best_price = price
        
        if best_venue is None:
            raise Exception("No suitable venue found")
        
        return RoutingDecision(
            primary_venue=best_venue,
            backup_venues=[v for v in venue_quotes.keys() if v != best_venue],
            allocation={best_venue: order.quantity},
            expected_price=best_price,
            expected_cost=lowest_cost,
            routing_strategy="LOWEST_COST",
            confidence=0.85,
            reasoning=f"Lowest total cost {lowest_cost} at {best_venue}"
        )
    
    async def _route_fastest(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> RoutingDecision:
        """Route to venue with fastest execution."""
        
        fastest_venue = None
        lowest_latency = None
        
        for exchange, quote in venue_quotes.items():
            venue_config = self.venues.get(exchange, {})
            latency = venue_config.get("latency_ms", 999.0)
            
            if lowest_latency is None or latency < lowest_latency:
                lowest_latency = latency
                fastest_venue = exchange
        
        if fastest_venue is None:
            raise Exception("No suitable venue found")
        
        quote = venue_quotes[fastest_venue]
        price = quote.ask if order.side == OrderSide.BUY else quote.bid
        
        return RoutingDecision(
            primary_venue=fastest_venue,
            backup_venues=[v for v in venue_quotes.keys() if v != fastest_venue],
            allocation={fastest_venue: order.quantity},
            expected_price=price,
            expected_cost=self._calculate_execution_cost(order, fastest_venue, price),
            routing_strategy="FASTEST_EXECUTION",
            confidence=0.8,
            reasoning=f"Fastest execution {lowest_latency}ms at {fastest_venue}"
        )
    
    async def _route_liquidity_seeking(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> RoutingDecision:
        """Route to venue with most liquidity."""
        
        best_venue = None
        best_liquidity = Decimal('0')
        
        for exchange, quote in venue_quotes.items():
            if order.side == OrderSide.BUY:
                liquidity = quote.ask_size or Decimal('0')
            else:
                liquidity = quote.bid_size or Decimal('0')
            
            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_venue = exchange
        
        if best_venue is None:
            raise Exception("No suitable venue found")
        
        quote = venue_quotes[best_venue]
        price = quote.ask if order.side == OrderSide.BUY else quote.bid
        
        return RoutingDecision(
            primary_venue=best_venue,
            backup_venues=[v for v in venue_quotes.keys() if v != best_venue],
            allocation={best_venue: order.quantity},
            expected_price=price,
            expected_cost=self._calculate_execution_cost(order, best_venue, price),
            routing_strategy="LIQUIDITY_SEEKING",
            confidence=0.75,
            reasoning=f"Best liquidity {best_liquidity} at {best_venue}"
        )
    
    async def _route_smart_split(self, order: BaseOrder, venue_quotes: Dict[Exchange, VenueQuote]) -> RoutingDecision:
        """Split order across multiple venues for optimal execution."""
        
        # Calculate optimal allocation
        allocation = {}
        total_allocated = Decimal('0')
        
        # Sort venues by attractiveness (price + liquidity + cost)
        venue_scores = []
        for exchange, quote in venue_quotes.items():
            score = self._calculate_venue_score(order, exchange, quote)
            venue_scores.append((exchange, score, quote))
        
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate quantity based on scores and liquidity
        remaining_quantity = order.quantity
        
        for exchange, score, quote in venue_scores:
            if remaining_quantity <= 0:
                break
            
            # Calculate allocation based on liquidity and score
            if order.side == OrderSide.BUY:
                available_liquidity = quote.ask_size or Decimal('1000')
            else:
                available_liquidity = quote.bid_size or Decimal('1000')
            
            # Allocate up to 40% of available liquidity
            max_allocation = min(remaining_quantity, available_liquidity * Decimal('0.4'))
            
            if max_allocation > 0:
                allocation[exchange] = max_allocation
                total_allocated += max_allocation
                remaining_quantity -= max_allocation
        
        # If not fully allocated, put remainder on best venue
        if remaining_quantity > 0 and venue_scores:
            best_venue = venue_scores[0][0]
            allocation[best_venue] = allocation.get(best_venue, Decimal('0')) + remaining_quantity
        
        # Calculate weighted average price
        total_cost = Decimal('0')
        total_value = Decimal('0')
        
        for exchange, quantity in allocation.items():
            quote = venue_quotes[exchange]
            price = quote.ask if order.side == OrderSide.BUY else quote.bid
            
            if price is not None:
                cost = self._calculate_execution_cost_for_quantity(order, exchange, price, quantity)
                total_cost += cost
                total_value += quantity * price
        
        weighted_avg_price = total_value / sum(allocation.values()) if allocation else Decimal('0')
        
        primary_venue = max(allocation.items(), key=lambda x: x[1])[0] if allocation else None
        backup_venues = [v for v in allocation.keys() if v != primary_venue]
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            allocation=allocation,
            expected_price=weighted_avg_price,
            expected_cost=total_cost,
            routing_strategy="SMART_SPLIT",
            confidence=0.95,
            reasoning=f"Split across {len(allocation)} venues for optimal execution"
        )
    
    def _calculate_venue_score(self, order: BaseOrder, exchange: Exchange, quote: VenueQuote) -> float:
        """Calculate venue attractiveness score."""
        
        venue_config = self.venues.get(exchange, {})
        
        # Price score (0-1, higher is better)
        price = quote.ask if order.side == OrderSide.BUY else quote.bid
        if price is None:
            return 0.0
        
        # Normalize price (simplified)
        price_score = 1.0 / float(price) if price > 0 else 0.0
        
        # Latency score (0-1, lower latency is better)
        latency = venue_config.get("latency_ms", 999.0)
        latency_score = max(0.0, 1.0 - latency / 100.0)
        
        # Reliability score
        reliability_score = venue_config.get("reliability", 0.5)
        
        # Liquidity score
        liquidity = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
        liquidity_score = min(1.0, float(liquidity or 0) / 10000.0)
        
        # Fee score (lower fees are better)
        fee_rate = venue_config.get("fee_rate", 0.001)
        fee_score = max(0.0, 1.0 - fee_rate * 1000)
        
        # Weighted combination
        total_score = (
            price_score * 0.3 +
            latency_score * 0.2 +
            reliability_score * 0.2 +
            liquidity_score * 0.2 +
            fee_score * 0.1
        )
        
        return total_score
    
    def _calculate_execution_cost(self, order: BaseOrder, exchange: Exchange, price: Decimal) -> Decimal:
        """Calculate total execution cost including fees."""
        return self._calculate_execution_cost_for_quantity(order, exchange, price, order.quantity)
    
    def _calculate_execution_cost_for_quantity(
        self,
        order: BaseOrder,
        exchange: Exchange,
        price: Decimal,
        quantity: Decimal
    ) -> Decimal:
        """Calculate execution cost for specific quantity."""
        
        venue_config = self.venues.get(exchange, {})
        fee_rate = Decimal(str(venue_config.get("fee_rate", 0.001)))
        
        notional_value = quantity * price
        fees = notional_value * fee_rate
        
        return notional_value + fees
    
    async def _get_venue_quotes(self, symbol: str) -> Dict[Exchange, VenueQuote]:
        """Get current quotes from all venues."""
        
        # Check cache first
        if symbol in self.market_data_cache:
            cache_time = min(quote.timestamp for quote in self.market_data_cache[symbol].values())
            if (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl_seconds:
                return self.market_data_cache[symbol]
        
        # Fetch fresh quotes
        quotes = {}
        
        for exchange in self.venues.keys():
            try:
                quote = await self._fetch_venue_quote(exchange, symbol)
                if quote:
                    quotes[exchange] = quote
            except Exception as e:
                self.logger.warning(f"Failed to get quote from {exchange}: {str(e)}")
        
        # Update cache
        self.market_data_cache[symbol] = quotes
        
        return quotes
    
    async def _fetch_venue_quote(self, exchange: Exchange, symbol: str) -> Optional[VenueQuote]:
        """Fetch quote from specific venue."""
        
        # Mock implementation - would integrate with actual venue APIs
        await asyncio.sleep(0.001)  # Simulate network latency
        
        # Generate mock quote
        base_price = Decimal('100.00')
        spread = Decimal('0.05')
        
        return VenueQuote(
            exchange=exchange,
            symbol=symbol,
            bid=base_price - spread/2,
            ask=base_price + spread/2,
            bid_size=Decimal('1000'),
            ask_size=Decimal('1000'),
            timestamp=datetime.utcnow(),
            latency_ms=self.venues[exchange]["latency_ms"]
        )
    
    async def _collect_market_data(self) -> None:
        """Continuously collect market data from venues."""
        while self.is_active:
            try:
                # This would continuously update market data cache
                await asyncio.sleep(0.1)  # 100ms update frequency
            except Exception as e:
                self.logger.error(f"Error collecting market data: {str(e)}")
    
    async def _monitor_venues(self) -> None:
        """Monitor venue health and performance."""
        while self.is_active:
            try:
                # Monitor venue connectivity, latency, etc.
                await asyncio.sleep(5)  # 5 second monitoring interval
            except Exception as e:
                self.logger.error(f"Error monitoring venues: {str(e)}")
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update latency statistics."""
        if self.routing_stats["total_orders"] == 1:
            self.routing_stats["average_latency_ms"] = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            current_avg = self.routing_stats["average_latency_ms"]
            self.routing_stats["average_latency_ms"] = alpha * latency_ms + (1 - alpha) * current_avg
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        return self.routing_stats.copy()
    
    def get_venue_status(self) -> Dict[Exchange, Dict[str, Any]]:
        """Get current venue status."""
        status = {}
        
        for exchange, config in self.venues.items():
            status[exchange] = {
                "latency_ms": config["latency_ms"],
                "reliability": config["reliability"],
                "fee_rate": config["fee_rate"],
                "status": "ACTIVE" if self.is_active else "INACTIVE"
            }
        
        return status
