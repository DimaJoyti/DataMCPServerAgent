"""
Institutional-Grade Order Management System (OMS)

High-performance order management system designed for hedge funds and
institutional trading operations.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ..core.base_models import BaseOrder, BaseTrade
from ..core.enums import OrderStatus, OrderType
from ..core.exceptions import (
    ExecutionError, OrderValidationError, RiskLimitExceededError
)
from .execution_algorithms import ExecutionAlgorithmManager
from .fill_manager import FillManager
from .order_validator import OrderValidator
from .smart_routing import SmartOrderRouter


class OrderManagementSystem:
    """
    Institutional-grade Order Management System.
    
    Features:
    - Real-time order lifecycle management
    - Smart order routing
    - Execution algorithms (TWAP, VWAP, Implementation Shortfall)
    - Pre-trade risk checks
    - Post-trade analysis
    - High-frequency trading support
    """
    
    def __init__(
        self,
        name: str = "InstitutionalOMS",
        enable_smart_routing: bool = True,
        enable_algorithms: bool = True,
        max_orders_per_second: int = 10000,
        latency_threshold_ms: float = 1.0
    ):
        self.name = name
        self.enable_smart_routing = enable_smart_routing
        self.enable_algorithms = enable_algorithms
        self.max_orders_per_second = max_orders_per_second
        self.latency_threshold_ms = latency_threshold_ms
        
        # Core components
        self.order_validator = OrderValidator()
        self.smart_router = SmartOrderRouter() if enable_smart_routing else None
        self.fill_manager = FillManager()
        self.algorithm_manager = ExecutionAlgorithmManager() if enable_algorithms else None
        
        # Order storage
        self.orders: Dict[str, BaseOrder] = {}
        self.orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        self.orders_by_status: Dict[OrderStatus, Set[str]] = defaultdict(set)
        self.orders_by_strategy: Dict[str, Set[str]] = defaultdict(set)
        
        # Trade storage
        self.trades: Dict[str, BaseTrade] = {}
        self.trades_by_order: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.total_orders_processed = 0
        self.total_trades_executed = 0
        self.average_latency_ms = 0.0
        self.orders_per_second = 0.0
        
        # Risk limits
        self.daily_order_limit = 100000
        self.daily_notional_limit = Decimal('1000000000')  # $1B
        self.position_limits: Dict[str, Decimal] = {}
        
        # Monitoring
        self.logger = logging.getLogger(f"OMS.{name}")
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Event handlers
        self.order_event_handlers: List[callable] = []
        self.trade_event_handlers: List[callable] = []
        self.risk_event_handlers: List[callable] = []
    
    async def start(self) -> None:
        """Start the OMS."""
        self.logger.info(f"Starting Order Management System: {self.name}")
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Start components
        if self.smart_router:
            await self.smart_router.start()
        
        if self.algorithm_manager:
            await self.algorithm_manager.start()
        
        await self.fill_manager.start()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._monitor_risk_limits())
        
        self.logger.info("OMS started successfully")
    
    async def stop(self) -> None:
        """Stop the OMS."""
        self.logger.info("Stopping Order Management System")
        
        self.is_running = False
        
        # Stop components
        if self.smart_router:
            await self.smart_router.stop()
        
        if self.algorithm_manager:
            await self.algorithm_manager.stop()
        
        await self.fill_manager.stop()
        
        self.logger.info("OMS stopped")
    
    async def submit_order(self, order: BaseOrder) -> str:
        """
        Submit an order to the OMS.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
            
        Raises:
            OrderValidationError: If order validation fails
            RiskLimitExceededError: If risk limits are exceeded
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate order
            validation_result = await self.order_validator.validate_order(order)
            if not validation_result.is_valid:
                raise OrderValidationError(
                    f"Order validation failed: {validation_result.errors}",
                    order.order_id,
                    validation_result.errors
                )
            
            # Check risk limits
            await self._check_risk_limits(order)
            
            # Store order
            self.orders[order.order_id] = order
            self.orders_by_symbol[order.symbol].add(order.order_id)
            self.orders_by_status[order.status].add(order.order_id)
            
            if order.strategy_id:
                self.orders_by_strategy[order.strategy_id].add(order.order_id)
            
            # Update order status
            order.status = OrderStatus.NEW
            order.submitted_at = datetime.utcnow()
            
            # Route order
            if self.smart_router and order.order_type in [OrderType.MARKET, OrderType.LIMIT]:
                await self.smart_router.route_order(order)
            elif self.algorithm_manager and order.order_type in [
                OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL
            ]:
                await self.algorithm_manager.execute_algorithmic_order(order)
            else:
                # Direct execution
                await self._execute_order_direct(order)
            
            # Update performance metrics
            self.total_orders_processed += 1
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_latency_metrics(latency)
            
            # Trigger events
            await self._trigger_order_event("ORDER_SUBMITTED", order)
            
            self.logger.info(f"Order submitted: {order.order_id} ({order.symbol})")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            await self._trigger_order_event("ORDER_REJECTED", order)
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order not found for cancellation: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.logger.warning(f"Cannot cancel order in status {order.status}: {order_id}")
            return False
        
        try:
            # Cancel with exchange/venue
            if self.smart_router:
                success = await self.smart_router.cancel_order(order)
            else:
                success = await self._cancel_order_direct(order)
            
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                
                # Update indices
                self._update_order_indices(order)
                
                # Trigger events
                await self._trigger_order_event("ORDER_CANCELLED", order)
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[Decimal] = None,
        new_price: Optional[Decimal] = None
    ) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of order to modify
            new_quantity: New quantity (optional)
            new_price: New price (optional)
            
        Returns:
            True if modification was successful
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order not found for modification: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            self.logger.warning(f"Cannot modify order in status {order.status}: {order_id}")
            return False
        
        try:
            # Store original values
            original_quantity = order.quantity
            original_price = order.price
            
            # Update order
            if new_quantity is not None:
                order.quantity = new_quantity
            if new_price is not None:
                order.price = new_price
            
            order.updated_at = datetime.utcnow()
            
            # Validate modified order
            validation_result = await self.order_validator.validate_order(order)
            if not validation_result.is_valid:
                # Revert changes
                order.quantity = original_quantity
                order.price = original_price
                raise OrderValidationError(
                    f"Order modification validation failed: {validation_result.errors}",
                    order_id,
                    validation_result.errors
                )
            
            # Modify with exchange/venue
            if self.smart_router:
                success = await self.smart_router.modify_order(order)
            else:
                success = await self._modify_order_direct(order)
            
            if success:
                # Trigger events
                await self._trigger_order_event("ORDER_MODIFIED", order)
                
                self.logger.info(f"Order modified: {order_id}")
                return True
            else:
                # Revert changes
                order.quantity = original_quantity
                order.price = original_price
                self.logger.error(f"Failed to modify order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Optional[BaseOrder]:
        """Get an order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[BaseOrder]:
        """Get all orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol, set())
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[BaseOrder]:
        """Get all orders with a specific status."""
        order_ids = self.orders_by_status.get(status, set())
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[BaseOrder]:
        """Get all orders for a strategy."""
        order_ids = self.orders_by_strategy.get(strategy_id, set())
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get OMS performance metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "uptime_seconds": uptime,
            "total_orders_processed": self.total_orders_processed,
            "total_trades_executed": self.total_trades_executed,
            "orders_per_second": self.orders_per_second,
            "average_latency_ms": self.average_latency_ms,
            "active_orders": len(self.get_orders_by_status(OrderStatus.NEW)) + 
                           len(self.get_orders_by_status(OrderStatus.PARTIALLY_FILLED)),
            "filled_orders": len(self.get_orders_by_status(OrderStatus.FILLED)),
            "cancelled_orders": len(self.get_orders_by_status(OrderStatus.CANCELLED)),
            "rejected_orders": len(self.get_orders_by_status(OrderStatus.REJECTED)),
        }
    
    # Private methods
    async def _check_risk_limits(self, order: BaseOrder) -> None:
        """Check risk limits for an order."""
        # Daily order count limit
        if self.total_orders_processed >= self.daily_order_limit:
            raise RiskLimitExceededError(
                "Daily order limit exceeded",
                "daily_order_count",
                self.total_orders_processed,
                self.daily_order_limit
            )
        
        # Position limits
        if order.symbol in self.position_limits:
            current_position = await self._get_current_position(order.symbol)
            new_position = current_position + (order.quantity if order.side.value == "BUY" else -order.quantity)
            
            if abs(new_position) > self.position_limits[order.symbol]:
                raise RiskLimitExceededError(
                    f"Position limit exceeded for {order.symbol}",
                    "position_limit",
                    float(abs(new_position)),
                    float(self.position_limits[order.symbol])
                )
    
    async def _get_current_position(self, symbol: str) -> Decimal:
        """Get current position for a symbol."""
        # This would integrate with position management system
        return Decimal('0')
    
    def _update_order_indices(self, order: BaseOrder) -> None:
        """Update order indices after status change."""
        # Remove from old status
        for status, order_ids in self.orders_by_status.items():
            order_ids.discard(order.order_id)
        
        # Add to new status
        self.orders_by_status[order.status].add(order.order_id)
    
    def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics."""
        if self.total_orders_processed == 1:
            self.average_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_latency_ms = alpha * latency_ms + (1 - alpha) * self.average_latency_ms
    
    async def _execute_order_direct(self, order: BaseOrder) -> None:
        """Execute order directly (mock implementation)."""
        # This would integrate with actual exchange APIs
        await asyncio.sleep(0.001)  # Simulate execution latency
        
        # Mock fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price
        order.filled_at = datetime.utcnow()
        
        self._update_order_indices(order)
    
    async def _cancel_order_direct(self, order: BaseOrder) -> bool:
        """Cancel order directly (mock implementation)."""
        await asyncio.sleep(0.001)  # Simulate cancellation latency
        return True
    
    async def _modify_order_direct(self, order: BaseOrder) -> bool:
        """Modify order directly (mock implementation)."""
        await asyncio.sleep(0.001)  # Simulate modification latency
        return True
    
    async def _monitor_performance(self) -> None:
        """Monitor OMS performance."""
        while self.is_running:
            await asyncio.sleep(1)
            
            # Calculate orders per second
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                if uptime > 0:
                    self.orders_per_second = self.total_orders_processed / uptime
    
    async def _monitor_risk_limits(self) -> None:
        """Monitor risk limits."""
        while self.is_running:
            await asyncio.sleep(5)
            
            # Check latency threshold
            if self.average_latency_ms > self.latency_threshold_ms:
                await self._trigger_risk_event("HIGH_LATENCY", {
                    "current_latency": self.average_latency_ms,
                    "threshold": self.latency_threshold_ms
                })
    
    async def _trigger_order_event(self, event_type: str, order: BaseOrder) -> None:
        """Trigger order event handlers."""
        for handler in self.order_event_handlers:
            try:
                await handler(event_type, order)
            except Exception as e:
                self.logger.error(f"Error in order event handler: {str(e)}")
    
    async def _trigger_trade_event(self, event_type: str, trade: BaseTrade) -> None:
        """Trigger trade event handlers."""
        for handler in self.trade_event_handlers:
            try:
                await handler(event_type, trade)
            except Exception as e:
                self.logger.error(f"Error in trade event handler: {str(e)}")
    
    async def _trigger_risk_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger risk event handlers."""
        for handler in self.risk_event_handlers:
            try:
                await handler(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in risk event handler: {str(e)}")
    
    def add_order_event_handler(self, handler: callable) -> None:
        """Add order event handler."""
        self.order_event_handlers.append(handler)
    
    def add_trade_event_handler(self, handler: callable) -> None:
        """Add trade event handler."""
        self.trade_event_handlers.append(handler)
    
    def add_risk_event_handler(self, handler: callable) -> None:
        """Add risk event handler."""
        self.risk_event_handlers.append(handler)
