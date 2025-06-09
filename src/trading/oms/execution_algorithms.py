"""
Execution algorithms for institutional trading.

Implements sophisticated execution algorithms including TWAP, VWAP,
Implementation Shortfall, and other institutional-grade strategies.
"""

import asyncio
import logging
import math
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.base_models import BaseOrder, MarketData
from ..core.enums import ExecutionAlgorithm, OrderSide, OrderStatus, OrderType
from .order_types import (
    AlgorithmicOrder, ArrivalPriceOrder, ImplementationShortfallOrder,
    PercentOfVolumeOrder, TWAPOrder, VWAPOrder
)


class BaseExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Algorithm.{name}")
        self.is_active = False
        self.orders: Dict[str, AlgorithmicOrder] = {}
        self.child_orders: Dict[str, List[BaseOrder]] = {}
    
    @abstractmethod
    async def execute(self, order: AlgorithmicOrder) -> None:
        """Execute the algorithmic order."""
        pass
    
    @abstractmethod
    async def cancel(self, order_id: str) -> bool:
        """Cancel the algorithmic order."""
        pass
    
    async def start(self) -> None:
        """Start the algorithm."""
        self.is_active = True
        self.logger.info(f"Started execution algorithm: {self.name}")
    
    async def stop(self) -> None:
        """Stop the algorithm."""
        self.is_active = False
        self.logger.info(f"Stopped execution algorithm: {self.name}")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol."""
        # This would integrate with market data feed
        # Mock implementation for now
        return MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bid=Decimal('100.00'),
            ask=Decimal('100.05'),
            last=Decimal('100.02'),
            volume=Decimal('1000000')
        )


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """Time Weighted Average Price algorithm."""
    
    def __init__(self):
        super().__init__("TWAP")
    
    async def execute(self, order: TWAPOrder) -> None:
        """Execute TWAP order by breaking into time-based slices."""
        self.logger.info(f"Starting TWAP execution for order {order.order_id}")
        
        self.orders[order.order_id] = order
        self.child_orders[order.order_id] = []
        
        # Calculate slice parameters
        total_duration = order.end_time - order.start_time
        slice_quantity = order.quantity / order.total_slices
        
        # Schedule slices
        for slice_num in range(order.total_slices):
            slice_time = order.start_time + (slice_num * order.slice_interval)
            
            # Add randomization if enabled
            if order.randomize_timing:
                randomization = order.slice_interval.total_seconds() * order.randomization_factor
                random_offset = random.uniform(-randomization, randomization)
                slice_time += timedelta(seconds=random_offset)
            
            # Schedule slice execution
            asyncio.create_task(self._execute_slice(order, slice_num, slice_time, slice_quantity))
        
        # Monitor execution
        asyncio.create_task(self._monitor_execution(order))
    
    async def _execute_slice(
        self,
        parent_order: TWAPOrder,
        slice_num: int,
        execution_time: datetime,
        quantity: Decimal
    ) -> None:
        """Execute a single TWAP slice."""
        
        # Wait until execution time
        now = datetime.utcnow()
        if execution_time > now:
            wait_seconds = (execution_time - now).total_seconds()
            await asyncio.sleep(wait_seconds)
        
        # Check if parent order is still active
        if parent_order.status in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
            return
        
        try:
            # Create child order
            child_order = BaseOrder(
                symbol=parent_order.symbol,
                side=parent_order.side,
                order_type=OrderType.MARKET,  # Use market orders for TWAP slices
                quantity=quantity,
                strategy_id=parent_order.strategy_id,
                portfolio_id=parent_order.portfolio_id,
                account_id=parent_order.account_id
            )
            
            # Execute child order (mock implementation)
            await self._execute_child_order(child_order)
            
            # Update parent order
            parent_order.filled_quantity += child_order.filled_quantity
            parent_order.slices_completed += 1
            parent_order.execution_progress = parent_order.slices_completed / parent_order.total_slices
            
            # Store child order
            self.child_orders[parent_order.order_id].append(child_order)
            parent_order.child_orders.append(child_order.order_id)
            
            self.logger.info(
                f"TWAP slice {slice_num + 1}/{parent_order.total_slices} executed for {parent_order.order_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP slice: {str(e)}")
    
    async def _execute_child_order(self, order: BaseOrder) -> None:
        """Execute a child order (mock implementation)."""
        await asyncio.sleep(0.01)  # Simulate execution latency
        
        # Mock fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = Decimal('100.00')  # Mock price
        order.filled_at = datetime.utcnow()
    
    async def _monitor_execution(self, order: TWAPOrder) -> None:
        """Monitor TWAP execution progress."""
        while order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            await asyncio.sleep(1)
            
            # Check if fully filled
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self.logger.info(f"TWAP order {order.order_id} fully executed")
                break
            
            # Check if execution window expired
            if datetime.utcnow() > order.end_time:
                if order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                else:
                    order.status = OrderStatus.EXPIRED
                self.logger.info(f"TWAP order {order.order_id} execution window expired")
                break
    
    async def cancel(self, order_id: str) -> bool:
        """Cancel TWAP order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Cancel any pending child orders
        for child_order in self.child_orders.get(order_id, []):
            if child_order.status == OrderStatus.NEW:
                child_order.status = OrderStatus.CANCELLED
        
        self.logger.info(f"TWAP order {order_id} cancelled")
        return True


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """Volume Weighted Average Price algorithm."""
    
    def __init__(self):
        super().__init__("VWAP")
        self.volume_profiles: Dict[str, List[float]] = {}
    
    async def execute(self, order: VWAPOrder) -> None:
        """Execute VWAP order based on historical volume patterns."""
        self.logger.info(f"Starting VWAP execution for order {order.order_id}")
        
        self.orders[order.order_id] = order
        self.child_orders[order.order_id] = []
        
        # Get volume profile
        volume_profile = await self._get_volume_profile(order.symbol)
        
        # Calculate execution schedule
        execution_schedule = self._calculate_vwap_schedule(order, volume_profile)
        
        # Execute according to schedule
        for schedule_item in execution_schedule:
            asyncio.create_task(self._execute_vwap_slice(order, schedule_item))
        
        # Monitor execution
        asyncio.create_task(self._monitor_execution(order))
    
    async def _get_volume_profile(self, symbol: str) -> List[float]:
        """Get historical volume profile for a symbol."""
        # Mock volume profile (normalized hourly volumes)
        if symbol not in self.volume_profiles:
            # Generate realistic intraday volume profile
            self.volume_profiles[symbol] = [
                0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,  # Morning
                0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08,  # Midday
                0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01   # Evening
            ]
        
        return self.volume_profiles[symbol]
    
    def _calculate_vwap_schedule(self, order: VWAPOrder, volume_profile: List[float]) -> List[Dict]:
        """Calculate VWAP execution schedule."""
        schedule = []
        total_duration = order.end_time - order.start_time
        interval_duration = total_duration / len(volume_profile)
        
        for i, volume_weight in enumerate(volume_profile):
            execution_time = order.start_time + (i * interval_duration)
            slice_quantity = order.quantity * Decimal(str(volume_weight))
            
            schedule.append({
                'execution_time': execution_time,
                'quantity': slice_quantity,
                'volume_weight': volume_weight,
                'slice_number': i
            })
        
        return schedule
    
    async def _execute_vwap_slice(self, parent_order: VWAPOrder, schedule_item: Dict) -> None:
        """Execute a single VWAP slice."""
        execution_time = schedule_item['execution_time']
        quantity = schedule_item['quantity']
        
        # Wait until execution time
        now = datetime.utcnow()
        if execution_time > now:
            wait_seconds = (execution_time - now).total_seconds()
            await asyncio.sleep(wait_seconds)
        
        # Check if parent order is still active
        if parent_order.status in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
            return
        
        try:
            # Get current market data
            market_data = await self.get_market_data(parent_order.symbol)
            
            # Calculate participation rate
            if market_data and market_data.volume:
                current_volume = market_data.volume
                participation_rate = min(
                    float(quantity / current_volume),
                    parent_order.max_participation_rate
                )
                
                # Adjust quantity based on participation rate
                adjusted_quantity = min(quantity, current_volume * Decimal(str(participation_rate)))
            else:
                adjusted_quantity = quantity
            
            # Create child order
            child_order = BaseOrder(
                symbol=parent_order.symbol,
                side=parent_order.side,
                order_type=OrderType.LIMIT,
                quantity=adjusted_quantity,
                price=market_data.mid_price if market_data else None,
                strategy_id=parent_order.strategy_id,
                portfolio_id=parent_order.portfolio_id,
                account_id=parent_order.account_id
            )
            
            # Execute child order
            await self._execute_child_order(child_order)
            
            # Update parent order
            parent_order.filled_quantity += child_order.filled_quantity
            
            # Store child order
            self.child_orders[parent_order.order_id].append(child_order)
            parent_order.child_orders.append(child_order.order_id)
            
            self.logger.info(f"VWAP slice executed for {parent_order.order_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing VWAP slice: {str(e)}")
    
    async def _execute_child_order(self, order: BaseOrder) -> None:
        """Execute a child order (mock implementation)."""
        await asyncio.sleep(0.01)  # Simulate execution latency
        
        # Mock fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price or Decimal('100.00')
        order.filled_at = datetime.utcnow()
    
    async def _monitor_execution(self, order: VWAPOrder) -> None:
        """Monitor VWAP execution progress."""
        while order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            await asyncio.sleep(1)
            
            # Check if fully filled
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self.logger.info(f"VWAP order {order.order_id} fully executed")
                break
            
            # Check if execution window expired
            if datetime.utcnow() > order.end_time:
                if order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                else:
                    order.status = OrderStatus.EXPIRED
                self.logger.info(f"VWAP order {order.order_id} execution window expired")
                break
    
    async def cancel(self, order_id: str) -> bool:
        """Cancel VWAP order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Cancel any pending child orders
        for child_order in self.child_orders.get(order_id, []):
            if child_order.status == OrderStatus.NEW:
                child_order.status = OrderStatus.CANCELLED
        
        self.logger.info(f"VWAP order {order_id} cancelled")
        return True


class ImplementationShortfallAlgorithm(BaseExecutionAlgorithm):
    """Implementation Shortfall algorithm."""
    
    def __init__(self):
        super().__init__("ImplementationShortfall")
    
    async def execute(self, order: ImplementationShortfallOrder) -> None:
        """Execute Implementation Shortfall order."""
        self.logger.info(f"Starting Implementation Shortfall execution for order {order.order_id}")
        
        self.orders[order.order_id] = order
        self.child_orders[order.order_id] = []
        
        # Calculate optimal execution strategy
        execution_strategy = await self._calculate_is_strategy(order)
        
        # Execute according to strategy
        asyncio.create_task(self._execute_is_strategy(order, execution_strategy))
        
        # Monitor execution
        asyncio.create_task(self._monitor_execution(order))
    
    async def _calculate_is_strategy(self, order: ImplementationShortfallOrder) -> Dict:
        """Calculate optimal Implementation Shortfall strategy."""
        # Simplified IS calculation
        # In practice, this would use sophisticated market impact models
        
        total_duration = order.end_time - order.start_time
        risk_aversion = order.risk_aversion
        
        # Calculate optimal trading rate
        # Higher risk aversion = slower trading
        # Lower risk aversion = faster trading
        
        if risk_aversion < 0.3:
            # Aggressive execution
            execution_rate = 0.8  # Execute 80% immediately
        elif risk_aversion < 0.7:
            # Balanced execution
            execution_rate = 0.5  # Execute 50% immediately
        else:
            # Conservative execution
            execution_rate = 0.2  # Execute 20% immediately
        
        immediate_quantity = order.quantity * Decimal(str(execution_rate))
        remaining_quantity = order.quantity - immediate_quantity
        
        return {
            'immediate_quantity': immediate_quantity,
            'remaining_quantity': remaining_quantity,
            'execution_rate': execution_rate,
            'total_duration': total_duration
        }
    
    async def _execute_is_strategy(self, order: ImplementationShortfallOrder, strategy: Dict) -> None:
        """Execute Implementation Shortfall strategy."""
        
        # Execute immediate portion
        if strategy['immediate_quantity'] > 0:
            await self._execute_immediate_portion(order, strategy['immediate_quantity'])
        
        # Execute remaining portion gradually
        if strategy['remaining_quantity'] > 0:
            await self._execute_gradual_portion(order, strategy['remaining_quantity'], strategy['total_duration'])
    
    async def _execute_immediate_portion(self, order: ImplementationShortfallOrder, quantity: Decimal) -> None:
        """Execute immediate portion with market orders."""
        
        child_order = BaseOrder(
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_id=order.strategy_id,
            portfolio_id=order.portfolio_id,
            account_id=order.account_id
        )
        
        await self._execute_child_order(child_order)
        
        # Update parent order
        order.filled_quantity += child_order.filled_quantity
        
        # Store child order
        self.child_orders[order.order_id].append(child_order)
        order.child_orders.append(child_order.order_id)
        
        self.logger.info(f"IS immediate portion executed for {order.order_id}")
    
    async def _execute_gradual_portion(
        self,
        order: ImplementationShortfallOrder,
        quantity: Decimal,
        duration: timedelta
    ) -> None:
        """Execute remaining portion gradually."""
        
        # Break into smaller slices
        num_slices = 10
        slice_quantity = quantity / num_slices
        slice_interval = duration / num_slices
        
        for i in range(num_slices):
            if order.status in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                break
            
            # Wait for slice interval
            await asyncio.sleep(slice_interval.total_seconds())
            
            # Execute slice
            child_order = BaseOrder(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=slice_quantity,
                strategy_id=order.strategy_id,
                portfolio_id=order.portfolio_id,
                account_id=order.account_id
            )
            
            # Set limit price based on current market
            market_data = await self.get_market_data(order.symbol)
            if market_data:
                child_order.price = market_data.mid_price
            
            await self._execute_child_order(child_order)
            
            # Update parent order
            order.filled_quantity += child_order.filled_quantity
            
            # Store child order
            self.child_orders[order.order_id].append(child_order)
            order.child_orders.append(child_order.order_id)
    
    async def _execute_child_order(self, order: BaseOrder) -> None:
        """Execute a child order (mock implementation)."""
        await asyncio.sleep(0.01)  # Simulate execution latency
        
        # Mock fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price or Decimal('100.00')
        order.filled_at = datetime.utcnow()
    
    async def _monitor_execution(self, order: ImplementationShortfallOrder) -> None:
        """Monitor Implementation Shortfall execution."""
        while order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            await asyncio.sleep(1)
            
            # Check if fully filled
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self.logger.info(f"IS order {order.order_id} fully executed")
                break
            
            # Check if execution window expired
            if datetime.utcnow() > order.end_time:
                if order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                else:
                    order.status = OrderStatus.EXPIRED
                self.logger.info(f"IS order {order.order_id} execution window expired")
                break
    
    async def cancel(self, order_id: str) -> bool:
        """Cancel Implementation Shortfall order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Cancel any pending child orders
        for child_order in self.child_orders.get(order_id, []):
            if child_order.status == OrderStatus.NEW:
                child_order.status = OrderStatus.CANCELLED
        
        self.logger.info(f"IS order {order_id} cancelled")
        return True


class ExecutionAlgorithmManager:
    """Manager for execution algorithms."""
    
    def __init__(self):
        self.algorithms = {
            ExecutionAlgorithm.TWAP: TWAPAlgorithm(),
            ExecutionAlgorithm.VWAP: VWAPAlgorithm(),
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: ImplementationShortfallAlgorithm(),
        }
        self.logger = logging.getLogger("ExecutionAlgorithmManager")
    
    async def start(self) -> None:
        """Start all algorithms."""
        for algorithm in self.algorithms.values():
            await algorithm.start()
        self.logger.info("Execution Algorithm Manager started")
    
    async def stop(self) -> None:
        """Stop all algorithms."""
        for algorithm in self.algorithms.values():
            await algorithm.stop()
        self.logger.info("Execution Algorithm Manager stopped")
    
    async def execute_algorithmic_order(self, order: AlgorithmicOrder) -> None:
        """Execute an algorithmic order."""
        algorithm_type = order.algorithm
        
        if algorithm_type not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}")
        
        algorithm = self.algorithms[algorithm_type]
        await algorithm.execute(order)
    
    async def cancel_algorithmic_order(self, order_id: str, algorithm_type: ExecutionAlgorithm) -> bool:
        """Cancel an algorithmic order."""
        if algorithm_type not in self.algorithms:
            return False
        
        algorithm = self.algorithms[algorithm_type]
        return await algorithm.cancel(order_id)
