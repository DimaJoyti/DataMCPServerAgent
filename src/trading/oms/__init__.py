"""
Order Management System (OMS)

Institutional-grade order management system for high-frequency trading
and multi-strategy hedge fund operations.

Features:
- Smart order routing
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Real-time order lifecycle management
- Pre-trade risk checks
- Post-trade analysis
- Order book management
- Fill management
"""

from .order_management_system import OrderManagementSystem
from .order_types import *
from .execution_algorithms import *
from .smart_routing import SmartOrderRouter
from .fill_manager import FillManager
from .order_validator import OrderValidator

__all__ = [
    'OrderManagementSystem',
    'SmartOrderRouter',
    'FillManager',
    'OrderValidator',
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    'ImplementationShortfallAlgorithm',
    'IcebergOrder',
    'AlgorithmicOrder',
]
