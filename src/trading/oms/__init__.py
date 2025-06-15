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

from .execution_algorithms import *
from .fill_manager import FillManager
from .order_management_system import OrderManagementSystem
from .order_types import *
from .order_validator import OrderValidator
from .smart_routing import SmartOrderRouter

__all__ = [
    "OrderManagementSystem",
    "SmartOrderRouter",
    "FillManager",
    "OrderValidator",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "ImplementationShortfallAlgorithm",
    "IcebergOrder",
    "AlgorithmicOrder",
]
