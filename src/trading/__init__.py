"""
Institutional-Grade Trading System

A comprehensive trading infrastructure designed for high-frequency trading,
multi-strategy hedge fund operations, and 24/7 automated systematic trading.

Features:
- High-performance Order Management System (OMS)
- Low-latency market data processing
- Multi-strategy execution framework
- Real-time risk management
- Smart order routing
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- 24/7 automated operations
- Comprehensive monitoring and alerting
"""

from .core import *
from .oms import *
from .market_data import *
from .risk import *
from .strategies import *
from .execution import *
from .monitoring import *
from .operations import *

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Trading Team"
__description__ = "Institutional-Grade High-Performance Trading System"
