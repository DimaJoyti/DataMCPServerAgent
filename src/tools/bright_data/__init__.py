"""
Enhanced Bright Data MCP Integration

This package provides a comprehensive, production-ready integration with Bright Data's
MCP (Model Context Protocol) server, featuring advanced caching, error handling,
rate limiting, and specialized tools for various use cases.

Key Features:
- Enhanced client with retry logic and circuit breaker
- Multi-level caching system with Redis support
- Advanced error handling and recovery
- Rate limiting and throttling
- Specialized tools for OSINT, market research, and competitive intelligence
- RESTful API and WebSocket support
- Real-time monitoring and analytics
- Integration with knowledge graph and distributed memory
"""

from .core.enhanced_client import EnhancedBrightDataClient
from .core.cache_manager import CacheManager
from .core.error_handler import BrightDataErrorHandler
from .core.rate_limiter import RateLimiter
from .core.config import BrightDataConfig

__version__ = "2.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    "EnhancedBrightDataClient",
    "CacheManager",
    "BrightDataErrorHandler",
    "RateLimiter",
    "BrightDataConfig",
]
