"""
Streaming Pipeline for Real-time Processing.

This module provides comprehensive streaming capabilities including:
- Real-time document processing
- Incremental vector updates
- Live monitoring and metrics
- Auto-scaling based on load
- Event-driven architecture
- Backpressure handling
"""

from .event_bus import (
    BusEvent,
    EventBus,
    EventFilter,
    EventHandler,
    EventPriority,
    EventSubscription,
)
from .incremental import (
    IncrementalProcessor,
    IncrementalUpdate,
    IndexManager,
    UpdateStrategy,
    UpdateType,
)
from .live_monitor import AlertManager, LiveMonitor, MetricsCollector, PerformanceMetrics
from .stream_processor import (
    ProcessingResult,
    StreamEvent,
    StreamEventType,
    StreamingConfig,
    StreamingPipeline,
    StreamProcessor,
)

__all__ = [
    # Stream Processing
    "StreamingPipeline",
    "StreamProcessor",
    "StreamingConfig",
    "StreamEvent",
    "StreamEventType",
    "ProcessingResult",
    # Incremental Updates
    "IncrementalProcessor",
    "IncrementalUpdate",
    "UpdateStrategy",
    "UpdateType",
    "IndexManager",
    # Live Monitoring
    "LiveMonitor",
    "MetricsCollector",
    "PerformanceMetrics",
    "AlertManager",
    # Event System
    "EventBus",
    "EventHandler",
    "EventSubscription",
    "EventFilter",
    "BusEvent",
    "EventPriority",
]
