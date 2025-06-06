"""
Streaming Data Ingestion Module.

This module provides real-time streaming data ingestion capabilities
for processing continuous data streams.
"""

from .stream_ingestion import StreamIngestionEngine, StreamIngestionConfig

__all__ = [
    "StreamIngestionEngine",
    "StreamIngestionConfig",
]
