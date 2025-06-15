"""
Stream Processing Module.

This module provides real-time stream processing capabilities
for continuous data processing operations.
"""

from .stream_processor import StreamProcessingConfig, StreamProcessor

__all__ = [
    "StreamProcessor",
    "StreamProcessingConfig",
]
