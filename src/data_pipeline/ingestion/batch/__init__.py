"""
Batch Data Ingestion Module.

This module provides batch data ingestion capabilities for processing
large datasets from various sources.
"""

from .batch_ingestion import BatchIngestionEngine, BatchIngestionConfig

__all__ = [
    "BatchIngestionEngine",
    "BatchIngestionConfig",
]
