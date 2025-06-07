"""
Data Ingestion Module for the Data Pipeline.

This module provides comprehensive data ingestion capabilities including:
- Batch data ingestion from various sources
- Real-time streaming data ingestion
- Data source connectors and adapters
- Data format detection and parsing
- Ingestion monitoring and metrics
"""

from .batch.batch_ingestion import BatchIngestionEngine
from .streaming.stream_ingestion import StreamIngestionEngine
from .connectors.database_connector import DatabaseConnector
from .connectors.file_connector import FileConnector
from .connectors.api_connector import APIConnector
from .connectors.object_storage_connector import ObjectStorageConnector

__all__ = [
    "BatchIngestionEngine",
    "StreamIngestionEngine",
    "DatabaseConnector",
    "FileConnector",
    "APIConnector",
    "ObjectStorageConnector",
]
