"""
Data Pipeline Module for Large-Scale Data Processing.

This module provides comprehensive data pipeline infrastructure including:
- Data ingestion from multiple sources
- ETL/ELT transformation pipelines
- Batch and stream processing
- Data quality validation
- Pipeline orchestration and scheduling
- Monitoring and observability
"""

from .core.orchestrator import PipelineOrchestrator
from .core.scheduler import PipelineScheduler
from .core.executor import PipelineExecutor
from .ingestion.batch.batch_ingestion import BatchIngestionEngine
from .ingestion.streaming.stream_ingestion import StreamIngestionEngine
from .transformation.etl.etl_engine import ETLEngine
from .transformation.validation.data_validator import DataValidator
from .storage.unified_access.data_access_layer import DataAccessLayer
from .processing.batch.batch_processor import BatchProcessor
from .processing.stream.stream_processor import StreamProcessor
from .monitoring.metrics.pipeline_metrics import PipelineMetrics

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    "PipelineOrchestrator",
    "PipelineScheduler", 
    "PipelineExecutor",
    "BatchIngestionEngine",
    "StreamIngestionEngine",
    "ETLEngine",
    "DataValidator",
    "DataAccessLayer",
    "BatchProcessor",
    "StreamProcessor",
    "PipelineMetrics",
]
