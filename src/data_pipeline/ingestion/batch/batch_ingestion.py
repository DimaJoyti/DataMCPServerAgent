"""
Batch Data Ingestion Engine.

This module provides comprehensive batch data ingestion capabilities
for processing large datasets from various sources.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
import polars as pl
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from ...core.pipeline_models import DataSource, DataDestination, QualityMetrics
from ..connectors.database_connector import DatabaseConnector
from ..connectors.file_connector import FileConnector
from ..connectors.api_connector import APIConnector
from ..connectors.object_storage_connector import ObjectStorageConnector


class BatchIngestionConfig(BaseModel):
    """Configuration for batch ingestion."""
    batch_size: int = Field(default=10000, description="Number of records per batch")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    chunk_size: int = Field(default=1000, description="Chunk size for processing")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    data_format: str = Field(default="auto", description="Data format (auto, csv, json, parquet, etc.)")
    compression: Optional[str] = Field(None, description="Compression type")
    encoding: str = Field(default="utf-8", description="Text encoding")
    
    # Quality and validation
    enable_data_profiling: bool = Field(default=True, description="Enable data profiling")
    enable_schema_validation: bool = Field(default=True, description="Enable schema validation")
    max_error_rate: float = Field(default=0.05, description="Maximum acceptable error rate")
    
    # Performance
    memory_limit: Optional[str] = Field(None, description="Memory limit (e.g., '1GB')")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")


class IngestionMetrics(BaseModel):
    """Metrics for ingestion operations."""
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    bytes_processed: int = 0
    processing_time: float = 0.0
    throughput_records_per_second: float = 0.0
    throughput_bytes_per_second: float = 0.0
    error_rate: float = 0.0
    
    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = None
    
    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class BatchIngestionEngine:
    """
    Engine for batch data ingestion.
    
    Supports ingestion from various sources including databases, files,
    APIs, and object storage systems.
    """
    
    def __init__(
        self,
        config: Optional[BatchIngestionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the batch ingestion engine.
        
        Args:
            config: Ingestion configuration
            logger: Logger instance
        """
        self.config = config or BatchIngestionConfig()
        self.logger = logger or structlog.get_logger("batch_ingestion")
        
        # Initialize connectors
        self.connectors = {
            "database": DatabaseConnector(logger=self.logger),
            "file": FileConnector(logger=self.logger),
            "api": APIConnector(logger=self.logger),
            "object_storage": ObjectStorageConnector(logger=self.logger),
        }
        
        self.logger.info("Batch ingestion engine initialized")
    
    async def ingest_data(
        self,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> IngestionMetrics:
        """
        Ingest data from source to destination.
        
        Args:
            source_config: Source configuration
            destination_config: Destination configuration
            transformation_config: Optional transformation configuration
            
        Returns:
            Ingestion metrics
        """
        metrics = IngestionMetrics(start_time=datetime.now(timezone.utc))
        
        try:
            self.logger.info(
                "Starting batch ingestion",
                source_type=source_config.get("type"),
                destination_type=destination_config.get("type")
            )
            
            # Get source connector
            source_connector = self._get_connector(source_config["type"])
            
            # Get destination connector
            destination_connector = self._get_connector(destination_config["type"])
            
            # Read data in batches
            async for batch_data in self._read_data_batches(source_connector, source_config):
                # Process batch
                processed_batch = await self._process_batch(
                    batch_data,
                    transformation_config,
                    metrics
                )
                
                # Write batch to destination
                await self._write_batch(
                    destination_connector,
                    destination_config,
                    processed_batch,
                    metrics
                )
            
            # Finalize metrics
            metrics.end_time = datetime.now(timezone.utc)
            if metrics.start_time:
                metrics.processing_time = (
                    metrics.end_time - metrics.start_time
                ).total_seconds()
                
                if metrics.processing_time > 0:
                    metrics.throughput_records_per_second = (
                        metrics.processed_records / metrics.processing_time
                    )
                    metrics.throughput_bytes_per_second = (
                        metrics.bytes_processed / metrics.processing_time
                    )
            
            if metrics.total_records > 0:
                metrics.error_rate = metrics.failed_records / metrics.total_records
            
            self.logger.info(
                "Batch ingestion completed",
                total_records=metrics.total_records,
                processed_records=metrics.processed_records,
                failed_records=metrics.failed_records,
                processing_time=metrics.processing_time,
                throughput_rps=metrics.throughput_records_per_second
            )
            
            return metrics
            
        except Exception as e:
            metrics.end_time = datetime.now(timezone.utc)
            self.logger.error(
                "Batch ingestion failed",
                error=str(e),
                exc_info=True
            )
            raise e
    
    async def _read_data_batches(
        self,
        connector: Any,
        source_config: Dict[str, Any]
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read data in batches from source."""
        try:
            # Connect to source
            await connector.connect(source_config)
            
            # Read data in batches
            async for batch in connector.read_batches(
                batch_size=self.config.batch_size,
                **source_config.get("read_options", {})
            ):
                yield batch
                
        finally:
            # Disconnect from source
            await connector.disconnect()
    
    async def _process_batch(
        self,
        batch_data: Union[pd.DataFrame, pl.DataFrame],
        transformation_config: Optional[Dict[str, Any]],
        metrics: IngestionMetrics
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Process a batch of data."""
        try:
            # Update metrics
            batch_size = len(batch_data)
            metrics.total_records += batch_size
            
            # Calculate batch size in bytes (approximate)
            if isinstance(batch_data, pd.DataFrame):
                batch_bytes = batch_data.memory_usage(deep=True).sum()
            else:  # polars DataFrame
                batch_bytes = batch_data.estimated_size()
            
            metrics.bytes_processed += batch_bytes
            
            # Apply transformations if configured
            if transformation_config:
                batch_data = await self._apply_transformations(
                    batch_data,
                    transformation_config
                )
            
            # Data quality checks
            if self.config.enable_data_profiling:
                await self._profile_batch(batch_data, metrics)
            
            # Schema validation
            if self.config.enable_schema_validation:
                await self._validate_batch_schema(batch_data, transformation_config)
            
            metrics.processed_records += batch_size
            
            return batch_data
            
        except Exception as e:
            metrics.failed_records += len(batch_data)
            self.logger.error(
                "Batch processing failed",
                batch_size=len(batch_data),
                error=str(e)
            )
            
            # Check error rate
            if metrics.total_records > 0:
                current_error_rate = metrics.failed_records / metrics.total_records
                if current_error_rate > self.config.max_error_rate:
                    raise RuntimeError(
                        f"Error rate {current_error_rate:.2%} exceeds maximum "
                        f"allowed rate {self.config.max_error_rate:.2%}"
                    )
            
            # Return empty DataFrame to continue processing
            if isinstance(batch_data, pd.DataFrame):
                return pd.DataFrame()
            else:
                return pl.DataFrame()
    
    async def _write_batch(
        self,
        connector: Any,
        destination_config: Dict[str, Any],
        batch_data: Union[pd.DataFrame, pl.DataFrame],
        metrics: IngestionMetrics
    ) -> None:
        """Write batch to destination."""
        if len(batch_data) == 0:
            return
        
        try:
            # Connect to destination
            await connector.connect(destination_config)
            
            # Write batch
            await connector.write_batch(
                batch_data,
                **destination_config.get("write_options", {})
            )
            
        finally:
            # Disconnect from destination
            await connector.disconnect()
    
    async def _apply_transformations(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        transformation_config: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Apply transformations to data."""
        # This would integrate with the transformation engine
        # For now, return data as-is
        return data
    
    async def _profile_batch(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        metrics: IngestionMetrics
    ) -> None:
        """Profile batch data for quality metrics."""
        try:
            if isinstance(data, pd.DataFrame):
                # Pandas profiling
                total_records = len(data)
                null_count = data.isnull().sum().sum()
                duplicate_count = data.duplicated().sum()
                
            else:  # polars DataFrame
                # Polars profiling
                total_records = data.height
                null_count = data.null_count().sum_horizontal()[0]
                duplicate_count = total_records - data.n_unique()
            
            # Update quality metrics
            if not metrics.quality_metrics:
                metrics.quality_metrics = QualityMetrics(
                    total_records=0,
                    valid_records=0,
                    invalid_records=0,
                    completeness_score=0.0,
                    validity_score=0.0,
                    consistency_score=0.0,
                    null_count=0,
                    duplicate_count=0
                )
            
            # Accumulate metrics
            metrics.quality_metrics.total_records += total_records
            metrics.quality_metrics.null_count += null_count
            metrics.quality_metrics.duplicate_count += duplicate_count
            
            # Calculate scores
            if metrics.quality_metrics.total_records > 0:
                metrics.quality_metrics.completeness_score = 1.0 - (
                    metrics.quality_metrics.null_count / 
                    (metrics.quality_metrics.total_records * len(data.columns))
                )
                
                metrics.quality_metrics.validity_score = (
                    metrics.quality_metrics.total_records - 
                    metrics.quality_metrics.duplicate_count
                ) / metrics.quality_metrics.total_records
            
        except Exception as e:
            self.logger.warning(
                "Data profiling failed",
                error=str(e)
            )
    
    async def _validate_batch_schema(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        transformation_config: Optional[Dict[str, Any]]
    ) -> None:
        """Validate batch schema."""
        # This would implement schema validation logic
        # For now, just log the schema
        if isinstance(data, pd.DataFrame):
            schema_info = data.dtypes.to_dict()
        else:  # polars DataFrame
            schema_info = data.schema
        
        self.logger.debug("Batch schema", schema=schema_info)
    
    def _get_connector(self, connector_type: str) -> Any:
        """Get connector by type."""
        if connector_type not in self.connectors:
            raise ValueError(f"Unsupported connector type: {connector_type}")
        
        return self.connectors[connector_type]
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status."""
        return {
            "engine": "batch_ingestion",
            "config": self.config.model_dump(),
            "connectors": list(self.connectors.keys()),
            "status": "ready"
        }
