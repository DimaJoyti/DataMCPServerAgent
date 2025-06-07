"""
Batch Processor for large-scale data processing.

This module provides batch processing capabilities for handling
large datasets with parallel and distributed processing support.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import polars as pl
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import structlog
from pydantic import BaseModel, Field

class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing."""
    # Processing options
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=mp.cpu_count(), description="Maximum number of worker processes")
    chunk_size: int = Field(default=10000, description="Chunk size for processing")
    use_processes: bool = Field(default=True, description="Use processes instead of threads")

    # Memory management
    memory_limit: Optional[str] = Field(None, description="Memory limit per worker")
    enable_memory_monitoring: bool = Field(default=True, description="Enable memory monitoring")

    # Performance options
    use_polars: bool = Field(default=False, description="Use Polars for processing")
    enable_lazy_evaluation: bool = Field(default=True, description="Enable lazy evaluation")

    # Error handling
    continue_on_error: bool = Field(default=False, description="Continue processing on errors")
    max_error_rate: float = Field(default=0.05, description="Maximum acceptable error rate")

class ProcessingMetrics(BaseModel):
    """Metrics for batch processing."""
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0

    # Performance metrics
    processing_time: float = 0.0
    throughput_records_per_second: float = 0.0
    average_chunk_time: float = 0.0

    # Resource metrics
    peak_memory_usage: Optional[float] = None
    cpu_utilization: Optional[float] = None

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class BatchProcessor:
    """
    Batch processor for large-scale data processing.

    Provides parallel and distributed processing capabilities
    for handling large datasets efficiently.
    """

    def __init__(
        self,
        config: Optional[BatchProcessingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the batch processor.

        Args:
            config: Processing configuration
            logger: Logger instance
        """
        self.config = config or BatchProcessingConfig()
        self.logger = logger or structlog.get_logger("batch_processor")

        # Processing state
        self.is_processing = False
        self.current_metrics = ProcessingMetrics()

        # Executor for parallel processing
        self.executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None

        self.logger.info("Batch processor initialized")

    async def process_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame, Any],
        processing_config: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Process data according to configuration.

        Args:
            data: Input data
            processing_config: Processing configuration

        Returns:
            Processed data
        """
        if self.is_processing:
            raise RuntimeError("Batch processor is already processing data")

        self.is_processing = True
        self.current_metrics = ProcessingMetrics(start_time=datetime.now(timezone.utc))

        try:
            self.logger.info("Starting batch processing")

            # Convert input data to DataFrame if needed
            if not isinstance(data, (pd.DataFrame, pl.DataFrame)):
                if isinstance(data, list):
                    if self.config.use_polars:
                        data = pl.DataFrame(data)
                    else:
                        data = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if self.config.use_polars:
                        data = pl.DataFrame([data])
                    else:
                        data = pd.DataFrame([data])
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")

            self.current_metrics.total_records = len(data)

            # Extract processing operations
            operations = processing_config.get("operations", [])

            # Process data
            if self.config.enable_parallel_processing and len(data) > self.config.chunk_size:
                result = await self._process_parallel(data, operations)
            else:
                result = await self._process_sequential(data, operations)

            # Finalize metrics
            self.current_metrics.end_time = datetime.now(timezone.utc)
            if self.current_metrics.start_time:
                self.current_metrics.processing_time = (
                    self.current_metrics.end_time - self.current_metrics.start_time
                ).total_seconds()

                if self.current_metrics.processing_time > 0:
                    self.current_metrics.throughput_records_per_second = (
                        self.current_metrics.processed_records / self.current_metrics.processing_time
                    )

            self.logger.info(
                "Batch processing completed",
                total_records=self.current_metrics.total_records,
                processed_records=self.current_metrics.processed_records,
                failed_records=self.current_metrics.failed_records,
                processing_time=self.current_metrics.processing_time,
                throughput_rps=self.current_metrics.throughput_records_per_second
            )

            return result

        except Exception as e:
            self.logger.error("Batch processing failed", error=str(e))
            raise e
        finally:
            self.is_processing = False
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None

    async def _process_parallel(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        operations: List[Dict[str, Any]]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Process data in parallel chunks."""
        try:
            # Initialize executor
            if self.config.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

            # Split data into chunks
            chunks = self._split_data_into_chunks(data)
            self.logger.info(f"Processing {len(chunks)} chunks in parallel")

            # Process chunks in parallel
            loop = asyncio.get_event_loop()
            tasks = []

            for i, chunk in enumerate(chunks):
                task = loop.run_in_executor(
                    self.executor,
                    self._process_chunk,
                    chunk,
                    operations,
                    i
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            processed_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.current_metrics.chunks_failed += 1
                    self.logger.error(f"Chunk {i} processing failed", error=str(result))

                    if not self.config.continue_on_error:
                        raise result
                else:
                    processed_chunks.append(result)
                    self.current_metrics.chunks_processed += 1
                    self.current_metrics.processed_records += len(result)

            # Combine all processed chunks
            if processed_chunks:
                if isinstance(processed_chunks[0], pl.DataFrame):
                    return pl.concat(processed_chunks)
                else:
                    return pd.concat(processed_chunks, ignore_index=True)
            else:
                # Return empty DataFrame of same type as input
                if isinstance(data, pl.DataFrame):
                    return pl.DataFrame()
                else:
                    return pd.DataFrame()

        except Exception as e:
            self.logger.error("Parallel processing failed", error=str(e))
            raise e

    async def _process_sequential(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        operations: List[Dict[str, Any]]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Process data sequentially."""
        try:
            result = self._process_chunk(data, operations, 0)
            self.current_metrics.processed_records = len(result)
            self.current_metrics.chunks_processed = 1
            return result

        except Exception as e:
            self.current_metrics.chunks_failed = 1
            self.current_metrics.failed_records = len(data)
            raise e

    def _process_chunk(
        self,
        chunk: Union[pd.DataFrame, pl.DataFrame],
        operations: List[Dict[str, Any]],
        chunk_id: int
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Process a single chunk of data."""
        try:
            current_data = chunk

            # Apply operations sequentially
            for operation in operations:
                current_data = self._apply_operation(current_data, operation)

            return current_data

        except Exception as e:
            self.logger.error(f"Chunk {chunk_id} processing failed", error=str(e))
            raise e

    def _apply_operation(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        operation: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Apply a single processing operation."""
        operation_type = operation.get("type")
        parameters = operation.get("parameters", {})

        if operation_type == "filter":
            return self._filter_data(data, parameters)
        elif operation_type == "transform":
            return self._transform_data(data, parameters)
        elif operation_type == "aggregate":
            return self._aggregate_data(data, parameters)
        elif operation_type == "sort":
            return self._sort_data(data, parameters)
        elif operation_type == "deduplicate":
            return self._deduplicate_data(data, parameters)
        elif operation_type == "custom":
            return self._custom_operation(data, parameters)
        else:
            self.logger.warning(f"Unknown operation type: {operation_type}")
            return data

    def _filter_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Filter data based on conditions."""
        condition = parameters.get("condition")
        if not condition:
            return data

        try:
            if isinstance(data, pl.DataFrame):
                return data.filter(pl.expr(condition))
            else:
                return data.query(condition)
        except Exception:
            return data

    def _transform_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Transform data columns."""
        transformations = parameters.get("transformations", {})

        for column, transformation in transformations.items():
            if column in data.columns:
                if transformation == "uppercase":
                    if isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.col(column).str.to_uppercase())
                    else:
                        data[column] = data[column].str.upper()
                elif transformation == "lowercase":
                    if isinstance(data, pl.DataFrame):
                        data = data.with_columns(pl.col(column).str.to_lowercase())
                    else:
                        data[column] = data[column].str.lower()

        return data

    def _aggregate_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Aggregate data."""
        group_by = parameters.get("group_by", [])
        aggregations = parameters.get("aggregations", {})

        if not aggregations:
            return data

        try:
            if isinstance(data, pl.DataFrame):
                if group_by:
                    return data.group_by(group_by).agg([
                        getattr(pl.col(col), agg_func)().alias(f"{col}_{agg_func}")
                        for col, agg_func in aggregations.items()
                    ])
                else:
                    return data.select([
                        getattr(pl.col(col), agg_func)().alias(f"{col}_{agg_func}")
                        for col, agg_func in aggregations.items()
                    ])
            else:
                if group_by:
                    return data.groupby(group_by).agg(aggregations).reset_index()
                else:
                    return data.agg(aggregations).to_frame().T
        except Exception:
            return data

    def _sort_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Sort data."""
        columns = parameters.get("columns", [])
        ascending = parameters.get("ascending", True)

        if not columns:
            return data

        try:
            if isinstance(data, pl.DataFrame):
                return data.sort(columns, descending=not ascending)
            else:
                return data.sort_values(columns, ascending=ascending)
        except Exception:
            return data

    def _deduplicate_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Remove duplicate rows."""
        columns = parameters.get("columns")

        try:
            if isinstance(data, pl.DataFrame):
                if columns:
                    return data.unique(subset=columns)
                else:
                    return data.unique()
            else:
                return data.drop_duplicates(subset=columns)
        except Exception:
            return data

    def _custom_operation(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Apply custom operation."""
        # This would allow for custom processing functions
        # For now, return data unchanged
        return data

    def _split_data_into_chunks(
        self,
        data: Union[pd.DataFrame, pl.DataFrame]
    ) -> List[Union[pd.DataFrame, pl.DataFrame]]:
        """Split data into chunks for parallel processing."""
        chunks = []
        total_rows = len(data)

        for start in range(0, total_rows, self.config.chunk_size):
            end = min(start + self.config.chunk_size, total_rows)

            if isinstance(data, pl.DataFrame):
                chunk = data.slice(start, end - start)
            else:
                chunk = data.iloc[start:end]

            chunks.append(chunk)

        return chunks

    async def get_processing_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.current_metrics

    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "is_processing": self.is_processing,
            "config": self.config.model_dump(),
            "metrics": self.current_metrics.model_dump(),
            "processor": "batch_processor"
        }
