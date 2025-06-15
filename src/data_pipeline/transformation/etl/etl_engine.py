"""
ETL Engine for data transformation.

This module provides Extract, Transform, Load capabilities
for processing and transforming data in pipelines.
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import polars as pl
import structlog
from pydantic import BaseModel, Field


class ETLConfig(BaseModel):
    """Configuration for ETL operations."""

    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    chunk_size: int = Field(default=10000, description="Chunk size for processing")
    memory_limit: Optional[str] = Field(None, description="Memory limit")

    # Transformation options
    enable_type_inference: bool = Field(default=True, description="Enable automatic type inference")
    enable_null_handling: bool = Field(default=True, description="Enable null value handling")
    date_format: str = Field(default="ISO", description="Date format for parsing")

    # Performance options
    use_polars: bool = Field(default=False, description="Use Polars for transformations")
    lazy_evaluation: bool = Field(default=True, description="Use lazy evaluation when possible")


class TransformationOperation(BaseModel):
    """Represents a single transformation operation."""

    operation_id: str = Field(..., description="Unique operation identifier")
    operation_type: str = Field(..., description="Type of operation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    condition: Optional[str] = Field(None, description="Condition for applying operation")
    description: Optional[str] = Field(None, description="Operation description")


class ETLEngine:
    """
    ETL Engine for data transformation.

    Provides comprehensive Extract, Transform, Load capabilities
    with support for various transformation operations.
    """

    def __init__(self, config: Optional[ETLConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the ETL engine.

        Args:
            config: ETL configuration
            logger: Logger instance
        """
        self.config = config or ETLConfig()
        self.logger = logger or structlog.get_logger("etl_engine")

        # Transformation registry
        self.transformations: Dict[str, Callable] = {
            "filter": self._filter_operation,
            "select": self._select_operation,
            "rename": self._rename_operation,
            "add_column": self._add_column_operation,
            "drop_column": self._drop_column_operation,
            "cast_type": self._cast_type_operation,
            "fill_null": self._fill_null_operation,
            "replace_value": self._replace_value_operation,
            "aggregate": self._aggregate_operation,
            "join": self._join_operation,
            "sort": self._sort_operation,
            "deduplicate": self._deduplicate_operation,
            "pivot": self._pivot_operation,
            "unpivot": self._unpivot_operation,
            "custom": self._custom_operation,
        }

        self.logger.info("ETL engine initialized")

    def register_transformation(self, operation_type: str, handler: Callable) -> None:
        """
        Register a custom transformation operation.

        Args:
            operation_type: Type of operation
            handler: Function to handle the operation
        """
        self.transformations[operation_type] = handler
        self.logger.info("Custom transformation registered", operation_type=operation_type)

    async def transform_data(
        self, data: Union[pd.DataFrame, pl.DataFrame, Any], transformation_config: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Transform data according to configuration.

        Args:
            data: Input data
            transformation_config: Transformation configuration

        Returns:
            Transformed data
        """
        try:
            self.logger.info("Starting data transformation")

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

            # Extract operations from config
            operations = transformation_config.get("operations", [])

            # Apply transformations sequentially
            current_data = data
            for operation_config in operations:
                operation = TransformationOperation(**operation_config)
                current_data = await self._apply_operation(current_data, operation)

            self.logger.info(
                "Data transformation completed",
                input_records=len(data),
                output_records=len(current_data),
                operations_applied=len(operations),
            )

            return current_data

        except Exception as e:
            self.logger.error("Data transformation failed", error=str(e))
            raise e

    async def _apply_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], operation: TransformationOperation
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Apply a single transformation operation."""
        try:
            # Get operation handler
            handler = self.transformations.get(operation.operation_type)
            if not handler:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")

            # Apply condition if specified
            if operation.condition:
                # This is a simplified condition check
                # In a real implementation, you'd want a proper expression evaluator
                pass

            # Apply transformation
            result = await handler(data, operation.parameters)

            self.logger.debug(
                "Operation applied",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                input_records=len(data),
                output_records=len(result),
            )

            return result

        except Exception as e:
            self.logger.error(
                "Operation failed",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                error=str(e),
            )
            raise e

    async def _filter_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Filter rows based on condition."""
        condition = parameters.get("condition")
        if not condition:
            return data

        if isinstance(data, pl.DataFrame):
            # Polars filtering
            return data.filter(pl.expr(condition))
        else:
            # Pandas filtering
            return data.query(condition)

    async def _select_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Select specific columns."""
        columns = parameters.get("columns", [])
        if not columns:
            return data

        return data.select(columns) if isinstance(data, pl.DataFrame) else data[columns]

    async def _rename_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Rename columns."""
        mapping = parameters.get("mapping", {})
        if not mapping:
            return data

        if isinstance(data, pl.DataFrame):
            return data.rename(mapping)
        else:
            return data.rename(columns=mapping)

    async def _add_column_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Add a new column."""
        column_name = parameters.get("column")
        expression = parameters.get("expression")
        value = parameters.get("value")

        if not column_name:
            return data

        if isinstance(data, pl.DataFrame):
            if expression:
                # Polars expression
                return data.with_columns(pl.expr(expression).alias(column_name))
            elif value is not None:
                # Static value
                return data.with_columns(pl.lit(value).alias(column_name))
        else:
            if expression:
                # Pandas expression (simplified)
                # In a real implementation, you'd want a proper expression evaluator
                if "case when" in expression.lower():
                    # Simple case when handling
                    return data.assign(**{column_name: value or ""})
                else:
                    return data.eval(f"{column_name} = {expression}")
            elif value is not None:
                # Static value
                data[column_name] = value
                return data

        return data

    async def _drop_column_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Drop columns."""
        columns = parameters.get("columns", [])
        if not columns:
            return data

        if isinstance(data, pl.DataFrame):
            return data.drop(columns)
        else:
            return data.drop(columns=columns)

    async def _cast_type_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Cast column types."""
        type_mapping = parameters.get("type_mapping", {})
        if not type_mapping:
            return data

        if isinstance(data, pl.DataFrame):
            # Polars type casting
            for column, dtype in type_mapping.items():
                if column in data.columns:
                    data = data.with_columns(pl.col(column).cast(dtype))
        else:
            # Pandas type casting
            data = data.astype(type_mapping)

        return data

    async def _fill_null_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Fill null values."""
        strategy = parameters.get("strategy", "value")
        value = parameters.get("value")
        columns = parameters.get("columns")

        if isinstance(data, pl.DataFrame):
            if strategy == "value" and value is not None:
                if columns:
                    return data.with_columns([pl.col(col).fill_null(value) for col in columns])
                else:
                    return data.fill_null(value)
            elif strategy == "forward":
                return data.fill_null(strategy="forward")
            elif strategy == "backward":
                return data.fill_null(strategy="backward")
        else:
            if strategy == "value" and value is not None:
                if columns:
                    data[columns] = data[columns].fillna(value)
                else:
                    data = data.fillna(value)
            elif strategy == "forward":
                data = data.fillna(method="ffill")
            elif strategy == "backward":
                data = data.fillna(method="bfill")

        return data

    async def _replace_value_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Replace values."""
        old_value = parameters.get("old_value")
        new_value = parameters.get("new_value")
        columns = parameters.get("columns")

        if old_value is None or new_value is None:
            return data

        if isinstance(data, pl.DataFrame):
            if columns:
                return data.with_columns(
                    [pl.col(col).str.replace(str(old_value), str(new_value)) for col in columns]
                )
            else:
                return data.with_columns(
                    [
                        pl.col(col).str.replace(str(old_value), str(new_value))
                        for col in data.columns
                    ]
                )
        else:
            if columns:
                data[columns] = data[columns].replace(old_value, new_value)
            else:
                data = data.replace(old_value, new_value)

        return data

    async def _aggregate_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Aggregate data."""
        group_by = parameters.get("group_by", [])
        aggregations = parameters.get("aggregations", {})

        if not aggregations:
            return data

        if isinstance(data, pl.DataFrame):
            if group_by:
                return data.group_by(group_by).agg(
                    [
                        getattr(pl.col(col), agg_func)().alias(f"{col}_{agg_func}")
                        for col, agg_func in aggregations.items()
                    ]
                )
            else:
                return data.select(
                    [
                        getattr(pl.col(col), agg_func)().alias(f"{col}_{agg_func}")
                        for col, agg_func in aggregations.items()
                    ]
                )
        else:
            if group_by:
                return data.groupby(group_by).agg(aggregations).reset_index()
            else:
                return data.agg(aggregations).to_frame().T

    async def _join_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Join with another dataset."""
        # This would require access to the other dataset
        # For now, return data unchanged
        self.logger.warning("Join operation not fully implemented")
        return data

    async def _sort_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Sort data."""
        columns = parameters.get("columns", [])
        ascending = parameters.get("ascending", True)

        if not columns:
            return data

        if isinstance(data, pl.DataFrame):
            return data.sort(columns, descending=not ascending)
        else:
            return data.sort_values(columns, ascending=ascending)

    async def _deduplicate_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Remove duplicate rows."""
        columns = parameters.get("columns")

        if isinstance(data, pl.DataFrame):
            if columns:
                return data.unique(subset=columns)
            else:
                return data.unique()
        else:
            return data.drop_duplicates(subset=columns)

    async def _pivot_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Pivot data."""
        # Simplified pivot implementation
        self.logger.warning("Pivot operation not fully implemented")
        return data

    async def _unpivot_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Unpivot data."""
        # Simplified unpivot implementation
        self.logger.warning("Unpivot operation not fully implemented")
        return data

    async def _custom_operation(
        self, data: Union[pd.DataFrame, pl.DataFrame], parameters: Dict[str, Any]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Apply custom transformation."""
        function_name = parameters.get("function")
        if function_name and hasattr(self, function_name):
            custom_func = getattr(self, function_name)
            return await custom_func(data, parameters)

        return data

    async def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about available transformations."""
        return {
            "available_operations": list(self.transformations.keys()),
            "config": self.config.model_dump(),
            "engine": "etl_engine",
        }
