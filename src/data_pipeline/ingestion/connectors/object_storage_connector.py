"""
Object Storage Connector for data pipeline ingestion.

This module provides object storage connectivity for S3-compatible storage,
Azure Blob Storage, Google Cloud Storage, and other object storage systems.
"""

import io
import logging
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl
import structlog
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field


class ObjectStorageConfig(BaseModel):
    """Object storage connection configuration."""

    storage_type: str = Field(..., description="Storage type (s3, minio, azure, gcs)")
    endpoint: Optional[str] = Field(None, description="Storage endpoint URL")
    access_key: str = Field(..., description="Access key")
    secret_key: str = Field(..., description="Secret key")
    bucket_name: str = Field(..., description="Bucket/container name")

    # S3/MinIO specific
    region: Optional[str] = Field(None, description="AWS region")
    secure: bool = Field(default=True, description="Use HTTPS")

    # Path configuration
    prefix: Optional[str] = Field(None, description="Object key prefix")
    file_pattern: Optional[str] = Field(None, description="File pattern to match")

    # Format configuration
    file_format: str = Field(default="auto", description="File format (csv, json, parquet, etc.)")
    compression: Optional[str] = Field(None, description="Compression type")
    encoding: str = Field(default="utf-8", description="Text encoding")


class ObjectStorageConnector:
    """
    Object storage connector for data pipeline ingestion.

    Supports S3-compatible storage systems including AWS S3, MinIO,
    and other cloud storage providers.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the object storage connector.

        Args:
            logger: Logger instance
        """
        self.logger = logger or structlog.get_logger("object_storage_connector")

        # Connection state
        self.config: Optional[ObjectStorageConfig] = None
        self.client: Optional[Minio] = None
        self.is_connected = False
        self.object_list: List[str] = []

        self.logger.info("Object storage connector initialized")

    async def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to object storage.

        Args:
            config: Object storage configuration
        """
        self.config = ObjectStorageConfig(**config)

        try:
            # Initialize MinIO client (works with S3-compatible storage)
            if self.config.storage_type in ["s3", "minio"]:
                self.client = Minio(
                    endpoint=self.config.endpoint or "s3.amazonaws.com",
                    access_key=self.config.access_key,
                    secret_key=self.config.secret_key,
                    secure=self.config.secure,
                    region=self.config.region,
                )

                # Test connection by checking if bucket exists
                if not self.client.bucket_exists(self.config.bucket_name):
                    raise Exception(f"Bucket {self.config.bucket_name} does not exist")

                # List objects
                self.object_list = await self._list_objects()

            else:
                raise ValueError(f"Unsupported storage type: {self.config.storage_type}")

            self.is_connected = True

            self.logger.info(
                "Object storage connected",
                storage_type=self.config.storage_type,
                bucket=self.config.bucket_name,
                object_count=len(self.object_list),
            )

        except Exception as e:
            self.logger.error(
                "Object storage connection failed",
                error=str(e),
                storage_type=self.config.storage_type,
                bucket=self.config.bucket_name,
            )
            raise e

    async def disconnect(self) -> None:
        """Disconnect from object storage."""
        self.is_connected = False
        self.client = None
        self.object_list = []
        self.logger.info("Object storage disconnected")

    async def read_batches(
        self, batch_size: int = 10000, **kwargs
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """
        Read data in batches from object storage.

        Args:
            batch_size: Number of records per batch
            **kwargs: Additional parameters

        Yields:
            DataFrames containing batch data
        """
        if not self.is_connected:
            raise RuntimeError("Object storage not connected")

        use_polars = kwargs.get("use_polars", False)

        try:
            for object_key in self.object_list:
                self.logger.debug("Reading object", object_key=object_key)

                # Download object data
                object_data = await self._download_object(object_key)

                # Detect format if auto
                file_format = self.config.file_format
                if file_format == "auto":
                    file_format = self._detect_format_from_key(object_key)

                # Parse data based on format
                if file_format == "csv":
                    async for batch in self._parse_csv_data(object_data, batch_size, use_polars):
                        yield batch

                elif file_format == "json":
                    async for batch in self._parse_json_data(object_data, batch_size, use_polars):
                        yield batch

                elif file_format == "parquet":
                    async for batch in self._parse_parquet_data(
                        object_data, batch_size, use_polars
                    ):
                        yield batch

                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

        except Exception as e:
            self.logger.error(
                "Object storage read error", error=str(e), bucket=self.config.bucket_name
            )
            raise e

    async def write_batch(
        self, data: Union[pd.DataFrame, pl.DataFrame], object_key: str, **kwargs
    ) -> None:
        """
        Write a batch of data to object storage.

        Args:
            data: Data to write
            object_key: Object key/path
            **kwargs: Additional parameters
        """
        if not self.is_connected:
            raise RuntimeError("Object storage not connected")

        try:
            # Determine format from object key or config
            file_format = kwargs.get("format", self.config.file_format)
            if file_format == "auto":
                file_format = self._detect_format_from_key(object_key)

            # Convert data to bytes
            data_bytes = await self._serialize_data(data, file_format)

            # Upload to object storage
            await self._upload_object(object_key, data_bytes)

            self.logger.debug(
                "Batch written to object storage",
                object_key=object_key,
                records=len(data),
                format=file_format,
            )

        except Exception as e:
            self.logger.error(
                "Object storage write error", error=str(e), object_key=object_key, records=len(data)
            )
            raise e

    async def _list_objects(self) -> List[str]:
        """List objects in the bucket."""
        try:
            objects = []

            # List objects with prefix
            for obj in self.client.list_objects(
                self.config.bucket_name, prefix=self.config.prefix, recursive=True
            ):
                # Apply file pattern filter if specified
                if self.config.file_pattern:
                    if not self._matches_pattern(obj.object_name, self.config.file_pattern):
                        continue

                objects.append(obj.object_name)

            return objects

        except S3Error as e:
            self.logger.error("Failed to list objects", error=str(e))
            raise e

    async def _download_object(self, object_key: str) -> bytes:
        """Download object data."""
        try:
            response = self.client.get_object(self.config.bucket_name, object_key)
            data = response.read()
            response.close()
            response.release_conn()

            return data

        except S3Error as e:
            self.logger.error("Failed to download object", error=str(e), object_key=object_key)
            raise e

    async def _upload_object(self, object_key: str, data: bytes) -> None:
        """Upload object data."""
        try:
            data_stream = io.BytesIO(data)

            self.client.put_object(
                self.config.bucket_name, object_key, data_stream, length=len(data)
            )

        except S3Error as e:
            self.logger.error("Failed to upload object", error=str(e), object_key=object_key)
            raise e

    async def _parse_csv_data(
        self, data: bytes, batch_size: int, use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Parse CSV data in batches."""
        data_str = data.decode(self.config.encoding)
        data_io = io.StringIO(data_str)

        if use_polars:
            # Read entire CSV with polars, then batch
            df = pl.read_csv(data_io)

            for i in range(0, len(df), batch_size):
                yield df.slice(i, batch_size)
        else:
            # Read CSV in chunks with pandas
            for chunk in pd.read_csv(data_io, chunksize=batch_size):
                yield chunk

    async def _parse_json_data(
        self, data: bytes, batch_size: int, use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Parse JSON data in batches."""
        import json

        data_str = data.decode(self.config.encoding)

        # Try to parse as JSON
        try:
            json_data = json.loads(data_str)

            if isinstance(json_data, list):
                # Process in batches
                for i in range(0, len(json_data), batch_size):
                    batch = json_data[i : i + batch_size]
                    if use_polars:
                        yield pl.DataFrame(batch)
                    else:
                        yield pd.DataFrame(batch)
            else:
                # Single object
                if use_polars:
                    yield pl.DataFrame([json_data])
                else:
                    yield pd.DataFrame([json_data])

        except json.JSONDecodeError:
            # Try line-delimited JSON
            lines = data_str.strip().split("\n")
            batch_data = []

            for line in lines:
                try:
                    record = json.loads(line)
                    batch_data.append(record)

                    if len(batch_data) >= batch_size:
                        if use_polars:
                            yield pl.DataFrame(batch_data)
                        else:
                            yield pd.DataFrame(batch_data)
                        batch_data = []

                except json.JSONDecodeError:
                    continue

            # Yield remaining data
            if batch_data:
                if use_polars:
                    yield pl.DataFrame(batch_data)
                else:
                    yield pd.DataFrame(batch_data)

    async def _parse_parquet_data(
        self, data: bytes, batch_size: int, use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Parse Parquet data in batches."""
        data_io = io.BytesIO(data)

        if use_polars:
            # Read with polars
            df = pl.read_parquet(data_io)

            for i in range(0, len(df), batch_size):
                yield df.slice(i, batch_size)
        else:
            # Read with pandas
            df = pd.read_parquet(data_io)

            for i in range(0, len(df), batch_size):
                yield df.iloc[i : i + batch_size]

    async def _serialize_data(
        self, data: Union[pd.DataFrame, pl.DataFrame], file_format: str
    ) -> bytes:
        """Serialize data to bytes."""
        if file_format == "csv":
            if isinstance(data, pl.DataFrame):
                return data.write_csv().encode(self.config.encoding)
            else:
                return data.to_csv(index=False).encode(self.config.encoding)

        elif file_format == "json":
            if isinstance(data, pl.DataFrame):
                data = data.to_pandas()

            return data.to_json(orient="records").encode(self.config.encoding)

        elif file_format == "parquet":
            buffer = io.BytesIO()

            if isinstance(data, pl.DataFrame):
                data.write_parquet(buffer)
            else:
                data.to_parquet(buffer, index=False)

            return buffer.getvalue()

        else:
            raise ValueError(f"Unsupported format for serialization: {file_format}")

    def _detect_format_from_key(self, object_key: str) -> str:
        """Detect file format from object key."""
        key_lower = object_key.lower()

        if key_lower.endswith(".csv"):
            return "csv"
        elif key_lower.endswith(".json") or key_lower.endswith(".jsonl"):
            return "json"
        elif key_lower.endswith(".parquet"):
            return "parquet"
        else:
            # Default to CSV
            return "csv"

    def _matches_pattern(self, object_key: str, pattern: str) -> bool:
        """Check if object key matches pattern."""
        import fnmatch

        return fnmatch.fnmatch(object_key, pattern)

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the object storage connection."""
        if not self.is_connected:
            raise RuntimeError("Object storage not connected")

        # Calculate total size
        total_size = 0
        for object_key in self.object_list:
            try:
                stat = self.client.stat_object(self.config.bucket_name, object_key)
                total_size += stat.size
            except S3Error:
                continue

        return {
            "storage_type": self.config.storage_type,
            "bucket": self.config.bucket_name,
            "object_count": len(self.object_list),
            "total_size": total_size,
            "prefix": self.config.prefix,
            "file_pattern": self.config.file_pattern,
            "connected": self.is_connected,
        }
