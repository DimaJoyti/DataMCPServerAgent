"""
File Connector for data pipeline ingestion.

This module provides file system connectivity for various file formats
including CSV, JSON, Parquet, Excel, and others.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import pandas as pd
import polars as pl
import aiofiles
import json

import structlog
from pydantic import BaseModel, Field


class FileConfig(BaseModel):
    """File connection configuration."""
    file_path: str = Field(..., description="File path or directory path")
    file_format: str = Field(default="auto", description="File format (csv, json, parquet, excel, etc.)")
    encoding: str = Field(default="utf-8", description="File encoding")
    compression: Optional[str] = Field(None, description="Compression type (gzip, bz2, xz, etc.)")
    
    # CSV specific options
    delimiter: str = Field(default=",", description="CSV delimiter")
    header: Union[int, List[int], None] = Field(default=0, description="Header row(s)")
    skip_rows: int = Field(default=0, description="Number of rows to skip")
    
    # JSON specific options
    json_lines: bool = Field(default=False, description="Whether JSON is line-delimited")
    json_orient: str = Field(default="records", description="JSON orientation")
    
    # Parquet specific options
    parquet_engine: str = Field(default="pyarrow", description="Parquet engine")
    
    # General options
    columns: Optional[List[str]] = Field(None, description="Columns to read")
    dtype: Optional[Dict[str, str]] = Field(None, description="Data types for columns")
    parse_dates: Optional[List[str]] = Field(None, description="Columns to parse as dates")


class FileConnector:
    """
    File connector for data pipeline ingestion.
    
    Supports various file formats and provides both sync and async operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the file connector.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or structlog.get_logger("file_connector")
        
        # Connection state
        self.config: Optional[FileConfig] = None
        self.is_connected = False
        self.file_list: List[Path] = []
        
        self.logger.info("File connector initialized")
    
    async def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to the file system.
        
        Args:
            config: File configuration
        """
        self.config = FileConfig(**config)
        
        try:
            file_path = Path(self.config.file_path)
            
            if file_path.is_file():
                # Single file
                self.file_list = [file_path]
            elif file_path.is_dir():
                # Directory - find matching files
                self.file_list = self._find_files_in_directory(file_path)
            else:
                # Pattern matching
                parent_dir = file_path.parent
                pattern = file_path.name
                self.file_list = list(parent_dir.glob(pattern))
            
            if not self.file_list:
                raise FileNotFoundError(f"No files found matching: {self.config.file_path}")
            
            # Detect file format if auto
            if self.config.file_format == "auto":
                self.config.file_format = self._detect_file_format(self.file_list[0])
            
            self.is_connected = True
            
            self.logger.info(
                "File connector connected",
                file_path=self.config.file_path,
                file_format=self.config.file_format,
                file_count=len(self.file_list)
            )
            
        except Exception as e:
            self.logger.error(
                "File connection failed",
                error=str(e),
                file_path=self.config.file_path
            )
            raise e
    
    async def disconnect(self) -> None:
        """Disconnect from the file system."""
        self.is_connected = False
        self.file_list = []
        self.logger.info("File connector disconnected")
    
    async def read_batches(
        self,
        batch_size: int = 10000,
        **kwargs
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """
        Read data in batches from files.
        
        Args:
            batch_size: Number of records per batch
            **kwargs: Additional parameters
            
        Yields:
            DataFrames containing batch data
        """
        if not self.is_connected:
            raise RuntimeError("File connector not connected")
        
        use_polars = kwargs.get("use_polars", False)
        
        try:
            for file_path in self.file_list:
                self.logger.debug("Reading file", file_path=str(file_path))
                
                if self.config.file_format == "csv":
                    async for batch in self._read_csv_batches(file_path, batch_size, use_polars):
                        yield batch
                
                elif self.config.file_format == "json":
                    async for batch in self._read_json_batches(file_path, batch_size, use_polars):
                        yield batch
                
                elif self.config.file_format == "parquet":
                    async for batch in self._read_parquet_batches(file_path, batch_size, use_polars):
                        yield batch
                
                elif self.config.file_format == "excel":
                    async for batch in self._read_excel_batches(file_path, batch_size, use_polars):
                        yield batch
                
                else:
                    raise ValueError(f"Unsupported file format: {self.config.file_format}")
                    
        except Exception as e:
            self.logger.error(
                "File read error",
                error=str(e),
                file_format=self.config.file_format
            )
            raise e
    
    async def write_batch(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        file_path: Optional[str] = None,
        mode: str = "append",
        **kwargs
    ) -> None:
        """
        Write a batch of data to file.
        
        Args:
            data: Data to write
            file_path: Target file path (uses config path if not provided)
            mode: Write mode ('append', 'overwrite')
            **kwargs: Additional parameters
        """
        if not file_path:
            file_path = self.config.file_path
        
        target_path = Path(file_path)
        
        try:
            # Ensure directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert polars to pandas if needed for certain formats
            if isinstance(data, pl.DataFrame) and self.config.file_format in ["excel"]:
                data = data.to_pandas()
            
            if self.config.file_format == "csv":
                await self._write_csv(data, target_path, mode)
            
            elif self.config.file_format == "json":
                await self._write_json(data, target_path, mode)
            
            elif self.config.file_format == "parquet":
                await self._write_parquet(data, target_path, mode)
            
            elif self.config.file_format == "excel":
                await self._write_excel(data, target_path, mode)
            
            else:
                raise ValueError(f"Unsupported file format for writing: {self.config.file_format}")
            
            self.logger.debug(
                "Batch written to file",
                file_path=str(target_path),
                records=len(data),
                mode=mode
            )
            
        except Exception as e:
            self.logger.error(
                "File write error",
                error=str(e),
                file_path=str(target_path),
                records=len(data)
            )
            raise e
    
    async def _read_csv_batches(
        self,
        file_path: Path,
        batch_size: int,
        use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read CSV file in batches."""
        if use_polars:
            # Polars lazy reading
            df = pl.scan_csv(
                str(file_path),
                separator=self.config.delimiter,
                encoding=self.config.encoding,
                skip_rows=self.config.skip_rows,
                has_header=self.config.header is not None
            )
            
            # Process in batches
            total_rows = df.select(pl.count()).collect().item()
            
            for offset in range(0, total_rows, batch_size):
                batch = df.slice(offset, batch_size).collect()
                yield batch
        else:
            # Pandas chunked reading
            for chunk in pd.read_csv(
                file_path,
                delimiter=self.config.delimiter,
                encoding=self.config.encoding,
                skiprows=self.config.skip_rows,
                header=self.config.header,
                usecols=self.config.columns,
                dtype=self.config.dtype,
                parse_dates=self.config.parse_dates,
                chunksize=batch_size,
                compression=self.config.compression
            ):
                yield chunk
    
    async def _read_json_batches(
        self,
        file_path: Path,
        batch_size: int,
        use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read JSON file in batches."""
        if self.config.json_lines:
            # Line-delimited JSON
            batch_data = []
            
            async with aiofiles.open(file_path, 'r', encoding=self.config.encoding) as f:
                async for line in f:
                    try:
                        record = json.loads(line.strip())
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
        else:
            # Regular JSON
            async with aiofiles.open(file_path, 'r', encoding=self.config.encoding) as f:
                content = await f.read()
                data = json.loads(content)
                
                if isinstance(data, list):
                    # Process in batches
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        if use_polars:
                            yield pl.DataFrame(batch)
                        else:
                            yield pd.DataFrame(batch)
                else:
                    # Single object
                    if use_polars:
                        yield pl.DataFrame([data])
                    else:
                        yield pd.DataFrame([data])
    
    async def _read_parquet_batches(
        self,
        file_path: Path,
        batch_size: int,
        use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read Parquet file in batches."""
        if use_polars:
            # Polars lazy reading
            df = pl.scan_parquet(str(file_path))
            total_rows = df.select(pl.count()).collect().item()
            
            for offset in range(0, total_rows, batch_size):
                batch = df.slice(offset, batch_size).collect()
                yield batch
        else:
            # Pandas reading
            df = pd.read_parquet(
                file_path,
                engine=self.config.parquet_engine,
                columns=self.config.columns
            )
            
            # Process in batches
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i + batch_size]
    
    async def _read_excel_batches(
        self,
        file_path: Path,
        batch_size: int,
        use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read Excel file in batches."""
        # Excel reading (pandas only)
        df = pd.read_excel(
            file_path,
            header=self.config.header,
            skiprows=self.config.skip_rows,
            usecols=self.config.columns,
            dtype=self.config.dtype,
            parse_dates=self.config.parse_dates
        )
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            if use_polars:
                yield pl.from_pandas(batch)
            else:
                yield batch
    
    async def _write_csv(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        file_path: Path,
        mode: str
    ) -> None:
        """Write data to CSV file."""
        write_header = mode == "overwrite" or not file_path.exists()
        write_mode = "w" if mode == "overwrite" else "a"
        
        if isinstance(data, pl.DataFrame):
            data.write_csv(
                str(file_path),
                separator=self.config.delimiter,
                include_header=write_header
            )
        else:
            data.to_csv(
                file_path,
                mode=write_mode,
                header=write_header,
                index=False,
                sep=self.config.delimiter,
                encoding=self.config.encoding,
                compression=self.config.compression
            )
    
    async def _write_json(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        file_path: Path,
        mode: str
    ) -> None:
        """Write data to JSON file."""
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        
        if self.config.json_lines:
            # Line-delimited JSON
            write_mode = "w" if mode == "overwrite" else "a"
            async with aiofiles.open(file_path, write_mode, encoding=self.config.encoding) as f:
                for record in data.to_dict(orient="records"):
                    await f.write(json.dumps(record) + "\n")
        else:
            # Regular JSON
            data.to_json(
                file_path,
                orient=self.config.json_orient,
                compression=self.config.compression
            )
    
    async def _write_parquet(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        file_path: Path,
        mode: str
    ) -> None:
        """Write data to Parquet file."""
        if isinstance(data, pl.DataFrame):
            data.write_parquet(str(file_path))
        else:
            data.to_parquet(
                file_path,
                engine=self.config.parquet_engine,
                compression=self.config.compression
            )
    
    async def _write_excel(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        file_path: Path,
        mode: str
    ) -> None:
        """Write data to Excel file."""
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        
        data.to_excel(file_path, index=False)
    
    def _find_files_in_directory(self, directory: Path) -> List[Path]:
        """Find files in directory based on format."""
        if self.config.file_format == "auto":
            # Common data file extensions
            patterns = ["*.csv", "*.json", "*.parquet", "*.xlsx", "*.xls"]
        else:
            # Specific format
            if self.config.file_format == "csv":
                patterns = ["*.csv"]
            elif self.config.file_format == "json":
                patterns = ["*.json", "*.jsonl"]
            elif self.config.file_format == "parquet":
                patterns = ["*.parquet"]
            elif self.config.file_format == "excel":
                patterns = ["*.xlsx", "*.xls"]
            else:
                patterns = [f"*.{self.config.file_format}"]
        
        files = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        
        return sorted(files)
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".csv":
            return "csv"
        elif suffix in [".json", ".jsonl"]:
            return "json"
        elif suffix == ".parquet":
            return "parquet"
        elif suffix in [".xlsx", ".xls"]:
            return "excel"
        else:
            # Default to CSV
            return "csv"
    
    async def get_file_info(self) -> Dict[str, Any]:
        """Get information about connected files."""
        if not self.is_connected:
            raise RuntimeError("File connector not connected")
        
        file_info = []
        for file_path in self.file_list:
            stat = file_path.stat()
            file_info.append({
                "path": str(file_path),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "format": self._detect_file_format(file_path)
            })
        
        return {
            "file_count": len(self.file_list),
            "total_size": sum(info["size"] for info in file_info),
            "files": file_info
        }
