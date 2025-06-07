"""
Database Connector for data pipeline ingestion.

This module provides database connectivity for various database systems
including PostgreSQL, MySQL, SQLite, and others.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncpg

import structlog
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    database_type: str = Field(..., description="Database type (postgresql, mysql, sqlite, etc.)")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: str = Field(..., description="Database name")
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password")

    # Connection options
    connection_string: Optional[str] = Field(None, description="Custom connection string")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")

    # SSL options
    ssl_mode: Optional[str] = Field(None, description="SSL mode")
    ssl_cert: Optional[str] = Field(None, description="SSL certificate path")
    ssl_key: Optional[str] = Field(None, description="SSL key path")
    ssl_ca: Optional[str] = Field(None, description="SSL CA path")

class DatabaseConnector:
    """
    Database connector for data pipeline ingestion.

    Supports various database systems and provides both sync and async operations.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the database connector.

        Args:
            logger: Logger instance
        """
        self.logger = logger or structlog.get_logger("database_connector")

        # Connection state
        self.config: Optional[DatabaseConfig] = None
        self.engine = None
        self.async_engine = None
        self.connection = None
        self.is_connected = False

        self.logger.info("Database connector initialized")

    async def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to the database.

        Args:
            config: Database configuration
        """
        self.config = DatabaseConfig(**config)

        try:
            # Build connection string if not provided
            if not self.config.connection_string:
                connection_string = self._build_connection_string()
            else:
                connection_string = self.config.connection_string

            # Create async engine
            self.async_engine = create_async_engine(
                connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                echo=False
            )

            # Test connection
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            self.is_connected = True

            self.logger.info(
                "Database connected",
                database_type=self.config.database_type,
                database=self.config.database,
                host=self.config.host
            )

        except Exception as e:
            self.logger.error(
                "Database connection failed",
                error=str(e),
                database_type=self.config.database_type,
                database=self.config.database
            )
            raise e

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()

            if self.engine:
                self.engine.dispose()

            self.is_connected = False

            self.logger.info("Database disconnected")

        except Exception as e:
            self.logger.error("Database disconnection error", error=str(e))

    async def read_batches(
        self,
        batch_size: int = 10000,
        query: Optional[str] = None,
        table: Optional[str] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """
        Read data in batches from the database.

        Args:
            batch_size: Number of records per batch
            query: Custom SQL query
            table: Table name (if not using custom query)
            columns: Columns to select
            where_clause: WHERE clause
            order_by: ORDER BY clause
            **kwargs: Additional parameters

        Yields:
            DataFrames containing batch data
        """
        if not self.is_connected:
            raise RuntimeError("Database not connected")

        try:
            # Build query if not provided
            if not query:
                query = self._build_select_query(table, columns, where_clause, order_by)

            # Use pandas for data reading with chunking
            use_polars = kwargs.get("use_polars", False)

            if use_polars:
                # Read with polars (if supported)
                async with self.async_engine.begin() as conn:
                    result = await conn.execute(text(query))

                    while True:
                        rows = result.fetchmany(batch_size)
                        if not rows:
                            break

                        # Convert to polars DataFrame
                        columns = list(result.keys())
                        data = [dict(zip(columns, row)) for row in rows]
                        df = pl.DataFrame(data)

                        yield df
            else:
                # Read with pandas
                connection_string = self._build_sync_connection_string()

                for chunk in pd.read_sql(
                    query,
                    connection_string,
                    chunksize=batch_size
                ):
                    yield chunk

        except Exception as e:
            self.logger.error(
                "Database read error",
                error=str(e),
                query=query[:100] if query else None
            )
            raise e

    async def write_batch(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        table: str,
        if_exists: str = "append",
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Write a batch of data to the database.

        Args:
            data: Data to write
            table: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
            **kwargs: Additional parameters
        """
        if not self.is_connected:
            raise RuntimeError("Database not connected")

        try:
            # Convert polars to pandas if needed
            if isinstance(data, pl.DataFrame):
                data = data.to_pandas()

            # Write data
            connection_string = self._build_sync_connection_string()

            data.to_sql(
                table,
                connection_string,
                if_exists=if_exists,
                index=index,
                method="multi",  # Use multi-row insert for better performance
                chunksize=kwargs.get("chunksize", 10000)
            )

            self.logger.debug(
                "Batch written to database",
                table=table,
                records=len(data),
                if_exists=if_exists
            )

        except Exception as e:
            self.logger.error(
                "Database write error",
                error=str(e),
                table=table,
                records=len(data)
            )
            raise e

    async def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """
        Execute a custom query.

        Args:
            query: SQL query to execute
            parameters: Query parameters

        Returns:
            Query result
        """
        if not self.is_connected:
            raise RuntimeError("Database not connected")

        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(query), parameters or {})

                if result.returns_rows:
                    return result.fetchall()
                else:
                    return result.rowcount

        except Exception as e:
            self.logger.error(
                "Query execution error",
                error=str(e),
                query=query[:100]
            )
            raise e

    async def get_table_info(self, table: str) -> Dict[str, Any]:
        """
        Get information about a table.

        Args:
            table: Table name

        Returns:
            Table information
        """
        if not self.is_connected:
            raise RuntimeError("Database not connected")

        try:
            # Get column information
            if self.config.database_type == "postgresql":
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
                """
            elif self.config.database_type == "mysql":
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
                """
            else:
                # Generic query
                query = f"PRAGMA table_info({table})"

            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(query), {"table_name": table})
                columns = result.fetchall()

            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table}"
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(count_query))
                row_count = result.scalar()

            return {
                "table_name": table,
                "columns": [dict(row._mapping) for row in columns],
                "row_count": row_count
            }

        except Exception as e:
            self.logger.error(
                "Table info error",
                error=str(e),
                table=table
            )
            raise e

    def _build_connection_string(self) -> str:
        """Build database connection string."""
        if self.config.database_type == "postgresql":
            if self.config.host:
                return (
                    f"postgresql+asyncpg://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port or 5432}/{self.config.database}"
                )
            else:
                return f"postgresql+asyncpg:///{self.config.database}"

        elif self.config.database_type == "mysql":
            return (
                f"mysql+aiomysql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port or 3306}/{self.config.database}"
            )

        elif self.config.database_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.config.database}"

        else:
            raise ValueError(f"Unsupported database type: {self.config.database_type}")

    def _build_sync_connection_string(self) -> str:
        """Build synchronous database connection string for pandas."""
        if self.config.database_type == "postgresql":
            if self.config.host:
                return (
                    f"postgresql://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port or 5432}/{self.config.database}"
                )
            else:
                return f"postgresql:///{self.config.database}"

        elif self.config.database_type == "mysql":
            return (
                f"mysql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port or 3306}/{self.config.database}"
            )

        elif self.config.database_type == "sqlite":
            return f"sqlite:///{self.config.database}"

        else:
            raise ValueError(f"Unsupported database type: {self.config.database_type}")

    def _build_select_query(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None
    ) -> str:
        """Build SELECT query."""
        # Select columns
        if columns:
            column_list = ", ".join(columns)
        else:
            column_list = "*"

        query = f"SELECT {column_list} FROM {table}"

        # Add WHERE clause
        if where_clause:
            query += f" WHERE {where_clause}"

        # Add ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"

        return query

    async def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection is successful
        """
        try:
            if not self.is_connected:
                return False

            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            return True

        except Exception as e:
            self.logger.error("Connection test failed", error=str(e))
            return False
