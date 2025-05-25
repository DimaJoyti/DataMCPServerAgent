"""
API Connector for data pipeline ingestion.

This module provides API connectivity for REST APIs, GraphQL,
and other web-based data sources.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import pandas as pd
import polars as pl
import aiohttp
import json
from urllib.parse import urljoin, urlencode

import structlog
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API connection configuration."""
    base_url: str = Field(..., description="Base URL for the API")
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    
    # Authentication
    auth_type: Optional[str] = Field(None, description="Authentication type (bearer, basic, api_key)")
    api_key: Optional[str] = Field(None, description="API key")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[str] = Field(None, description="Password for basic auth")
    bearer_token: Optional[str] = Field(None, description="Bearer token")
    
    # Request configuration
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Pagination
    pagination_type: Optional[str] = Field(None, description="Pagination type (offset, cursor, page)")
    page_size: int = Field(default=100, description="Number of records per page")
    page_param: str = Field(default="page", description="Page parameter name")
    size_param: str = Field(default="size", description="Size parameter name")
    offset_param: str = Field(default="offset", description="Offset parameter name")
    cursor_param: str = Field(default="cursor", description="Cursor parameter name")
    
    # Response parsing
    data_path: Optional[str] = Field(None, description="JSON path to data array")
    next_page_path: Optional[str] = Field(None, description="JSON path to next page info")
    total_count_path: Optional[str] = Field(None, description="JSON path to total count")


class APIConnector:
    """
    API connector for data pipeline ingestion.
    
    Supports REST APIs with various authentication methods and pagination strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the API connector.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or structlog.get_logger("api_connector")
        
        # Connection state
        self.config: Optional[APIConfig] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        
        self.logger.info("API connector initialized")
    
    async def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to the API.
        
        Args:
            config: API configuration
        """
        self.config = APIConfig(**config)
        
        try:
            # Prepare headers
            headers = self.config.headers.copy()
            
            # Add authentication headers
            if self.config.auth_type == "bearer" and self.config.bearer_token:
                headers["Authorization"] = f"Bearer {self.config.bearer_token}"
            elif self.config.auth_type == "api_key" and self.config.api_key:
                headers[self.config.api_key_header] = self.config.api_key
            
            # Create session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            if self.config.auth_type == "basic" and self.config.username and self.config.password:
                auth = aiohttp.BasicAuth(self.config.username, self.config.password)
                self.session = aiohttp.ClientSession(
                    headers=headers,
                    timeout=timeout,
                    auth=auth
                )
            else:
                self.session = aiohttp.ClientSession(
                    headers=headers,
                    timeout=timeout
                )
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            
            self.logger.info(
                "API connected",
                base_url=self.config.base_url,
                endpoint=self.config.endpoint
            )
            
        except Exception as e:
            self.logger.error(
                "API connection failed",
                error=str(e),
                base_url=self.config.base_url
            )
            raise e
    
    async def disconnect(self) -> None:
        """Disconnect from the API."""
        try:
            if self.session:
                await self.session.close()
            
            self.is_connected = False
            self.logger.info("API disconnected")
            
        except Exception as e:
            self.logger.error("API disconnection error", error=str(e))
    
    async def read_batches(
        self,
        batch_size: int = 100,
        **kwargs
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """
        Read data in batches from the API.
        
        Args:
            batch_size: Number of records per batch
            **kwargs: Additional parameters
            
        Yields:
            DataFrames containing batch data
        """
        if not self.is_connected:
            raise RuntimeError("API not connected")
        
        use_polars = kwargs.get("use_polars", False)
        
        try:
            if self.config.pagination_type:
                # Paginated API
                async for batch in self._read_paginated_data(batch_size, use_polars):
                    yield batch
            else:
                # Single request API
                data = await self._make_request()
                
                # Extract data from response
                if self.config.data_path:
                    data = self._extract_data_from_path(data, self.config.data_path)
                
                # Convert to DataFrame
                if isinstance(data, list):
                    # Process in batches
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        if use_polars:
                            yield pl.DataFrame(batch)
                        else:
                            yield pd.DataFrame(batch)
                else:
                    # Single record
                    if use_polars:
                        yield pl.DataFrame([data])
                    else:
                        yield pd.DataFrame([data])
                        
        except Exception as e:
            self.logger.error(
                "API read error",
                error=str(e),
                endpoint=self.config.endpoint
            )
            raise e
    
    async def write_batch(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        **kwargs
    ) -> None:
        """
        Write a batch of data to the API.
        
        Args:
            data: Data to write
            **kwargs: Additional parameters
        """
        if not self.is_connected:
            raise RuntimeError("API not connected")
        
        try:
            # Convert to records
            if isinstance(data, pl.DataFrame):
                records = data.to_dicts()
            else:
                records = data.to_dict(orient="records")
            
            # Send data in batches
            batch_size = kwargs.get("batch_size", 100)
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                await self._send_batch(batch)
            
            self.logger.debug(
                "Batch written to API",
                records=len(data),
                endpoint=self.config.endpoint
            )
            
        except Exception as e:
            self.logger.error(
                "API write error",
                error=str(e),
                records=len(data)
            )
            raise e
    
    async def _test_connection(self) -> None:
        """Test API connection."""
        try:
            url = urljoin(self.config.base_url, self.config.endpoint)
            
            # Make a simple request to test connectivity
            async with self.session.request(
                "HEAD" if self.config.method == "GET" else self.config.method,
                url,
                params=self.config.params
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API test failed with status {response.status}"
                    )
                    
        except Exception as e:
            self.logger.error("API connection test failed", error=str(e))
            raise e
    
    async def _read_paginated_data(
        self,
        batch_size: int,
        use_polars: bool
    ) -> AsyncGenerator[Union[pd.DataFrame, pl.DataFrame], None]:
        """Read data from paginated API."""
        page = 1
        offset = 0
        cursor = None
        has_more = True
        
        while has_more:
            # Prepare pagination parameters
            params = self.config.params.copy()
            
            if self.config.pagination_type == "page":
                params[self.config.page_param] = page
                params[self.config.size_param] = batch_size
            elif self.config.pagination_type == "offset":
                params[self.config.offset_param] = offset
                params[self.config.size_param] = batch_size
            elif self.config.pagination_type == "cursor" and cursor:
                params[self.config.cursor_param] = cursor
                params[self.config.size_param] = batch_size
            
            # Make request
            response_data = await self._make_request(params)
            
            # Extract data
            if self.config.data_path:
                data = self._extract_data_from_path(response_data, self.config.data_path)
            else:
                data = response_data
            
            if not data or len(data) == 0:
                break
            
            # Convert to DataFrame
            if use_polars:
                yield pl.DataFrame(data)
            else:
                yield pd.DataFrame(data)
            
            # Check for next page
            if self.config.pagination_type == "page":
                page += 1
                has_more = len(data) == batch_size
            elif self.config.pagination_type == "offset":
                offset += len(data)
                has_more = len(data) == batch_size
            elif self.config.pagination_type == "cursor":
                if self.config.next_page_path:
                    cursor = self._extract_data_from_path(response_data, self.config.next_page_path)
                    has_more = cursor is not None
                else:
                    has_more = len(data) == batch_size
            
            # Check total count if available
            if self.config.total_count_path:
                total_count = self._extract_data_from_path(response_data, self.config.total_count_path)
                if total_count and offset >= total_count:
                    has_more = False
    
    async def _make_request(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make HTTP request to API."""
        url = urljoin(self.config.base_url, self.config.endpoint)
        request_params = params or self.config.params
        
        async with self.session.request(
            self.config.method,
            url,
            params=request_params
        ) as response:
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                return await response.json()
            else:
                text = await response.text()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"data": text}
    
    async def _send_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Send a batch of data to API."""
        url = urljoin(self.config.base_url, self.config.endpoint)
        
        async with self.session.request(
            "POST",  # Assume POST for writing
            url,
            json=batch,
            params=self.config.params
        ) as response:
            response.raise_for_status()
    
    def _extract_data_from_path(self, data: Any, path: str) -> Any:
        """Extract data from nested JSON using dot notation path."""
        try:
            current = data
            for key in path.split("."):
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Get information about the API connection."""
        if not self.is_connected:
            raise RuntimeError("API not connected")
        
        return {
            "base_url": self.config.base_url,
            "endpoint": self.config.endpoint,
            "method": self.config.method,
            "auth_type": self.config.auth_type,
            "pagination_type": self.config.pagination_type,
            "page_size": self.config.page_size,
            "connected": self.is_connected
        }
