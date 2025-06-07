"""
Enhanced Bright Data MCP Client

This module provides a production-ready client for Bright Data MCP with:
- Automatic retry with exponential backoff
- Circuit breaker pattern
- Connection pooling
- Request/response compression
- Intelligent failover
- Comprehensive monitoring
- Integration with caching and rate limiting
"""

import asyncio
import aiohttp
import json
import gzip
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

from .config import BrightDataConfig, get_config
from .cache_manager import CacheManager
from .rate_limiter import RateLimiter
from .error_handler import (
    BrightDataErrorHandler, BrightDataException, NetworkException,
    AuthenticationException, RateLimitException, ServerException,
    TimeoutException, ErrorCategory
)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker implementation"""
    failure_threshold: int
    recovery_timeout: int
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0
    success_count: int = 0

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record successful request"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

@dataclass
class RequestMetrics:
    """Request metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0
    min_response_time: float = float('inf')
    max_response_time: float = 0

    def record_request(self, response_time: float, success: bool) -> None:
        """Record request metrics"""
        self.total_requests += 1
        self.total_response_time += response_time

        if response_time < self.min_response_time:
            self.min_response_time = response_time
        if response_time > self.max_response_time:
            self.max_response_time = response_time

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_average_response_time(self) -> float:
        """Get average response time"""
        return self.total_response_time / self.total_requests if self.total_requests > 0 else 0

    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0

class EnhancedBrightDataClient:
    """Enhanced Bright Data MCP client with advanced features"""

    def __init__(self, config: Optional[BrightDataConfig] = None,
                 cache_manager: Optional[CacheManager] = None,
                 rate_limiter: Optional[RateLimiter] = None,
                 error_handler: Optional[BrightDataErrorHandler] = None):

        self.config = config or get_config()
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter
        self.error_handler = error_handler or BrightDataErrorHandler()

        self.logger = logging.getLogger(__name__)

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker.failure_threshold,
            recovery_timeout=self.config.circuit_breaker.recovery_timeout
        )

        # Connection management
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

        # Metrics
        self.metrics = RequestMetrics()

        # Endpoints for failover
        self.endpoints = [self.config.api_base_url]
        self.current_endpoint_index = 0

        # Request hooks
        self.before_request_hooks: List[Callable] = []
        self.after_request_hooks: List[Callable] = []

    async def _initialize_session(self) -> None:
        """Initialize HTTP session with connection pooling"""
        if self.config.enable_connection_pooling:
            self._connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests,
                limit_per_host=self.config.max_concurrent_requests,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )

        timeout = aiohttp.ClientTimeout(total=self.config.api_timeout)

        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={
                'User-Agent': self.config.user_agent,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            await self._initialize_session()
        return self._session

    def _get_current_endpoint(self) -> str:
        """Get current endpoint for requests"""
        return self.endpoints[self.current_endpoint_index]

    def _rotate_endpoint(self) -> None:
        """Rotate to next endpoint for failover"""
        if len(self.endpoints) > 1:
            self.current_endpoint_index = (self.current_endpoint_index + 1) % len(self.endpoints)
            self.logger.info(f"Rotated to endpoint: {self._get_current_endpoint()}")

    def _should_compress_request(self, data: str) -> bool:
        """Check if request should be compressed"""
        return (self.config.enable_compression and
                len(data.encode('utf-8')) > 1024)

    def _compress_request_data(self, data: str) -> bytes:
        """Compress request data"""
        return gzip.compress(data.encode('utf-8'))

    def _decompress_response_data(self, data: bytes) -> str:
        """Decompress response data"""
        return gzip.decompress(data).decode('utf-8')

    async def _execute_hooks(self, hooks: List[Callable], *args, **kwargs) -> None:
        """Execute request hooks"""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook execution failed: {e}")

    def add_before_request_hook(self, hook: Callable) -> None:
        """Add before request hook"""
        self.before_request_hooks.append(hook)

    def add_after_request_hook(self, hook: Callable) -> None:
        """Add after request hook"""
        self.after_request_hooks.append(hook)

    def add_endpoint(self, endpoint: str) -> None:
        """Add endpoint for failover"""
        if endpoint not in self.endpoints:
            self.endpoints.append(endpoint)

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": self.metrics.get_success_rate(),
            "average_response_time": self.metrics.get_average_response_time(),
            "min_response_time": self.metrics.min_response_time if self.metrics.min_response_time != float('inf') else 0,
            "max_response_time": self.metrics.max_response_time,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "current_endpoint": self._get_current_endpoint(),
            "available_endpoints": len(self.endpoints)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = time.time()
            # Make a simple request to check connectivity
            await self._make_single_request("GET", f"{self._get_current_endpoint()}/health")
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "endpoint": self._get_current_endpoint(),
                "circuit_breaker_state": self.circuit_breaker.state.value
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "endpoint": self._get_current_endpoint(),
                "circuit_breaker_state": self.circuit_breaker.state.value
            }

    async def _make_request_with_retry(self, method: str, url: str,
                                     data: Optional[Dict[str, Any]] = None,
                                     headers: Optional[Dict[str, str]] = None,
                                     user_id: str = "default") -> Dict[str, Any]:
        """Make HTTP request with retry logic"""

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise BrightDataException(
                "Circuit breaker is open",
                ErrorCategory.SERVER_ERROR,
                context={"circuit_state": self.circuit_breaker.state.value}
            )

        # Rate limiting
        if self.rate_limiter:
            if not await self.rate_limiter.acquire(user_id, url):
                raise RateLimitException(
                    "Rate limit exceeded",
                    context={"user_id": user_id, "endpoint": url}
                )

        last_exception = None

        for attempt in range(self.config.retry.max_retries + 1):
            start_time = time.time()

            try:
                # Execute before request hooks
                await self._execute_hooks(self.before_request_hooks, method, url, data, headers)

                # Make the actual request
                response_data = await self._make_single_request(method, url, data, headers)

                # Record success
                response_time = time.time() - start_time
                self.metrics.record_request(response_time, True)
                self.circuit_breaker.record_success()

                if self.rate_limiter:
                    self.rate_limiter.record_response(user_id, response_time, True)

                # Execute after request hooks
                await self._execute_hooks(self.after_request_hooks, method, url, response_data, response_time)

                return response_data

            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time

                # Handle error
                error_info = await self.error_handler.handle_error(e, {
                    "method": method,
                    "url": url,
                    "attempt": attempt,
                    "user_id": user_id
                })

                # Record failure
                self.metrics.record_request(response_time, False)
                self.circuit_breaker.record_failure()

                if self.rate_limiter:
                    self.rate_limiter.record_response(user_id, response_time, False)

                # Check if we should retry
                if attempt < self.config.retry.max_retries and error_info.recoverable:
                    # Calculate delay
                    delay = min(
                        self.config.retry.base_delay * (self.config.retry.exponential_base ** attempt),
                        self.config.retry.max_delay
                    )

                    # Add jitter if enabled
                    if self.config.retry.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)

                    self.logger.info(f"Retrying request in {delay:.2f} seconds (attempt {attempt + 1})")
                    await asyncio.sleep(delay)

                    # Try failover if available
                    if len(self.endpoints) > 1:
                        self._rotate_endpoint()
                        url = url.replace(self.endpoints[self.current_endpoint_index - 1],
                                        self._get_current_endpoint())
                else:
                    break

        # All retries exhausted
        if last_exception:
            raise last_exception

        raise BrightDataException("Request failed after all retries")

    async def _make_single_request(self, method: str, url: str,
                                 data: Optional[Dict[str, Any]] = None,
                                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a single HTTP request"""
        session = await self._get_session()

        # Prepare headers
        request_headers = headers or {}
        if self.config.api_key:
            request_headers['Authorization'] = f'Bearer {self.config.api_key}'

        # Prepare data
        request_data = None
        if data:
            json_data = json.dumps(data)

            # Compress if needed
            if self._should_compress_request(json_data):
                request_data = self._compress_request_data(json_data)
                request_headers['Content-Encoding'] = 'gzip'
            else:
                request_data = json_data.encode('utf-8')

        # Make request
        async with session.request(method, url, data=request_data, headers=request_headers) as response:
            # Check for HTTP errors
            if response.status == 401:
                raise AuthenticationException(
                    "Authentication failed",
                    context={"status_code": response.status, "url": url}
                )
            elif response.status == 429:
                retry_after = response.headers.get('Retry-After')
                raise RateLimitException(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None,
                    context={"status_code": response.status, "url": url}
                )
            elif response.status >= 500:
                raise ServerException(
                    f"Server error: {response.status}",
                    status_code=response.status,
                    context={"url": url}
                )
            elif response.status >= 400:
                raise BrightDataException(
                    f"Client error: {response.status}",
                    ErrorCategory.CLIENT_ERROR,
                    context={"status_code": response.status, "url": url}
                )

            # Read response
            response_data = await response.read()

            # Decompress if needed
            if response.headers.get('Content-Encoding') == 'gzip':
                response_text = self._decompress_response_data(response_data)
            else:
                response_text = response_data.decode('utf-8')

            # Parse JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                raise BrightDataException(
                    f"Invalid JSON response: {e}",
                    ErrorCategory.SERVER_ERROR,
                    context={"response_text": response_text[:500]}
                )

    # Public API methods

    async def scrape_url(self, url: str, user_id: str = "default",
                        use_cache: bool = True, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Scrape a URL with caching support"""
        cache_key = f"scrape:{url}"

        # Try cache first
        if use_cache and self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Make request
        endpoint = f"{self._get_current_endpoint()}/scrape"
        data = {"url": url}

        result = await self._make_request_with_retry("POST", endpoint, data, user_id=user_id)

        # Cache result
        if use_cache and self.cache_manager:
            await self.cache_manager.set(cache_key, result, cache_ttl)

        return result

    async def search_web(self, query: str, count: int = 10, user_id: str = "default",
                        use_cache: bool = True, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Search the web with caching support"""
        cache_key = f"search:{query}:{count}"

        # Try cache first
        if use_cache and self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Make request
        endpoint = f"{self._get_current_endpoint()}/search"
        data = {"query": query, "count": count}

        result = await self._make_request_with_retry("POST", endpoint, data, user_id=user_id)

        # Cache result
        if use_cache and self.cache_manager:
            await self.cache_manager.set(cache_key, result, cache_ttl)

        return result

    async def get_product_data(self, product_url: str, user_id: str = "default",
                              use_cache: bool = True, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Get product data with caching support"""
        cache_key = f"product:{product_url}"

        # Try cache first
        if use_cache and self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Make request
        endpoint = f"{self._get_current_endpoint()}/product"
        data = {"url": product_url}

        result = await self._make_request_with_retry("POST", endpoint, data, user_id=user_id)

        # Cache result
        if use_cache and self.cache_manager:
            await self.cache_manager.set(cache_key, result, cache_ttl)

        return result

    async def get_social_media_data(self, social_url: str, user_id: str = "default",
                                   use_cache: bool = True, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Get social media data with caching support"""
        cache_key = f"social:{social_url}"

        # Try cache first
        if use_cache and self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Make request
        endpoint = f"{self._get_current_endpoint()}/social"
        data = {"url": social_url}

        result = await self._make_request_with_retry("POST", endpoint, data, user_id=user_id)

        # Cache result
        if use_cache and self.cache_manager:
            await self.cache_manager.set(cache_key, result, cache_ttl)

        return result

    async def close(self) -> None:
        """Close client connections"""
        if self._session and not self._session.closed:
            await self._session.close()

        if self._connector:
            await self._connector.close()

        if self.cache_manager:
            await self.cache_manager.close()
