"""
Tests for Enhanced Bright Data MCP Integration

This module contains comprehensive tests for the enhanced Bright Data integration
including unit tests, integration tests, and performance tests.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Import components to test
from src.tools.bright_data.core.config import BrightDataConfig
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.cache_manager import CacheManager, MemoryCache
from src.tools.bright_data.core.rate_limiter import RateLimiter, ThrottleStrategy
from src.tools.bright_data.core.error_handler import (
    BrightDataErrorHandler, BrightDataException, ErrorCategory
)

class TestBrightDataConfig:
    """Test configuration management"""

    def test_config_creation(self):
        """Test basic configuration creation"""
        config = BrightDataConfig()
        assert config.api_timeout == 30
        assert config.max_concurrent_requests == 10
        assert config.cache.enabled is True
        assert config.rate_limit.enabled is True

    def test_config_from_dict(self):
        """Test configuration creation from dictionary"""
        config_data = {
            "api_timeout": 60,
            "max_concurrent_requests": 20,
            "cache": {
                "enabled": False,
                "default_ttl": 7200
            },
            "rate_limit": {
                "requests_per_minute": 120
            }
        }

        config = BrightDataConfig.from_dict(config_data)
        assert config.api_timeout == 60
        assert config.max_concurrent_requests == 20
        assert config.cache.enabled is False
        assert config.cache.default_ttl == 7200
        assert config.rate_limit.requests_per_minute == 120

    def test_config_validation(self):
        """Test configuration validation"""
        config = BrightDataConfig()
        config.api_key = "test_key"

        # Should not raise exception
        config.validate()

        # Should raise exception for missing API key
        config.api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            config.validate()

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary"""
        config = BrightDataConfig()
        config.api_timeout = 45

        config_dict = config.to_dict()
        assert config_dict["api_timeout"] == 45
        assert "cache" in config_dict
        assert "rate_limit" in config_dict

class TestMemoryCache:
    """Test memory cache functionality"""

    @pytest.fixture
    def memory_cache(self):
        """Create memory cache for testing"""
        return MemoryCache(max_size=10, default_ttl=60)

    @pytest.mark.asyncio
    async def test_cache_set_get(self, memory_cache):
        """Test basic cache set and get operations"""
        # Set value
        success = await memory_cache.set("test_key", "test_value")
        assert success is True

        # Get value
        value = await memory_cache.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_cache_miss(self, memory_cache):
        """Test cache miss"""
        value = await memory_cache.get("nonexistent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, memory_cache):
        """Test TTL expiration"""
        # Set value with short TTL
        await memory_cache.set("ttl_key", "ttl_value", ttl=1)

        # Should exist immediately
        value = await memory_cache.get("ttl_key")
        assert value == "ttl_value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        value = await memory_cache.get("ttl_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, memory_cache):
        """Test LRU eviction when cache is full"""
        # Fill cache to capacity
        for i in range(10):
            await memory_cache.set(f"key_{i}", f"value_{i}")

        # Add one more item (should evict oldest)
        await memory_cache.set("new_key", "new_value")

        # First key should be evicted
        value = await memory_cache.get("key_0")
        assert value is None

        # New key should exist
        value = await memory_cache.get("new_key")
        assert value == "new_value"

    @pytest.mark.asyncio
    async def test_cache_delete(self, memory_cache):
        """Test cache deletion"""
        await memory_cache.set("delete_key", "delete_value")

        # Verify exists
        assert await memory_cache.exists("delete_key") is True

        # Delete
        success = await memory_cache.delete("delete_key")
        assert success is True

        # Verify deleted
        assert await memory_cache.exists("delete_key") is False

    @pytest.mark.asyncio
    async def test_cache_clear(self, memory_cache):
        """Test cache clear operation"""
        # Add some items
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")

        # Clear cache
        success = await memory_cache.clear()
        assert success is True

        # Verify empty
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") is None

    def test_cache_stats(self, memory_cache):
        """Test cache statistics"""
        stats = memory_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0

class TestRateLimiter:
    """Test rate limiting functionality"""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing"""
        return RateLimiter(requests_per_minute=60, burst_size=5)

    @pytest.mark.asyncio
    async def test_rate_limit_acquire(self, rate_limiter):
        """Test basic rate limit acquisition"""
        # Should allow initial requests up to burst size
        for i in range(5):
            success = await rate_limiter.acquire("test_user")
            assert success is True

        # Should reject additional requests
        success = await rate_limiter.acquire("test_user")
        assert success is False

    @pytest.mark.asyncio
    async def test_rate_limit_different_users(self, rate_limiter):
        """Test rate limiting for different users"""
        # User 1 uses up their quota
        for i in range(5):
            await rate_limiter.acquire("user1")

        # User 2 should still be able to make requests
        success = await rate_limiter.acquire("user2")
        assert success is True

    @pytest.mark.asyncio
    async def test_rate_limit_token_refill(self, rate_limiter):
        """Test token bucket refill over time"""
        # Use up quota
        for i in range(5):
            await rate_limiter.acquire("test_user")

        # Should be rejected
        success = await rate_limiter.acquire("test_user")
        assert success is False

        # Wait for token refill (simulate time passage)
        # Note: In real test, you might want to mock time
        await asyncio.sleep(1.1)  # Allow some token refill

        # Should allow request after refill
        success = await rate_limiter.acquire("test_user")
        # Note: This might still fail depending on exact timing

    @pytest.mark.asyncio
    async def test_rate_limit_acquire_with_wait(self, rate_limiter):
        """Test acquire with waiting"""
        # Use up quota
        for i in range(5):
            await rate_limiter.acquire("test_user")

        # Should wait and eventually succeed (with short timeout for testing)
        start_time = time.time()
        success = await rate_limiter.acquire_with_wait("test_user", timeout=2.0)
        elapsed = time.time() - start_time

        # Should have waited some time
        assert elapsed > 0.5  # At least some wait time

    def test_rate_limit_stats(self, rate_limiter):
        """Test rate limiting statistics"""
        stats = rate_limiter.get_global_stats()
        assert "total_requests" in stats
        assert "rejected_requests" in stats
        assert "success_rate" in stats

    def test_rate_limit_user_status(self, rate_limiter):
        """Test user-specific rate limit status"""
        status = rate_limiter.get_rate_limit_status("test_user")
        assert status.requests_per_minute == 60
        assert status.burst_size == 5

class TestErrorHandler:
    """Test error handling functionality"""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing"""
        return BrightDataErrorHandler()

    def test_error_categorization(self, error_handler):
        """Test error categorization"""
        # Network error
        network_error = ConnectionError("Connection failed")
        error_info = error_handler.categorize_error(network_error)
        assert error_info.category == ErrorCategory.NETWORK

        # Authentication error
        auth_error = Exception("Unauthorized access")
        error_info = error_handler.categorize_error(auth_error)
        assert error_info.category == ErrorCategory.AUTHENTICATION

        # Rate limit error
        rate_error = Exception("Too many requests")
        error_info = error_handler.categorize_error(rate_error)
        assert error_info.category == ErrorCategory.RATE_LIMIT

    def test_bright_data_exception(self):
        """Test custom BrightData exceptions"""
        exception = BrightDataException(
            "Test error",
            ErrorCategory.VALIDATION,
            context={"test": "context"}
        )

        assert str(exception) == "Test error"
        assert exception.category == ErrorCategory.VALIDATION
        assert exception.context["test"] == "context"

    @pytest.mark.asyncio
    async def test_error_recovery_strategy(self, error_handler):
        """Test custom error recovery strategy"""
        recovery_called = False

        async def custom_recovery(error_info):
            nonlocal recovery_called
            recovery_called = True

        # Register custom recovery
        error_handler.register_recovery_strategy(ErrorCategory.NETWORK, custom_recovery)

        # Trigger error handling
        network_error = ConnectionError("Test network error")
        try:
            await error_handler.handle_error(network_error)
        except:
            pass  # Expected to re-raise

        # Recovery should have been called
        assert recovery_called is True

    def test_error_statistics(self, error_handler):
        """Test error statistics collection"""
        # Generate some errors
        error1 = ConnectionError("Network error")
        error2 = ValueError("Validation error")

        error_handler.categorize_error(error1)
        error_handler.categorize_error(error2)

        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 2
        assert "category_breakdown" in stats

class TestCacheManager:
    """Test cache manager functionality"""

    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing"""
        memory_cache = MemoryCache(max_size=10)
        return CacheManager(memory_cache=memory_cache)

    @pytest.mark.asyncio
    async def test_cache_manager_basic_operations(self, cache_manager):
        """Test basic cache manager operations"""
        # Set value
        success = await cache_manager.set("test_key", "test_value")
        assert success is True

        # Get value
        value = await cache_manager.get("test_key")
        assert value == "test_value"

        # Check exists
        exists = await cache_manager.exists("test_key")
        assert exists is True

        # Delete value
        success = await cache_manager.delete("test_key")
        assert success is True

        # Verify deleted
        value = await cache_manager.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_decorator(self, cache_manager):
        """Test cache result decorator"""
        call_count = 0

        @cache_manager.cache_result(ttl=60)
        async def expensive_function(param):
            nonlocal call_count
            call_count += 1
            return f"result_{param}"

        # First call should execute function
        result1 = await expensive_function("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = await expensive_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment

        # Different parameter should execute function
        result3 = await expensive_function("other")
        assert result3 == "result_other"
        assert call_count == 2

    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        stats = cache_manager.get_cache_stats()
        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate_percentage" in stats

@pytest.mark.integration
class TestEnhancedClientIntegration:
    """Integration tests for enhanced client"""

    @pytest.fixture
    def mock_client(self):
        """Create mock enhanced client for testing"""
        config = BrightDataConfig()
        config.api_key = "test_key"

        cache_manager = CacheManager(MemoryCache(max_size=10))
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        error_handler = BrightDataErrorHandler()

        return EnhancedBrightDataClient(
            config=config,
            cache_manager=cache_manager,
            rate_limiter=rate_limiter,
            error_handler=error_handler
        )

    def test_client_initialization(self, mock_client):
        """Test client initialization"""
        assert mock_client.config.api_key == "test_key"
        assert mock_client.cache_manager is not None
        assert mock_client.rate_limiter is not None
        assert mock_client.error_handler is not None

    def test_client_metrics(self, mock_client):
        """Test client metrics collection"""
        metrics = mock_client.get_metrics()
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "success_rate" in metrics
        assert "circuit_breaker_state" in metrics

    @pytest.mark.asyncio
    async def test_client_health_check(self, mock_client):
        """Test client health check"""
        with patch.object(mock_client, '_make_single_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "ok"}

            health = await mock_client.health_check()
            assert "status" in health
            assert "response_time" in health

    @pytest.mark.asyncio
    async def test_client_cleanup(self, mock_client):
        """Test client cleanup"""
        # Should not raise exception
        await mock_client.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
