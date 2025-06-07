#!/usr/bin/env python3
"""
Quick test script for Enhanced Bright Data MCP Integration

This script provides a quick way to test the enhanced Bright Data integration
without running the full example. It includes basic functionality tests and
performance benchmarks.
"""

import asyncio
import time
import logging
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
try:
    from src.tools.bright_data.core.config import BrightDataConfig
    from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
    from src.tools.bright_data.core.cache_manager import CacheManager, MemoryCache
    from src.tools.bright_data.core.rate_limiter import RateLimiter, ThrottleStrategy
    from src.tools.bright_data.core.error_handler import BrightDataErrorHandler
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

class BrightDataTester:
    """Test runner for Bright Data enhanced integration"""

    def __init__(self):
        self.results = {}
        self.start_time = None

    def start_test_suite(self):
        """Start the test suite"""
        self.start_time = time.time()
        logger.info("🚀 Starting Enhanced Bright Data Integration Tests")
        logger.info("=" * 60)

    def finish_test_suite(self):
        """Finish the test suite and show summary"""
        total_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"✅ Test suite completed in {total_time:.2f} seconds")

        # Show results summary
        passed = sum(1 for result in self.results.values() if result["status"] == "PASS")
        failed = sum(1 for result in self.results.values() if result["status"] == "FAIL")

        logger.info(f"📊 Results: {passed} passed, {failed} failed")

        if failed > 0:
            logger.info("❌ Failed tests:")
            for test_name, result in self.results.items():
                if result["status"] == "FAIL":
                    logger.info(f"  - {test_name}: {result['error']}")

    def record_test_result(self, test_name: str, status: str, error: str = None, duration: float = None):
        """Record test result"""
        self.results[test_name] = {
            "status": status,
            "error": error,
            "duration": duration
        }

        status_emoji = "✅" if status == "PASS" else "❌"
        duration_str = f" ({duration:.3f}s)" if duration else ""
        logger.info(f"{status_emoji} {test_name}{duration_str}")

        if error:
            logger.info(f"   Error: {error}")

    async def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        start_time = time.time()
        try:
            await test_func()
            duration = time.time() - start_time
            self.record_test_result(test_name, "PASS", duration=duration)
        except Exception as e:
            duration = time.time() - start_time
            self.record_test_result(test_name, "FAIL", str(e), duration)

    async def test_config_management(self):
        """Test configuration management"""
        # Test basic config creation
        config = BrightDataConfig()
        assert config.api_timeout == 30

        # Test config from dict
        config_data = {
            "api_timeout": 60,
            "cache": {"enabled": False}
        }
        config = BrightDataConfig.from_dict(config_data)
        assert config.api_timeout == 60
        assert config.cache.enabled is False

        # Test config validation
        config.api_key = "test_key"
        config.validate()  # Should not raise

    async def test_memory_cache(self):
        """Test memory cache functionality"""
        cache = MemoryCache(max_size=5, default_ttl=60)

        # Test set/get
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

        # Test cache miss
        value = await cache.get("nonexistent")
        assert value is None

        # Test TTL expiration
        await cache.set("ttl_key", "ttl_value", ttl=1)
        await asyncio.sleep(1.1)
        value = await cache.get("ttl_key")
        assert value is None

        # Test LRU eviction
        for i in range(6):  # Exceed max_size
            await cache.set(f"key_{i}", f"value_{i}")

        # First key should be evicted
        value = await cache.get("key_0")
        assert value is None

    async def test_rate_limiter(self):
        """Test rate limiting functionality"""
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=3)

        # Test burst allowance
        for i in range(3):
            success = await rate_limiter.acquire("test_user")
            assert success is True

        # Should reject next request
        success = await rate_limiter.acquire("test_user")
        assert success is False

        # Test different users
        success = await rate_limiter.acquire("other_user")
        assert success is True

        # Test statistics
        stats = rate_limiter.get_global_stats()
        assert stats["total_requests"] > 0

    async def test_error_handler(self):
        """Test error handling functionality"""
        error_handler = BrightDataErrorHandler()

        # Test error categorization
        network_error = ConnectionError("Network failed")
        error_info = error_handler.categorize_error(network_error)
        assert error_info.category.value == "network"

        # Test custom recovery strategy
        recovery_called = False

        async def custom_recovery(error_info):
            nonlocal recovery_called
            recovery_called = True

        from src.tools.bright_data.core.error_handler import ErrorCategory
        error_handler.register_recovery_strategy(ErrorCategory.NETWORK, custom_recovery)

        # Test statistics
        stats = error_handler.get_error_statistics()
        assert "total_errors" in stats

    async def test_cache_manager(self):
        """Test cache manager functionality"""
        memory_cache = MemoryCache(max_size=10)
        cache_manager = CacheManager(memory_cache=memory_cache)

        # Test basic operations
        await cache_manager.set("test_key", "test_value")
        value = await cache_manager.get("test_key")
        assert value == "test_value"

        # Test cache decorator
        call_count = 0

        @cache_manager.cache_result(ttl=60)
        async def expensive_function(param):
            nonlocal call_count
            call_count += 1
            return f"result_{param}"

        # First call
        result1 = await expensive_function("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call (should use cache)
        result2 = await expensive_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment

        # Test statistics
        stats = cache_manager.get_cache_stats()
        assert "hit_rate_percentage" in stats

    async def test_enhanced_client_basic(self):
        """Test enhanced client basic functionality"""
        config = BrightDataConfig()
        config.api_key = "test_key"

        cache_manager = CacheManager(MemoryCache(max_size=10))
        rate_limiter = RateLimiter(requests_per_minute=60)
        error_handler = BrightDataErrorHandler()

        client = EnhancedBrightDataClient(
            config=config,
            cache_manager=cache_manager,
            rate_limiter=rate_limiter,
            error_handler=error_handler
        )

        try:
            # Test metrics
            metrics = client.get_metrics()
            assert "total_requests" in metrics
            assert "circuit_breaker_state" in metrics

            # Test health check (will fail without real API, but should not crash)
            try:
                health = await client.health_check()
                # If it succeeds, great! If it fails, that's expected without real API
            except Exception:
                pass  # Expected without real API

        finally:
            await client.close()

    async def test_performance_benchmark(self):
        """Run basic performance benchmarks"""
        logger.info("🏃 Running performance benchmarks...")

        # Cache performance test
        cache = MemoryCache(max_size=1000)

        # Benchmark cache writes
        start_time = time.time()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time

        # Benchmark cache reads
        start_time = time.time()
        for i in range(1000):
            await cache.get(f"key_{i}")
        read_time = time.time() - start_time

        logger.info(f"📈 Cache Performance:")
        logger.info(f"   Writes: {1000/write_time:.0f} ops/sec")
        logger.info(f"   Reads: {1000/read_time:.0f} ops/sec")

        # Rate limiter performance test
        rate_limiter = RateLimiter(requests_per_minute=6000, burst_size=100)

        start_time = time.time()
        successful_requests = 0
        for i in range(100):
            if await rate_limiter.acquire("perf_test"):
                successful_requests += 1
        rate_limit_time = time.time() - start_time

        logger.info(f"📈 Rate Limiter Performance:")
        logger.info(f"   Processed: {100/rate_limit_time:.0f} requests/sec")
        logger.info(f"   Success rate: {successful_requests}%")

    async def run_all_tests(self):
        """Run all tests"""
        self.start_test_suite()

        # Core component tests
        await self.run_test("Configuration Management", self.test_config_management)
        await self.run_test("Memory Cache", self.test_memory_cache)
        await self.run_test("Rate Limiter", self.test_rate_limiter)
        await self.run_test("Error Handler", self.test_error_handler)
        await self.run_test("Cache Manager", self.test_cache_manager)
        await self.run_test("Enhanced Client Basic", self.test_enhanced_client_basic)

        # Performance benchmarks
        await self.run_test("Performance Benchmark", self.test_performance_benchmark)

        self.finish_test_suite()

async def main():
    """Main test runner"""
    logger.info("Enhanced Bright Data MCP Integration - Quick Test")
    logger.info("This script tests the core functionality without requiring API keys")

    tester = BrightDataTester()
    await tester.run_all_tests()

    # Exit with appropriate code
    failed_tests = sum(1 for result in tester.results.values() if result["status"] == "FAIL")
    sys.exit(1 if failed_tests > 0 else 0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        sys.exit(1)
