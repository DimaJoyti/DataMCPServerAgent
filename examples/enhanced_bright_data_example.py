"""
Enhanced Bright Data MCP Integration Example

This example demonstrates the advanced features of the enhanced Bright Data integration:
- Enhanced client with retry logic and circuit breaker
- Multi-level caching with Redis support
- Rate limiting and throttling
- Advanced error handling
- Competitive intelligence tools
- Real-time monitoring
"""

import asyncio
import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced Bright Data components
from src.tools.bright_data.core.cache_manager import CacheManager, MemoryCache, RedisCache
from src.tools.bright_data.core.config import BrightDataConfig
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.error_handler import BrightDataErrorHandler
from src.tools.bright_data.core.rate_limiter import RateLimiter, ThrottleStrategy
from src.tools.bright_data.tools.competitive_intelligence import CompetitiveIntelligenceTools


async def setup_enhanced_bright_data_system() -> Dict[str, Any]:
    """Setup the enhanced Bright Data system with all components"""

    # 1. Load configuration
    config = BrightDataConfig.from_env()
    logger.info("Configuration loaded")

    # 2. Setup caching system
    memory_cache = MemoryCache(max_size=1000, default_ttl=3600)

    # Try to setup Redis cache (optional)
    redis_cache = None
    try:
        redis_cache = RedisCache(
            redis_url=config.cache.redis_url,
            default_ttl=config.cache.default_ttl
        )
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.warning(f"Redis cache not available: {e}")

    cache_manager = CacheManager(
        memory_cache=memory_cache,
        redis_cache=redis_cache,
        compression_threshold=config.cache.compression_threshold,
        enable_compression=config.cache.compression_enabled
    )

    # 3. Setup rate limiting
    rate_limiter = RateLimiter(
        requests_per_minute=config.rate_limit.requests_per_minute,
        burst_size=config.rate_limit.burst_size,
        strategy=ThrottleStrategy.ADAPTIVE if config.rate_limit.adaptive_throttling else ThrottleStrategy.FIXED
    )

    # 4. Setup error handling
    error_handler = BrightDataErrorHandler()

    # 5. Create enhanced client
    client = EnhancedBrightDataClient(
        config=config,
        cache_manager=cache_manager,
        rate_limiter=rate_limiter,
        error_handler=error_handler
    )

    # 6. Setup specialized tools
    competitive_tools = CompetitiveIntelligenceTools(client)

    return {
        "config": config,
        "client": client,
        "cache_manager": cache_manager,
        "rate_limiter": rate_limiter,
        "error_handler": error_handler,
        "competitive_tools": competitive_tools
    }

async def demonstrate_basic_scraping(client: EnhancedBrightDataClient):
    """Demonstrate basic web scraping with enhanced features"""
    logger.info("=== Basic Web Scraping Demo ===")

    try:
        # Scrape a website with caching
        result = await client.scrape_url(
            "https://example.com",
            user_id="demo_user",
            use_cache=True,
            cache_ttl=1800  # 30 minutes
        )

        logger.info(f"Scraping successful: {len(str(result))} characters")

        # Scrape the same URL again (should hit cache)
        cached_result = await client.scrape_url(
            "https://example.com",
            user_id="demo_user",
            use_cache=True
        )

        logger.info("Second request completed (likely from cache)")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")

async def demonstrate_web_search(client: EnhancedBrightDataClient):
    """Demonstrate web search with caching and rate limiting"""
    logger.info("=== Web Search Demo ===")

    try:
        # Perform web search
        search_results = await client.search_web(
            query="artificial intelligence trends 2024",
            count=5,
            user_id="demo_user",
            use_cache=True
        )

        logger.info(f"Search completed: {len(search_results.get('results', []))} results")

        # Perform multiple searches to test rate limiting
        queries = [
            "machine learning frameworks",
            "data science tools",
            "web scraping best practices"
        ]

        for query in queries:
            try:
                results = await client.search_web(
                    query=query,
                    count=3,
                    user_id="demo_user"
                )
                logger.info(f"Search '{query}': {len(results.get('results', []))} results")

                # Small delay to demonstrate rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Search '{query}' failed: {e}")

    except Exception as e:
        logger.error(f"Web search demo failed: {e}")

async def demonstrate_competitive_intelligence(competitive_tools: CompetitiveIntelligenceTools):
    """Demonstrate competitive intelligence features"""
    logger.info("=== Competitive Intelligence Demo ===")

    try:
        # Create competitive intelligence tools
        tools = competitive_tools.create_tools()
        logger.info(f"Created {len(tools)} competitive intelligence tools")

        # Example: Monitor competitor prices
        price_monitoring_tool = next(
            (tool for tool in tools if tool.name == "competitive_price_monitoring"),
            None
        )

        if price_monitoring_tool:
            # Example product URLs (replace with real URLs for testing)
            product_urls = [
                "https://example-store.com/product1",
                "https://example-store.com/product2"
            ]

            try:
                result = await price_monitoring_tool.func(
                    product_urls=product_urls,
                    competitor_name="Example Competitor"
                )
                logger.info("Price monitoring completed")
                logger.info(f"Result preview: {result[:200]}...")

            except Exception as e:
                logger.warning(f"Price monitoring failed: {e}")

        # Example: Feature comparison
        feature_comparison_tool = next(
            (tool for tool in tools if tool.name == "feature_comparison"),
            None
        )

        if feature_comparison_tool:
            try:
                result = await feature_comparison_tool.func(
                    product_urls=product_urls,
                    comparison_criteria=["price", "features", "rating"]
                )
                logger.info("Feature comparison completed")

            except Exception as e:
                logger.warning(f"Feature comparison failed: {e}")

    except Exception as e:
        logger.error(f"Competitive intelligence demo failed: {e}")

async def demonstrate_monitoring_and_metrics(system_components: Dict[str, Any]):
    """Demonstrate monitoring and metrics collection"""
    logger.info("=== Monitoring and Metrics Demo ===")

    client = system_components["client"]
    cache_manager = system_components["cache_manager"]
    rate_limiter = system_components["rate_limiter"]
    error_handler = system_components["error_handler"]

    # Get client metrics
    client_metrics = client.get_metrics()
    logger.info("Client Metrics:")
    for key, value in client_metrics.items():
        logger.info(f"  {key}: {value}")

    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    logger.info("Cache Statistics:")
    for key, value in cache_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

    # Get rate limiting statistics
    rate_limit_stats = rate_limiter.get_global_stats()
    logger.info("Rate Limiting Statistics:")
    for key, value in rate_limit_stats.items():
        logger.info(f"  {key}: {value}")

    # Get error statistics
    error_stats = error_handler.get_error_statistics()
    logger.info("Error Statistics:")
    for key, value in error_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

    # Perform health check
    health_status = await client.health_check()
    logger.info("Health Check:")
    for key, value in health_status.items():
        logger.info(f"  {key}: {value}")

async def demonstrate_error_handling(client: EnhancedBrightDataClient):
    """Demonstrate advanced error handling and recovery"""
    logger.info("=== Error Handling Demo ===")

    # Test with invalid URL to trigger error handling
    try:
        await client.scrape_url("https://invalid-url-that-does-not-exist.com")
    except Exception as e:
        logger.info(f"Expected error handled: {type(e).__name__}: {e}")

    # Test rate limiting by making many rapid requests
    logger.info("Testing rate limiting with rapid requests...")
    for i in range(10):
        try:
            await client.search_web(
                query=f"test query {i}",
                count=1,
                user_id="rate_limit_test"
            )
            logger.info(f"Request {i+1} succeeded")
        except Exception as e:
            logger.info(f"Request {i+1} failed (expected): {type(e).__name__}")

async def main():
    """Main demonstration function"""
    logger.info("Starting Enhanced Bright Data MCP Integration Demo")

    try:
        # Setup the enhanced system
        system_components = await setup_enhanced_bright_data_system()
        logger.info("Enhanced Bright Data system initialized successfully")

        client = system_components["client"]
        competitive_tools = system_components["competitive_tools"]

        # Run demonstrations
        await demonstrate_basic_scraping(client)
        await demonstrate_web_search(client)
        await demonstrate_competitive_intelligence(competitive_tools)
        await demonstrate_error_handling(client)
        await demonstrate_monitoring_and_metrics(system_components)

        logger.info("All demonstrations completed successfully")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

    finally:
        # Cleanup
        if 'client' in locals():
            await client.close()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
