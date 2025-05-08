"""
Example demonstrating the enhanced error recovery capabilities.
This example shows how to use the ErrorRecoverySystem for sophisticated retry strategies,
automatic fallback to alternative tools, and self-healing capabilities.
"""

import asyncio
import logging
import os

# Add the project root to the Python path
import sys
import time
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import (
    AuthenticationError,
    ConnectionError,
    ContentExtractionError,
    RateLimitError,
)
from src.utils.error_recovery import ErrorRecoverySystem, RetryStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Example URLs with different error scenarios
EXAMPLE_URLS = [
    # Working URL
    "https://www.example.com",
    # Rate-limited URL (simulated)
    "https://www.rate-limited-example.com",
    # Connection error URL (non-existent domain)
    "https://www.non-existent-domain-12345.com",
    # Authentication error URL (simulated)
    "https://www.auth-error-example.com",
    # Content extraction error URL (simulated)
    "https://www.content-error-example.com",
]


async def setup_error_recovery_system() -> tuple:
    """Set up the error recovery system with tools and model.

    Returns:
        Tuple of (error_recovery_system, tools, model, session)
    """
    # Initialize model
    model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

    # Initialize memory database
    db_path = os.getenv("MEMORY_DB_PATH", "error_recovery_example.db")
    db = MemoryDatabase(db_path)

    # Set up MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        env={
            "API_TOKEN": os.getenv("API_TOKEN"),
            "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
            "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
        },
        args=["@brightdata/mcp"],
    )

    # Initialize MCP client session
    async with stdio_client(server_params) as session:
        # Load MCP tools
        mcp_tools = await load_mcp_tools(session)

        # Create Bright Data toolkit
        bright_data_toolkit = BrightDataToolkit(session)

        # Create custom tools
        custom_tools = await bright_data_toolkit.create_custom_tools()

        # Combine all tools
        all_tools = mcp_tools + custom_tools

        # Create error recovery system
        error_recovery = ErrorRecoverySystem(model, db, all_tools)

        return error_recovery, all_tools, model, session


async def simulate_error_for_tool(tool: BaseTool, url: str) -> None:
    """Simulate an error for a specific tool and URL.

    Args:
        tool: The tool to use
        url: The URL to process
    """
    if "non-existent" in url:
        raise ConnectionError(f"Failed to connect to {url}: Connection refused")
    elif "rate-limited" in url:
        raise RateLimitError(f"Rate limit exceeded for {url}", retry_after=5)
    elif "auth-error" in url:
        raise AuthenticationError(f"Authentication failed for {url}")
    elif "content-error" in url:
        raise ContentExtractionError(f"Failed to extract content from {url}")
    else:
        # No error for regular URLs
        return await tool.invoke({"url": url})


async def demonstrate_retry_strategies(
    error_recovery: ErrorRecoverySystem, tools: List[BaseTool]
) -> None:
    """Demonstrate different retry strategies.

    Args:
        error_recovery: The error recovery system
        tools: List of available tools
    """
    logger.info("=== Demonstrating Retry Strategies ===")

    # Find a scraping tool
    scrape_tool = next((t for t in tools if "scrape" in t.name.lower()), None)
    if not scrape_tool:
        logger.error("No scraping tool found")
        return

    # Test different retry strategies with a rate-limited URL
    retry_strategies = [
        RetryStrategy.EXPONENTIAL,
        RetryStrategy.LINEAR,
        RetryStrategy.JITTERED,
        RetryStrategy.CONSTANT,
        RetryStrategy.ADAPTIVE,
    ]

    for strategy in retry_strategies:
        logger.info(f"Testing {strategy.value} retry strategy...")

        try:
            # Mock the tool's invoke method to simulate rate limiting
            original_invoke = scrape_tool.invoke
            scrape_tool.invoke = lambda args: simulate_error_for_tool(
                scrape_tool, "https://www.rate-limited-example.com"
            )

            # Try with the current retry strategy
            start_time = time.time()
            await error_recovery.with_advanced_retry(
                scrape_tool.invoke,
                {"url": "https://www.rate-limited-example.com"},
                tool_name=scrape_tool.name,
                retry_strategy=strategy,
                max_retries=2,
                base_delay=0.5,  # Short delay for the example
                max_delay=2.0,
            )
        except Exception as e:
            end_time = time.time()
            logger.info(
                f"{strategy.value} strategy failed after {end_time - start_time:.2f}s: {str(e)}"
            )
        finally:
            # Restore original invoke method
            scrape_tool.invoke = original_invoke


async def demonstrate_fallback_mechanisms(
    error_recovery: ErrorRecoverySystem, tools: List[BaseTool]
) -> None:
    """Demonstrate fallback mechanisms.

    Args:
        error_recovery: The error recovery system
        tools: List of available tools
    """
    logger.info("\n=== Demonstrating Fallback Mechanisms ===")

    # Find a primary tool and potential fallbacks
    primary_tool = next(
        (t for t in tools if "scrape_as_markdown" in t.name.lower()), None
    )
    if not primary_tool:
        logger.error("Primary tool not found")
        return

    # Test fallback for different error types
    error_urls = [
        "https://www.non-existent-domain-12345.com",  # Connection error
        "https://www.auth-error-example.com",  # Authentication error
        "https://www.content-error-example.com",  # Content extraction error
    ]

    for url in error_urls:
        logger.info(f"Testing fallback for URL: {url}")

        try:
            # Mock the primary tool's invoke method to simulate an error
            for tool in tools:
                original_invoke = tool.invoke
                tool.invoke = lambda args: simulate_error_for_tool(tool, url)

            # Try with fallbacks
            result, tool_used, success = await error_recovery.try_with_fallbacks(
                primary_tool.name,
                {"url": url},
                {"operation": "web_scraping", "url": url},
                max_fallbacks=2,
            )

            logger.info(f"Fallback succeeded with tool: {tool_used}")
        except Exception as e:
            logger.info(f"All fallbacks failed: {str(e)}")
        finally:
            # Restore original invoke methods
            for tool in tools:
                tool.invoke = original_invoke


async def demonstrate_self_healing(
    error_recovery: ErrorRecoverySystem, tools: List[BaseTool]
) -> None:
    """Demonstrate self-healing capabilities.

    Args:
        error_recovery: The error recovery system
        tools: List of available tools
    """
    logger.info("\n=== Demonstrating Self-Healing Capabilities ===")

    # Simulate a series of errors to generate patterns
    error_types = [
        ("https://www.non-existent-domain-12345.com", "connection"),
        ("https://www.rate-limited-example.com", "rate_limit"),
        ("https://www.auth-error-example.com", "authentication"),
        ("https://www.content-error-example.com", "content_extraction"),
    ]

    # Find a scraping tool
    scrape_tool = next((t for t in tools if "scrape" in t.name.lower()), None)
    if not scrape_tool:
        logger.error("No scraping tool found")
        return

    # Generate error patterns
    for url, error_type in error_types:
        logger.info(f"Generating {error_type} error pattern...")

        try:
            # Mock the tool's invoke method to simulate an error
            original_invoke = scrape_tool.invoke
            scrape_tool.invoke = lambda args: simulate_error_for_tool(scrape_tool, url)

            # Analyze the error
            try:
                await error_recovery.with_advanced_retry(
                    scrape_tool.invoke,
                    {"url": url},
                    tool_name=scrape_tool.name,
                    max_retries=1,
                )
            except Exception as e:
                # Analyze the error
                analysis = await error_recovery.analyze_error(
                    e, {"operation": "web_scraping", "url": url}, scrape_tool.name
                )
                logger.info(
                    f"Error analysis: {analysis['error_type']}, Severity: {analysis['severity']}"
                )
        finally:
            # Restore original invoke method
            scrape_tool.invoke = original_invoke

    # Learn from the error patterns
    logger.info("Learning from error patterns...")
    learning_results = await error_recovery.learn_from_errors()

    logger.info("Learning results:")
    logger.info(f"Identified patterns: {learning_results['identified_patterns']}")
    logger.info(f"Retry improvements: {learning_results['retry_improvements']}")
    logger.info(f"Fallback improvements: {learning_results['fallback_improvements']}")
    logger.info(
        f"Self-healing improvements: {learning_results['self_healing_improvements']}"
    )


async def main():
    """Main function to run the example."""
    logger.info("Starting Enhanced Error Recovery Example")

    try:
        # Set up error recovery system
        error_recovery, tools, model, session = await setup_error_recovery_system()

        # Demonstrate retry strategies
        await demonstrate_retry_strategies(error_recovery, tools)

        # Demonstrate fallback mechanisms
        await demonstrate_fallback_mechanisms(error_recovery, tools)

        # Demonstrate self-healing capabilities
        await demonstrate_self_healing(error_recovery, tools)

        logger.info("Enhanced Error Recovery Example completed successfully")
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
