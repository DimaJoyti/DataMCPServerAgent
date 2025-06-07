"""
Example demonstrating the advanced error analysis capabilities.
This example shows how to use the AdvancedErrorAnalysis class for error clustering,
root cause analysis, error correlation analysis, and predictive error detection.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.advanced_error_analysis import AdvancedErrorAnalysis
from src.utils.error_handlers import (
    AuthenticationError,
    ConnectionError,
    ContentExtractionError,
    RateLimitError,
    WebsiteError,
)
from src.utils.error_recovery import ErrorRecoverySystem, RetryStrategy
from src.utils.env_config import get_mcp_server_params, get_model_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def simulate_error_for_tool(tool: BaseTool, url: str) -> Dict[str, Any]:
    """Simulate an error for a tool based on the URL.

    Args:
        tool: The tool to simulate an error for
        url: The URL to use for error simulation

    Returns:
        Simulated result (never reached, always raises an exception)
    """
    if "rate-limited" in url:
        raise RateLimitError(
            f"Rate limit exceeded for {tool.name}. Try again later.", retry_after=5
        )
    elif "connection-error" in url:
        raise ConnectionError(f"Connection error for {tool.name}. Network unreachable.")
    elif "authentication-error" in url:
        raise AuthenticationError(
            f"Authentication error for {tool.name}. Invalid credentials."
        )
    elif "content-extraction-error" in url:
        raise ContentExtractionError(
            f"Content extraction error for {tool.name}. Could not find selector."
        )
    elif "website-error" in url:
        raise WebsiteError(
            f"Website error for {tool.name}. Status code: 404.", status_code=404
        )
    else:
        raise Exception(f"Unknown error for {tool.name}.")

async def generate_error_patterns(
    error_recovery: ErrorRecoverySystem, tools: List[BaseTool]
) -> None:
    """Generate error patterns for testing.

    Args:
        error_recovery: Error recovery system
        tools: List of tools
    """
    # Define error URLs and types
    error_types = [
        ("https://www.rate-limited-example.com", "rate_limit"),
        ("https://www.connection-error-example.com", "connection"),
        ("https://www.authentication-error-example.com", "authentication"),
        ("https://www.content-extraction-error-example.com", "content_extraction"),
        ("https://www.website-error-example.com", "website"),
        ("https://www.unknown-error-example.com", "unknown"),
    ]

    # Get a tool for testing
    scrape_tool = next(
        (tool for tool in tools if "scrape" in tool.name.lower()), tools[0]
    )

    # Generate error patterns
    for url, error_type in error_types:
        logger.info(f"Generating {error_type} error pattern...")

        # Generate multiple instances of each error type
        for i in range(3):
            try:
                # Mock the tool's invoke method to simulate an error
                original_invoke = scrape_tool.invoke
                scrape_tool.invoke = lambda args: simulate_error_for_tool(
                    scrape_tool, url
                )

                # Try to execute the tool
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

            # Add a small delay between errors
            await asyncio.sleep(0.1)

async def main():
    """Main entry point for the example."""
    try:
        # Get model configuration
        model_config = get_model_config()
        model = ChatAnthropic(**model_config)

        # Get MCP server parameters
        server_params = get_mcp_server_params()

        # Create memory database
        db = MemoryDatabase("advanced_error_analysis_example.db")

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

            # Create advanced error analysis system
            advanced_analysis = AdvancedErrorAnalysis(
                model, db, all_tools, error_recovery
            )

            # Generate error patterns for testing
            logger.info("Generating error patterns...")
            await generate_error_patterns(error_recovery, all_tools)

            # Run error clustering
            logger.info("Running error clustering...")
            clusters = await advanced_analysis.cluster_errors()
            logger.info(f"Found {len(clusters)} error clusters")
            for cluster in clusters:
                logger.info(
                    f"Cluster {cluster.cluster_id}: {cluster.error_type} - {cluster.frequency} occurrences"
                )

            # Run root cause analysis
            logger.info("Running root cause analysis...")
            root_causes = await advanced_analysis.analyze_root_causes()
            for cluster_id, analysis in root_causes.items():
                logger.info(
                    f"Root cause for cluster {cluster_id}: {analysis.get('root_cause', 'Unknown')}"
                )
                logger.info(
                    f"Prevention strategies: {', '.join(analysis.get('prevention_strategies', []))}"
                )

            # Run error correlation analysis
            logger.info("Running error correlation analysis...")
            correlations = await advanced_analysis.analyze_error_correlations()
            logger.info(
                f"Found {len(correlations.get('correlations', []))} error correlations"
            )
            for correlation in correlations.get("correlations", []):
                logger.info(f"Correlation: {correlation}")

            # Run predictive error detection
            logger.info("Running predictive error detection...")
            predictions = await advanced_analysis.predict_potential_errors()
            logger.info(
                f"Found {len(predictions.get('predicted_errors', []))} potential future errors"
            )
            for prediction in predictions.get("predicted_errors", []):
                logger.info(f"Predicted error: {prediction}")

            # Run comprehensive analysis
            logger.info("Running comprehensive analysis...")
            comprehensive_results = await advanced_analysis.run_comprehensive_analysis()
            logger.info("Comprehensive analysis complete")

            logger.info("Advanced error analysis example complete")
    except Exception as e:
        logger.error(f"Error in advanced error analysis example: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
