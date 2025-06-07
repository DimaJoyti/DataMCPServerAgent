"""
Example script demonstrating how to use the Fetch.ai Advanced Crypto Trading System.
"""

import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from .sentiment_agent import SentimentIntelligenceAgent
from .technical_agent import TechnicalAnalysisAgent
from .risk_agent import RiskManagementAgent
from .regulatory_agent import RegulatoryComplianceAgent
from .macro_agent import MacroCorrelationAgent
from .learning_agent import LearningOptimizationAgent
from .trading_system import AdvancedCryptoTradingSystem, TradeRecommendation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_single_agent_example():
    """Run an example with a single agent."""
    logger.info("Running single agent example")

    # Create a technical analysis agent
    technical_agent = TechnicalAnalysisAgent(
        name="example_technical_agent",
        exchange_id="binance"
    )

    # Start the agent
    agent_task = asyncio.create_task(technical_agent.run_async())

    # Wait for a few seconds to let the agent initialize
    await asyncio.sleep(5)

    # Stop the agent
    agent_task.cancel()

    logger.info("Single agent example completed")

async def run_multi_agent_example():
    """Run an example with multiple agents."""
    logger.info("Running multi-agent example")

    # Create agents
    sentiment_agent = SentimentIntelligenceAgent(name="example_sentiment_agent")
    technical_agent = TechnicalAnalysisAgent(name="example_technical_agent", exchange_id="binance")
    risk_agent = RiskManagementAgent(name="example_risk_agent")

    # Start agents
    agent_tasks = [
        asyncio.create_task(sentiment_agent.run_async()),
        asyncio.create_task(technical_agent.run_async()),
        asyncio.create_task(risk_agent.run_async())
    ]

    # Wait for a few seconds to let the agents initialize
    await asyncio.sleep(5)

    # Stop agents
    for task in agent_tasks:
        task.cancel()

    logger.info("Multi-agent example completed")

async def run_full_system_example():
    """Run an example with the full trading system."""
    logger.info("Running full system example")

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="example_trading_system",
        exchange_id="binance"
    )

    # Update symbols to track
    trading_system.state.symbols_to_track = ["BTC/USD", "ETH/USD"]

    # Start the trading system
    system_task = asyncio.create_task(trading_system.start_all_agents())

    # Wait for a few seconds to let the system initialize
    await asyncio.sleep(10)

    # Generate a recommendation manually
    symbol = "BTC/USD"
    recommendation = await trading_system._generate_recommendation(None, symbol)

    if recommendation:
        logger.info(f"Recommendation for {symbol}:")
        logger.info(f"Signal: {recommendation.signal}")
        logger.info(f"Strength: {recommendation.strength}")
        logger.info(f"Entry Price: {recommendation.entry_price}")
        logger.info(f"Stop Loss: {recommendation.stop_loss}")
        logger.info(f"Take Profit: {recommendation.take_profit}")
        logger.info(f"Position Size: {recommendation.position_size}")
        logger.info(f"Confidence: {recommendation.confidence}")
        logger.info(f"Reasoning: {recommendation.reasoning}")
    else:
        logger.info(f"No recommendation generated for {symbol}")

    # Stop the trading system
    system_task.cancel()

    logger.info("Full system example completed")

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Run examples
    await run_single_agent_example()
    await run_multi_agent_example()
    await run_full_system_example()

if __name__ == "__main__":
    asyncio.run(main())
