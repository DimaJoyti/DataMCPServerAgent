"""
Run script for the enhanced Fetch.ai Advanced Crypto Trading System.

This script runs all the enhancements:
1. Testing with real market data
2. Integration with n8n workflows
3. Enhanced machine learning models
4. Additional data sources
5. Visualization dashboard
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
from typing import Dict, Optional

from dotenv import load_dotenv

from .advanced_ml_models import ModelManager, ModelType, PredictionTarget
from .dashboard import Dashboard
from .data_sources import DataSourceManager
from .n8n_integration import N8nIntegration
from .test_real_data import RealDataTester
from .trading_system import AdvancedCryptoTradingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_ai_enhanced.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Fetch.ai Advanced Crypto Trading System')

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange to use (default: binance)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTC/USDT', 'ETH/USDT'],
        help='Symbols to track (default: BTC/USDT ETH/USDT)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for the exchange'
    )

    parser.add_argument(
        '--api-secret',
        type=str,
        help='API secret for the exchange'
    )

    parser.add_argument(
        '--n8n-url',
        type=str,
        default='http://localhost:5678',
        help='URL for n8n (default: http://localhost:5678)'
    )

    parser.add_argument(
        '--n8n-api-key',
        type=str,
        help='API key for n8n'
    )

    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8050,
        help='Port for the dashboard (default: 8050)'
    )

    parser.add_argument(
        '--test-duration',
        type=int,
        default=3600,
        help='Duration of the test in seconds (default: 3600)'
    )

    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        default=['trading', 'test', 'n8n', 'ml', 'data', 'dashboard'],
        help='Components to run (default: all)'
    )

    return parser.parse_args()

async def run_trading_system(args):
    """Run the trading system.

    Args:
        args: Command line arguments
    """
    logger.info("Starting trading system")

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="enhanced_trading_system",
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret
    )

    # Update symbols to track
    trading_system.state.symbols_to_track = args.symbols

    # Run the trading system
    await trading_system.start_all_agents()

async def run_real_data_test(args):
    """Run the real data test.

    Args:
        args: Command line arguments
    """
    logger.info("Starting real data test")

    # Create tester
    tester = RealDataTester(
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbols=args.symbols,
        test_duration=args.test_duration
    )

    # Run test
    await tester.run_test()

async def run_n8n_integration(args):
    """Run the n8n integration.

    Args:
        args: Command line arguments
    """
    logger.info("Starting n8n integration")

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="n8n_trading_system",
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret
    )

    # Update symbols to track
    trading_system.state.symbols_to_track = args.symbols

    # Create n8n integration
    n8n_integration = N8nIntegration(
        name="n8n_integration",
        n8n_base_url=args.n8n_url,
        n8n_api_key=args.n8n_api_key,
        trading_system=trading_system
    )

    # Start agents
    await asyncio.gather(
        trading_system.start_all_agents(),
        n8n_integration.run_async()
    )

async def run_enhanced_ml(args):
    """Run the enhanced machine learning models.

    Args:
        args: Command line arguments
    """
    logger.info("Starting enhanced machine learning models")

    # Create model manager
    manager = ModelManager()

    # Create and train models
    for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.NEURAL_NETWORK, ModelType.ENSEMBLE]:
        for target in [PredictionTarget.PRICE_DIRECTION, PredictionTarget.VOLATILITY, PredictionTarget.SENTIMENT_IMPACT]:
            # Create model
            model = manager.get_model(
                model_type=model_type,
                target=target
            )

            # Generate random data for demonstration
            import numpy as np
            X = np.random.rand(1000, 10)
            y = np.random.randint(0, 2, size=1000)

            # Train model
            logger.info(f"Training {model_type} model for {target}")
            result = model.train(X, y)
            logger.info(f"Training result: {result}")

            # Save model
            manager.save_model(
                model_type=model_type,
                target=target
            )

async def run_data_sources(args):
    """Run the additional data sources.

    Args:
        args: Command line arguments
    """
    logger.info("Starting additional data sources")

    # Create data source manager
    manager = DataSourceManager()

    # Fetch data for each symbol
    for symbol in args.symbols:
        # Fetch market data
        logger.info(f"Fetching market data for {symbol}")
        market_data = await manager.fetch_market_data(symbol)
        logger.info(f"Market data: {market_data}")

        # Fetch news
        logger.info(f"Fetching news for {symbol}")
        news = await manager.fetch_news(symbol, limit=5)
        logger.info(f"News: {len(news)} articles")

        # Fetch social media
        logger.info(f"Fetching social media for {symbol}")
        social = await manager.fetch_social_media(symbol, limit=5)
        logger.info(f"Social media: {len(social)} posts")

    # Fetch Fear & Greed Index
    logger.info("Fetching Fear & Greed Index")
    fear_greed = await manager.fetch_fear_greed_index()
    logger.info(f"Fear & Greed Index: {fear_greed}")

def run_dashboard(args):
    """Run the visualization dashboard.

    Args:
        args: Command line arguments
    """
    logger.info("Starting visualization dashboard")

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="dashboard_trading_system",
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret
    )

    # Update symbols to track
    trading_system.state.symbols_to_track = args.symbols

    # Create dashboard
    dashboard = Dashboard(
        trading_system=trading_system,
        port=args.dashboard_port,
        debug=True
    )

    # Run dashboard
    dashboard.run()

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Get API credentials from environment variables if not provided
    args.api_key = args.api_key or os.getenv('EXCHANGE_API_KEY')
    args.api_secret = args.api_secret or os.getenv('EXCHANGE_API_SECRET')
    args.n8n_api_key = args.n8n_api_key or os.getenv('N8N_API_KEY')

    if not args.api_key or not args.api_secret:
        logger.warning(
            "API key and secret not provided. "
            "The system will run in read-only mode."
        )

    try:
        # Run components
        tasks = []

        if 'trading' in args.components:
            tasks.append(run_trading_system(args))

        if 'test' in args.components:
            tasks.append(run_real_data_test(args))

        if 'n8n' in args.components:
            tasks.append(run_n8n_integration(args))

        if 'ml' in args.components:
            tasks.append(run_enhanced_ml(args))

        if 'data' in args.components:
            tasks.append(run_data_sources(args))

        if 'dashboard' in args.components:
            # Run dashboard in a separate thread
            dashboard_thread = threading.Thread(target=run_dashboard, args=(args,))
            dashboard_thread.daemon = True
            dashboard_thread.start()

        # Run all tasks
        if tasks:
            await asyncio.gather(*tasks)
        else:
            # If only dashboard is running, wait forever
            while True:
                await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
