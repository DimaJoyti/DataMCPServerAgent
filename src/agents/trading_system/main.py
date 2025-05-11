"""
Main entry point for the Fetch.ai Advanced Crypto Trading System.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .trading_system import AdvancedCryptoTradingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_ai_trading.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch.ai Advanced Crypto Trading System')
    
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
        default=['BTC/USD', 'ETH/USD'],
        help='Symbols to track (default: BTC/USD ETH/USD)'
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
        '--port',
        type=int,
        default=8000,
        help='Port for the agent server (default: 8000)'
    )
    
    parser.add_argument(
        '--endpoint',
        type=str,
        help='Endpoint for the agent server'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Get API credentials from environment variables if not provided
    api_key = args.api_key or os.getenv('EXCHANGE_API_KEY')
    api_secret = args.api_secret or os.getenv('EXCHANGE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.warning(
            "API key and secret not provided. "
            "The system will run in read-only mode."
        )
    
    try:
        # Create trading system
        trading_system = AdvancedCryptoTradingSystem(
            name="fetch_ai_trading_system",
            port=args.port,
            endpoint=args.endpoint,
            exchange_id=args.exchange,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Update symbols to track
        trading_system.state.symbols_to_track = args.symbols
        
        logger.info(f"Starting Fetch.ai Advanced Crypto Trading System")
        logger.info(f"Exchange: {args.exchange}")
        logger.info(f"Symbols: {', '.join(args.symbols)}")
        
        # Run the trading system
        trading_system.run_all()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
