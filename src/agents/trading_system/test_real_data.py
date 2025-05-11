"""
Test script for the Fetch.ai Advanced Crypto Trading System using real market data.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentIntelligenceAgent
from .risk_agent import RiskManagementAgent
from .trading_system import AdvancedCryptoTradingSystem, TradeRecommendation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_ai_test.log')
    ]
)
logger = logging.getLogger(__name__)

class RealDataTester:
    """Class for testing the trading system with real market data."""
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: List[str] = ["BTC/USDT", "ETH/USDT"],
        timeframes: List[str] = ["1h", "4h", "1d"],
        test_duration: int = 3600  # 1 hour in seconds
    ):
        """Initialize the tester.
        
        Args:
            exchange_id: ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            symbols: Symbols to test
            timeframes: Timeframes to test
            test_duration: Duration of the test in seconds
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.timeframes = timeframes
        self.test_duration = test_duration
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Initialize trading system
        self.trading_system = AdvancedCryptoTradingSystem(
            name="test_trading_system",
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Initialize results storage
        self.results = {
            "recommendations": [],
            "market_data": {},
            "performance": {}
        }
    
    async def run_test(self):
        """Run the test."""
        logger.info(f"Starting test with {self.exchange_id} for {len(self.symbols)} symbols")
        
        # Start time
        start_time = time.time()
        
        # Initialize trading system
        system_task = asyncio.create_task(self.trading_system.start_all_agents())
        
        # Wait for system to initialize
        await asyncio.sleep(10)
        
        # Test loop
        while time.time() - start_time < self.test_duration:
            # Get market data
            await self._fetch_market_data()
            
            # Generate recommendations
            await self._generate_recommendations()
            
            # Wait for next iteration
            await asyncio.sleep(60)  # Check every minute
        
        # Stop trading system
        system_task.cancel()
        
        # Analyze results
        self._analyze_results()
        
        logger.info("Test completed")
    
    async def _fetch_market_data(self):
        """Fetch market data for all symbols and timeframes."""
        for symbol in self.symbols:
            if symbol not in self.results["market_data"]:
                self.results["market_data"][symbol] = {}
            
            for timeframe in self.timeframes:
                try:
                    # Fetch OHLCV data
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Store data
                    self.results["market_data"][symbol][timeframe] = df
                    
                    logger.info(f"Fetched {len(df)} OHLCV data points for {symbol} on {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error fetching OHLCV data for {symbol} on {timeframe}: {str(e)}")
    
    async def _generate_recommendations(self):
        """Generate trading recommendations for all symbols."""
        for symbol in self.symbols:
            try:
                # Generate recommendation
                recommendation = await self.trading_system._generate_recommendation(None, symbol)
                
                if recommendation:
                    # Add timestamp
                    recommendation_dict = recommendation.dict()
                    recommendation_dict["timestamp"] = datetime.now().isoformat()
                    
                    # Store recommendation
                    self.results["recommendations"].append(recommendation_dict)
                    
                    logger.info(
                        f"Generated recommendation for {symbol}: "
                        f"{recommendation.signal.upper()} "
                        f"({recommendation.strength}) "
                        f"with {recommendation.confidence:.2f} confidence"
                    )
                else:
                    logger.warning(f"No recommendation generated for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error generating recommendation for {symbol}: {str(e)}")
    
    def _analyze_results(self):
        """Analyze test results."""
        # Count recommendations by signal
        signal_counts = {}
        for rec in self.results["recommendations"]:
            signal = rec["signal"]
            if signal not in signal_counts:
                signal_counts[signal] = 0
            signal_counts[signal] += 1
        
        # Calculate average confidence
        if self.results["recommendations"]:
            avg_confidence = sum(rec["confidence"] for rec in self.results["recommendations"]) / len(self.results["recommendations"])
        else:
            avg_confidence = 0.0
        
        # Store performance metrics
        self.results["performance"] = {
            "total_recommendations": len(self.results["recommendations"]),
            "signal_counts": signal_counts,
            "average_confidence": avg_confidence
        }
        
        # Log results
        logger.info(f"Test results:")
        logger.info(f"Total recommendations: {self.results['performance']['total_recommendations']}")
        logger.info(f"Signal counts: {self.results['performance']['signal_counts']}")
        logger.info(f"Average confidence: {self.results['performance']['average_confidence']:.2f}")
        
        # Save results to file
        with open("fetch_ai_test_results.json", "w") as f:
            json.dump(self.results["performance"], f, indent=2)
        
        # Save recommendations to file
        with open("fetch_ai_recommendations.json", "w") as f:
            json.dump(self.results["recommendations"], f, indent=2)

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment variables
    api_key = os.getenv('EXCHANGE_API_KEY')
    api_secret = os.getenv('EXCHANGE_API_SECRET')
    
    # Create tester
    tester = RealDataTester(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        symbols=["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"],
        test_duration=1800  # 30 minutes
    )
    
    # Run test
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())
