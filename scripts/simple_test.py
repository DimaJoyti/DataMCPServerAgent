#!/usr/bin/env python3
"""
Simple test for TradingView tools without complex dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.tradingview_tools import TradingViewToolkit, CryptoSymbol, CryptoExchange


async def test_basic_functionality():
    """Test basic TradingView tools functionality."""
    print("🧪 Testing Basic TradingView Tools")
    print("=" * 40)
    
    # Mock session for testing
    class MockSession:
        async def list_plugins(self):
            class MockPlugin:
                def __init__(self):
                    self.tools = [
                        type('MockTool', (), {'name': 'scrape_as_markdown_Bright_Data'}),
                    ]
            return [MockPlugin()]
    
    session = MockSession()
    toolkit = TradingViewToolkit(session)
    
    # Test crypto symbols
    print("📊 Testing crypto symbols...")
    symbols = toolkit.get_popular_crypto_symbols()
    print(f"✅ Found {len(symbols)} popular crypto symbols")
    
    for symbol in symbols[:5]:
        print(f"   - {symbol.tradingview_symbol}")
    
    # Test exchanges
    print("\n🏦 Testing supported exchanges...")
    exchanges = toolkit.get_supported_exchanges()
    print(f"✅ Found {len(exchanges)} supported exchanges")
    
    for exchange in exchanges:
        print(f"   - {exchange.value}")
    
    # Test timeframes
    print("\n⏰ Testing timeframes...")
    timeframes = toolkit.get_supported_timeframes()
    print(f"✅ Found {len(timeframes)} timeframes")
    
    for tf in timeframes:
        print(f"   - {tf.value}")
    
    print("\n✅ Basic functionality test completed!")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
