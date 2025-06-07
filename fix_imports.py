#!/usr/bin/env python3
"""
Fix import issues in the crypto portfolio system.
"""

import os
import re

def fix_error_recovery_imports():
    """Fix ErrorRecoveryManager imports to ErrorRecoverySystem."""
    files_to_fix = [
        "src/agents/crypto_portfolio_agent.py",
        "src/core/crypto_portfolio_main.py",
        "test_crypto_portfolio_system.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"Fixing imports in {file_path}...")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace ErrorRecoveryManager with ErrorRecoverySystem
            content = content.replace('ErrorRecoveryManager', 'ErrorRecoverySystem')
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Fixed {file_path}")
        else:
            print(f"⚠️ File not found: {file_path}")

def create_simple_test():
    """Create a simple test that doesn't require complex imports."""
    test_content = '''#!/usr/bin/env python3
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
    print("\\n🏦 Testing supported exchanges...")
    exchanges = toolkit.get_supported_exchanges()
    print(f"✅ Found {len(exchanges)} supported exchanges")
    
    for exchange in exchanges:
        print(f"   - {exchange.value}")
    
    # Test timeframes
    print("\\n⏰ Testing timeframes...")
    timeframes = toolkit.get_supported_timeframes()
    print(f"✅ Found {len(timeframes)} timeframes")
    
    for tf in timeframes:
        print(f"   - {tf.value}")
    
    print("\\n✅ Basic functionality test completed!")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
'''
    
    with open('simple_test.py', 'w') as f:
        f.write(test_content)
    
    print("✅ Created simple_test.py")

if __name__ == "__main__":
    print("🔧 Fixing import issues...")
    fix_error_recovery_imports()
    create_simple_test()
    print("🎉 All fixes completed!")
