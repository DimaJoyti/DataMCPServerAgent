#!/usr/bin/env python3
"""
Simple test to verify the crypto portfolio fixes.
"""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.crypto_portfolio_complete_example import CryptoPortfolioDemo


async def test_crypto_portfolio():
    """Test the crypto portfolio system."""
    print("🧪 Testing Crypto Portfolio System...")
    
    try:
        demo = CryptoPortfolioDemo()
        print("✅ CryptoPortfolioDemo instance created successfully")
        
        # Test individual components
        print("\n📝 Testing individual components...")
        
        # Test portfolio initialization
        await demo.initialize_portfolio()
        print("✅ Portfolio initialization: PASSED")
        
        # Test market data fetching
        await demo.fetch_market_data()
        print("✅ Market data fetching: PASSED")
        
        # Test portfolio analysis
        await demo.analyze_portfolio_performance()
        print("✅ Portfolio analysis: PASSED")
        
        # Test risk assessment
        await demo.assess_portfolio_risk()
        print("✅ Risk assessment: PASSED")
        
        print("\n🎉 All tests passed! The crypto portfolio system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_crypto_portfolio())
    if success:
        print("\n✅ Crypto portfolio system is fixed and working!")
    else:
        print("\n❌ Crypto portfolio system still has issues.")
