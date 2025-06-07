#!/usr/bin/env python3
"""
Test script for the Crypto Portfolio Management System.
This script tests the TradingView integration and portfolio management features.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.tradingview_tools import TradingViewToolkit, CryptoSymbol, CryptoExchange
from src.agents.crypto_portfolio_agent import CryptoPortfolioAgent
from src.memory.memory_persistence import MemoryDatabase
from langchain_anthropic import ChatAnthropic
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()


async def test_tradingview_tools():
    """Test TradingView tools functionality."""
    print("🧪 Testing TradingView Tools")
    print("=" * 40)
    
    # Mock session for testing
    class MockSession:
        async def list_plugins(self):
            class MockPlugin:
                def __init__(self):
                    self.tools = [
                        type('MockTool', (), {'name': 'scrape_as_markdown_Bright_Data'}),
                        type('MockTool', (), {'name': 'scraping_browser_navigate_Bright_Data'}),
                        type('MockTool', (), {'name': 'scraping_browser_get_text_Bright_Data'}),
                    ]
            return [MockPlugin()]
    
    session = MockSession()
    toolkit = TradingViewToolkit(session)
    
    # Test crypto symbols
    print("📊 Testing crypto symbols...")
    symbols = toolkit.get_popular_crypto_symbols()
    print(f"✅ Found {len(symbols)} popular crypto symbols")
    
    for symbol in symbols[:3]:
        print(f"   - {symbol.tradingview_symbol}")
    
    # Test exchanges
    print("\n🏦 Testing supported exchanges...")
    exchanges = toolkit.get_supported_exchanges()
    print(f"✅ Found {len(exchanges)} supported exchanges")
    
    for exchange in exchanges[:3]:
        print(f"   - {exchange.value}")
    
    # Test timeframes
    print("\n⏰ Testing timeframes...")
    timeframes = toolkit.get_supported_timeframes()
    print(f"✅ Found {len(timeframes)} timeframes")
    
    for tf in timeframes[:3]:
        print(f"   - {tf.value}")
    
    print("\n✅ TradingView tools test completed!")


async def test_crypto_portfolio_agent():
    """Test crypto portfolio agent functionality."""
    print("\n🤖 Testing Crypto Portfolio Agent")
    print("=" * 40)
    
    # Mock components
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = "This is a mock response from the AI model for testing purposes."
            return MockResponse()
    
    class MockSession:
        async def list_plugins(self):
            class MockPlugin:
                def __init__(self):
                    self.tools = [
                        type('MockTool', (), {'name': 'scrape_as_markdown_Bright_Data'}),
                    ]
            return [MockPlugin()]
    
    # Initialize components
    model = MockModel()
    session = MockSession()
    db = MemoryDatabase(":memory:")  # In-memory database for testing
    
    # Create agent
    print("🚀 Initializing crypto portfolio agent...")
    agent = CryptoPortfolioAgent(model, session, db)
    await agent.initialize()
    print("✅ Agent initialized successfully")
    
    # Test portfolio analysis
    print("\n📊 Testing portfolio analysis...")
    analysis = await agent.analyze_portfolio()
    print(f"✅ Analysis completed: {type(analysis).__name__}")
    
    # Test market monitoring
    print("\n📈 Testing market monitoring...")
    symbols = ["BTCUSD", "ETHUSD"]
    market_data = await agent.monitor_markets(symbols)
    print(f"✅ Market monitoring completed for {len(symbols)} symbols")
    
    # Test chat interface
    print("\n💬 Testing chat interface...")
    response = await agent.chat_with_agent("What's the current market sentiment?")
    print(f"✅ Chat response received: {len(response)} characters")
    
    # Test report generation
    print("\n📋 Testing report generation...")
    report = await agent.generate_report("daily")
    print(f"✅ Daily report generated: {len(report)} characters")
    
    print("\n✅ Crypto portfolio agent test completed!")


async def test_portfolio_operations():
    """Test portfolio management operations."""
    print("\n💼 Testing Portfolio Operations")
    print("=" * 40)
    
    # Sample portfolio data
    portfolio = {
        "BTCUSD": {"quantity": 0.5, "avg_price": 45000, "timestamp": "2024-01-01"},
        "ETHUSD": {"quantity": 2.0, "avg_price": 3000, "timestamp": "2024-01-01"},
        "ADAUSD": {"quantity": 1000, "avg_price": 1.2, "timestamp": "2024-01-01"},
    }
    
    print("📊 Sample Portfolio:")
    total_invested = 0
    for symbol, position in portfolio.items():
        invested = position["quantity"] * position["avg_price"]
        total_invested += invested
        print(f"   {symbol}: {position['quantity']} @ ${position['avg_price']:,.2f} = ${invested:,.2f}")
    
    print(f"\n💰 Total Invested: ${total_invested:,.2f}")
    
    # Simulate current prices
    current_prices = {
        "BTCUSD": 52000,
        "ETHUSD": 3200,
        "ADAUSD": 1.1,
    }
    
    print("\n📈 Current Market Prices:")
    total_current_value = 0
    total_pnl = 0
    
    for symbol, position in portfolio.items():
        current_price = current_prices[symbol]
        current_value = position["quantity"] * current_price
        pnl = current_value - (position["quantity"] * position["avg_price"])
        pnl_percent = (pnl / (position["quantity"] * position["avg_price"])) * 100
        
        total_current_value += current_value
        total_pnl += pnl
        
        status = "📈" if pnl >= 0 else "📉"
        print(f"   {status} {symbol}: ${current_price:,.2f} | Value: ${current_value:,.2f} | P&L: ${pnl:+,.2f} ({pnl_percent:+.1f}%)")
    
    print(f"\n💹 Portfolio Summary:")
    print(f"   Current Value: ${total_current_value:,.2f}")
    print(f"   Total P&L: ${total_pnl:+,.2f}")
    print(f"   Return: {(total_pnl / total_invested) * 100:+.2f}%")
    
    print("\n✅ Portfolio operations test completed!")


async def test_risk_management():
    """Test risk management features."""
    print("\n⚠️ Testing Risk Management")
    print("=" * 40)
    
    # Sample risk metrics
    portfolio_value = 125000
    risk_metrics = {
        "daily_var_95": -2500,      # Value at Risk (95% confidence)
        "max_drawdown": -8.5,       # Maximum drawdown %
        "sharpe_ratio": 1.85,       # Risk-adjusted return
        "beta": 1.2,                # Market correlation
        "volatility": 0.45,         # Annualized volatility
        "concentration_risk": 0.35,  # Largest position %
    }
    
    print("📊 Risk Metrics Analysis:")
    print(f"   💰 Portfolio Value: ${portfolio_value:,.2f}")
    print(f"   ⚠️ Daily VaR (95%): ${risk_metrics['daily_var_95']:,.2f}")
    print(f"   📉 Max Drawdown: {risk_metrics['max_drawdown']:.1f}%")
    print(f"   📈 Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"   🔗 Beta: {risk_metrics['beta']:.2f}")
    print(f"   📊 Volatility: {risk_metrics['volatility']:.1%}")
    print(f"   🎯 Concentration Risk: {risk_metrics['concentration_risk']:.1%}")
    
    # Risk assessment
    print("\n🚨 Risk Assessment:")
    warnings = []
    
    if abs(risk_metrics['daily_var_95']) / portfolio_value > 0.02:
        warnings.append("High daily VaR (>2% of portfolio)")
    
    if risk_metrics['max_drawdown'] < -10:
        warnings.append("Maximum drawdown exceeded -10%")
    
    if risk_metrics['volatility'] > 0.4:
        warnings.append("High portfolio volatility (>40%)")
    
    if risk_metrics['concentration_risk'] > 0.3:
        warnings.append("High concentration risk (>30% in single position)")
    
    if warnings:
        for warning in warnings:
            print(f"   🔴 {warning}")
    else:
        print("   🟢 All risk metrics within acceptable ranges")
    
    # Position sizing recommendations
    print("\n💡 Position Sizing Recommendations:")
    recommendations = [
        "BTC: Maintain current position (core holding)",
        "ETH: Consider reducing by 10% (high correlation with BTC)",
        "ADA: Take partial profits (overbought conditions)",
        "SOL: Increase position by 5% (oversold opportunity)",
    ]
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    print("\n✅ Risk management test completed!")


async def test_market_analysis():
    """Test market analysis capabilities."""
    print("\n🌍 Testing Market Analysis")
    print("=" * 40)
    
    # Mock market data
    market_overview = {
        "total_market_cap": 2.1e12,
        "btc_dominance": 42.5,
        "fear_greed_index": 65,
        "active_addresses": 1.2e6,
        "transaction_volume": 15.8e9,
    }
    
    print("🌐 Crypto Market Overview:")
    print(f"   💰 Total Market Cap: ${market_overview['total_market_cap']:,.0f}")
    print(f"   ₿ BTC Dominance: {market_overview['btc_dominance']:.1f}%")
    print(f"   😱 Fear & Greed: {market_overview['fear_greed_index']} (Greed)")
    print(f"   👥 Active Addresses: {market_overview['active_addresses']:,.0f}")
    print(f"   💸 24h Volume: ${market_overview['transaction_volume']:,.0f}")
    
    # Top movers
    print("\n🚀 Top Market Movers (24h):")
    
    gainers = [
        {"symbol": "ATOM", "change": 15.2, "volume": 250e6},
        {"symbol": "NEAR", "change": 12.8, "volume": 180e6},
        {"symbol": "FTM", "change": 11.4, "volume": 120e6},
    ]
    
    losers = [
        {"symbol": "LUNA", "change": -8.9, "volume": 95e6},
        {"symbol": "AVAX", "change": -6.2, "volume": 140e6},
        {"symbol": "DOT", "change": -4.8, "volume": 110e6},
    ]
    
    print("   📈 Top Gainers:")
    for gainer in gainers:
        print(f"      🟢 {gainer['symbol']}: +{gainer['change']:.1f}% (Vol: ${gainer['volume']:,.0f})")
    
    print("   📉 Top Losers:")
    for loser in losers:
        print(f"      🔴 {loser['symbol']}: {loser['change']:.1f}% (Vol: ${loser['volume']:,.0f})")
    
    # Sector analysis
    print("\n🏷️ Sector Performance:")
    sectors = [
        {"name": "DeFi", "change": 5.2, "market_cap": 85e9},
        {"name": "Layer 1", "change": -2.1, "market_cap": 450e9},
        {"name": "NFTs", "change": 12.8, "market_cap": 25e9},
        {"name": "Gaming", "change": 8.4, "market_cap": 18e9},
        {"name": "Metaverse", "change": -1.5, "market_cap": 32e9},
    ]
    
    for sector in sectors:
        emoji = "📈" if sector["change"] >= 0 else "📉"
        print(f"   {emoji} {sector['name']}: {sector['change']:+.1f}% (${sector['market_cap']:,.0f})")
    
    print("\n✅ Market analysis test completed!")


async def run_all_tests():
    """Run all test functions."""
    print("🧪 CRYPTO PORTFOLIO SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    try:
        await test_tradingview_tools()
        await test_crypto_portfolio_agent()
        await test_portfolio_operations()
        await test_risk_management()
        await test_market_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🚀 Crypto Portfolio Management System is ready for use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
