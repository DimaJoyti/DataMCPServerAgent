#!/usr/bin/env python3
"""
Example usage of TradingView Crypto Tools for portfolio management.
This script demonstrates how to use the TradingView scraping tools for cryptocurrency analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.tools.tradingview_tools import TradingViewToolkit, create_tradingview_tools
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()

async def demo_crypto_price_analysis():
    """Demonstrate cryptocurrency price analysis."""
    print("🚀 TradingView Crypto Tools Demo")
    print("=" * 50)

    # Mock session for demonstration
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

    # Create TradingView tools
    print("📊 Creating TradingView crypto tools...")
    tools = await create_tradingview_tools(session)

    print(f"✅ Created {len(tools)} specialized crypto tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")

    print("\n" + "=" * 50)

    # Demo popular crypto symbols
    toolkit = TradingViewToolkit(session)
    popular_symbols = toolkit.get_popular_crypto_symbols()

    print("💰 Popular Cryptocurrency Symbols:")
    for symbol in popular_symbols[:5]:
        print(f"   - {symbol.tradingview_symbol} ({symbol.base}/{symbol.quote})")

    print("\n" + "=" * 50)

    # Demo supported exchanges
    exchanges = toolkit.get_supported_exchanges()
    print("🏦 Supported Exchanges:")
    for exchange in exchanges:
        print(f"   - {exchange.value}")

    print("\n" + "=" * 50)

    # Demo timeframes
    timeframes = toolkit.get_supported_timeframes()
    print("⏰ Supported Timeframes:")
    for tf in timeframes:
        print(f"   - {tf.value}")

    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")

async def demo_portfolio_analysis():
    """Demonstrate portfolio analysis workflow."""
    print("\n📈 Portfolio Analysis Workflow Demo")
    print("=" * 50)

    # Sample portfolio
    portfolio = [
        {"symbol": "BTCUSD", "amount": 0.5, "avg_price": 45000},
        {"symbol": "ETHUSD", "amount": 2.0, "avg_price": 3000},
        {"symbol": "ADAUSD", "amount": 1000, "avg_price": 1.2},
        {"symbol": "SOLUSD", "amount": 10, "avg_price": 150},
    ]

    print("💼 Sample Portfolio:")
    total_value = 0
    for asset in portfolio:
        value = asset["amount"] * asset["avg_price"]
        total_value += value
        print(f"   - {asset['symbol']}: {asset['amount']} @ ${asset['avg_price']:,.2f} = ${value:,.2f}")

    print(f"\n💰 Total Portfolio Value: ${total_value:,.2f}")

    # Simulate analysis workflow
    print("\n🔍 Analysis Workflow:")
    print("   1. ✅ Extract current prices from TradingView")
    print("   2. ✅ Calculate P&L for each position")
    print("   3. ✅ Analyze technical indicators")
    print("   4. ✅ Check market sentiment")
    print("   5. ✅ Monitor news and events")
    print("   6. ✅ Generate risk metrics")
    print("   7. ✅ Create alerts and notifications")

    # Simulate P&L calculation
    print("\n📊 P&L Analysis (Simulated):")
    current_prices = {"BTCUSD": 52000, "ETHUSD": 3200, "ADAUSD": 1.1, "SOLUSD": 180}

    total_pnl = 0
    for asset in portfolio:
        symbol = asset["symbol"]
        current_price = current_prices.get(symbol, asset["avg_price"])
        pnl = (current_price - asset["avg_price"]) * asset["amount"]
        pnl_percent = ((current_price / asset["avg_price"]) - 1) * 100
        total_pnl += pnl

        status = "📈" if pnl >= 0 else "📉"
        print(f"   {status} {symbol}: ${pnl:+,.2f} ({pnl_percent:+.2f}%)")

    print(f"\n💹 Total P&L: ${total_pnl:+,.2f}")

async def demo_risk_management():
    """Demonstrate risk management features."""
    print("\n⚠️ Risk Management Demo")
    print("=" * 50)

    # Sample risk metrics
    risk_metrics = {
        "portfolio_value": 125000,
        "daily_var_95": -2500,  # Value at Risk (95% confidence)
        "max_drawdown": -8.5,   # Maximum drawdown %
        "sharpe_ratio": 1.85,   # Risk-adjusted return
        "beta": 1.2,            # Market correlation
        "volatility": 0.45,     # Annualized volatility
    }

    print("📊 Risk Metrics:")
    print(f"   💰 Portfolio Value: ${risk_metrics['portfolio_value']:,.2f}")
    print(f"   ⚠️ Daily VaR (95%): ${risk_metrics['daily_var_95']:,.2f}")
    print(f"   📉 Max Drawdown: {risk_metrics['max_drawdown']:.1f}%")
    print(f"   📈 Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"   🔗 Beta: {risk_metrics['beta']:.2f}")
    print(f"   📊 Volatility: {risk_metrics['volatility']:.1%}")

    # Risk alerts
    print("\n🚨 Risk Alerts:")
    alerts = []

    if risk_metrics["daily_var_95"] < -2000:
        alerts.append("High daily VaR detected")

    if risk_metrics["max_drawdown"] < -10:
        alerts.append("Maximum drawdown exceeded threshold")

    if risk_metrics["volatility"] > 0.4:
        alerts.append("High portfolio volatility")

    if alerts:
        for alert in alerts:
            print(f"   🔴 {alert}")
    else:
        print("   🟢 All risk metrics within acceptable ranges")

    # Position sizing recommendations
    print("\n💡 Position Sizing Recommendations:")
    print("   - BTC: Reduce position by 10% (high correlation risk)")
    print("   - ETH: Maintain current position")
    print("   - ADA: Consider taking profits (overbought)")
    print("   - SOL: Increase position by 5% (oversold)")

async def demo_market_analysis():
    """Demonstrate market analysis capabilities."""
    print("\n🌍 Market Analysis Demo")
    print("=" * 50)

    # Market overview
    market_data = {
        "total_market_cap": 2.1e12,  # $2.1T
        "btc_dominance": 42.5,       # 42.5%
        "fear_greed_index": 65,      # Greed
        "active_addresses": 1.2e6,   # 1.2M
        "transaction_volume": 15.8e9, # $15.8B
    }

    print("🌐 Crypto Market Overview:")
    print(f"   💰 Total Market Cap: ${market_data['total_market_cap']:,.0f}")
    print(f"   ₿ BTC Dominance: {market_data['btc_dominance']:.1f}%")
    print(f"   😱 Fear & Greed Index: {market_data['fear_greed_index']} (Greed)")
    print(f"   👥 Active Addresses: {market_data['active_addresses']:,.0f}")
    print(f"   💸 24h Volume: ${market_data['transaction_volume']:,.0f}")

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

    # Top movers
    print("\n🚀 Top Movers (24h):")
    gainers = [
        {"symbol": "ATOM", "change": 15.2},
        {"symbol": "NEAR", "change": 12.8},
        {"symbol": "FTM", "change": 11.4},
    ]

    losers = [
        {"symbol": "LUNA", "change": -8.9},
        {"symbol": "AVAX", "change": -6.2},
        {"symbol": "DOT", "change": -4.8},
    ]

    print("   📈 Top Gainers:")
    for gainer in gainers:
        print(f"      🟢 {gainer['symbol']}: +{gainer['change']:.1f}%")

    print("   📉 Top Losers:")
    for loser in losers:
        print(f"      🔴 {loser['symbol']}: {loser['change']:.1f}%")

async def main():
    """Run all demo functions."""
    try:
        await demo_crypto_price_analysis()
        await demo_portfolio_analysis()
        await demo_risk_management()
        await demo_market_analysis()

        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("🚀 Ready to build your crypto portfolio management system!")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
