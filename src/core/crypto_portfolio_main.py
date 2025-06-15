#!/usr/bin/env python3
"""
Main entry point for the Crypto Portfolio Management System.
This module provides an interactive interface for managing cryptocurrency portfolios using TradingView data.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_anthropic import ChatAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.crypto_portfolio_agent import CryptoPortfolioAgent
from src.memory.memory_persistence import MemoryDatabase
from src.utils.env_config import load_dotenv
from src.utils.error_recovery import ErrorRecoverySystem

# Load environment variables
load_dotenv()


class CryptoPortfolioSystem:
    """Main system for cryptocurrency portfolio management."""

    def __init__(self):
        """Initialize the crypto portfolio system."""
        self.agent: Optional[CryptoPortfolioAgent] = None
        self.session: Optional[ClientSession] = None
        self.db: Optional[MemoryDatabase] = None
        self.model: Optional[ChatAnthropic] = None

    async def initialize(self):
        """Initialize all system components."""
        try:
            print("🚀 Initializing Crypto Portfolio Management System...")

            # Initialize database
            print("📊 Setting up memory database...")
            self.db = MemoryDatabase()

            # Initialize language model
            print("🧠 Initializing AI model...")
            self.model = ChatAnthropic(
                model="claude-3-sonnet-20240229", temperature=0.1, max_tokens=4000
            )

            # Initialize MCP session for Bright Data tools
            print("🔗 Connecting to Bright Data MCP server...")
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@brightdata/mcp-server-bright-data"],
                env=dict(os.environ),
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    # Initialize error recovery
                    error_recovery = ErrorRecoverySystem(self.db)

                    # Initialize crypto portfolio agent
                    print("💰 Setting up crypto portfolio agent...")
                    self.agent = CryptoPortfolioAgent(
                        model=self.model, session=session, db=self.db, error_recovery=error_recovery
                    )

                    await self.agent.initialize()

                    print("✅ System initialization completed!")

                    # Start interactive session
                    await self.run_interactive_session()

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            raise

    async def run_interactive_session(self):
        """Run the interactive crypto portfolio management session."""
        print("\n" + "=" * 60)
        print("🎯 CRYPTO PORTFOLIO MANAGEMENT SYSTEM")
        print("=" * 60)
        print("Welcome to your AI-powered cryptocurrency portfolio manager!")
        print("Powered by TradingView data and advanced analytics.")
        print("\nAvailable commands:")
        print("  📊 'analyze' - Analyze current portfolio")
        print("  📈 'monitor [symbols]' - Monitor specific cryptocurrencies")
        print("  📰 'news [symbol]' - Get latest crypto news")
        print("  📋 'report [daily/weekly/monthly]' - Generate performance report")
        print("  ⚙️ 'settings' - View/modify system settings")
        print("  ❓ 'help' - Show detailed help")
        print("  🚪 'quit' - Exit the system")
        print("\nOr just ask me anything about cryptocurrency markets and portfolio management!")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("👋 Thank you for using Crypto Portfolio Management System!")
                    break

                # Handle special commands
                if user_input.lower() == "analyze":
                    await self.handle_analyze_command()
                elif user_input.lower().startswith("monitor"):
                    await self.handle_monitor_command(user_input)
                elif user_input.lower().startswith("news"):
                    await self.handle_news_command(user_input)
                elif user_input.lower().startswith("report"):
                    await self.handle_report_command(user_input)
                elif user_input.lower() == "settings":
                    await self.handle_settings_command()
                elif user_input.lower() == "help":
                    await self.handle_help_command()
                else:
                    # General chat with agent
                    response = await self.agent.chat_with_agent(user_input)
                    print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    async def handle_analyze_command(self):
        """Handle portfolio analysis command."""
        print("\n📊 Analyzing your cryptocurrency portfolio...")

        try:
            analysis = await self.agent.analyze_portfolio()

            if "error" in analysis:
                print(f"❌ Analysis failed: {analysis['error']}")
                return

            # Display analysis results
            print("\n" + "=" * 50)
            print("📈 PORTFOLIO ANALYSIS RESULTS")
            print("=" * 50)
            print(f"💰 Total Portfolio Value: ${analysis['portfolio_value']:,.2f}")
            print(f"📊 Total P&L: ${analysis['total_pnl']:+,.2f}")
            print(f"🏦 Number of Positions: {len(analysis['positions'])}")

            if analysis["positions"]:
                print("\n📋 Position Details:")
                for pos in analysis["positions"]:
                    emoji = "📈" if pos.get("pnl", 0) >= 0 else "📉"
                    print(
                        f"  {emoji} {pos['symbol']}: ${pos.get('current_value', 0):,.2f} (P&L: ${pos.get('pnl', 0):+,.2f})"
                    )

            if analysis["recommendations"]:
                print("\n💡 Recommendations:")
                for rec in analysis["recommendations"]:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"❌ Error during analysis: {e}")

    async def handle_monitor_command(self, user_input: str):
        """Handle market monitoring command."""
        parts = user_input.split()
        symbols = parts[1:] if len(parts) > 1 else ["BTCUSD", "ETHUSD", "ADAUSD"]

        print(f"\n📈 Monitoring markets for: {', '.join(symbols)}")

        try:
            market_data = await self.agent.monitor_markets(symbols)

            if "error" in market_data:
                print(f"❌ Monitoring failed: {market_data['error']}")
                return

            print("\n" + "=" * 50)
            print("🌍 MARKET MONITORING RESULTS")
            print("=" * 50)

            for symbol in symbols:
                print(f"\n💰 {symbol}:")
                if symbol in market_data.get("price_data", {}):
                    print("  📊 Price Data: Available")
                if symbol in market_data.get("technical_signals", {}):
                    print("  📈 Technical Analysis: Available")

            if market_data.get("alerts"):
                print("\n🚨 Active Alerts:")
                for alert in market_data["alerts"]:
                    print(f"  ⚠️ {alert}")

        except Exception as e:
            print(f"❌ Error during monitoring: {e}")

    async def handle_news_command(self, user_input: str):
        """Handle crypto news command."""
        parts = user_input.split()
        symbol = parts[1] if len(parts) > 1 else "BTCUSD"

        print(f"\n📰 Fetching latest news for {symbol}...")

        try:
            # Use TradingView news tool
            news_tool = next(
                tool for tool in self.agent.tools if tool.name == "tradingview_crypto_news"
            )
            news_result = await news_tool.invoke({"symbol": symbol, "limit": 5})

            print("\n" + "=" * 50)
            print(f"📰 LATEST NEWS FOR {symbol}")
            print("=" * 50)
            print(news_result)

        except Exception as e:
            print(f"❌ Error fetching news: {e}")

    async def handle_report_command(self, user_input: str):
        """Handle report generation command."""
        parts = user_input.split()
        report_type = parts[1] if len(parts) > 1 else "daily"

        if report_type not in ["daily", "weekly", "monthly"]:
            print("❌ Invalid report type. Use 'daily', 'weekly', or 'monthly'.")
            return

        print(f"\n📋 Generating {report_type} portfolio report...")

        try:
            report = await self.agent.generate_report(report_type)

            print("\n" + "=" * 50)
            print(f"📊 {report_type.upper()} PORTFOLIO REPORT")
            print("=" * 50)
            print(report)

        except Exception as e:
            print(f"❌ Error generating report: {e}")

    async def handle_settings_command(self):
        """Handle settings command."""
        print("\n⚙️ System Settings:")
        print("=" * 30)
        print(f"🏦 Portfolio Positions: {len(self.agent.portfolio)}")
        print(f"👀 Watchlist Items: {len(self.agent.watchlist)}")
        print(f"🚨 Active Alerts: {len(self.agent.alerts)}")
        print("\n📊 Risk Limits:")
        for key, value in self.agent.risk_limits.items():
            print(f"  • {key}: {value}")

    async def handle_help_command(self):
        """Handle help command."""
        help_text = """
🎯 CRYPTO PORTFOLIO MANAGEMENT SYSTEM - HELP

📊 PORTFOLIO COMMANDS:
  • analyze                    - Analyze current portfolio performance
  • report [daily/weekly/monthly] - Generate performance reports

📈 MARKET COMMANDS:
  • monitor [symbols]          - Monitor specific cryptocurrencies
  • news [symbol]              - Get latest crypto news

⚙️ SYSTEM COMMANDS:
  • settings                   - View/modify system settings
  • help                       - Show this help message
  • quit                       - Exit the system

💬 NATURAL LANGUAGE:
You can also ask questions in natural language, such as:
  • "What's the current price of Bitcoin?"
  • "Should I buy Ethereum now?"
  • "Analyze the risk in my portfolio"
  • "What are the top performing cryptocurrencies today?"

🔧 FEATURES:
  • Real-time TradingView data integration
  • Advanced technical analysis
  • Risk management and position sizing
  • Portfolio performance tracking
  • Market sentiment analysis
  • Automated alerts and notifications

For more detailed information, visit our documentation.
"""
        print(help_text)


async def main():
    """Main entry point for the crypto portfolio system."""
    system = CryptoPortfolioSystem()
    await system.initialize()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)
