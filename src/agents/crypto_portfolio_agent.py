"""
Crypto Portfolio Management Agent.
This agent specializes in cryptocurrency portfolio management using TradingView data.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession

from src.tools.tradingview_tools import create_tradingview_tools, TradingViewToolkit
from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_recovery import ErrorRecoverySystem
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()

class CryptoPortfolioAgent:
    """An intelligent agent for cryptocurrency portfolio management."""

    def __init__(
        self,
        model: ChatAnthropic,
        session: ClientSession,
        db: MemoryDatabase,
        error_recovery: Optional[ErrorRecoverySystem] = None
    ):
        """Initialize the crypto portfolio agent.

        Args:
            model: The language model for decision making
            session: MCP client session for tools
            db: Memory database for persistence
            error_recovery: Error recovery manager
        """
        self.model = model
        self.session = session
        self.db = db
        self.error_recovery = error_recovery or ErrorRecoverySystem(db)
        self.tools = []
        self.toolkit = TradingViewToolkit(session)

        # Portfolio state
        self.portfolio = {}
        self.watchlist = []
        self.alerts = []
        self.risk_limits = {
            "max_position_size": 0.2,  # 20% max per position
            "max_daily_loss": 0.05,    # 5% max daily loss
            "min_cash_reserve": 0.1,   # 10% cash reserve
        }

    async def initialize(self):
        """Initialize the agent with tools and load portfolio state."""
        try:
            # Create TradingView tools
            self.tools = await create_tradingview_tools(self.session)

            # Load portfolio state from memory
            await self._load_portfolio_state()

            print("✅ Crypto Portfolio Agent initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise

    async def analyze_portfolio(self) -> Dict[str, Any]:
        """Analyze the current portfolio performance and risk."""
        try:
            analysis = {
                "timestamp": datetime.now(),
                "portfolio_value": 0.0,
                "total_pnl": 0.0,
                "positions": [],
                "risk_metrics": {},
                "recommendations": []
            }

            # Analyze each position
            for symbol, position in self.portfolio.items():
                position_analysis = await self._analyze_position(symbol, position)
                analysis["positions"].append(position_analysis)
                analysis["portfolio_value"] += position_analysis["current_value"]
                analysis["total_pnl"] += position_analysis["pnl"]

            # Calculate risk metrics
            analysis["risk_metrics"] = await self._calculate_risk_metrics(analysis)

            # Generate recommendations
            analysis["recommendations"] = await self._generate_recommendations(analysis)

            # Save analysis to memory
            await self._save_analysis(analysis)

            return analysis

        except Exception as e:
            await self.error_recovery.handle_error(e, "analyze_portfolio")
            return {"error": str(e)}

    async def monitor_markets(self, symbols: List[str]) -> Dict[str, Any]:
        """Monitor cryptocurrency markets for opportunities and risks."""
        try:
            market_data = {
                "timestamp": datetime.now(),
                "symbols": symbols,
                "price_data": {},
                "technical_signals": {},
                "sentiment_data": {},
                "alerts": []
            }

            # Get price data for each symbol
            for symbol in symbols:
                try:
                    # Use TradingView price tool
                    price_tool = next(tool for tool in self.tools if tool.name == "tradingview_crypto_price")
                    price_result = await price_tool.invoke({"symbol": symbol})
                    market_data["price_data"][symbol] = price_result

                    # Get technical analysis
                    analysis_tool = next(tool for tool in self.tools if tool.name == "tradingview_crypto_analysis")
                    analysis_result = await analysis_tool.invoke({"symbol": symbol})
                    market_data["technical_signals"][symbol] = analysis_result

                    # Check for alerts
                    alerts = await self._check_alerts(symbol, price_result)
                    market_data["alerts"].extend(alerts)

                except Exception as e:
                    print(f"Error monitoring {symbol}: {e}")
                    continue

            return market_data

        except Exception as e:
            await self.error_recovery.handle_error(e, "monitor_markets")
            return {"error": str(e)}

    async def execute_trade_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading signal with risk management."""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return {"status": "rejected", "reason": "Invalid signal"}

            # Check risk limits
            risk_check = await self._check_risk_limits(signal)
            if not risk_check["approved"]:
                return {"status": "rejected", "reason": risk_check["reason"]}

            # Calculate position size
            position_size = await self._calculate_position_size(signal)

# Simulate trade execution (in real implementation, this would connect to
# exchange APIs)
            trade_result = {
                "status": "executed",
                "symbol": signal["symbol"],
                "action": signal["action"],
                "quantity": position_size,
                "price": signal["price"],
                "timestamp": datetime.now(),
                "trade_id": f"trade_{datetime.now().timestamp()}"
            }

            # Update portfolio
            await self._update_portfolio(trade_result)

            # Log trade
            await self._log_trade(trade_result)

            return trade_result

        except Exception as e:
            await self.error_recovery.handle_error(e, "execute_trade_signal")
            return {"status": "error", "error": str(e)}

    async def generate_report(self, report_type: str = "daily") -> str:
        """Generate a portfolio performance report."""
        try:
            analysis = await self.analyze_portfolio()

            if report_type == "daily":
                return await self._generate_daily_report(analysis)
            elif report_type == "weekly":
                return await self._generate_weekly_report(analysis)
            elif report_type == "monthly":
                return await self._generate_monthly_report(analysis)
            else:
                return "Invalid report type. Use 'daily', 'weekly', or 'monthly'."

        except Exception as e:
            return f"Error generating report: {str(e)}"

    async def chat_with_agent(self, user_message: str) -> str:
        """Chat interface for the crypto portfolio agent."""
        try:
            # Prepare system message
            system_message = SystemMessage(content=f"""
You are a professional cryptocurrency portfolio manager with access to TradingView data and analysis tools.

Current Portfolio Status:
- Total Positions: {len(self.portfolio)}
- Watchlist: {len(self.watchlist)} symbols
- Active Alerts: {len(self.alerts)}

Available Tools:
{', '.join([tool.name for tool in self.tools])}

Your capabilities include:
- Real-time market analysis using TradingView data
- Portfolio performance tracking and optimization
- Risk management and position sizing
- Technical analysis and sentiment monitoring
- Trade signal generation and execution
- Comprehensive reporting and insights

Always provide actionable insights and consider risk management in your recommendations.
Use the available tools to gather current market data when needed.
""")

            # Get current market context if needed
            context = await self._get_market_context(user_message)

            # Prepare messages
            messages = [
                system_message,
                HumanMessage(content=f"{user_message}\n\nCurrent Market Context:\n{context}")
            ]

            # Get response from model
            response = await self.model.ainvoke(messages)

            return response.content

        except Exception as e:
            return f"Error processing request: {str(e)}"

    # Private helper methods
    async def _analyze_position(self, symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single portfolio position."""
        # This would use TradingView tools to get current price and calculate metrics
        return {
            "symbol": symbol,
            "quantity": position.get("quantity", 0),
            "avg_price": position.get("avg_price", 0),
            "current_price": 0,  # Would be fetched from TradingView
            "current_value": 0,
            "pnl": 0,
            "pnl_percent": 0,
        }

    async def _calculate_risk_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        return {
            "total_value": analysis["portfolio_value"],
            "daily_var": 0,  # Value at Risk
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "beta": 0,
            "volatility": 0,
        }

    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate portfolio recommendations based on analysis."""
        recommendations = []

        # Add sample recommendations
        if analysis["total_pnl"] < 0:
            recommendations.append("Consider reducing position sizes to limit losses")

        if len(analysis["positions"]) > 10:
            recommendations.append("Portfolio may be over-diversified, consider consolidating positions")

        return recommendations

    async def _check_alerts(self, symbol: str, price_data: str) -> List[Dict[str, Any]]:
        """Check for price alerts and notifications."""
        alerts = []

        # This would parse price_data and check against user-defined alerts
        # For now, return empty list

        return alerts

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trading signal."""
        required_fields = ["symbol", "action", "price", "confidence"]
        return all(field in signal for field in required_fields)

    async def _check_risk_limits(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Check if signal complies with risk limits."""
        # Implement risk limit checks
        return {"approved": True, "reason": ""}

    async def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate appropriate position size based on risk management."""
        # Implement position sizing logic
        return 0.1  # 10% of portfolio

    async def _update_portfolio(self, trade_result: Dict[str, Any]):
        """Update portfolio with trade result."""
        # Update internal portfolio state
        pass

    async def _log_trade(self, trade_result: Dict[str, Any]):
        """Log trade to database."""
        await self.db.store_memory(
            "trade_log",
            json.dumps(trade_result),
            {"type": "trade", "symbol": trade_result["symbol"]}
        )

    async def _load_portfolio_state(self):
        """Load portfolio state from memory."""
        try:
            portfolio_data = await self.db.retrieve_memory("portfolio_state")
            if portfolio_data:
                self.portfolio = json.loads(portfolio_data[0]["content"])
        except Exception:
            self.portfolio = {}

    async def _save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis to memory."""
        await self.db.store_memory(
            "portfolio_analysis",
            json.dumps(analysis, default=str),
            {"type": "analysis", "timestamp": str(analysis["timestamp"])}
        )

    async def _get_market_context(self, user_message: str) -> str:
        """Get relevant market context for user message."""
        # This would analyze the user message and fetch relevant market data
        return "Market context would be fetched here based on user query"

    async def _generate_daily_report(self, analysis: Dict[str, Any]) -> str:
        """Generate daily portfolio report."""
        return f"""
# 📊 Daily Portfolio Report - {analysis['timestamp'].strftime('%Y-%m-%d')}

## Portfolio Summary
- **Total Value**: ${analysis['portfolio_value']:,.2f}
- **Total P&L**: ${analysis['total_pnl']:+,.2f}
- **Number of Positions**: {len(analysis['positions'])}

## Top Performers
{self._format_top_performers(analysis['positions'])}

## Risk Metrics
{self._format_risk_metrics(analysis['risk_metrics'])}

## Recommendations
{chr(10).join([f"- {rec}" for rec in analysis['recommendations']])}
"""

    async def _generate_weekly_report(self, analysis: Dict[str, Any]) -> str:
        """Generate weekly portfolio report."""
        return "Weekly report would be generated here"

    async def _generate_monthly_report(self, analysis: Dict[str, Any]) -> str:
        """Generate monthly portfolio report."""
        return "Monthly report would be generated here"

    def _format_top_performers(self, positions: List[Dict[str, Any]]) -> str:
        """Format top performing positions."""
        if not positions:
            return "No positions to display"

        # Sort by P&L and take top 3
        sorted_positions = sorted(positions, key=lambda x: x.get('pnl', 0), reverse=True)[:3]

        result = ""
        for pos in sorted_positions:
            emoji = "📈" if pos.get('pnl', 0) >= 0 else "📉"
            result += f"{emoji} {pos['symbol']}: ${pos.get('pnl', 0):+,.2f}\n"

        return result

    def _format_risk_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format risk metrics for display."""
        return f"""
- **Portfolio Value**: ${metrics.get('total_value', 0):,.2f}
- **Daily VaR**: ${metrics.get('daily_var', 0):,.2f}
- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2f}%
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}
"""
