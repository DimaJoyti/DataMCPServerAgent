"""
Visualization dashboard for the Fetch.ai Advanced Crypto Trading System.

This module provides a web-based dashboard for visualizing trading recommendations,
market data, and system performance.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dotenv import load_dotenv

from .trading_system import AdvancedCryptoTradingSystem, TradeRecommendation, TradingSignal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    """Dashboard for the Fetch.ai Advanced Crypto Trading System."""

    def __init__(
        self,
        trading_system: Optional[AdvancedCryptoTradingSystem] = None,
        port: int = 8050,
        debug: bool = False
    ):
        """Initialize the dashboard.

        Args:
            trading_system: Trading system to visualize
            port: Port for the dashboard server
            debug: Whether to run in debug mode
        """
        self.trading_system = trading_system
        self.port = port
        self.debug = debug

        # Initialize data storage
        self.data = {
            "recommendations": [],
            "market_data": {},
            "performance": {}
        }

        # Initialize dashboard
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Fetch.ai Trading Dashboard"
        )

        # Set up layout
        self._setup_layout()

        # Set up callbacks
        self._setup_callbacks()

    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Fetch.ai Advanced Crypto Trading System", className="text-center my-4"),
                    html.Hr()
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.H3("Trading Recommendations", className="text-center"),
                    dcc.Graph(id="recommendations-chart"),
                    html.Div(id="recommendations-table")
                ], width=12)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H3("Market Data", className="text-center"),
                    dcc.Dropdown(
                        id="symbol-dropdown",
                        options=[
                            {"label": "BTC/USD", "value": "BTC/USD"},
                            {"label": "ETH/USD", "value": "ETH/USD"},
                            {"label": "ADA/USD", "value": "ADA/USD"},
                            {"label": "SOL/USD", "value": "SOL/USD"}
                        ],
                        value="BTC/USD",
                        clearable=False
                    ),
                    dcc.Graph(id="price-chart")
                ], width=6),

                dbc.Col([
                    html.H3("System Performance", className="text-center"),
                    dcc.Graph(id="performance-chart"),
                    html.Div(id="performance-stats")
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H3("Agent Insights", className="text-center"),
                    dbc.Tabs([
                        dbc.Tab(label="Sentiment", tab_id="sentiment-tab"),
                        dbc.Tab(label="Technical", tab_id="technical-tab"),
                        dbc.Tab(label="Risk", tab_id="risk-tab"),
                        dbc.Tab(label="Macro", tab_id="macro-tab"),
                        dbc.Tab(label="Learning", tab_id="learning-tab")
                    ], id="agent-tabs"),
                    html.Div(id="agent-content")
                ], width=12)
            ]),

            dcc.Interval(
                id="interval-component",
                interval=60 * 1000,  # 1 minute in milliseconds
                n_intervals=0
            )
        ], fluid=True)

    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""

        @self.app.callback(
            [
                Output("recommendations-chart", "figure"),
                Output("recommendations-table", "children")
            ],
            [Input("interval-component", "n_intervals")]
        )
        def update_recommendations(n):
            """Update recommendations chart and table."""
            # Load recommendations
            recommendations = self._load_recommendations()

            if not recommendations:
                return self._empty_figure("No recommendations available"), html.Div("No recommendations available")

            # Create figure
            fig = go.Figure()

            # Add buy signals
            buy_recs = [r for r in recommendations if r["signal"] == "buy"]
            if buy_recs:
                buy_times = [datetime.fromisoformat(r["timestamp"]) for r in buy_recs]
                buy_prices = [r["entry_price"] for r in buy_recs]
                buy_confidences = [r["confidence"] for r in buy_recs]
                buy_sizes = [c * 20 for c in buy_confidences]

                fig.add_trace(go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(
                        size=buy_sizes,
                        color="green",
                        symbol="triangle-up"
                    ),
                    name="Buy Signals"
                ))

            # Add sell signals
            sell_recs = [r for r in recommendations if r["signal"] == "sell"]
            if sell_recs:
                sell_times = [datetime.fromisoformat(r["timestamp"]) for r in sell_recs]
                sell_prices = [r["entry_price"] for r in sell_recs]
                sell_confidences = [r["confidence"] for r in sell_recs]
                sell_sizes = [c * 20 for c in sell_confidences]

                fig.add_trace(go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(
                        size=sell_sizes,
                        color="red",
                        symbol="triangle-down"
                    ),
                    name="Sell Signals"
                ))

            # Update layout
            fig.update_layout(
                title="Trading Recommendations",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=400
            )

            # Create table
            table = dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Time"),
                    html.Th("Symbol"),
                    html.Th("Signal"),
                    html.Th("Strength"),
                    html.Th("Entry Price"),
                    html.Th("Stop Loss"),
                    html.Th("Take Profit"),
                    html.Th("Confidence")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(datetime.fromisoformat(r["timestamp"]).strftime("%Y-%m-%d %H:%M")),
                        html.Td(r["symbol"]),
                        html.Td(r["signal"].upper(), style={"color": "green" if r["signal"] == "buy" else "red"}),
                        html.Td(r["strength"]),
                        html.Td(f"${r['entry_price']:.2f}"),
                        html.Td(f"${r['stop_loss']:.2f}"),
                        html.Td(f"${r['take_profit']:.2f}"),
                        html.Td(f"{r['confidence']:.2f}")
                    ]) for r in recommendations[:5]
                ])
            ], bordered=True, dark=True, hover=True, responsive=True, striped=True)

            return fig, table

        @self.app.callback(
            Output("price-chart", "figure"),
            [
                Input("interval-component", "n_intervals"),
                Input("symbol-dropdown", "value")
            ]
        )
        def update_price_chart(n, symbol):
            """Update price chart."""
            # Load market data
            market_data = self._load_market_data(symbol)

            if not market_data:
                return self._empty_figure(f"No market data available for {symbol}")

            # Create figure
            fig = go.Figure()

            # Add price line
            times = [datetime.fromisoformat(d["timestamp"]) for d in market_data]
            prices = [d["price"] for d in market_data]

            fig.add_trace(go.Scatter(
                x=times,
                y=prices,
                mode="lines",
                name="Price"
            ))

            # Update layout
            fig.update_layout(
                title=f"{symbol} Price",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                height=400
            )

            return fig

        @self.app.callback(
            [
                Output("performance-chart", "figure"),
                Output("performance-stats", "children")
            ],
            [Input("interval-component", "n_intervals")]
        )
        def update_performance(n):
            """Update performance chart and stats."""
            # Load performance data
            performance = self._load_performance()

            if not performance:
                return self._empty_figure("No performance data available"), html.Div("No performance data available")

            # Create figure
            fig = go.Figure()

            # Add performance metrics
            if "accuracy_over_time" in performance:
                times = [datetime.fromisoformat(d["timestamp"]) for d in performance["accuracy_over_time"]]
                values = [d["value"] for d in performance["accuracy_over_time"]]

                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode="lines",
                    name="Prediction Accuracy"
                ))

            # Update layout
            fig.update_layout(
                title="System Performance",
                xaxis_title="Time",
                yaxis_title="Accuracy",
                template="plotly_dark",
                height=400
            )

            # Create stats
            stats = dbc.Card([
                dbc.CardHeader("Performance Statistics"),
                dbc.CardBody([
                    html.P(f"Total Recommendations: {performance.get('total_recommendations', 0)}"),
                    html.P(f"Accuracy: {performance.get('accuracy', 0):.2f}"),
                    html.P(f"Profit/Loss: {performance.get('profit_loss', 0):.2f}%")
                ])
            ])

            return fig, stats

        @self.app.callback(
            Output("agent-content", "children"),
            [
                Input("agent-tabs", "active_tab"),
                Input("interval-component", "n_intervals")
            ]
        )
        def update_agent_content(active_tab, n):
            """Update agent content."""
            if active_tab == "sentiment-tab":
                return self._render_sentiment_tab()
            elif active_tab == "technical-tab":
                return self._render_technical_tab()
            elif active_tab == "risk-tab":
                return self._render_risk_tab()
            elif active_tab == "macro-tab":
                return self._render_macro_tab()
            elif active_tab == "learning-tab":
                return self._render_learning_tab()
            else:
                return html.Div("Select a tab to view agent insights")

    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message.

        Args:
            message: Message to display

        Returns:
            Empty figure
        """
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        return fig

    def _load_recommendations(self) -> List[Dict[str, Any]]:
        """Load trading recommendations.

        Returns:
            List of recommendations
        """
        if self.trading_system:
            # Get recommendations from trading system
            return [rec.dict() for rec in self.trading_system.state.recent_recommendations]
        else:
            # Try to load from file
            try:
                with open("fetch_ai_recommendations.json", "r") as f:
                    return json.load(f)
            except:
                return []

    def _load_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Load market data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of market data points
        """
        if symbol in self.data["market_data"]:
            return self.data["market_data"][symbol]

        # Generate mock data
        now = datetime.now()
        data = []

        for i in range(100):
            time = now - timedelta(minutes=i)
            price = 50000 - i * 10 + (i % 10) * 20  # Mock price data

            data.append({
                "timestamp": time.isoformat(),
                "price": price,
                "volume": 1000000 - i * 1000
            })

        self.data["market_data"][symbol] = data
        return data

    def _load_performance(self) -> Dict[str, Any]:
        """Load system performance data.

        Returns:
            Performance data
        """
        if self.data["performance"]:
            return self.data["performance"]

        # Generate mock data
        now = datetime.now()
        accuracy_over_time = []

        for i in range(20):
            time = now - timedelta(hours=i)
            accuracy = 0.7 + (i % 10) * 0.02  # Mock accuracy data

            accuracy_over_time.append({
                "timestamp": time.isoformat(),
                "value": accuracy
            })

        self.data["performance"] = {
            "total_recommendations": 50,
            "accuracy": 0.75,
            "profit_loss": 12.5,
            "accuracy_over_time": accuracy_over_time
        }

        return self.data["performance"]

    def _render_sentiment_tab(self) -> html.Div:
        """Render sentiment tab content.

        Returns:
            Tab content
        """
        return html.Div([
            html.H4("Sentiment Analysis", className="text-center my-3"),
            dcc.Graph(
                figure=self._create_sentiment_chart()
            ),
            html.Div([
                html.H5("Recent News", className="mt-3"),
                html.Ul([
                    html.Li([
                        html.A(
                            "Bitcoin price surges after positive regulatory news",
                            href="#",
                            className="text-info"
                        ),
                        html.Span(" - CryptoNews (Sentiment: Positive)")
                    ]),
                    html.Li([
                        html.A(
                            "Ethereum upgrade delayed, developers cite security concerns",
                            href="#",
                            className="text-info"
                        ),
                        html.Span(" - CoinDesk (Sentiment: Negative)")
                    ]),
                    html.Li([
                        html.A(
                            "Major bank announces crypto custody service",
                            href="#",
                            className="text-info"
                        ),
                        html.Span(" - Bloomberg (Sentiment: Positive)")
                    ])
                ])
            ])
        ])

    def _create_sentiment_chart(self) -> go.Figure:
        """Create sentiment chart.

        Returns:
            Sentiment chart
        """
        fig = go.Figure()

        # Add sentiment data
        now = datetime.now()
        times = [now - timedelta(days=i) for i in range(7)]
        sentiments = [0.6, 0.2, -0.3, 0.1, 0.5, 0.7, 0.4]  # Mock sentiment data

        fig.add_trace(go.Scatter(
            x=times,
            y=sentiments,
            mode="lines+markers",
            name="Sentiment Score"
        ))

        # Add zero line
        fig.add_shape(
            type="line",
            x0=times[-1],
            y0=0,
            x1=times[0],
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="dash"
            )
        )

        # Update layout
        fig.update_layout(
            title="Sentiment Analysis Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score (-1 to 1)",
            template="plotly_dark",
            height=400,
            yaxis=dict(
                range=[-1, 1]
            )
        )

        return fig

    def _render_technical_tab(self) -> html.Div:
        """Render technical tab content.

        Returns:
            Tab content
        """
        return html.Div([
            html.H4("Technical Analysis", className="text-center my-3"),
            dcc.Graph(
                figure=self._create_technical_chart()
            ),
            html.Div([
                html.H5("Technical Indicators", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("RSI"),
                            dbc.CardBody([
                                html.H3("42.5", className="text-warning"),
                                html.P("Neutral")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("MACD"),
                            dbc.CardBody([
                                html.H3("-0.15", className="text-danger"),
                                html.P("Bearish")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Bollinger Bands"),
                            dbc.CardBody([
                                html.H3("Lower Band", className="text-success"),
                                html.P("Bullish")
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def _create_technical_chart(self) -> go.Figure:
        """Create technical chart.

        Returns:
            Technical chart
        """
        fig = go.Figure()

        # Add price data
        now = datetime.now()
        times = [now - timedelta(hours=i) for i in range(24)]
        prices = [50000 - i * 10 + (i % 10) * 20 for i in range(24)]  # Mock price data

        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode="lines",
            name="Price"
        ))

        # Add moving average
        ma = [sum(prices[max(0, i-5):i+1]) / min(i+1, 5) for i in range(len(prices))]

        fig.add_trace(go.Scatter(
            x=times,
            y=ma,
            mode="lines",
            line=dict(
                color="orange",
                width=2
            ),
            name="5-period MA"
        ))

        # Update layout
        fig.update_layout(
            title="Price and Moving Average",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=400
        )

        return fig

    def _render_risk_tab(self) -> html.Div:
        """Render risk tab content.

        Returns:
            Tab content
        """
        return html.Div([
            html.H4("Risk Management", className="text-center my-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Position Sizing"),
                        dbc.CardBody([
                            html.H3("0.25 BTC", className="text-info"),
                            html.P("5% of account balance")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stop Loss"),
                        dbc.CardBody([
                            html.H3("$47,500", className="text-danger"),
                            html.P("5% below entry")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Take Profit"),
                        dbc.CardBody([
                            html.H3("$52,500", className="text-success"),
                            html.P("5% above entry")
                        ])
                    ])
                ])
            ]),
            html.Div([
                html.H5("Risk Assessment", className="mt-3"),
                dbc.Progress(value=65, color="warning", className="mb-3"),
                html.P("Current Risk Level: Medium")
            ])
        ])

    def _render_macro_tab(self) -> html.Div:
        """Render macro tab content.

        Returns:
            Tab content
        """
        return html.Div([
            html.H4("Macro Correlation Analysis", className="text-center my-3"),
            dcc.Graph(
                figure=self._create_correlation_chart()
            ),
            html.Div([
                html.H5("Upcoming Economic Events", className="mt-3"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Date"),
                        html.Th("Event"),
                        html.Th("Impact"),
                        html.Th("Expected Crypto Impact")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td("2023-06-15"),
                            html.Td("Federal Reserve Interest Rate Decision"),
                            html.Td("High"),
                            html.Td("Negative")
                        ]),
                        html.Tr([
                            html.Td("2023-06-20"),
                            html.Td("US CPI Data Release"),
                            html.Td("Medium"),
                            html.Td("Neutral")
                        ]),
                        html.Tr([
                            html.Td("2023-06-25"),
                            html.Td("ECB Monetary Policy Statement"),
                            html.Td("Medium"),
                            html.Td("Neutral")
                        ])
                    ])
                ], bordered=True, dark=True, hover=True, responsive=True, striped=True)
            ])
        ])

    def _create_correlation_chart(self) -> go.Figure:
        """Create correlation chart.

        Returns:
            Correlation chart
        """
        fig = go.Figure()

        # Add correlation data
        assets = ["S&P 500", "Gold", "US Dollar", "Oil", "10Y Treasury"]
        correlations = [0.65, -0.45, -0.7, 0.2, -0.3]  # Mock correlation data

        # Create bar chart
        fig.add_trace(go.Bar(
            x=assets,
            y=correlations,
            marker_color=["green" if c > 0 else "red" for c in correlations]
        ))

        # Add zero line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0,
            x1=len(assets) - 0.5,
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="dash"
            )
        )

        # Update layout
        fig.update_layout(
            title="Bitcoin Correlation with Traditional Assets",
            xaxis_title="Asset",
            yaxis_title="Correlation Coefficient (-1 to 1)",
            template="plotly_dark",
            height=400,
            yaxis=dict(
                range=[-1, 1]
            )
        )

        return fig

    def _render_learning_tab(self) -> html.Div:
        """Render learning tab content.

        Returns:
            Tab content
        """
        return html.Div([
            html.H4("Learning Optimization", className="text-center my-3"),
            dcc.Graph(
                figure=self._create_learning_chart()
            ),
            html.Div([
                html.H5("Model Performance", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Price Direction Model"),
                            dbc.CardBody([
                                html.H3("75.2%", className="text-success"),
                                html.P("Accuracy")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Volatility Model"),
                            dbc.CardBody([
                                html.H3("68.7%", className="text-warning"),
                                html.P("Accuracy")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Sentiment Impact Model"),
                            dbc.CardBody([
                                html.H3("72.1%", className="text-info"),
                                html.P("Accuracy")
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def _create_learning_chart(self) -> go.Figure:
        """Create learning chart.

        Returns:
            Learning chart
        """
        fig = go.Figure()

        # Add learning data
        now = datetime.now()
        times = [now - timedelta(days=i) for i in range(10)]
        accuracy = [0.65, 0.67, 0.68, 0.7, 0.69, 0.72, 0.73, 0.74, 0.75, 0.75]  # Mock accuracy data

        fig.add_trace(go.Scatter(
            x=times,
            y=accuracy,
            mode="lines+markers",
            name="Model Accuracy"
        ))

        # Update layout
        fig.update_layout(
            title="Model Accuracy Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            template="plotly_dark",
            height=400,
            yaxis=dict(
                range=[0.6, 0.8]
            )
        )

        return fig

    def run(self):
        """Run the dashboard."""
        self.app.run_server(debug=self.debug, port=self.port)

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="dashboard_trading_system",
        exchange_id="binance",
        api_key=os.getenv('EXCHANGE_API_KEY'),
        api_secret=os.getenv('EXCHANGE_API_SECRET')
    )

    # Create dashboard
    dashboard = Dashboard(trading_system=trading_system)

    # Run dashboard
    dashboard.run()

if __name__ == "__main__":
    asyncio.run(main())
