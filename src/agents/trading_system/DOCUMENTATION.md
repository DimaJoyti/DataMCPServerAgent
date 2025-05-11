# Fetch.ai Advanced Crypto Trading System - Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Agents](#core-agents)
4. [Installation and Setup](#installation-and-setup)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Trading Execution](#trading-execution)
8. [Integration with n8n](#integration-with-n8n)
9. [Visualization Dashboard](#visualization-dashboard)
10. [Advanced Machine Learning](#advanced-machine-learning)
11. [Additional Data Sources](#additional-data-sources)
12. [Testing Framework](#testing-framework)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)
15. [Contributing](#contributing)

## Introduction

The Fetch.ai Advanced Crypto Trading System is a sophisticated crypto leverage advisory system that combines n8n workflows with fetch.ai agents to create an intelligent trading ecosystem. The system analyzes market data, sentiment, technical indicators, and macroeconomic factors to provide optimized trading recommendations.

### Key Features

- **Multi-Agent Architecture**: Six specialized agents working together
- **Real-Time Analysis**: Continuous monitoring of market conditions
- **Risk Management**: Sophisticated position sizing and risk assessment
- **Regulatory Compliance**: Swiss tax reporting and banking regulations
- **Machine Learning**: Continuous improvement through ML models
- **Visualization**: Interactive dashboard for monitoring
- **n8n Integration**: Seamless workflow automation
- **Multiple Data Sources**: Comprehensive market data analysis
- **Real Trading Execution**: Connect to exchange APIs for real trading

## System Architecture

The system is built on the Fetch.ai uAgents framework, which allows for creating autonomous AI agents in Python. Each agent is specialized for a specific task and communicates with other agents to share information and insights.

### Component Layers

1. **Core Agents Layer**: The six specialized agents that form the intelligence of the system
2. **Integration Layer**: Connects with n8n workflows and external systems
3. **Data Layer**: Handles market data, sentiment analysis, and technical indicators
4. **Trading Layer**: Implements trading strategies and risk management
5. **Execution Layer**: Connects to cryptocurrency exchanges for real trading
6. **Visualization Layer**: Provides a dashboard for monitoring the system

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Visualization Layer                       │
│                        (Dashboard, Charts)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                        Integration Layer                         │
│                      (n8n Workflows, APIs)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                         Core Agents Layer                        │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│  Sentiment  │  Technical  │    Risk     │ Regulatory  │  Macro  │
│    Agent    │    Agent    │    Agent    │    Agent    │  Agent  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                          Data Layer                              │
│            (Market Data, News, Social Media, On-Chain)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                        Trading Layer                             │
│           (Strategies, Signals, Recommendations)                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                       Execution Layer                            │
│                 (Order Management, Positions)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                        ┌───────┴───────┐
                        │  Exchanges    │
                        │  (CCXT API)   │
                        └───────────────┘
```

## Core Agents

### Sentiment Intelligence Agent

The Sentiment Intelligence Agent analyzes news and social media content to determine market sentiment. It uses the VADER sentiment analysis library to calculate sentiment scores and incorporates source credibility weighting to provide more accurate sentiment analysis.

**Key Features:**

- News and social media analysis
- Source credibility weighting
- VADER sentiment analysis
- Real-time sentiment tracking

### Technical Analysis Agent

The Technical Analysis Agent performs multi-timeframe analysis using various technical indicators. It calculates primary indicators (RSI, MACD, Bollinger Bands) and secondary indicators (Volume, ATR, ADX) to identify trends and potential entry/exit points.

**Key Features:**

- Multi-timeframe analysis
- Primary and secondary indicators
- Trend identification
- Support and resistance detection

### Risk Management Agent

The Risk Management Agent implements position sizing, stop-loss calculation, and a curve.fi-inspired soft liquidation process. It helps manage risk by determining appropriate position sizes and stop-loss levels based on account balance and market conditions.

**Key Features:**

- Position sizing based on risk percentage
- Stop-loss calculation
- Take-profit calculation
- Soft liquidation process
- Risk level assessment

### Regulatory Compliance Agent

The Regulatory Compliance Agent manages Swiss tax reporting and banking regulations. It ensures that all trading activities comply with relevant regulations and provides tax reporting capabilities.

**Key Features:**

- Swiss tax reporting
- Banking regulations compliance
- AML/KYC compliance
- Transaction monitoring
- Tax liability calculation

### Macro-Correlation Agent

The Macro-Correlation Agent analyzes relationships between crypto and traditional markets. It calculates correlations between cryptocurrencies and traditional market indices, commodities, and currencies to identify potential market trends.

**Key Features:**

- Correlation analysis
- Traditional market monitoring
- Economic event tracking
- Macro sentiment assessment

### Learning Optimization Agent

The Learning Optimization Agent continuously improves system performance through machine learning. It trains models to predict price movements, volatility, trading volume, and sentiment impact, and uses these predictions to enhance trading recommendations.

**Key Features:**

- Machine learning models
- Continuous improvement
- Performance tracking
- Model evaluation

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Access to cryptocurrency exchange API (optional for real trading)
- n8n instance (optional for workflow integration)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
   cd DataMCPServerAgent
   ```

2. Install the required dependencies:

   ```bash
   pip install -r src/agents/fetch_ai/requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:

   ```env
   # Exchange API credentials
   ```

EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret

# n8n integration

N8N_BASE_URL=<http://localhost:5678>
N8N_API_KEY=your_n8n_api_key

# Additional data sources

CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key
LUNARCRUSH_API_KEY=your_lunarcrush_api_key
GLASSNODE_API_KEY=your_glassnode_api_key

````

## Configuration

The system can be configured through command-line arguments or by editing the configuration files. The main configuration options include:

### Exchange Configuration

- `exchange`: The exchange to use (default: binance)
- `symbols`: The symbols to track (default: BTC/USD ETH/USD)
- `api_key`: API key for the exchange
- `api_secret`: API secret for the exchange

### Agent Configuration

- `port`: Port for the agent server (default: 8000)
- `endpoint`: Endpoint for the agent server
- `seed`: Seed for deterministic address generation

### Trading Execution Configuration

- `dry_run`: Whether to run in dry run mode (no real trades)
- `auto_execute`: Whether to automatically execute recommendations
- `execution_strategy`: Strategy for executing trades (SIMPLE, TWAP, ICEBERG, SMART)
- `max_slippage`: Maximum allowed slippage for trades

### Dashboard Configuration

- `dashboard_port`: Port for the dashboard server (default: 8050)

## Usage

### Running the Basic Trading System

To run the complete trading system:

```bash
python -m src.agents.trading_system.main
````

### Running the Enhanced System

To run the enhanced system with all components:

```bash
python -m src.agents.trading_system.run_enhanced_system
```

To run specific components:

```bash
python -m src.agents.trading_system.run_enhanced_system --components trading dashboard
```

### Command Line Arguments

The system supports several command line arguments:

- `--exchange`: Exchange to use (default: binance)
- `--symbols`: Symbols to track (default: BTC/USD ETH/USD)
- `--api-key`: API key for the exchange
- `--api-secret`: API secret for the exchange
- `--port`: Port for the agent server (default: 8000)
- `--endpoint`: Endpoint for the agent server
- `--n8n-url`: URL for n8n (default: <http://localhost:5678>)
- `--n8n-api-key`: API key for n8n
- `--dashboard-port`: Port for the dashboard (default: 8050)
- `--test-duration`: Duration of the test in seconds (default: 3600)
- `--components`: Components to run (default: all)

Example:

```bash
python -m src.agents.trading_system.run_enhanced_system --exchange kraken --symbols BTC/USD ETH/USD ADA/USD --components trading dashboard
```

## Trading Execution

The Trading Execution module connects the system to cryptocurrency exchanges for real trading using the CCXT library. It provides comprehensive order management, position tracking, and execution strategies for automated trading.

### Features

- **Real Trading Capabilities**:

  - **Multi-Exchange Support**: Compatible with 100+ cryptocurrency exchanges through CCXT
  - **Order Types**: Market, limit, stop-loss, take-profit, and trailing stop orders
  - **Position Management**: Open, modify, and close positions with proper risk management
  - **P&L Tracking**: Real-time tracking of unrealized and realized profit/loss
  - **Portfolio Management**: Track overall portfolio performance and exposure

- **Execution Strategies**:

  - **Simple Execution**: Immediate market order execution
  - **TWAP (Time-Weighted Average Price)**: Split orders over time to minimize market impact
  - **Iceberg Orders**: Split large orders into smaller ones to hide true order size
  - **Smart Execution**: Adaptive strategy based on market conditions and liquidity

- **Risk Management Integration**:

  - **Automatic Stop-Loss**: Create stop-loss orders based on risk parameters
  - **Take-Profit Orders**: Set take-profit levels for automatic profit realization
  - **Position Sizing**: Determine position size based on account balance and risk tolerance
  - **Maximum Slippage Control**: Limit execution to prevent excessive slippage

- **Safety Features**:
  - **Dry Run Mode**: Test execution without placing real orders
  - **API Key Permissions**: Operate with read-only permissions when not trading
  - **Error Handling**: Robust error handling for exchange API issues
  - **Execution Logging**: Detailed logging of all trading activities

### Implementation Details

The module is implemented in `trading_execution.py` and includes:

1. **TradingExecution Class**: Main class for trading execution with methods for:

   - Connecting to exchanges using CCXT
   - Executing orders of various types
   - Managing positions and calculating P&L
   - Implementing different execution strategies
   - Handling risk management integration

2. **Order and Position Models**:

   - **Order**: Represents a trading order with type, side, amount, price, and status
   - **Position**: Represents an open position with entry price, current price, and P&L

3. **Execution Process**:
   - Receive trading recommendations from the trading system
   - Validate recommendations against risk parameters
   - Determine optimal execution strategy
   - Execute orders on the exchange
   - Track order status and update positions
   - Implement stop-loss and take-profit mechanisms

### Exchange Integration

The module uses CCXT (CryptoCurrency eXchange Trading Library) to connect to exchanges, providing:

- **Unified API**: Consistent interface across different exchanges
- **Market Data**: Access to order books, trades, and OHLCV data
- **Trading Operations**: Place, modify, and cancel orders
- **Account Information**: Balance, positions, and order history
- **Authentication**: Secure API key management

### Usage

To run the trading execution:

```bash
python -m src.agents.trading_system.trading_execution
```

### Configuration

The trading execution can be configured through the following options:

- `exchange_id`: Exchange to use (e.g., "binance", "kraken", "coinbase")
- `api_key`: API key for the exchange (required for trading)
- `api_secret`: API secret for the exchange (required for trading)
- `dry_run`: Whether to run in dry run mode (default: true)
- `auto_execute`: Whether to automatically execute recommendations (default: false)
- `execution_strategy`: Strategy for executing trades (SIMPLE, TWAP, ICEBERG, SMART)
- `max_slippage`: Maximum allowed slippage for trades (default: 0.01)

### Example Code

```python
# Create trading execution instance
execution = TradingExecution(
    exchange_id="binance",
    api_key=os.getenv('EXCHANGE_API_KEY'),
    api_secret=os.getenv('EXCHANGE_API_SECRET'),
    trading_system=trading_system,
    risk_agent=risk_agent,
    dry_run=True  # Start in dry run mode for safety
)

# Configure execution
execution.set_execution_strategy(ExecutionStrategy.SMART)
execution.set_auto_execute(True)

# Execute a specific recommendation
recommendation = TradeRecommendation(
    symbol="BTC/USDT",
    signal=TradingSignal.BUY,
    entry_price=50000.0,
    stop_loss=48000.0,
    take_profit=55000.0,
    position_size=0.1,
    confidence=0.85
)
order = await execution.execute_recommendation(recommendation)

# Get current positions
positions = await execution.get_positions()
for position in positions:
    print(f"{position.symbol}: {position.amount} @ {position.entry_price}, P&L: {position.unrealized_pnl}")

# Close a position
await execution.close_position("BTC/USDT")
```

## Integration with n8n

The system integrates with n8n workflows for automation and external system integration. The n8n integration module provides webhooks and API calls for communication between the trading system and n8n workflows.

### Features

- **Webhooks**: Send data to n8n workflows through webhooks
- **API Calls**: Call n8n APIs to trigger workflows
- **Data Exchange**: Exchange data between the trading system and n8n
- **Event Triggers**: Trigger workflows based on specific events

### Usage

To run the n8n integration:

```bash
python -m src.agents.trading_system.n8n_integration
```

## Visualization Dashboard

The Visualization Dashboard provides a web-based interface for monitoring the trading system. It displays trading recommendations, market data, agent insights, and system performance.

### Features

- **Trading Recommendations**: View and analyze trading recommendations
- **Market Data**: Monitor market prices and volumes
- **Agent Insights**: View insights from each specialized agent
- **System Performance**: Track system performance metrics
- **Real-Time Updates**: Automatic updates as new data becomes available

### Usage

To run the dashboard:

```bash
python -m src.agents.trading_system.dashboard
```

## Advanced Machine Learning

The Advanced Machine Learning module provides sophisticated machine learning models for prediction and continuous improvement. It includes ensemble models, hyperparameter optimization, and model evaluation to enhance the trading system's predictive capabilities.

### Features

- **Multiple Model Types**:

  - **Random Forest**: Robust ensemble method using multiple decision trees
  - **Gradient Boosting**: Sequential ensemble technique that builds models to correct errors
  - **Neural Networks**: Multi-layer perceptron for complex pattern recognition
  - **Ensemble**: Combined approach using all three models with majority voting for classification and averaging for regression

- **Multiple Prediction Targets**:

  - **Price Direction**: Predict whether prices will go up, down, or sideways
  - **Volatility**: Forecast expected market volatility
  - **Trading Volume**: Predict changes in trading volume
  - **Sentiment Impact**: Estimate how sentiment will affect price movements
  - **Optimal Entry/Exit**: Identify optimal entry and exit points

- **Advanced Features**:
  - **Model Persistence**: Save and load trained models for continuous use
  - **Automated Hyperparameter Optimization**: Find optimal model parameters
  - **Feature Engineering**: Transform raw data into meaningful features
  - **Performance Metrics**: Track accuracy, precision, recall, and F1-score
  - **Ensemble Techniques**: Combine multiple models for improved predictions
  - **Continuous Learning**: Models improve over time with new data

### Implementation Details

The module is implemented in `advanced_ml_models.py` and includes:

1. **AdvancedMLModel Class**: Base class for all machine learning models with methods for:

   - Model creation based on type (Random Forest, Gradient Boosting, Neural Network, Ensemble)
   - Training with automatic feature scaling
   - Prediction with confidence scores
   - Model persistence (save/load)
   - Performance evaluation

2. **ModelManager Class**: Manages multiple models with capabilities for:

   - Creating and retrieving models by type and prediction target
   - Loading models from disk if available
   - Saving models to disk for persistence
   - Managing model lifecycle

3. **Prediction Pipeline**:
   - Data preprocessing and feature engineering
   - Model selection based on prediction task
   - Prediction with confidence scores
   - Result interpretation and recommendation generation

### Usage

To run the advanced machine learning:

```bash
python -m src.agents.trading_system.advanced_ml_models
```

### Example Code

```python
# Create model manager
manager = ModelManager()

# Get or create an ensemble model for price direction prediction
model = manager.get_model(
    model_type=ModelType.ENSEMBLE,
    target=PredictionTarget.PRICE_DIRECTION
)

# Train the model with market data
result = model.train(features, labels)
print(f"Training result: Accuracy: {result.accuracy}, F1: {result.f1_score}")

# Make predictions with confidence scores
predictions = model.predict(new_features)
probabilities = model.predict_proba(new_features)

# Save the model for future use
manager.save_model(ModelType.ENSEMBLE, PredictionTarget.PRICE_DIRECTION)
```

## Additional Data Sources

The Additional Data Sources module integrates with various data sources for market data, news, social media, and other relevant information. It provides a unified interface for accessing all data sources.

### Features

- **Market Data**: CoinGecko, CryptoCompare
- **News**: CryptoPanic
- **Social Media**: LunarCrush
- **Economic Calendar**: ForexFactory
- **On-Chain Metrics**: Glassnode
- **Alternative Data**: Fear & Greed Index

### Usage

To run the additional data sources:

```bash
python -m src.agents.trading_system.data_sources
```

## Testing Framework

The system includes a comprehensive testing framework for evaluating the trading system with real market data. It fetches real-time market data from exchanges, generates trading recommendations, and performs detailed analysis of the results.

### Testing Capabilities

- **Real Market Data Testing**:

  - **Live Exchange Data**: Connect to real exchanges for current market data
  - **Multi-Symbol Testing**: Test across multiple cryptocurrency pairs simultaneously
  - **Multi-Timeframe Analysis**: Test across different timeframes (1h, 4h, 1d)
  - **Historical Backtesting**: Test against historical data for validation
  - **Forward Testing**: Run tests in real-time with live market data

- **Performance Evaluation**:

  - **Signal Analysis**: Analyze distribution of buy/sell/hold signals
  - **Confidence Metrics**: Evaluate confidence levels of recommendations
  - **Recommendation Quality**: Assess accuracy of entry/exit points
  - **Strategy Comparison**: Compare different trading strategies
  - **Risk/Reward Analysis**: Evaluate risk/reward ratios of recommendations

- **Reporting and Visualization**:
  - **JSON Result Export**: Save test results in structured JSON format
  - **Performance Metrics**: Calculate key performance indicators
  - **Detailed Logging**: Comprehensive logging of test execution
  - **Result Persistence**: Store test results for comparison and analysis
  - **Test Configuration**: Flexible configuration of test parameters

### Implementation Details

The testing framework is implemented in `test_real_data.py` and includes:

1. **RealDataTester Class**: Main class for testing with methods for:

   - Connecting to exchanges and fetching market data
   - Running the trading system against real data
   - Generating and collecting trading recommendations
   - Analyzing test results and calculating performance metrics
   - Saving results to files for further analysis

2. **Testing Process**:

   - Initialize exchange connection and trading system
   - Fetch market data for specified symbols and timeframes
   - Generate trading recommendations based on real data
   - Collect and store recommendations and market data
   - Analyze results and calculate performance metrics
   - Generate detailed test reports

3. **Result Analysis**:
   - Count recommendations by signal type (buy, sell, hold)
   - Calculate average confidence of recommendations
   - Analyze recommendation distribution by symbol
   - Evaluate recommendation timing against price movements
   - Generate performance summary and detailed metrics

### Usage

To run the testing framework:

```bash
python -m src.agents.trading_system.test_real_data
```

### Configuration Options

The testing framework can be configured with the following options:

- `exchange_id`: Exchange to use for testing (default: "binance")
- `symbols`: List of symbols to test (default: ["BTC/USDT", "ETH/USDT"])
- `timeframes`: List of timeframes to test (default: ["1h", "4h", "1d"])
- `test_duration`: Duration of the test in seconds (default: 3600)
- `api_key`: API key for the exchange (optional, for private API access)
- `api_secret`: API secret for the exchange (optional, for private API access)

### Example Code

```python
# Create tester with custom configuration
tester = RealDataTester(
    exchange_id="kraken",
    api_key=os.getenv('EXCHANGE_API_KEY'),
    api_secret=os.getenv('EXCHANGE_API_SECRET'),
    symbols=["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD"],
    timeframes=["15m", "1h", "4h", "1d"],
    test_duration=7200  # 2 hours
)

# Run the test
await tester.run_test()

# Access test results
print(f"Total recommendations: {tester.results['performance']['total_recommendations']}")
print(f"Signal distribution: {tester.results['performance']['signal_counts']}")
print(f"Average confidence: {tester.results['performance']['average_confidence']:.2f}")

# Analyze specific recommendations
for rec in tester.results["recommendations"]:
    if rec["confidence"] > 0.8:
        print(f"High confidence recommendation: {rec['symbol']} {rec['signal']} at {rec['timestamp']}")
```

### Test Result Format

The test results are saved in JSON format with the following structure:

```json
{
  "performance": {
    "total_recommendations": 42,
    "signal_counts": {
      "buy": 15,
      "sell": 12,
      "hold": 15
    },
    "average_confidence": 0.76
  },
  "recommendations": [
    {
      "symbol": "BTC/USDT",
      "signal": "buy",
      "entry_price": 50000.0,
      "stop_loss": 48000.0,
      "take_profit": 55000.0,
      "confidence": 0.85,
      "strength": "strong",
      "timestamp": "2023-06-15T14:30:00.000Z"
    },
    ...
  ],
  "market_data": {
    "BTC/USDT": {
      "1h": [...],
      "4h": [...],
      "1d": [...]
    },
    ...
  }
}
```

## API Reference

The system provides a Python API for programmatic access to all functionality. The main classes and functions are:

- `AdvancedCryptoTradingSystem`: Main trading system class
- `SentimentIntelligenceAgent`: Sentiment analysis agent
- `TechnicalAnalysisAgent`: Technical analysis agent
- `RiskManagementAgent`: Risk management agent
- `RegulatoryComplianceAgent`: Regulatory compliance agent
- `MacroCorrelationAgent`: Macro correlation agent
- `LearningOptimizationAgent`: Learning optimization agent
- `TradingExecution`: Trading execution class
- `N8nIntegration`: n8n integration class
- `Dashboard`: Visualization dashboard class
- `ModelManager`: Machine learning model manager
- `DataSourceManager`: Data source manager

## Troubleshooting

### Common Issues

- **API Key Issues**: Ensure that your exchange API keys are correctly set up and have the necessary permissions.
- **Connection Issues**: Check your internet connection and ensure that the exchange API is accessible.
- **Dependency Issues**: Make sure all dependencies are correctly installed.
- **Configuration Issues**: Check your configuration settings and ensure they are correct.

### Logging

The system uses Python's logging module for logging. Logs are written to the console and to log files:

- `fetch_ai_trading.log`: Main trading system log
- `fetch_ai_test.log`: Test log
- `fetch_ai_enhanced.log`: Enhanced system log

## Contributing

Contributions to the Fetch.ai Advanced Crypto Trading System are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

Please ensure that your code follows the project's coding style and includes appropriate tests and documentation.
