# Fetch.ai Advanced Crypto Trading System

A sophisticated crypto leverage advisory system that combines n8n workflows with fetch.ai agents to create an intelligent trading ecosystem. The system analyzes market data, sentiment, technical indicators, and macroeconomic factors to provide optimized trading recommendations.

## Overview

This system consists of six specialized agents that work together to provide comprehensive trading recommendations:

1. **Sentiment Intelligence Agent**: Analyzes news and social media with VADER sentiment analysis, incorporating source credibility weighting.
2. **Technical Analysis Agent**: Performs multi-timeframe analysis with primary and secondary indicators.
3. **Risk Management Agent**: Implements position sizing, stop-loss calculation, and a curve.fi-inspired soft liquidation process.
4. **Regulatory Compliance Agent**: Manages Swiss tax reporting and banking regulations.
5. **Macro-Correlation Agent**: Analyzes relationships between crypto and traditional markets.
6. **Learning Optimization Agent**: Continuously improves system performance.

## Architecture

The system is built on the Fetch.ai uAgents framework, which allows for creating autonomous AI agents in Python. Each agent is specialized for a specific task and communicates with other agents to share information and insights.

The main components of the system are:

- **Core Agents**: The six specialized agents mentioned above.
- **Integration Layer**: Connects with n8n workflows.
- **Data Layer**: Handles market data, sentiment analysis, and technical indicators.
- **Trading Layer**: Implements trading strategies and risk management.
- **Execution Layer**: Connects to cryptocurrency exchanges for real trading.
- **Visualization Layer**: Provides a dashboard for monitoring the system.

## Key Features

- **Multi-Agent Architecture**: Six specialized agents working together
- **Real-Time Analysis**: Continuous monitoring of market conditions
- **Risk Management**: Sophisticated position sizing and risk assessment
- **Regulatory Compliance**: Swiss tax reporting and banking regulations
- **Machine Learning**: Continuous improvement through ML models
- **Visualization**: Interactive dashboard for monitoring
- **n8n Integration**: Seamless workflow automation
- **Multiple Data Sources**: Comprehensive market data analysis
- **Real Trading Execution**: Connect to exchange APIs for real trading

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

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
   EXCHANGE_API_KEY=your_api_key
   EXCHANGE_API_SECRET=your_api_secret
   ```

## Usage

### Running the Trading System

To run the complete trading system:

```
python -m src.agents.fetch_ai.main
```

### Command Line Arguments

The system supports several command line arguments:

- `--exchange`: Exchange to use (default: binance)
- `--symbols`: Symbols to track (default: BTC/USD ETH/USD)
- `--api-key`: API key for the exchange
- `--api-secret`: API secret for the exchange
- `--port`: Port for the agent server (default: 8000)
- `--endpoint`: Endpoint for the agent server

Example:

```bash
python -m src.agents.fetch_ai.main --exchange kraken --symbols BTC/USD ETH/USD ADA/USD
```

## Agent Details

### Sentiment Intelligence Agent

The Sentiment Intelligence Agent analyzes news and social media content to determine market sentiment. It uses the VADER sentiment analysis library to calculate sentiment scores and incorporates source credibility weighting to provide more accurate sentiment analysis.

### Technical Analysis Agent

The Technical Analysis Agent performs multi-timeframe analysis using various technical indicators. It calculates primary indicators (RSI, MACD, Bollinger Bands) and secondary indicators (Volume, ATR, ADX) to identify trends and potential entry/exit points.

### Risk Management Agent

The Risk Management Agent implements position sizing, stop-loss calculation, and a curve.fi-inspired soft liquidation process. It helps manage risk by determining appropriate position sizes and stop-loss levels based on account balance and market conditions.

### Regulatory Compliance Agent

The Regulatory Compliance Agent manages Swiss tax reporting and banking regulations. It ensures that all trading activities comply with relevant regulations and provides tax reporting capabilities.

### Macro-Correlation Agent

The Macro-Correlation Agent analyzes relationships between crypto and traditional markets. It calculates correlations between cryptocurrencies and traditional market indices, commodities, and currencies to identify potential market trends.

### Learning Optimization Agent

The Learning Optimization Agent continuously improves system performance through machine learning. It trains models to predict price movements, volatility, trading volume, and sentiment impact, and uses these predictions to enhance trading recommendations.

## Enhanced Features

### Trading Execution

The Trading Execution module connects the system to cryptocurrency exchanges for real trading. It provides order management, position tracking, and execution strategies.

```bash
python -m src.agents.fetch_ai.trading_execution
```

### Visualization Dashboard

The Visualization Dashboard provides a web-based interface for monitoring the trading system. It displays trading recommendations, market data, agent insights, and system performance.

```bash
python -m src.agents.fetch_ai.dashboard
```

### Advanced Machine Learning

The Advanced Machine Learning module provides more sophisticated machine learning models for prediction and continuous improvement. It includes ensemble models, hyperparameter optimization, and model evaluation.

```bash
python -m src.agents.fetch_ai.advanced_ml_models
```

### Additional Data Sources

The Additional Data Sources module integrates with various data sources for market data, news, social media, and other relevant information. It provides a unified interface for accessing all data sources.

```bash
python -m src.agents.fetch_ai.data_sources
```

### Testing with Real Market Data

The system includes a testing module for testing with real market data. It fetches real-time market data, generates trading recommendations, and analyzes the results.

```bash
python -m src.agents.fetch_ai.test_real_data
```

### Running the Enhanced System

To run the enhanced system with all components:

```bash
python -m src.agents.fetch_ai.run_enhanced_system
```

To run specific components:

```bash
python -m src.agents.fetch_ai.run_enhanced_system --components trading dashboard
```

## Integration with n8n

The system is designed to integrate with n8n workflows. The integration layer allows for:

- Sending trading signals to n8n workflows
- Receiving data from n8n workflows
- Triggering n8n workflows based on specific events
- Processing data from n8n workflows

```bash
python -m src.agents.fetch_ai.n8n_integration
```

## Documentation

For detailed documentation, see the [DOCUMENTATION.md](DOCUMENTATION.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Fetch.ai](https://fetch.ai/) for the uAgents framework
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange API integration
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [TA-Lib](https://github.com/mrjbq7/ta-lib) for technical analysis indicators
- [Dash](https://dash.plotly.com/) for the visualization dashboard
- [Plotly](https://plotly.com/) for interactive charts
- [scikit-learn](https://scikit-learn.org/) for machine learning models
