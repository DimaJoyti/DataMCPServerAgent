# Advanced Crypto Trading System

## Project Overview

We're developing a sophisticated crypto leverage advisory system that combines n8n workflows with fetch.ai agents to create an intelligent trading ecosystem. The system analyzes market data, sentiment, technical indicators, and macroeconomic factors to provide optimized trading recommendations.

The system is built on the Fetch.ai uAgents framework, which allows for creating autonomous AI agents in Python. Each agent is specialized for a specific task and communicates with other agents to share information and insights.

## Core Agents

The system consists of six specialized agents that work together to provide comprehensive trading recommendations:

1. **Sentiment Intelligence Agent**: Analyzes news and social media with VADER sentiment analysis, incorporates source credibility weighting.
2. **Technical Analysis Agent**: Performs multi-timeframe analysis with primary and secondary indicators.
3. **Risk Management Agent**: Implements position sizing, stop-loss calculation, and a curve.fi-inspired soft liquidation process.
4. **Regulatory Compliance Agent**: Manages Swiss tax reporting and banking regulations.
5. **Macro-Correlation Agent**: Analyzes relationships between crypto and traditional markets.
6. **Learning Optimization Agent**: Continuously improves system performance.

## Implementation Details

### Architecture

The system is built with the following components:

- **Core Agents**: The six specialized agents mentioned above.
- **Integration Layer**: Connects with n8n workflows.
- **Data Layer**: Handles market data, sentiment analysis, and technical indicators.
- **Trading Layer**: Implements trading strategies and risk management.
- **API Layer**: Connects to cryptocurrency exchanges using CCXT.

### Technologies Used

- **Fetch.ai uAgents**: Framework for creating autonomous AI agents
- **CCXT**: Library for cryptocurrency exchange API integration
- **VADER Sentiment Analysis**: Library for sentiment analysis
- **Technical Analysis Libraries**: For calculating technical indicators
- **Machine Learning**: For continuous improvement and prediction

### Installation

1. Install the required dependencies:

   ```bash
   pip install -r src/agents/fetch_ai/requirements.txt
   ```

2. Set up environment variables:
   Create a `.env` file with the following variables:

   ```env
   EXCHANGE_API_KEY=your_api_key
   EXCHANGE_API_SECRET=your_api_secret
   ```

### Usage

To run the complete trading system:

```bash
python -m src.agents.fetch_ai.main
```

The system supports several command line arguments:

- `--exchange`: Exchange to use (default: binance)
- `--symbols`: Symbols to track (default: BTC/USD ETH/USD)
- `--api-key`: API key for the exchange
- `--api-secret`: API secret for the exchange
- `--port`: Port for the agent server (default: 8000)
- `--endpoint`: Endpoint for the agent server

## Integration with n8n

The system is designed to integrate with n8n workflows. The integration layer allows for:

- Sending trading signals to n8n workflows
- Receiving data from n8n workflows
- Triggering n8n workflows based on specific events
- Processing data from n8n workflows

## Future Enhancements

- **Advanced Machine Learning Models**: Implement more sophisticated machine learning models for prediction
- **Additional Data Sources**: Integrate more data sources for sentiment analysis and market data
- **Enhanced Visualization**: Develop a dashboard for visualizing trading recommendations and system performance
- **Backtesting Framework**: Create a framework for backtesting trading strategies
- **Mobile Notifications**: Implement mobile notifications for trading signals
