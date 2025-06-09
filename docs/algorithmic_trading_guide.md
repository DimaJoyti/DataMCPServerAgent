# Algorithmic Trading Strategies & TradingView Integration

This guide covers the comprehensive algorithmic trading framework implemented in the DataMCPServerAgent, featuring advanced trading strategies, backtesting capabilities, and TradingView integration.

## 🚀 Features

### Algorithmic Trading Strategies

#### Momentum Strategies
- **RSI Strategy**: Relative Strength Index-based momentum trading
- **MACD Strategy**: Moving Average Convergence Divergence signals
- **Moving Average Crossover**: Fast/slow MA crossover system

#### Mean Reversion Strategies
- **Bollinger Bands Strategy**: Price reversion to statistical bands
- **Z-Score Strategy**: Statistical mean reversion trading
- **RSI Mean Reversion**: RSI-based mean reversion with divergence detection

#### Arbitrage Strategies
- **Pairs Trading**: Statistical arbitrage between correlated assets
- **Statistical Arbitrage**: Multi-asset statistical relationships

#### Machine Learning Strategies
- **Random Forest Strategy**: ML-based signal generation
- **LSTM Strategy**: Neural network time series prediction

### Advanced Features
- **Strategy Manager**: Multi-strategy portfolio management
- **Backtesting Engine**: Comprehensive performance analysis
- **Risk Management**: Position sizing and drawdown control
- **Real-time Execution**: Live trading signal generation
- **TradingView Integration**: Advanced charting and visualization

## 📊 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install optional TA-Lib for advanced technical analysis
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install ta-lib

# On macOS:
brew install ta-lib
pip install ta-lib

# On Windows:
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl
```

### 2. Run the Demo

```bash
# Run the comprehensive demo
python examples/algorithmic_trading_demo.py
```

### 3. Start the Trading Server

```bash
# Start the trading API server
python scripts/start_trading_server.py
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Strategy Management**: http://localhost:8000/api/strategies/
- **WebSocket Trading**: ws://localhost:8000/ws/trading
- **WebSocket Market Data**: ws://localhost:8000/ws/market-data

## 🔧 Strategy Configuration

### Creating a Strategy

```python
from src.trading.strategies import RSIStrategy, StrategyManager

# Create RSI strategy
strategy = RSIStrategy(
    strategy_id="rsi_001",
    symbols=['BTC/USD', 'ETH/USD'],
    timeframe="1h",
    parameters={
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70
    },
    risk_parameters={
        'max_position_size': 1000,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }
)

# Create strategy manager
manager = StrategyManager(total_capital=100000)
await manager.add_strategy(strategy, allocation_percentage=0.25)
```

### Strategy Types and Parameters

#### RSI Strategy
```python
parameters = {
    'rsi_period': 14,                    # RSI calculation period
    'oversold_threshold': 30,            # Buy signal threshold
    'overbought_threshold': 70,          # Sell signal threshold
    'extreme_oversold': 20,              # Strong buy threshold
    'extreme_overbought': 80,            # Strong sell threshold
    'min_volume': 1000                   # Minimum volume filter
}
```

#### MACD Strategy
```python
parameters = {
    'fast_period': 12,                   # Fast EMA period
    'slow_period': 26,                   # Slow EMA period
    'signal_period': 9,                  # Signal line period
    'min_histogram_threshold': 0.001,    # Minimum histogram value
    'divergence_lookback': 5             # Divergence detection period
}
```

#### Bollinger Bands Strategy
```python
parameters = {
    'bb_period': 20,                     # Bollinger Bands period
    'bb_std_dev': 2.0,                   # Standard deviation multiplier
    'oversold_threshold': 0.1,           # Distance from lower band
    'overbought_threshold': 0.1,         # Distance from upper band
    'rsi_filter': True,                  # Enable RSI confirmation
    'volume_confirmation': True          # Enable volume confirmation
}
```

#### Pairs Trading Strategy
```python
parameters = {
    'lookback_period': 60,               # Historical data period
    'entry_threshold': 2.0,              # Z-score entry threshold
    'exit_threshold': 0.5,               # Z-score exit threshold
    'min_correlation': 0.7,              # Minimum correlation
    'cointegration_pvalue': 0.05,        # Cointegration p-value
    'half_life_max': 30                  # Maximum mean reversion half-life
}
```

## 📈 Backtesting

### Running a Backtest

```python
from src.trading.strategies import BacktestingEngine

# Create backtesting engine
backtest_engine = BacktestingEngine(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005
)

# Run backtest
metrics = await backtest_engine.run_backtest(
    strategy=strategy,
    historical_data=historical_data,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Get detailed report
report = backtest_engine.generate_report()
```

### Backtest Metrics

The backtesting engine provides comprehensive performance metrics:

- **Basic Metrics**: Total trades, win rate, PnL
- **Risk Metrics**: Maximum drawdown, Sharpe ratio, Sortino ratio
- **Advanced Metrics**: Profit factor, expectancy, Calmar ratio
- **Time-based Metrics**: Average trade duration, consecutive wins/losses

## 🌐 TradingView Integration

### Chart Configuration

```python
from src.tools.tradingview_tools import TradingViewChartingEngine

# Create charting engine
charting_engine = TradingViewChartingEngine()

# Configure chart
chart_config = charting_engine.create_chart_config(
    symbol="BINANCE:BTCUSDT",
    timeframe="1H",
    indicators=["RSI", "MACD", "Bollinger Bands"],
    overlays=["Strategy Signals"]
)

# Generate chart HTML
chart_html = charting_engine.generate_chart_html("BINANCE:BTCUSDT")
```

### Real-time Data Integration

```python
from src.tools.tradingview_tools import TradingViewWebSocketClient

# Create WebSocket client
ws_client = TradingViewWebSocketClient(
    symbols=['BTCUSDT', 'ETHUSDT'],
    callback=handle_market_data
)

# Connect and subscribe
await ws_client.connect()
await ws_client.subscribe_symbols(['BTCUSDT', 'ETHUSDT'])
```

## 🔌 API Usage

### Create Strategy via API

```bash
curl -X POST "http://localhost:8000/api/strategies/" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "rsi",
    "name": "My RSI Strategy",
    "symbols": ["BTC/USD", "ETH/USD"],
    "timeframe": "1h",
    "allocation_percentage": 0.25,
    "parameters": {
      "rsi_period": 14,
      "oversold_threshold": 30,
      "overbought_threshold": 70
    }
  }'
```

### Run Backtest via API

```bash
curl -X POST "http://localhost:8000/api/strategies/{strategy_id}/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2023-01-01T00:00:00",
    "end_date": "2023-12-31T23:59:59",
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.0005
  }'
```

### Get Portfolio Summary

```bash
curl -X GET "http://localhost:8000/api/strategies/portfolio/summary"
```

## 📊 Technical Indicators

The framework includes a comprehensive technical indicators library:

### Available Indicators

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, VWAP
- **Support/Resistance**: Pivot Points, Fibonacci Retracements

### Usage Example

```python
from src.trading.strategies import TechnicalIndicators

# Calculate RSI
rsi = TechnicalIndicators.rsi(price_data, period=14)

# Calculate MACD
macd_data = TechnicalIndicators.macd(price_data, fast=12, slow=26, signal=9)

# Calculate Bollinger Bands
bb_data = TechnicalIndicators.bollinger_bands(price_data, period=20, std_dev=2)

# Calculate all indicators at once
df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
```

## 🎯 Risk Management

### Position Sizing

```python
# Kelly Criterion position sizing
def kelly_position_size(win_rate, avg_win, avg_loss, capital):
    if avg_loss == 0:
        return 0
    
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    return max(0, min(kelly_fraction * capital, capital * 0.25))  # Cap at 25%
```

### Risk Parameters

```python
risk_parameters = {
    'max_position_size': 10000,          # Maximum position size
    'max_drawdown_limit': 0.15,          # 15% maximum drawdown
    'stop_loss_pct': 0.02,               # 2% stop loss
    'take_profit_pct': 0.04,             # 4% take profit
    'max_correlation_threshold': 0.7,     # Maximum strategy correlation
    'position_concentration_limit': 0.3   # Maximum single position weight
}
```

## 🔄 Real-time Trading

### Signal Generation

```python
# Process real-time signals
async def process_market_data(symbol, market_data):
    # Update strategy with new data
    await strategy.update_market_data(symbol, market_data)
    
    # Generate signal
    signal = await strategy.generate_signal(symbol, market_data)
    
    if signal and signal.signal != StrategySignal.HOLD:
        # Calculate position size
        position_size = await strategy.calculate_position_size(symbol, signal)
        
        # Execute trade
        order = await execute_trade(symbol, signal, position_size)
        
        return order
```

### WebSocket Integration

```python
# WebSocket message handler
async def handle_websocket_message(websocket, message):
    data = json.loads(message)
    
    if data['type'] == 'market_data':
        symbol = data['symbol']
        price = data['price']
        volume = data['volume']
        
        # Create market data object
        market_data = MarketData(
            symbol=symbol,
            price=Decimal(str(price)),
            volume=Decimal(str(volume)),
            timestamp=datetime.now()
        )
        
        # Process with strategy manager
        orders = await strategy_manager.process_signals()
        
        # Broadcast orders to clients
        for order in orders:
            await broadcast_order_update(order)
```

## 📚 Examples

### Complete Strategy Implementation

```python
import asyncio
from src.trading.strategies import *

async def main():
    # Create strategy manager
    manager = StrategyManager(total_capital=1000000)
    
    # Add multiple strategies
    strategies = [
        RSIStrategy("rsi_001", ["BTC/USD", "ETH/USD"]),
        MACDStrategy("macd_001", ["BTC/USD", "ETH/USD"]),
        BollingerBandsStrategy("bb_001", ["ADA/USD", "DOT/USD"]),
        PairsTradingStrategy("pairs_001", [("BTC/USD", "ETH/USD")])
    ]
    
    for i, strategy in enumerate(strategies):
        await manager.add_strategy(strategy, allocation_percentage=0.25)
    
    # Start trading
    await manager.start()
    
    # Monitor performance
    while True:
        summary = manager.get_portfolio_summary()
        print(f"Total PnL: ${summary['portfolio_metrics']['total_pnl']:.2f}")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Advanced Configuration

### Custom Strategy Development

```python
class CustomStrategy(EnhancedBaseStrategy):
    async def generate_signal(self, symbol: str, market_data: MarketData):
        # Implement custom signal logic
        df = self.market_data.get(symbol)
        
        # Your custom indicators and logic here
        signal_strength = calculate_custom_signal(df)
        
        return StrategySignalData(
            signal=StrategySignal.BUY if signal_strength > 0.7 else StrategySignal.HOLD,
            strength=abs(signal_strength),
            confidence=0.8,
            timestamp=datetime.now(),
            price=market_data.price
        )
    
    async def calculate_position_size(self, symbol: str, signal: StrategySignalData):
        # Implement custom position sizing
        return self.max_position_size * Decimal(str(signal.strength))
```

## 📖 Next Steps

1. **Explore the Demo**: Run `python examples/algorithmic_trading_demo.py`
2. **Start the Server**: Launch the trading API server
3. **Create Strategies**: Use the API to create and manage strategies
4. **Run Backtests**: Test strategies on historical data
5. **Monitor Performance**: Use the dashboard for real-time monitoring
6. **Integrate TradingView**: Set up advanced charting
7. **Deploy Live**: Configure for live trading (paper trading recommended first)

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [API Reference](api_reference.md)
- [System Architecture](system_architecture_blueprint.md)
- [Contributing Guide](contributing.md)
