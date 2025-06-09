#!/usr/bin/env python3
"""
Algorithmic Trading Strategies Demo

Demonstrates the comprehensive algorithmic trading framework with:
- Multiple strategy types (momentum, mean reversion, arbitrage, ML)
- Strategy management and allocation
- Backtesting capabilities
- Real-time signal generation
- Performance analytics
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

# Set up the path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.strategies import (
    StrategyManager,
    RSIStrategy,
    MACDStrategy,
    MovingAverageCrossoverStrategy,
    BollingerBandsStrategy,
    ZScoreStrategy,
    PairsTradingStrategy,
    BacktestingEngine,
    TechnicalIndicators
)
from src.trading.core.base_models import MarketData
from src.tools.tradingview_tools import TradingViewChartingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate price data with realistic patterns
    n_points = len(timestamps)
    
    # Base price with trend
    base_price = 100
    trend = np.linspace(0, 20, n_points)  # Upward trend
    
    # Add volatility
    volatility = np.random.normal(0, 2, n_points)
    
    # Add some cyclical patterns
    cycle = 5 * np.sin(np.linspace(0, 4 * np.pi, n_points))
    
    # Combine components
    close_prices = base_price + trend + volatility + cycle
    close_prices = np.maximum(close_prices, 1)  # Ensure positive prices
    
    # Generate OHLC from close prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Generate volume
    volume = np.random.lognormal(10, 0.5, n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })


async def demo_strategy_creation():
    """Demonstrate creating different types of strategies."""
    logger.info("=== Strategy Creation Demo ===")
    
    # Create strategy manager
    strategy_manager = StrategyManager(
        total_capital=Decimal('1000000'),  # $1M
        max_strategies=10,
        rebalance_interval=3600
    )
    
    await strategy_manager.start()
    
    # Define symbols for testing
    symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD']
    
    # 1. RSI Strategy
    rsi_strategy = RSIStrategy(
        strategy_id="rsi_001",
        symbols=symbols[:2],
        timeframe="1h",
        parameters={
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
    )
    
    await strategy_manager.add_strategy(rsi_strategy, allocation_percentage=0.25)
    logger.info("✓ Created RSI Strategy with 25% allocation")
    
    # 2. MACD Strategy
    macd_strategy = MACDStrategy(
        strategy_id="macd_001",
        symbols=symbols[:2],
        timeframe="1h",
        parameters={
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    )
    
    await strategy_manager.add_strategy(macd_strategy, allocation_percentage=0.25)
    logger.info("✓ Created MACD Strategy with 25% allocation")
    
    # 3. Bollinger Bands Strategy
    bb_strategy = BollingerBandsStrategy(
        strategy_id="bb_001",
        symbols=symbols[2:],
        timeframe="1h",
        parameters={
            'bb_period': 20,
            'bb_std_dev': 2.0
        }
    )
    
    await strategy_manager.add_strategy(bb_strategy, allocation_percentage=0.25)
    logger.info("✓ Created Bollinger Bands Strategy with 25% allocation")
    
    # 4. Pairs Trading Strategy
    pairs_strategy = PairsTradingStrategy(
        strategy_id="pairs_001",
        symbol_pairs=[('BTC/USD', 'ETH/USD')],
        timeframe="1h",
        parameters={
            'lookback_period': 60,
            'entry_threshold': 2.0
        }
    )
    
    await strategy_manager.add_strategy(pairs_strategy, allocation_percentage=0.25)
    logger.info("✓ Created Pairs Trading Strategy with 25% allocation")
    
    return strategy_manager, symbols


async def demo_backtesting():
    """Demonstrate backtesting capabilities."""
    logger.info("\n=== Backtesting Demo ===")
    
    # Create a simple RSI strategy for backtesting
    strategy = RSIStrategy(
        strategy_id="backtest_rsi",
        symbols=['BTC/USD'],
        timeframe="1h",
        parameters={
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
    )
    
    # Generate historical data
    historical_data = {
        'BTC/USD': generate_sample_data('BTC/USD', days=90)
    }
    
    # Create backtesting engine
    backtest_engine = BacktestingEngine(
        initial_capital=Decimal('100000'),
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Define backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Run backtest
    metrics = await backtest_engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Display results
    logger.info("Backtest Results:")
    logger.info(f"  Total Trades: {metrics.total_trades}")
    logger.info(f"  Win Rate: {metrics.win_rate:.2%}")
    logger.info(f"  Total PnL: ${metrics.total_pnl:.2f}")
    logger.info(f"  Total Return: {metrics.total_pnl_percentage:.2f}%")
    logger.info(f"  Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
    
    return backtest_engine.generate_report()


async def demo_real_time_signals(strategy_manager, symbols):
    """Demonstrate real-time signal generation."""
    logger.info("\n=== Real-Time Signal Generation Demo ===")
    
    # Generate sample market data for each symbol
    market_data_feeds = {}
    for symbol in symbols:
        df = generate_sample_data(symbol, days=30)
        market_data_feeds[symbol] = df
    
    # Simulate real-time data updates
    for i in range(10):  # Simulate 10 time periods
        logger.info(f"\n--- Time Period {i+1} ---")
        
        # Update market data for each symbol
        for symbol in symbols:
            df = market_data_feeds[symbol]
            
            # Get current data point
            if i < len(df):
                current_row = df.iloc[-(len(df)-i)]
                
                # Create market data object
                market_data = MarketData(
                    symbol=symbol,
                    price=Decimal(str(current_row['close'])),
                    volume=Decimal(str(current_row['volume'])),
                    timestamp=current_row['timestamp'],
                    open_price=Decimal(str(current_row['open'])),
                    high_price=Decimal(str(current_row['high'])),
                    low_price=Decimal(str(current_row['low']))
                )
                
                # Update strategy manager with new data
                await strategy_manager.update_market_data(symbol, market_data)
        
        # Process signals from all strategies
        orders = await strategy_manager.process_signals()
        
        if orders:
            logger.info(f"Generated {len(orders)} trading signals:")
            for order in orders:
                logger.info(f"  {order['symbol']}: {order['side'].value} "
                          f"{order['quantity']} @ ${order['price']} "
                          f"(Strategy: {order['strategy_id']}, "
                          f"Strength: {order['signal_strength']:.2f})")
        else:
            logger.info("No trading signals generated")
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.5)


async def demo_technical_indicators():
    """Demonstrate technical indicator calculations."""
    logger.info("\n=== Technical Indicators Demo ===")
    
    # Generate sample data
    df = generate_sample_data('DEMO/USD', days=100)
    
    # Calculate all indicators
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    # Display latest values
    latest = df_with_indicators.iloc[-1]
    
    logger.info("Latest Technical Indicators:")
    logger.info(f"  Price: ${latest['close']:.2f}")
    logger.info(f"  RSI: {latest['rsi']:.2f}")
    logger.info(f"  MACD: {latest['macd']:.4f}")
    logger.info(f"  MACD Signal: {latest['macd_signal']:.4f}")
    logger.info(f"  Bollinger Upper: ${latest['bb_upper']:.2f}")
    logger.info(f"  Bollinger Lower: ${latest['bb_lower']:.2f}")
    logger.info(f"  Z-Score: {latest['z_score']:.2f}")
    logger.info(f"  ATR: {latest['atr']:.2f}")
    
    return df_with_indicators


async def demo_tradingview_integration():
    """Demonstrate TradingView integration."""
    logger.info("\n=== TradingView Integration Demo ===")
    
    # Create charting engine
    charting_engine = TradingViewChartingEngine()
    
    # Create chart configuration
    chart_config = charting_engine.create_chart_config(
        symbol="BINANCE:BTCUSDT",
        timeframe="1H",
        indicators=["RSI", "MACD", "Bollinger Bands"],
        overlays=["Strategy Signals"]
    )
    
    logger.info("Created TradingView chart configuration:")
    logger.info(f"  Symbol: {chart_config['symbol']}")
    logger.info(f"  Timeframe: {chart_config['interval']}")
    logger.info(f"  Theme: {chart_config['theme']}")
    logger.info(f"  Studies: {chart_config['studies']}")
    
    # Generate chart HTML
    chart_html = charting_engine.generate_chart_html("BINANCE:BTCUSDT")
    
    # Save chart to file
    chart_file = "tradingview_chart_demo.html"
    with open(chart_file, 'w') as f:
        f.write(chart_html)
    
    logger.info(f"✓ Generated TradingView chart HTML: {chart_file}")
    logger.info("  Open this file in a web browser to view the interactive chart")
    
    return chart_config


async def demo_portfolio_analytics(strategy_manager):
    """Demonstrate portfolio analytics."""
    logger.info("\n=== Portfolio Analytics Demo ===")
    
    # Get portfolio summary
    portfolio_summary = strategy_manager.get_portfolio_summary()
    
    logger.info("Portfolio Summary:")
    logger.info(f"  Total Capital: ${portfolio_summary['total_capital']:,.2f}")
    logger.info(f"  Active Strategies: {portfolio_summary['portfolio_metrics']['active_strategies']}")
    logger.info(f"  Total PnL: ${portfolio_summary['portfolio_metrics']['total_pnl']:,.2f}")
    logger.info(f"  Win Rate: {portfolio_summary['portfolio_metrics']['win_rate']:.2%}")
    
    logger.info("\nStrategy Performance:")
    for strategy_id, strategy_data in portfolio_summary['strategies'].items():
        logger.info(f"  {strategy_data['name']}:")
        logger.info(f"    Allocation: {strategy_data['allocation_percentage']:.1%}")
        logger.info(f"    State: {strategy_data['state']}")
        logger.info(f"    Total Trades: {strategy_data['total_trades']}")
        logger.info(f"    Win Rate: {strategy_data['win_rate']:.2%}")
        logger.info(f"    PnL: ${strategy_data['total_pnl']:,.2f}")


async def main():
    """Main demo function."""
    logger.info("🚀 Starting Algorithmic Trading Strategies Demo")
    logger.info("=" * 60)
    
    try:
        # 1. Strategy Creation Demo
        strategy_manager, symbols = await demo_strategy_creation()
        
        # 2. Technical Indicators Demo
        await demo_technical_indicators()
        
        # 3. Backtesting Demo
        backtest_report = await demo_backtesting()
        
        # 4. Real-time Signals Demo
        await demo_real_time_signals(strategy_manager, symbols)
        
        # 5. Portfolio Analytics Demo
        await demo_portfolio_analytics(strategy_manager)
        
        # 6. TradingView Integration Demo
        await demo_tradingview_integration()
        
        # Cleanup
        await strategy_manager.stop()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Demo completed successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("  ✓ Multiple algorithmic trading strategies")
        logger.info("  ✓ Strategy management and allocation")
        logger.info("  ✓ Comprehensive backtesting framework")
        logger.info("  ✓ Real-time signal generation")
        logger.info("  ✓ Technical indicator calculations")
        logger.info("  ✓ Portfolio analytics and monitoring")
        logger.info("  ✓ TradingView chart integration")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Start the trading server: python scripts/start_trading_server.py")
        logger.info("  2. Access the API at: http://localhost:8000/docs")
        logger.info("  3. Create strategies via API: POST /api/strategies/")
        logger.info("  4. Run backtests: POST /api/strategies/{id}/backtest")
        logger.info("  5. Monitor real-time performance")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
