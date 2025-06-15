"""
Strategy Management API

FastAPI endpoints for managing algorithmic trading strategies,
backtesting, and real-time monitoring.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..tools.tradingview_tools import TradingViewChartingEngine
from ..trading.strategies import (
    BacktestingEngine,
    BollingerBandsStrategy,
    EnhancedBaseStrategy,
    LSTMStrategy,
    MACDStrategy,
    MovingAverageCrossoverStrategy,
    PairsTradingStrategy,
    RandomForestStrategy,
    RSIMeanReversionStrategy,
    RSIStrategy,
    StatisticalArbitrageStrategy,
    StrategyManager,
    ZScoreStrategy,
)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Global strategy manager instance
strategy_manager: Optional[StrategyManager] = None
charting_engine = TradingViewChartingEngine()


# Pydantic models for API
class StrategyCreateRequest(BaseModel):
    strategy_type: str = Field(
        ..., description="Type of strategy (rsi, macd, bollinger_bands, etc.)"
    )
    name: str = Field(..., description="Custom name for the strategy")
    symbols: List[str] = Field(..., description="List of trading symbols")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    allocation_percentage: float = Field(
        ..., ge=0.01, le=1.0, description="Allocation percentage (0.01-1.0)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy-specific parameters"
    )
    risk_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Risk management parameters"
    )


class BacktestRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy ID to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000, description="Initial capital for backtest")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    slippage_rate: float = Field(default=0.0005, description="Slippage rate")


class StrategyUpdateRequest(BaseModel):
    allocation_percentage: Optional[float] = Field(default=None, ge=0.01, le=1.0)
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    risk_parameters: Optional[Dict[str, Any]] = Field(default=None)
    is_active: Optional[bool] = Field(default=None)


# Strategy factory
def create_strategy(
    strategy_type: str,
    strategy_id: str,
    name: str,
    symbols: List[str],
    timeframe: str,
    parameters: Optional[Dict[str, Any]] = None,
    risk_parameters: Optional[Dict[str, Any]] = None,
) -> EnhancedBaseStrategy:
    """Factory function to create strategy instances."""

    strategy_classes = {
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "ma_crossover": MovingAverageCrossoverStrategy,
        "bollinger_bands": BollingerBandsStrategy,
        "z_score": ZScoreStrategy,
        "rsi_mean_reversion": RSIMeanReversionStrategy,
        "pairs_trading": PairsTradingStrategy,
        "statistical_arbitrage": StatisticalArbitrageStrategy,
        "random_forest": RandomForestStrategy,
        "lstm": LSTMStrategy,
    }

    if strategy_type not in strategy_classes:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    strategy_class = strategy_classes[strategy_type]

    # Special handling for pairs trading
    if strategy_type == "pairs_trading":
        # Convert symbols list to pairs
        if len(symbols) % 2 != 0:
            raise ValueError("Pairs trading requires even number of symbols")

        symbol_pairs = [(symbols[i], symbols[i + 1]) for i in range(0, len(symbols), 2)]
        return strategy_class(
            strategy_id=strategy_id,
            symbol_pairs=symbol_pairs,
            timeframe=timeframe,
            parameters=parameters,
            risk_parameters=risk_parameters,
        )

    return strategy_class(
        strategy_id=strategy_id,
        symbols=symbols,
        timeframe=timeframe,
        parameters=parameters,
        risk_parameters=risk_parameters,
    )


@router.on_event("startup")
async def startup_strategy_manager():
    """Initialize strategy manager on startup."""
    global strategy_manager
    strategy_manager = StrategyManager(
        total_capital=Decimal("1000000"),  # $1M default capital
        max_strategies=20,
        rebalance_interval=3600,
    )
    await strategy_manager.start()


@router.on_event("shutdown")
async def shutdown_strategy_manager():
    """Shutdown strategy manager."""
    global strategy_manager
    if strategy_manager:
        await strategy_manager.stop()


@router.get("/")
async def list_strategies():
    """List all strategies."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    return strategy_manager.get_portfolio_summary()


@router.post("/")
async def create_strategy_endpoint(request: StrategyCreateRequest):
    """Create a new trading strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    try:
        # Generate unique strategy ID
        strategy_id = f"{request.strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create strategy instance
        strategy = create_strategy(
            strategy_type=request.strategy_type,
            strategy_id=strategy_id,
            name=request.name,
            symbols=request.symbols,
            timeframe=request.timeframe,
            parameters=request.parameters,
            risk_parameters=request.risk_parameters,
        )

        # Add to strategy manager
        success = await strategy_manager.add_strategy(
            strategy=strategy, allocation_percentage=request.allocation_percentage
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add strategy")

        return {
            "strategy_id": strategy_id,
            "message": "Strategy created successfully",
            "strategy": strategy.get_performance_summary(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get strategy details."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy = strategy_manager.strategies[strategy_id]
    allocation = strategy_manager.allocations[strategy_id]

    return {
        "strategy": strategy.get_performance_summary(),
        "allocation": {
            "allocation_percentage": allocation.allocation_percentage,
            "max_allocation": float(allocation.max_allocation),
            "current_allocation": float(allocation.current_allocation),
            "is_active": allocation.is_active,
            "priority": allocation.priority,
        },
    }


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, request: StrategyUpdateRequest):
    """Update strategy configuration."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        strategy = strategy_manager.strategies[strategy_id]
        allocation = strategy_manager.allocations[strategy_id]

        # Update allocation
        if request.allocation_percentage is not None:
            allocation.allocation_percentage = request.allocation_percentage
            allocation.max_allocation = strategy_manager.total_capital * Decimal(
                str(request.allocation_percentage)
            )

        # Update parameters
        if request.parameters is not None:
            strategy.parameters.update(request.parameters)

        if request.risk_parameters is not None:
            strategy.risk_parameters.update(request.risk_parameters)

        # Update active status
        if request.is_active is not None:
            allocation.is_active = request.is_active
            if request.is_active:
                await strategy.resume()
            else:
                await strategy.pause()

        return {"message": "Strategy updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    success = await strategy_manager.remove_strategy(strategy_id)

    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return {"message": "Strategy deleted successfully"}


@router.post("/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """Start a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy = strategy_manager.strategies[strategy_id]
    await strategy.start()

    return {"message": "Strategy started successfully"}


@router.post("/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """Stop a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy = strategy_manager.strategies[strategy_id]
    await strategy.stop()

    return {"message": "Strategy stopped successfully"}


@router.post("/{strategy_id}/backtest")
async def backtest_strategy(strategy_id: str, request: BacktestRequest):
    """Run backtest for a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        strategy = strategy_manager.strategies[strategy_id]

        # Create backtesting engine
        backtest_engine = BacktestingEngine(
            initial_capital=Decimal(str(request.initial_capital)),
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
        )

        # Generate sample historical data (in production, fetch real data)
        historical_data = {}
        for symbol in strategy.symbols:
            # This is placeholder data - in production, fetch from data provider
            dates = pd.date_range(start=request.start_date, end=request.end_date, freq="1H")
            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": 100 + np.random.randn(len(dates)).cumsum(),
                    "high": 100 + np.random.randn(len(dates)).cumsum() + 1,
                    "low": 100 + np.random.randn(len(dates)).cumsum() - 1,
                    "close": 100 + np.random.randn(len(dates)).cumsum(),
                    "volume": np.random.randint(1000, 10000, len(dates)),
                }
            )
            historical_data[symbol] = data

        # Run backtest
        metrics = await backtest_engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Generate report
        report = backtest_engine.generate_report()

        return {
            "strategy_id": strategy_id,
            "backtest_period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
            },
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{strategy_id}/chart")
async def get_strategy_chart(strategy_id: str, symbol: str):
    """Get TradingView chart for strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    if strategy_id not in strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        # Create chart configuration
        chart_config = charting_engine.create_chart_config(
            symbol=symbol,
            timeframe="1H",
            indicators=["RSI", "MACD", "Bollinger Bands"],
            overlays=["Strategy Signals"],
        )

        # Generate chart HTML
        chart_html = charting_engine.generate_chart_html(symbol)

        return HTMLResponse(content=chart_html)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/types/available")
async def get_available_strategy_types():
    """Get list of available strategy types."""
    return {
        "strategy_types": [
            {
                "id": "rsi",
                "name": "RSI Strategy",
                "description": "Relative Strength Index momentum strategy",
                "category": "momentum",
                "parameters": {
                    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50},
                    "oversold_threshold": {"type": "float", "default": 30, "min": 10, "max": 40},
                    "overbought_threshold": {"type": "float", "default": 70, "min": 60, "max": 90},
                },
            },
            {
                "id": "macd",
                "name": "MACD Strategy",
                "description": "Moving Average Convergence Divergence strategy",
                "category": "momentum",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 5, "max": 20},
                    "slow_period": {"type": "int", "default": 26, "min": 20, "max": 50},
                    "signal_period": {"type": "int", "default": 9, "min": 5, "max": 15},
                },
            },
            {
                "id": "bollinger_bands",
                "name": "Bollinger Bands Strategy",
                "description": "Mean reversion strategy using Bollinger Bands",
                "category": "mean_reversion",
                "parameters": {
                    "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
                    "bb_std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
                },
            },
            {
                "id": "pairs_trading",
                "name": "Pairs Trading Strategy",
                "description": "Statistical arbitrage between correlated assets",
                "category": "arbitrage",
                "parameters": {
                    "lookback_period": {"type": "int", "default": 60, "min": 30, "max": 200},
                    "entry_threshold": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
                },
            },
            {
                "id": "random_forest",
                "name": "Random Forest Strategy",
                "description": "Machine learning strategy using Random Forest",
                "category": "ml",
                "parameters": {
                    "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500},
                    "max_depth": {"type": "int", "default": 10, "min": 5, "max": 20},
                },
            },
        ]
    }


@router.get("/portfolio/summary")
async def get_portfolio_summary():
    """Get portfolio summary with all strategies."""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")

    return strategy_manager.get_portfolio_summary()
