"""
API routes for Trading Infinite Loop system.

This module provides REST API endpoints for managing trading strategy generation
using the Infinite Agentic Loop system.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ...agents.trading_infinite_loop.trading_strategy_orchestrator import TradingStrategyConfig
from ..services.trading_strategy_service import TradingStrategyService


# Pydantic models for API
class StrategyGenerationRequest(BaseModel):
    """Request model for strategy generation."""

    count: Union[int, str] = Field(
        default=10, description="Number of strategies to generate or 'infinite'"
    )
    target_symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT"], description="Trading symbols to target"
    )
    strategy_types: List[str] = Field(
        default=["momentum", "mean_reversion"], description="Types of strategies to generate"
    )
    risk_tolerance: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Risk tolerance (0.1% to 10%)"
    )
    min_profit_threshold: float = Field(
        default=0.005, ge=0.001, le=0.05, description="Minimum profit threshold"
    )
    backtest_period_days: int = Field(
        default=30, ge=7, le=365, description="Backtesting period in days"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional configuration parameters"
    )


class StrategyDeploymentRequest(BaseModel):
    """Request model for strategy deployment."""

    strategy_id: str = Field(description="ID of the strategy to deploy")
    allocation: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Portfolio allocation (1% to 100%)"
    )
    max_position_size: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Maximum position size"
    )
    stop_loss: float = Field(default=0.02, ge=0.005, le=0.1, description="Stop loss percentage")


class StrategyResponse(BaseModel):
    """Response model for strategy information."""

    strategy_id: str
    performance: Dict[str, Any]
    created_at: str
    status: str
    backtest_results: Optional[Dict[str, Any]] = None


class GenerationStatusResponse(BaseModel):
    """Response model for generation status."""

    session_id: str
    status: str
    progress: float
    strategies_generated: int
    strategies_accepted: int
    current_wave: int
    execution_time: float
    errors: List[str]


# Initialize router
router = APIRouter(prefix="/api/trading-infinite-loop", tags=["Trading Infinite Loop"])

# Initialize service
trading_service = TradingStrategyService()


@router.post("/generate", response_model=Dict[str, Any])
async def start_strategy_generation(
    background_tasks: BackgroundTasks, request: StrategyGenerationRequest
) -> Dict[str, Any]:
    """
    Start trading strategy generation using infinite agentic loop.

    This endpoint initiates the strategy generation process and returns
    a session ID for tracking progress.
    """
    try:
        # Create configuration
        config = TradingStrategyConfig(
            target_symbols=request.target_symbols,
            strategy_types=request.strategy_types,
            risk_tolerance=request.risk_tolerance,
            min_profit_threshold=request.min_profit_threshold,
            backtest_period_days=request.backtest_period_days,
        )

        # Override with custom config if provided
        if request.config:
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Start generation in background
        session_id = await trading_service.start_generation(count=request.count, config=config)

        return {
            "success": True,
            "session_id": session_id,
            "message": "Strategy generation started",
            "estimated_time": "5-30 minutes depending on count",
            "status_endpoint": f"/api/trading-infinite-loop/status/{session_id}",
        }

    except Exception as e:
        logging.error(f"Error starting strategy generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=GenerationStatusResponse)
async def get_generation_status(session_id: str) -> GenerationStatusResponse:
    """
    Get the status of a strategy generation session.

    Returns real-time progress information including number of strategies
    generated, current performance metrics, and any errors.
    """
    try:
        status = await trading_service.get_generation_status(session_id)

        if not status:
            raise HTTPException(status_code=404, detail="Session not found")

        return GenerationStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting generation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{session_id}")
async def stop_strategy_generation(session_id: str) -> Dict[str, Any]:
    """
    Stop a running strategy generation session.

    Gracefully stops the infinite loop and returns final results.
    """
    try:
        result = await trading_service.stop_generation(session_id)

        return {"success": True, "message": "Strategy generation stopped", "final_results": result}

    except Exception as e:
        logging.error(f"Error stopping strategy generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=List[StrategyResponse])
async def list_strategies(
    limit: int = 20,
    sort_by: str = "performance",
    min_sharpe_ratio: Optional[float] = None,
    max_drawdown: Optional[float] = None,
) -> List[StrategyResponse]:
    """
    List generated trading strategies with filtering and sorting options.

    Returns a list of strategies sorted by performance metrics.
    """
    try:
        strategies = await trading_service.list_strategies(
            limit=limit,
            sort_by=sort_by,
            filters={"min_sharpe_ratio": min_sharpe_ratio, "max_drawdown": max_drawdown},
        )

        return [StrategyResponse(**strategy) for strategy in strategies]

    except Exception as e:
        logging.error(f"Error listing strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: str) -> StrategyResponse:
    """
    Get detailed information about a specific strategy.

    Returns complete strategy details including backtest results,
    performance metrics, and implementation code.
    """
    try:
        strategy = await trading_service.get_strategy(strategy_id)

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        return StrategyResponse(**strategy)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/deploy")
async def deploy_strategy(strategy_id: str, request: StrategyDeploymentRequest) -> Dict[str, Any]:
    """
    Deploy a strategy for live trading.

    Sets up the strategy in the trading system with specified parameters
    and begins live execution.
    """
    try:
        deployment_result = await trading_service.deploy_strategy(
            strategy_id=strategy_id,
            allocation=request.allocation,
            max_position_size=request.max_position_size,
            stop_loss=request.stop_loss,
        )

        return {
            "success": True,
            "deployment_id": deployment_result["deployment_id"],
            "message": "Strategy deployed successfully",
            "live_trading_started": deployment_result["live_trading_started"],
            "monitoring_dashboard": f"/trading/strategies/{strategy_id}/monitor",
        }

    except Exception as e:
        logging.error(f"Error deploying strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str) -> Dict[str, Any]:
    """
    Delete a generated strategy.

    Removes the strategy from the system and stops any live trading.
    """
    try:
        await trading_service.delete_strategy(strategy_id)

        return {"success": True, "message": "Strategy deleted successfully"}

    except Exception as e:
        logging.error(f"Error deleting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/backtest")
async def get_backtest_results(strategy_id: str) -> Dict[str, Any]:
    """
    Get detailed backtest results for a strategy.

    Returns comprehensive backtesting data including trade history,
    performance metrics, and risk analysis.
    """
    try:
        backtest_results = await trading_service.get_backtest_results(strategy_id)

        if not backtest_results:
            raise HTTPException(status_code=404, detail="Backtest results not found")

        return backtest_results

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/rebacktest")
async def rerun_backtest(
    strategy_id: str, period_days: int = 30, symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Re-run backtest for a strategy with different parameters.

    Useful for testing strategy performance on different time periods
    or market conditions.
    """
    try:
        backtest_results = await trading_service.rerun_backtest(
            strategy_id=strategy_id, period_days=period_days, symbols=symbols
        )

        return {
            "success": True,
            "backtest_results": backtest_results,
            "message": "Backtest completed successfully",
        }

    except Exception as e:
        logging.error(f"Error re-running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary")
async def get_performance_summary() -> Dict[str, Any]:
    """
    Get overall performance summary of the strategy generation system.

    Returns aggregate statistics about generated strategies, success rates,
    and system performance metrics.
    """
    try:
        summary = await trading_service.get_performance_summary()

        return summary

    except Exception as e:
        logging.error(f"Error getting performance summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the trading infinite loop system.

    Returns system status and connectivity information.
    """
    try:
        health_status = await trading_service.health_check()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": health_status,
        }

    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "timestamp": datetime.now().isoformat(), "error": str(e)}
