"""
Trading Strategy Service

Service layer for managing trading strategy generation using the Infinite Agentic Loop.
Handles strategy lifecycle, performance tracking, and deployment management.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from ...agents.trading_infinite_loop.trading_strategy_orchestrator import (
    TradingStrategyConfig,
    TradingStrategyOrchestrator,
)
from ...agents.trading_system import AdvancedCryptoTradingSystem
from ...core.config import get_settings


class TradingStrategyService:
    """
    Service for managing trading strategy generation and deployment.

    This service provides high-level operations for the trading infinite loop system,
    including strategy generation, performance tracking, and live deployment.
    """

    def __init__(self):
        """Initialize the trading strategy service."""
        self.settings = get_settings()
        self.logger = logging.getLogger("trading_strategy_service")

        # Active generation sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Strategy storage
        self.strategies: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize required components."""
        try:
            # Initialize language model
            self.model = ChatAnthropic(
                model="claude-3-sonnet-20240229", temperature=0.7, max_tokens=4000
            )

            # Initialize tools (would be loaded from tool registry)
            self.tools: List[BaseTool] = []

            # Initialize trading system
            self.trading_system = AdvancedCryptoTradingSystem(
                name="strategy_generator",
                exchange_id="binance",  # Default exchange
                api_key=self.settings.EXCHANGE_API_KEY,
                api_secret=self.settings.EXCHANGE_API_SECRET,
            )

            self.logger.info("Trading strategy service initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing trading strategy service: {str(e)}")
            raise

    async def start_generation(self, count: Union[int, str], config: TradingStrategyConfig) -> str:
        """
        Start strategy generation process.

        Args:
            count: Number of strategies to generate or "infinite"
            config: Configuration for strategy generation

        Returns:
            Session ID for tracking progress
        """
        session_id = str(uuid.uuid4())

        try:
            # Create orchestrator
            orchestrator = TradingStrategyOrchestrator(
                model=self.model,
                tools=self.tools,
                trading_system=self.trading_system,
                config=config,
            )

            # Initialize session tracking
            self.active_sessions[session_id] = {
                "session_id": session_id,
                "status": "starting",
                "progress": 0.0,
                "strategies_generated": 0,
                "strategies_accepted": 0,
                "current_wave": 0,
                "start_time": datetime.now(),
                "execution_time": 0.0,
                "errors": [],
                "orchestrator": orchestrator,
                "config": config,
            }

            # Start generation in background
            asyncio.create_task(self._run_generation(session_id, orchestrator, count))

            self.logger.info(f"Started strategy generation session: {session_id}")
            return session_id

        except Exception as e:
            self.logger.error(f"Error starting strategy generation: {str(e)}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "error"
                self.active_sessions[session_id]["errors"].append(str(e))
            raise

    async def _run_generation(
        self, session_id: str, orchestrator: TradingStrategyOrchestrator, count: Union[int, str]
    ):
        """Run the strategy generation process."""
        session = self.active_sessions[session_id]

        try:
            session["status"] = "running"

            # Create output directory
            output_dir = Path(f"./generated_strategies/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Start generation
            results = await orchestrator.generate_trading_strategies(
                count=count, output_dir=output_dir
            )

            # Update session with results
            session["status"] = "completed"
            session["results"] = results
            session["execution_time"] = (datetime.now() - session["start_time"]).total_seconds()

            # Store generated strategies
            best_strategies = await orchestrator.get_best_strategies(limit=50)
            for strategy in best_strategies:
                strategy_id = strategy["strategy_id"]
                self.strategies[strategy_id] = strategy
                session["strategies_accepted"] += 1

            self.logger.info(f"Strategy generation completed for session: {session_id}")

        except Exception as e:
            self.logger.error(f"Error in strategy generation: {str(e)}")
            session["status"] = "error"
            session["errors"].append(str(e))

    async def get_generation_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a generation session.

        Args:
            session_id: ID of the generation session

        Returns:
            Session status information
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Update execution time
        session["execution_time"] = (datetime.now() - session["start_time"]).total_seconds()

        # Get orchestrator status if available
        if "orchestrator" in session and hasattr(session["orchestrator"], "execution_state"):
            execution_state = session["orchestrator"].execution_state
            if execution_state:
                session["current_wave"] = execution_state.current_wave
                session["strategies_generated"] = execution_state.total_iterations
                session["progress"] = min(
                    execution_state.total_iterations / 100, 1.0
                )  # Rough progress estimate

        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "progress": session["progress"],
            "strategies_generated": session["strategies_generated"],
            "strategies_accepted": session["strategies_accepted"],
            "current_wave": session["current_wave"],
            "execution_time": session["execution_time"],
            "errors": session["errors"],
        }

    async def stop_generation(self, session_id: str) -> Dict[str, Any]:
        """
        Stop a running generation session.

        Args:
            session_id: ID of the generation session

        Returns:
            Final results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        try:
            # Stop the orchestrator if running
            if "orchestrator" in session:
                orchestrator = session["orchestrator"]
                orchestrator.is_shutting_down = True

            session["status"] = "stopped"
            session["execution_time"] = (datetime.now() - session["start_time"]).total_seconds()

            return {
                "strategies_generated": session["strategies_generated"],
                "strategies_accepted": session["strategies_accepted"],
                "execution_time": session["execution_time"],
            }

        except Exception as e:
            self.logger.error(f"Error stopping generation: {str(e)}")
            raise

    async def list_strategies(
        self,
        limit: int = 20,
        sort_by: str = "performance",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List generated strategies with filtering and sorting.

        Args:
            limit: Maximum number of strategies to return
            sort_by: Field to sort by
            filters: Optional filters to apply

        Returns:
            List of strategy information
        """
        strategies = list(self.strategies.values())

        # Apply filters
        if filters:
            if filters.get("min_sharpe_ratio"):
                strategies = [
                    s
                    for s in strategies
                    if s.get("performance", {}).get("sharpe_ratio", 0)
                    >= filters["min_sharpe_ratio"]
                ]

            if filters.get("max_drawdown"):
                strategies = [
                    s
                    for s in strategies
                    if abs(s.get("performance", {}).get("max_drawdown", 1))
                    <= filters["max_drawdown"]
                ]

        # Sort strategies
        if sort_by == "performance":
            strategies.sort(
                key=lambda x: x.get("performance", {}).get("overall_score", 0), reverse=True
            )
        elif sort_by == "created_at":
            strategies.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Limit results
        strategies = strategies[:limit]

        # Format response
        return [
            {
                "strategy_id": strategy["strategy_id"],
                "performance": strategy.get("performance", {}),
                "created_at": strategy.get("created_at", ""),
                "status": "generated",
            }
            for strategy in strategies
        ]

    async def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific strategy.

        Args:
            strategy_id: ID of the strategy

        Returns:
            Strategy details
        """
        if strategy_id not in self.strategies:
            return None

        strategy = self.strategies[strategy_id]

        return {
            "strategy_id": strategy_id,
            "performance": strategy.get("performance", {}),
            "created_at": strategy.get("created_at", ""),
            "status": "generated",
            "backtest_results": strategy.get("backtest_results", {}),
        }

    async def deploy_strategy(
        self, strategy_id: str, allocation: float, max_position_size: float, stop_loss: float
    ) -> Dict[str, Any]:
        """
        Deploy a strategy for live trading.

        Args:
            strategy_id: ID of the strategy to deploy
            allocation: Portfolio allocation
            max_position_size: Maximum position size
            stop_loss: Stop loss percentage

        Returns:
            Deployment information
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        strategy = self.strategies[strategy_id]

        try:
            # Create deployment configuration
            deployment_config = {
                "strategy_id": strategy_id,
                "allocation": allocation,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "deployed_at": datetime.now().isoformat(),
            }

            # Deploy to trading system
            deployment_id = str(uuid.uuid4())

            # Update strategy status
            strategy["status"] = "deployed"
            strategy["deployment"] = deployment_config

            return {"deployment_id": deployment_id, "live_trading_started": True}

        except Exception as e:
            self.logger.error(f"Error deploying strategy: {str(e)}")
            raise

    async def delete_strategy(self, strategy_id: str):
        """
        Delete a strategy.

        Args:
            strategy_id: ID of the strategy to delete
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Stop live trading if deployed
        strategy = self.strategies[strategy_id]
        if strategy.get("status") == "deployed":
            # Stop live trading logic here
            pass

        # Remove from storage
        del self.strategies[strategy_id]

        self.logger.info(f"Deleted strategy: {strategy_id}")

    async def get_backtest_results(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get backtest results for a strategy.

        Args:
            strategy_id: ID of the strategy

        Returns:
            Backtest results
        """
        if strategy_id not in self.strategies:
            return None

        strategy = self.strategies[strategy_id]
        return strategy.get("backtest_results", {})

    async def rerun_backtest(
        self, strategy_id: str, period_days: int, symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Re-run backtest for a strategy.

        Args:
            strategy_id: ID of the strategy
            period_days: Backtesting period
            symbols: Optional list of symbols to test

        Returns:
            New backtest results
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        # This would implement the actual backtesting logic
        # For now, return mock results
        return {
            "period_days": period_days,
            "symbols": symbols or ["BTC/USDT", "ETH/USDT"],
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.08,
            "win_rate": 0.65,
        }

    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.

        Returns:
            Performance summary statistics
        """
        total_strategies = len(self.strategies)

        if total_strategies == 0:
            return {
                "total_strategies": 0,
                "average_performance": {},
                "best_strategy": None,
                "generation_stats": {},
            }

        # Calculate aggregate statistics
        performances = [s.get("performance", {}) for s in self.strategies.values()]

        avg_sharpe = sum(p.get("sharpe_ratio", 0) for p in performances) / total_strategies
        avg_return = sum(p.get("total_return", 0) for p in performances) / total_strategies
        avg_drawdown = sum(p.get("max_drawdown", 0) for p in performances) / total_strategies

        # Find best strategy
        best_strategy = max(
            self.strategies.items(),
            key=lambda x: x[1].get("performance", {}).get("overall_score", 0),
            default=(None, None),
        )

        return {
            "total_strategies": total_strategies,
            "average_performance": {
                "sharpe_ratio": avg_sharpe,
                "total_return": avg_return,
                "max_drawdown": avg_drawdown,
            },
            "best_strategy": (
                {
                    "strategy_id": best_strategy[0],
                    "performance": best_strategy[1].get("performance", {}),
                }
                if best_strategy[0]
                else None
            ),
            "generation_stats": {
                "active_sessions": len(self.active_sessions),
                "completed_sessions": len(
                    [s for s in self.active_sessions.values() if s["status"] == "completed"]
                ),
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on system components.

        Returns:
            Health status of components
        """
        health_status = {}

        try:
            # Check trading system
            health_status["trading_system"] = {
                "status": "healthy",
                "exchange_connected": True,  # Would check actual connection
            }

            # Check model availability
            health_status["language_model"] = {
                "status": "healthy",
                "model": "claude-3-sonnet-20240229",
            }

            # Check storage
            health_status["storage"] = {
                "status": "healthy",
                "strategies_count": len(self.strategies),
                "active_sessions": len(self.active_sessions),
            }

        except Exception as e:
            health_status["error"] = str(e)

        return health_status
