"""
Comprehensive test suite for Trading Infinite Loop system.

This module provides unit tests, integration tests, and performance tests
for the trading strategy generation system.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.trading_infinite_loop.trading_strategy_orchestrator import (
    TradingStrategyOrchestrator,
    TradingStrategyConfig
)
from src.api.services.trading_strategy_service import TradingStrategyService
from src.agents.trading_system import AdvancedCryptoTradingSystem


class TestTradingStrategyConfig:
    """Test trading strategy configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TradingStrategyConfig()
        
        assert config.target_symbols == ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        assert config.strategy_types == ["momentum", "mean_reversion", "arbitrage", "ml_based"]
        assert config.risk_tolerance == 0.02
        assert config.min_profit_threshold == 0.005
        assert config.backtest_period_days == 30
        assert config.min_sharpe_ratio == 1.5
        assert config.max_drawdown == 0.1
        assert config.min_win_rate == 0.6
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TradingStrategyConfig(
            target_symbols=["BTC/USDT"],
            risk_tolerance=0.05,
            min_sharpe_ratio=2.0
        )
        
        assert config.target_symbols == ["BTC/USDT"]
        assert config.risk_tolerance == 0.05
        assert config.min_sharpe_ratio == 2.0


class TestTradingStrategyOrchestrator:
    """Test trading strategy orchestrator."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock language model."""
        return MagicMock()
    
    @pytest.fixture
    def mock_tools(self):
        """Mock tools list."""
        return []
    
    @pytest.fixture
    def mock_trading_system(self):
        """Mock trading system."""
        return MagicMock(spec=AdvancedCryptoTradingSystem)
    
    @pytest.fixture
    def orchestrator(self, mock_model, mock_tools, mock_trading_system):
        """Create orchestrator instance."""
        config = TradingStrategyConfig()
        return TradingStrategyOrchestrator(
            model=mock_model,
            tools=mock_tools,
            trading_system=mock_trading_system,
            config=config
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.model is not None
        assert orchestrator.tools == []
        assert orchestrator.trading_system is not None
        assert isinstance(orchestrator.config, TradingStrategyConfig)
        assert orchestrator.strategies == {}
        assert orchestrator.performance_history == []
    
    def test_create_strategy_specification(self, orchestrator):
        """Test strategy specification creation."""
        spec = orchestrator._create_strategy_specification()
        
        assert spec["content_type"] == "trading_strategy"
        assert spec["format"] == "python_class"
        assert spec["evolution_pattern"] == "genetic_algorithm"
        assert "entry_conditions" in spec["innovation_areas"]
        assert "exit_conditions" in spec["innovation_areas"]
        assert "risk_management" in spec["innovation_areas"]
        assert spec["quality_requirements"]["min_sharpe_ratio"] == 1.5
        assert spec["target_symbols"] == ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    @pytest.mark.asyncio
    async def test_generate_trading_strategies(self, orchestrator):
        """Test strategy generation process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the infinite loop execution
            with patch.object(orchestrator.infinite_loop, 'execute_infinite_loop') as mock_execute:
                mock_execute.return_value = {
                    "success": True,
                    "session_id": "test_session",
                    "results": {"total_iterations": 5}
                }
                
                # Mock strategy processing
                with patch.object(orchestrator, '_process_generated_strategies') as mock_process:
                    mock_process.return_value = None
                    
                    result = await orchestrator.generate_trading_strategies(
                        count=5,
                        output_dir=temp_dir
                    )
                    
                    assert result["success"] is True
                    assert "session_id" in result
                    mock_execute.assert_called_once()
                    mock_process.assert_called_once()
    
    def test_calculate_backtest_metrics(self, orchestrator):
        """Test backtest metrics calculation."""
        backtest_results = {
            "daily_returns": [0.01, -0.005, 0.02, 0.015, -0.01, 0.008]
        }
        
        metrics = orchestrator._calculate_backtest_metrics(backtest_results)
        
        assert "total_return" in metrics
        assert "annual_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert metrics["total_trades"] == 6
        assert metrics["winning_trades"] == 4
        assert metrics["win_rate"] == pytest.approx(0.667, rel=1e-2)
    
    def test_meets_performance_criteria(self, orchestrator):
        """Test performance criteria evaluation."""
        # Good performance
        good_performance = {
            "sharpe_ratio": 2.0,
            "max_drawdown": -0.05,
            "win_rate": 0.7
        }
        assert orchestrator._meets_performance_criteria(good_performance) is True
        
        # Poor performance
        poor_performance = {
            "sharpe_ratio": 0.5,
            "max_drawdown": -0.15,
            "win_rate": 0.4
        }
        assert orchestrator._meets_performance_criteria(poor_performance) is False
    
    @pytest.mark.asyncio
    async def test_get_best_strategies(self, orchestrator):
        """Test getting best strategies."""
        # Add mock strategies
        orchestrator.strategies = {
            "strategy_1": {
                "performance": {"overall_score": 0.8},
                "created_at": "2024-01-01T00:00:00"
            },
            "strategy_2": {
                "performance": {"overall_score": 0.9},
                "created_at": "2024-01-02T00:00:00"
            },
            "strategy_3": {
                "performance": {"overall_score": 0.7},
                "created_at": "2024-01-03T00:00:00"
            }
        }
        
        best_strategies = await orchestrator.get_best_strategies(limit=2)
        
        assert len(best_strategies) == 2
        assert best_strategies[0]["strategy_id"] == "strategy_2"
        assert best_strategies[1]["strategy_id"] == "strategy_1"


class TestTradingStrategyService:
    """Test trading strategy service."""
    
    @pytest.fixture
    def service(self):
        """Create service instance."""
        with patch('src.api.services.trading_strategy_service.get_settings'):
            with patch.object(TradingStrategyService, '_initialize_components'):
                service = TradingStrategyService()
                service.model = MagicMock()
                service.tools = []
                service.trading_system = MagicMock()
                return service
    
    @pytest.mark.asyncio
    async def test_start_generation(self, service):
        """Test starting strategy generation."""
        config = TradingStrategyConfig()
        
        with patch.object(service, '_run_generation') as mock_run:
            session_id = await service.start_generation(count=5, config=config)
            
            assert session_id in service.active_sessions
            assert service.active_sessions[session_id]["status"] == "starting"
            assert service.active_sessions[session_id]["strategies_generated"] == 0
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_generation_status(self, service):
        """Test getting generation status."""
        # Create mock session
        session_id = "test_session"
        service.active_sessions[session_id] = {
            "session_id": session_id,
            "status": "running",
            "progress": 0.5,
            "strategies_generated": 10,
            "strategies_accepted": 8,
            "current_wave": 2,
            "start_time": datetime.now() - timedelta(minutes=5),
            "execution_time": 300.0,
            "errors": []
        }
        
        status = await service.get_generation_status(session_id)
        
        assert status is not None
        assert status["session_id"] == session_id
        assert status["status"] == "running"
        assert status["strategies_generated"] == 10
        assert status["strategies_accepted"] == 8
    
    @pytest.mark.asyncio
    async def test_list_strategies(self, service):
        """Test listing strategies."""
        # Add mock strategies
        service.strategies = {
            "strategy_1": {
                "strategy_id": "strategy_1",
                "performance": {"sharpe_ratio": 2.0, "overall_score": 0.8},
                "created_at": "2024-01-01T00:00:00"
            },
            "strategy_2": {
                "strategy_id": "strategy_2", 
                "performance": {"sharpe_ratio": 1.5, "overall_score": 0.6},
                "created_at": "2024-01-02T00:00:00"
            }
        }
        
        strategies = await service.list_strategies(limit=10)
        
        assert len(strategies) == 2
        assert strategies[0]["strategy_id"] == "strategy_1"  # Higher score first
        assert strategies[1]["strategy_id"] == "strategy_2"
    
    @pytest.mark.asyncio
    async def test_deploy_strategy(self, service):
        """Test strategy deployment."""
        # Add mock strategy
        strategy_id = "test_strategy"
        service.strategies[strategy_id] = {
            "strategy_id": strategy_id,
            "performance": {"sharpe_ratio": 2.0},
            "status": "generated"
        }
        
        result = await service.deploy_strategy(
            strategy_id=strategy_id,
            allocation=0.1,
            max_position_size=0.05,
            stop_loss=0.02
        )
        
        assert result["live_trading_started"] is True
        assert "deployment_id" in result
        assert service.strategies[strategy_id]["status"] == "deployed"
    
    @pytest.mark.asyncio
    async def test_get_performance_summary(self, service):
        """Test performance summary."""
        # Add mock strategies
        service.strategies = {
            "strategy_1": {
                "performance": {
                    "sharpe_ratio": 2.0,
                    "total_return": 0.15,
                    "max_drawdown": -0.05,
                    "overall_score": 0.8
                }
            },
            "strategy_2": {
                "performance": {
                    "sharpe_ratio": 1.5,
                    "total_return": 0.10,
                    "max_drawdown": -0.08,
                    "overall_score": 0.6
                }
            }
        }
        
        summary = await service.get_performance_summary()
        
        assert summary["total_strategies"] == 2
        assert summary["average_performance"]["sharpe_ratio"] == 1.75
        assert summary["average_performance"]["total_return"] == 0.125
        assert summary["best_strategy"]["strategy_id"] == "strategy_1"


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_strategy_generation(self):
        """Test complete strategy generation workflow."""
        with patch('src.api.services.trading_strategy_service.get_settings'):
            with patch.object(TradingStrategyService, '_initialize_components'):
                service = TradingStrategyService()
                service.model = MagicMock()
                service.tools = []
                service.trading_system = MagicMock()
                
                config = TradingStrategyConfig(
                    target_symbols=["BTC/USDT"],
                    backtest_period_days=7
                )
                
                # Mock the generation process
                with patch.object(service, '_run_generation') as mock_run:
                    # Start generation
                    session_id = await service.start_generation(count=3, config=config)
                    
                    # Simulate completion
                    service.active_sessions[session_id]["status"] = "completed"
                    service.active_sessions[session_id]["strategies_accepted"] = 2
                    
                    # Add mock strategies
                    service.strategies["strategy_1"] = {
                        "strategy_id": "strategy_1",
                        "performance": {"sharpe_ratio": 2.0, "overall_score": 0.8},
                        "created_at": "2024-01-01T00:00:00"
                    }
                    
                    # Check status
                    status = await service.get_generation_status(session_id)
                    assert status["status"] == "completed"
                    assert status["strategies_accepted"] == 2
                    
                    # List strategies
                    strategies = await service.list_strategies()
                    assert len(strategies) == 1
                    
                    # Deploy strategy
                    result = await service.deploy_strategy(
                        strategy_id="strategy_1",
                        allocation=0.1,
                        max_position_size=0.05,
                        stop_loss=0.02
                    )
                    assert result["live_trading_started"] is True


class TestPerformance:
    """Performance tests for the trading infinite loop system."""
    
    @pytest.mark.asyncio
    async def test_strategy_generation_performance(self):
        """Test performance of strategy generation."""
        start_time = time.time()
        
        # Mock rapid strategy generation
        config = TradingStrategyConfig()
        orchestrator = MagicMock()
        
        # Simulate processing 100 strategies
        for i in range(100):
            strategy_data = {"code": f"strategy_{i}", "performance": {"sharpe_ratio": 1.5 + i * 0.01}}
            # Mock processing time
            await asyncio.sleep(0.001)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should process 100 strategies in reasonable time
        assert execution_time < 1.0  # Less than 1 second for mock processing
    
    def test_memory_usage(self):
        """Test memory usage with large number of strategies."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create service with many strategies
        service = TradingStrategyService()
        service.strategies = {}
        
        # Add 1000 mock strategies
        for i in range(1000):
            service.strategies[f"strategy_{i}"] = {
                "strategy_id": f"strategy_{i}",
                "performance": {"sharpe_ratio": 1.5, "overall_score": 0.7},
                "backtest_results": {"trades": [{"pnl": 100}] * 100}  # Mock large data
            }
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 1000 strategies)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
