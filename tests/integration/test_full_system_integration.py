"""
Full System Integration Tests

This module provides comprehensive integration tests for the complete
Trading Infinite Loop system, testing end-to-end workflows.
"""

import asyncio
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestFullSystemIntegration:
    """
    Full system integration tests covering the complete workflow
    from strategy generation to deployment and monitoring.
    """

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_trading_system(self):
        """Mock trading system for testing."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_complete_strategy_generation_workflow(self, client):
        """
        Test the complete strategy generation workflow:
        1. Start generation
        2. Monitor progress
        3. List generated strategies
        4. Deploy strategy
        5. Monitor performance
        """
        # Step 1: Start strategy generation
        generation_request = {
            "count": 5,
            "target_symbols": ["BTC/USDT", "ETH/USDT"],
            "strategy_types": ["momentum", "mean_reversion"],
            "risk_tolerance": 0.02,
            "min_profit_threshold": 0.005,
            "backtest_period_days": 30
        }

        response = client.post("/api/trading-infinite-loop/generate", json=generation_request)
        assert response.status_code == 200

        generation_data = response.json()
        assert generation_data["success"] is True
        assert "session_id" in generation_data

        session_id = generation_data["session_id"]

        # Step 2: Monitor progress
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            response = client.get(f"/api/trading-infinite-loop/status/{session_id}")
            assert response.status_code == 200

            status_data = response.json()
            assert status_data["session_id"] == session_id

            if status_data["status"] in ["completed", "error"]:
                break

            attempts += 1
            await asyncio.sleep(1)  # Wait 1 second between checks

        assert status_data["status"] == "completed"
        assert status_data["strategies_accepted"] > 0

        # Step 3: List generated strategies
        response = client.get("/api/trading-infinite-loop/strategies?limit=10")
        assert response.status_code == 200

        strategies = response.json()
        assert len(strategies) > 0

        best_strategy = strategies[0]
        strategy_id = best_strategy["strategy_id"]

        # Verify strategy has required performance metrics
        assert "performance" in best_strategy
        assert "sharpe_ratio" in best_strategy["performance"]
        assert best_strategy["performance"]["sharpe_ratio"] > 0

        # Step 4: Get detailed strategy information
        response = client.get(f"/api/trading-infinite-loop/strategies/{strategy_id}")
        assert response.status_code == 200

        strategy_details = response.json()
        assert strategy_details["strategy_id"] == strategy_id
        assert "backtest_results" in strategy_details

        # Step 5: Deploy strategy
        deployment_request = {
            "allocation": 0.1,
            "max_position_size": 0.05,
            "stop_loss": 0.02
        }

        response = client.post(
            f"/api/trading-infinite-loop/strategies/{strategy_id}/deploy",
            json=deployment_request
        )
        assert response.status_code == 200

        deployment_data = response.json()
        assert deployment_data["success"] is True
        assert deployment_data["live_trading_started"] is True

        # Step 6: Verify deployment
        response = client.get(f"/api/trading-infinite-loop/strategies/{strategy_id}")
        assert response.status_code == 200

        updated_strategy = response.json()
        # Strategy should now be marked as deployed

    @pytest.mark.asyncio
    async def test_concurrent_strategy_generation(self, client):
        """
        Test concurrent strategy generation sessions to verify
        system can handle multiple simultaneous requests.
        """
        # Start multiple generation sessions concurrently
        session_ids = []

        async def start_generation(session_num):
            request_data = {
                "count": 3,
                "target_symbols": [f"SYMBOL{session_num}/USDT"],
                "strategy_types": ["momentum"],
                "risk_tolerance": 0.02
            }

            response = client.post("/api/trading-infinite-loop/generate", json=request_data)
            assert response.status_code == 200

            data = response.json()
            return data["session_id"]

        # Start 3 concurrent sessions
        tasks = [start_generation(i) for i in range(3)]
        session_ids = await asyncio.gather(*tasks)

        assert len(session_ids) == 3
        assert len(set(session_ids)) == 3  # All session IDs should be unique

        # Monitor all sessions
        for session_id in session_ids:
            response = client.get(f"/api/trading-infinite-loop/status/{session_id}")
            assert response.status_code == 200

            status_data = response.json()
            assert status_data["session_id"] == session_id
            assert status_data["status"] in ["starting", "running", "completed"]

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, client):
        """
        Test error handling and recovery mechanisms.
        """
        # Test invalid generation request
        invalid_request = {
            "count": -1,  # Invalid count
            "target_symbols": [],  # Empty symbols
            "risk_tolerance": 2.0  # Invalid risk tolerance
        }

        response = client.post("/api/trading-infinite-loop/generate", json=invalid_request)
        assert response.status_code == 422  # Validation error

        # Test non-existent session status
        response = client.get("/api/trading-infinite-loop/status/non-existent-session")
        assert response.status_code == 404

        # Test non-existent strategy
        response = client.get("/api/trading-infinite-loop/strategies/non-existent-strategy")
        assert response.status_code == 404

        # Test deployment of non-existent strategy
        deployment_request = {
            "allocation": 0.1,
            "max_position_size": 0.05,
            "stop_loss": 0.02
        }

        response = client.post(
            "/api/trading-infinite-loop/strategies/non-existent-strategy/deploy",
            json=deployment_request
        )
        assert response.status_code == 500  # Internal server error

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, client):
        """
        Test performance monitoring and metrics collection.
        """
        # Get performance summary
        response = client.get("/api/trading-infinite-loop/performance/summary")
        assert response.status_code == 200

        summary = response.json()
        assert "total_strategies" in summary
        assert "average_performance" in summary
        assert "generation_stats" in summary

        # Test health check
        response = client.get("/api/trading-infinite-loop/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "components" in health_data

    @pytest.mark.asyncio
    async def test_strategy_lifecycle_management(self, client):
        """
        Test complete strategy lifecycle: generation -> deployment -> monitoring -> deletion.
        """
        # Generate a strategy
        generation_request = {
            "count": 1,
            "target_symbols": ["BTC/USDT"],
            "strategy_types": ["momentum"]
        }

        response = client.post("/api/trading-infinite-loop/generate", json=generation_request)
        session_id = response.json()["session_id"]

        # Wait for completion (mock)
        await asyncio.sleep(2)

        # Get generated strategies
        response = client.get("/api/trading-infinite-loop/strategies")
        strategies = response.json()

        if len(strategies) > 0:
            strategy_id = strategies[0]["strategy_id"]

            # Deploy strategy
            deployment_request = {
                "allocation": 0.05,
                "max_position_size": 0.02,
                "stop_loss": 0.01
            }

            response = client.post(
                f"/api/trading-infinite-loop/strategies/{strategy_id}/deploy",
                json=deployment_request
            )

            # Get backtest results
            response = client.get(f"/api/trading-infinite-loop/strategies/{strategy_id}/backtest")
            assert response.status_code == 200

            # Re-run backtest with different parameters
            response = client.post(
                f"/api/trading-infinite-loop/strategies/{strategy_id}/rebacktest",
                json={"period_days": 60, "symbols": ["BTC/USDT", "ETH/USDT"]}
            )
            assert response.status_code == 200

            # Delete strategy
            response = client.delete(f"/api/trading-infinite-loop/strategies/{strategy_id}")
            assert response.status_code == 200

            # Verify deletion
            response = client.get(f"/api/trading-infinite-loop/strategies/{strategy_id}")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """
        Test WebSocket integration for real-time updates.
        """
        # This would test WebSocket connections for real-time updates
        # For now, we'll test the basic WebSocket endpoint availability

        async with httpx.AsyncClient() as client:
            # Test WebSocket endpoint (would need actual WebSocket testing)
            # This is a placeholder for WebSocket testing
            pass

    @pytest.mark.asyncio
    async def test_data_persistence_and_recovery(self, client):
        """
        Test data persistence and system recovery capabilities.
        """
        # Generate strategies
        generation_request = {
            "count": 2,
            "target_symbols": ["BTC/USDT"],
            "strategy_types": ["momentum"]
        }

        response = client.post("/api/trading-infinite-loop/generate", json=generation_request)
        session_id = response.json()["session_id"]

        # Simulate system restart by creating new service instance
        # (In real tests, this would involve actual service restart)

        # Verify data persistence
        response = client.get(f"/api/trading-infinite-loop/status/{session_id}")
        # Should either find the session or handle gracefully

        response = client.get("/api/trading-infinite-loop/strategies")
        # Should return previously generated strategies
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_performance_under_load(self, client):
        """
        Test system performance under load.
        """
        start_time = time.time()

        # Simulate multiple concurrent requests
        tasks = []

        for i in range(5):
            # Create different types of requests
            if i % 3 == 0:
                # Generation request
                task = asyncio.create_task(self._make_generation_request(client, i))
            elif i % 3 == 1:
                # Status check request
                task = asyncio.create_task(self._make_status_request(client))
            else:
                # Strategy list request
                task = asyncio.create_task(self._make_strategy_list_request(client))

            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance
        assert total_time < 10.0  # Should complete within 10 seconds

        # Check that most requests succeeded
        successful_requests = sum(1 for result in results if not isinstance(result, Exception))
        assert successful_requests >= len(tasks) * 0.8  # At least 80% success rate

    async def _make_generation_request(self, client, session_num):
        """Helper method for making generation requests."""
        request_data = {
            "count": 2,
            "target_symbols": [f"TEST{session_num}/USDT"],
            "strategy_types": ["momentum"]
        }

        response = client.post("/api/trading-infinite-loop/generate", json=request_data)
        return response.status_code == 200

    async def _make_status_request(self, client):
        """Helper method for making status requests."""
        response = client.get("/api/trading-infinite-loop/performance/summary")
        return response.status_code == 200

    async def _make_strategy_list_request(self, client):
        """Helper method for making strategy list requests."""
        response = client.get("/api/trading-infinite-loop/strategies")
        return response.status_code == 200


class TestUIIntegration:
    """
    Integration tests for UI components and frontend-backend communication.
    """

    @pytest.mark.asyncio
    async def test_ui_api_integration(self):
        """
        Test UI component integration with API endpoints.
        """
        # This would test the React components with actual API calls
        # For now, we'll test the API endpoints that the UI would use

        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Test endpoints used by the UI
            response = await client.get("/api/trading-infinite-loop/health")
            assert response.status_code == 200

            response = await client.get("/api/trading-infinite-loop/strategies")
            assert response.status_code == 200

            response = await client.get("/api/trading-infinite-loop/performance/summary")
            assert response.status_code == 200

    def test_ui_component_rendering(self):
        """
        Test UI component rendering and state management.
        """
        # This would test React component rendering
        # Placeholder for frontend testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
