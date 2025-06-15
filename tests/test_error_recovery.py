"""
Tests for the error recovery module.
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.error_handlers import ConnectionError
from src.utils.error_recovery import (
    CircuitBreaker,
    CircuitBreakerState,
    ErrorRecoverySystem,
    RetryStrategy,
)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for the CircuitBreaker class."""

    def setUp(self):
        """Set up test environment."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
        )

    def test_initial_state(self):
        """Test initial state of circuit breaker."""
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertTrue(self.circuit_breaker.allow_request())

    def test_record_failure(self):
        """Test recording failures."""
        # Record failures up to threshold
        for i in range(3):
            self.circuit_breaker.record_failure()

        # Circuit should be open now
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())

    def test_recovery_timeout(self):
        """Test recovery timeout."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.2)

        # Circuit should be half-open now
        self.assertTrue(self.circuit_breaker.allow_request())
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.HALF_OPEN)

    def test_half_open_success(self):
        """Test successful recovery from half-open state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.2)

        # Allow a request (moves to half-open)
        self.assertTrue(self.circuit_breaker.allow_request())

        # Record success
        self.circuit_breaker.record_success()

        # Circuit should be closed now
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)

    def test_half_open_failure(self):
        """Test failed recovery from half-open state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.2)

        # Allow a request (moves to half-open)
        self.assertTrue(self.circuit_breaker.allow_request())

        # Record failure
        self.circuit_breaker.record_failure()

        # Circuit should be open again
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)

    def test_half_open_max_calls(self):
        """Test maximum calls in half-open state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.2)

        # Allow first request (moves to half-open)
        self.assertTrue(self.circuit_breaker.allow_request())
        self.assertEqual(self.circuit_breaker.half_open_calls, 1)

        # Allow second request (still half-open)
        self.assertTrue(self.circuit_breaker.allow_request())
        self.assertEqual(self.circuit_breaker.half_open_calls, 2)

        # Third request should be blocked because half_open_max_calls is 2
        self.assertEqual(self.circuit_breaker.half_open_max_calls, 2)
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.HALF_OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())

class TestErrorRecoverySystem(unittest.IsolatedAsyncioTestCase):
    """Tests for the ErrorRecoverySystem class."""

    async def asyncSetUp(self):
        """Set up test environment."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.ainvoke = AsyncMock()

        # Create mock database
        self.mock_db = MagicMock()
        self.mock_db.save_entity = MagicMock()
        self.mock_db.get_entity = MagicMock()
        self.mock_db.get_entities_by_prefix = MagicMock()

        # Create mock tools
        self.mock_tool1 = MagicMock()
        self.mock_tool1.name = "test_tool1"
        self.mock_tool1.description = "Test tool 1"
        self.mock_tool1.invoke = AsyncMock()

        self.mock_tool2 = MagicMock()
        self.mock_tool2.name = "test_tool2"
        self.mock_tool2.description = "Test tool 2"
        self.mock_tool2.invoke = AsyncMock()

        # Create error recovery system
        self.recovery_system = ErrorRecoverySystem(
            model=self.mock_model,
            db=self.mock_db,
            tools=[self.mock_tool1, self.mock_tool2],
        )

    async def test_with_advanced_retry_success(self):
        """Test successful execution with retry."""
        # Mock function that succeeds
        mock_func = AsyncMock(return_value="success")

        # Execute with retry
        result = await self.recovery_system.with_advanced_retry(
            mock_func,
            "arg1",
            "arg2",
            tool_name="test_tool1",
            retry_strategy=RetryStrategy.EXPONENTIAL,
            max_retries=3,
            kwarg1="value1",
        )

        # Check result
        self.assertEqual(result, "success")

        # Check function was called with correct arguments
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

        # Check success was recorded
        self.mock_db.save_entity.assert_called_once()

    async def test_with_advanced_retry_failure(self):
        """Test failed execution with retry."""
        # Mock function that fails
        mock_func = AsyncMock(side_effect=ConnectionError("Connection failed"))

        # Execute with retry and expect exception
        with self.assertRaises(ConnectionError):
            await self.recovery_system.with_advanced_retry(
                mock_func,
                "arg1",
                tool_name="test_tool1",
                retry_strategy=RetryStrategy.EXPONENTIAL,
                max_retries=2,
                base_delay=0.01,  # Short delay for testing
            )

        # Check function was called multiple times (initial + retries)
        self.assertEqual(mock_func.call_count, 3)

        # Check failures were recorded
        self.assertEqual(self.mock_db.save_entity.call_count, 3)

    async def test_analyze_error(self):
        """Test error analysis."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "error_type": "connection",
                "severity": "medium",
                "retry_strategy": "exponential",
                "max_retries": 3,
                "alternative_tools": ["test_tool2"],
                "recovery_actions": ["Wait and retry"],
                "self_healing_suggestions": ["Improve error handling"],
            }
        )
        self.mock_model.ainvoke.return_value = mock_response

        # Analyze error
        error = ConnectionError("Connection failed")
        context = {"operation": "web_search", "query": "test query"}
        analysis = await self.recovery_system.analyze_error(
            error, context, "test_tool1"
        )

        # Check analysis
        self.assertEqual(analysis["error_type"], "connection")
        self.assertEqual(analysis["severity"], "medium")
        self.assertEqual(analysis["retry_strategy"], "exponential")
        self.assertEqual(analysis["alternative_tools"], ["test_tool2"])

    async def test_try_with_fallbacks_primary_success(self):
        """Test successful execution with primary tool."""
        # Mock primary tool success
        self.mock_tool1.invoke.return_value = "primary result"

        # Try with fallbacks
        result, tool_used, success = await self.recovery_system.try_with_fallbacks(
            "test_tool1", {"arg1": "value1"}, {"operation": "test_operation"}
        )

        # Check result
        self.assertEqual(result, "primary result")
        self.assertEqual(tool_used, "test_tool1")
        self.assertTrue(success)

        # Check primary tool was called
        self.mock_tool1.invoke.assert_called_once_with({"arg1": "value1"})

        # Check fallback tool was not called
        self.mock_tool2.invoke.assert_not_called()

    async def test_try_with_fallbacks_primary_failure(self):
        """Test fallback execution when primary tool fails."""
        # Mock primary tool failure and fallback success
        self.mock_tool1.invoke.side_effect = ConnectionError("Connection failed")
        self.mock_tool2.invoke.return_value = "fallback result"

        # Mock get_alternative_tools
        self.recovery_system.get_alternative_tools = AsyncMock(
            return_value=["test_tool2"]
        )

        # Try with fallbacks
        result, tool_used, success = await self.recovery_system.try_with_fallbacks(
            "test_tool1", {"arg1": "value1"}, {"operation": "test_operation"}
        )

        # Check result
        self.assertEqual(result, "fallback result")
        self.assertEqual(tool_used, "test_tool2")
        self.assertTrue(success)

        # Check both tools were called
        # The primary tool may be called multiple times due to retries
        self.assertTrue(self.mock_tool1.invoke.call_count >= 1)
        self.mock_tool2.invoke.assert_called_once()

if __name__ == "__main__":
    unittest.main()
