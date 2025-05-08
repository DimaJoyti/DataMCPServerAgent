# Error Recovery

This document describes the enhanced error recovery capabilities of the DataMCPServerAgent.

## Overview

The error recovery system provides sophisticated retry strategies, automatic fallback mechanisms, and a self-healing system that learns from errors. It is designed to make the agent more resilient to failures and improve its ability to recover from errors.

## Key Components

### 1. Sophisticated Retry Strategies

The error recovery system implements several retry strategies:

- **Exponential Backoff**: Increases the delay between retries exponentially (e.g., 1s, 2s, 4s, 8s)
- **Linear Backoff**: Increases the delay between retries linearly (e.g., 1s, 2s, 3s, 4s)
- **Jittered Backoff**: Adds random jitter to the exponential backoff to prevent thundering herd problems
- **Constant Delay**: Uses a constant delay between retries
- **Adaptive Backoff**: Adapts the retry strategy based on the error type

Example usage:

```python
from src.utils.error_recovery import ErrorRecoverySystem, RetryStrategy

# Create error recovery system
error_recovery = ErrorRecoverySystem(model, db, tools)

# Execute a function with retry
result = await error_recovery.with_advanced_retry(
    function_to_retry,
    arg1, arg2,
    tool_name="tool_name",
    retry_strategy=RetryStrategy.EXPONENTIAL,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0
)
```

### 2. Automatic Fallback Mechanisms

The error recovery system can automatically fall back to alternative tools when a primary tool fails. It selects fallback tools based on:

- **Error Analysis**: Analyzes the error to determine the most appropriate fallback tools
- **Tool Similarity**: Selects tools that are similar to the failed tool
- **Success Rates**: Prefers tools with higher success rates
- **Predefined Alternatives**: Uses predefined alternative tools for common scenarios

Example usage:

```python
from src.utils.error_recovery import ErrorRecoverySystem

# Create error recovery system
error_recovery = ErrorRecoverySystem(model, db, tools)

# Try with fallbacks
result, tool_used, success = await error_recovery.try_with_fallbacks(
    primary_tool="scrape_as_markdown_Bright_Data",
    args={"url": "https://example.com"},
    context={"operation": "web_scraping"},
    max_fallbacks=2
)
```

### 3. Self-Healing System

The self-healing system learns from errors to improve future error recovery. It:

- **Analyzes Error Patterns**: Identifies common error patterns across tools
- **Evaluates Recovery Strategies**: Determines which recovery strategies are most effective
- **Suggests Improvements**: Provides suggestions for improving retry strategies and fallback mechanisms
- **Adapts Over Time**: Continuously improves based on experience

Example usage:

```python
from src.utils.error_recovery import ErrorRecoverySystem

# Create error recovery system
error_recovery = ErrorRecoverySystem(model, db, tools)

# Learn from errors
learning_results = await error_recovery.learn_from_errors()
```

### 4. Circuit Breaker Pattern

The error recovery system implements the circuit breaker pattern to prevent cascading failures:

- **Closed State**: Normal operation, requests are allowed
- **Open State**: After multiple failures, requests are blocked to prevent further failures
- **Half-Open State**: After a recovery timeout, allows limited requests to test if the service is back

Example usage:

```python
from src.utils.error_recovery import CircuitBreaker

# Create circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=1
)

# Check if request should be allowed
if circuit_breaker.allow_request():
    try:
        # Execute request
        result = await execute_request()
        circuit_breaker.record_success()
    except Exception:
        circuit_breaker.record_failure()
```

## Error Analysis

The error recovery system analyzes errors to determine the most appropriate recovery strategy. It considers:

- **Error Type**: The type of error (e.g., connection, authentication, rate limit)
- **Error Severity**: The severity of the error (low, medium, high, critical)
- **Context**: The context in which the error occurred
- **Available Tools**: The tools available for fallback

Example error analysis result:

```json
{
  "error_type": "connection",
  "severity": "medium",
  "retry_strategy": "exponential",
  "max_retries": 3,
  "alternative_tools": ["scrape_as_html_Bright_Data", "enhanced_web_scraper"],
  "recovery_actions": [
    "Wait and retry with exponential backoff",
    "Check service status",
    "Verify input parameters"
  ],
  "self_healing_suggestions": [
    "Implement more robust error handling",
    "Add circuit breaker pattern",
    "Improve input validation"
  ]
}
```

## Learning from Errors

The error recovery system learns from past errors to improve future recovery. It:

- **Identifies Patterns**: Identifies common error patterns
- **Evaluates Strategies**: Evaluates the effectiveness of different recovery strategies
- **Suggests Improvements**: Suggests improvements to retry strategies and fallback mechanisms
- **Adapts Over Time**: Continuously improves based on experience

Example learning result:

```json
{
  "identified_patterns": [
    "Connection errors are common for web scraping tools",
    "Rate limit errors often occur in batches"
  ],
  "successful_strategies": {
    "connection": "jittered",
    "rate_limit": "adaptive"
  },
  "retry_improvements": [
    "Use jittered backoff for connection errors",
    "Use adaptive backoff for rate limit errors"
  ],
  "fallback_improvements": [
    "Prefer enhanced_web_scraper for connection errors",
    "Use search tools as fallbacks for content extraction errors"
  ],
  "self_healing_improvements": [
    "Implement automatic circuit breaker for frequently failing tools",
    "Add automatic parameter adjustment for retries"
  ]
}
```

## Integration with Agent Architecture

The error recovery system is integrated with the agent architecture through the `ErrorRecoveryCoordinatorAgent` class. This agent:

- **Selects Tools**: Selects the most appropriate tools for a request
- **Handles Errors**: Handles errors that occur during tool execution
- **Applies Retry Strategies**: Applies the most appropriate retry strategy for each error
- **Uses Fallbacks**: Falls back to alternative tools when a primary tool fails
- **Learns from Errors**: Learns from errors to improve future recovery

## Usage

To use the error recovery agent, run:

```bash
python main.py --mode error_recovery
```

Or use the Python API:

```python
import asyncio
from src.core.error_recovery_main import chat_with_error_recovery_agent

asyncio.run(chat_with_error_recovery_agent())
```

## Example

See `examples/enhanced_error_recovery_example.py` for a complete example of using the error recovery system.

## Future Improvements

1. **More Sophisticated Error Analysis**: Enhance error analysis with more advanced techniques
2. **Improved Fallback Selection**: Improve the selection of fallback tools based on more factors
3. **Adaptive Retry Strategies**: Develop more adaptive retry strategies based on error patterns
4. **Integration with Reinforcement Learning**: Integrate with reinforcement learning for improved decision-making
5. **Distributed Circuit Breakers**: Implement distributed circuit breakers for multi-agent systems
