"""
Advanced error recovery module for DataMCPServerAgent.
This module provides sophisticated retry strategies, automatic fallback mechanisms,
and a self-healing system that learns from errors.
"""

import asyncio
import json
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import (
    ConnectionError,
    MCPError,
    RateLimitError,
    classify_error,
)

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")


class RetryStrategy(Enum):
    """Enum for different retry strategies."""

    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    JITTERED = "jittered"  # Jittered exponential backoff
    CONSTANT = "constant"  # Constant delay
    ADAPTIVE = "adaptive"  # Adaptive based on error type


class CircuitBreakerState(Enum):
    """Enum for circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before trying to recover
            half_open_max_calls: Maximum number of calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            # If we're in half-open state and get a success, close the circuit
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # If we're testing the service and it fails, go back to open
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker reopened after failed recovery attempt")

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if the request should be allowed, False otherwise
        """
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
                # First call in half-open state
                self.half_open_calls = 1
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            else:
                # If we've reached the maximum number of calls in half-open state,
                # don't allow any more requests until we get a success or failure
                return False

        return False


class ErrorRecoverySystem:
    """Advanced error recovery system with retry strategies, fallbacks, and learning."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the error recovery system.

        Args:
            model: Language model for error analysis and recovery
            db: Memory database for persistence
            tools: Optional list of available tools
        """
        self.model = model
        self.db = db
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools} if tools else {}

        # Circuit breakers for different services/tools
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Create the error analysis prompt
        self.error_analysis_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced error analysis system responsible for diagnosing errors and suggesting recovery strategies.
Your job is to analyze error information and determine the best approach for recovery.

For each error, you should:
1. Analyze the error type and message in detail
2. Consider the context in which the error occurred
3. Determine the most appropriate retry strategy
4. Suggest alternative tools or approaches
5. Provide specific recovery actions

Respond with a JSON object containing:
- "error_type": Classified error type
- "severity": Error severity (low, medium, high, critical)
- "retry_strategy": Recommended retry strategy
- "max_retries": Recommended maximum number of retries
- "alternative_tools": Array of alternative tools to try
- "recovery_actions": Specific actions to take for recovery
- "self_healing_suggestions": Suggestions for preventing this error in the future
"""
                ),
                HumanMessage(
                    content="""
Error information:
{error_info}

Context:
{context}

Available tools:
{tool_descriptions}

Analyze this error and suggest recovery strategies.
"""
                ),
            ]
        )

        # Create the learning prompt
        self.learning_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced error recovery learning system responsible for improving error handling over time.
Your job is to analyze patterns in errors and successful recoveries to suggest improvements.

For the error patterns provided, you should:
1. Identify common patterns in errors
2. Analyze which recovery strategies were most successful
3. Suggest improvements to retry strategies
4. Recommend changes to fallback mechanisms
5. Provide self-healing suggestions

Respond with a JSON object containing:
- "identified_patterns": Array of identified error patterns
- "successful_strategies": Object mapping error types to successful strategies
- "retry_improvements": Suggestions for improving retry strategies
- "fallback_improvements": Suggestions for improving fallback mechanisms
- "self_healing_improvements": Suggestions for self-healing capabilities
"""
                ),
                HumanMessage(
                    content="""
Error patterns:
{error_patterns}

Recovery successes:
{recovery_successes}

Recovery failures:
{recovery_failures}

Analyze these patterns and suggest improvements.
"""
                ),
            ]
        )

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the error recovery system.

        Args:
            tool: Tool to register
        """
        self.tools.append(tool)
        self.tool_map[tool.name] = tool

        # Create a circuit breaker for the tool
        self.circuit_breakers[tool.name] = CircuitBreaker()

    def register_tools(self, tools: List[BaseTool]) -> None:
        """Register multiple tools with the error recovery system.

        Args:
            tools: Tools to register
        """
        for tool in tools:
            self.register_tool(tool)

    async def with_advanced_retry(
        self,
        func: Callable[..., T],
        *args,
        tool_name: Optional[str] = None,
        retry_strategy: RetryStrategy = RetryStrategy.ADAPTIVE,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter_factor: float = 0.1,
        **kwargs,
    ) -> T:
        """Execute a function with advanced retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            tool_name: Optional name of the tool being used
            retry_strategy: Retry strategy to use
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter_factor: Factor for jitter in jittered strategy
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        attempt = 0

        # Check circuit breaker if tool_name is provided
        if tool_name and tool_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[tool_name]
            if not circuit_breaker.allow_request():
                raise MCPError(
                    f"Circuit breaker open for {tool_name}",
                    "circuit_breaker",
                    f"Wait {circuit_breaker.recovery_timeout} seconds before retrying",
                )

        while attempt <= max_retries:
            try:
                result = await func(*args, **kwargs)

                # Record success in circuit breaker
                if tool_name and tool_name in self.circuit_breakers:
                    self.circuit_breakers[tool_name].record_success()

                # Save successful execution
                if tool_name:
                    self._save_execution_result(tool_name, True, None)

                return result
            except Exception as e:
                last_exception = e
                attempt += 1

                # Record failure in circuit breaker
                if tool_name and tool_name in self.circuit_breakers:
                    self.circuit_breakers[tool_name].record_failure()

                # Save failed execution
                if tool_name:
                    self._save_execution_result(tool_name, False, str(e))

                # If we've exhausted all retries, raise the exception
                if attempt > max_retries:
                    break

                # Calculate delay based on retry strategy
                delay = self._calculate_delay(
                    attempt, retry_strategy, base_delay, max_delay, jitter_factor, e
                )

                logger.info(
                    f"Retry attempt {attempt}/{max_retries} for {tool_name or func.__name__} after {delay:.2f}s delay"
                )
                await asyncio.sleep(delay)

        # If we've exhausted all retries, raise the last exception
        raise last_exception

    def _calculate_delay(
        self,
        attempt: int,
        strategy: RetryStrategy,
        base_delay: float,
        max_delay: float,
        jitter_factor: float,
        error: Exception,
    ) -> float:
        """Calculate delay for retry based on strategy.

        Args:
            attempt: Current attempt number (1-based)
            strategy: Retry strategy to use
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter_factor: Factor for jitter in jittered strategy
            error: The exception that occurred

        Returns:
            Delay in seconds
        """
        if strategy == RetryStrategy.EXPONENTIAL:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        elif strategy == RetryStrategy.LINEAR:
            delay = min(base_delay * attempt, max_delay)
        elif strategy == RetryStrategy.JITTERED:
            # Exponential backoff with jitter
            exponential_delay = base_delay * (2 ** (attempt - 1))
            jitter = exponential_delay * jitter_factor * random.uniform(-1, 1)
            delay = min(exponential_delay + jitter, max_delay)
        elif strategy == RetryStrategy.CONSTANT:
            delay = base_delay
        elif strategy == RetryStrategy.ADAPTIVE:
            # Adapt based on error type
            if (
                isinstance(error, RateLimitError)
                and hasattr(error, "retry_after")
                and error.retry_after
            ):
                # Use the retry-after value if provided
                delay = error.retry_after
            elif isinstance(error, ConnectionError):
                # Use exponential backoff for connection errors
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            else:
                # Default to jittered exponential backoff
                exponential_delay = base_delay * (2 ** (attempt - 1))
                jitter = exponential_delay * jitter_factor * random.uniform(-1, 1)
                delay = min(exponential_delay + jitter, max_delay)
        else:
            # Default to exponential backoff
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

        return delay

    def _save_execution_result(
        self, tool_name: str, success: bool, error_message: Optional[str]
    ) -> None:
        """Save execution result for learning.

        Args:
            tool_name: Name of the tool
            success: Whether the execution was successful
            error_message: Error message if execution failed
        """
        execution_data = {
            "tool_name": tool_name,
            "success": success,
            "timestamp": time.time(),
        }

        if error_message:
            execution_data["error_message"] = error_message

        self.db.save_entity("error_recovery", f"execution_{int(time.time())}", execution_data)

    async def analyze_error(
        self, error: Exception, context: Dict[str, Any], tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze an error and suggest recovery strategies.

        Args:
            error: The exception that occurred
            context: Context information about the operation
            tool_name: Optional name of the tool that failed

        Returns:
            Analysis results with recovery strategies
        """
        # Classify the error if it's not already an MCPError
        if not isinstance(error, MCPError):
            error = classify_error(error)

        # Format error information
        error_info = f"Error type: {error.__class__.__name__}\n"
        error_info += f"Error message: {str(error)}\n"

        if hasattr(error, "error_type"):
            error_info += f"Classified error type: {error.error_type}\n"

        if hasattr(error, "recovery_suggestion"):
            error_info += f"Recovery suggestion: {error.recovery_suggestion}\n"

        # Format context information
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

        # Format tool descriptions
        tool_descriptions = "\n\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools]
        )

        # Prepare the input for the prompt
        input_values = {
            "error_info": error_info,
            "context": context_str,
            "tool_descriptions": tool_descriptions,
        }

        # Get the error analysis from the model
        messages = self.error_analysis_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            analysis = json.loads(json_str)

            # Save the analysis to the database
            if tool_name:
                self.db.save_entity(
                    "error_recovery",
                    f"analysis_{tool_name}_{int(time.time())}",
                    analysis,
                )

            return analysis
        except Exception as e:
            # If parsing fails, return a default analysis
            default_analysis = {
                "error_type": error.error_type if hasattr(error, "error_type") else "unknown",
                "severity": "medium",
                "retry_strategy": "exponential",
                "max_retries": 3,
                "alternative_tools": [],
                "recovery_actions": [
                    "Wait and retry with exponential backoff",
                    "Check service status",
                    "Verify input parameters",
                ],
                "self_healing_suggestions": [
                    "Implement more robust error handling",
                    "Add circuit breaker pattern",
                    "Improve input validation",
                ],
            }

            # Save the default analysis to the database
            if tool_name:
                self.db.save_entity(
                    "error_recovery",
                    f"analysis_{tool_name}_{int(time.time())}",
                    {**default_analysis, "parsing_error": str(e)},
                )

            return default_analysis

    async def get_alternative_tools(self, failed_tool: str, context: Dict[str, Any]) -> List[str]:
        """Get alternative tools when a specific tool fails.

        Args:
            failed_tool: Name of the tool that failed
            context: Context information about the operation

        Returns:
            List of alternative tool names
        """
        # Check if we have an error analysis for this tool
        analysis_key = f"analysis_{failed_tool}"
        analysis = self.db.get_entity("error_recovery", analysis_key)

        # Log context information for debugging
        logger.debug(f"Getting alternative tools for {failed_tool} with context: {context}")

        if analysis and "alternative_tools" in analysis:
            # Filter to ensure all tools exist
            alternatives = [t for t in analysis["alternative_tools"] if t in self.tool_map]
            if alternatives:
                return alternatives

        # Fallback to predefined alternatives
        predefined_alternatives = {
            "scrape_as_markdown_Bright_Data": [
                "scrape_as_html_Bright_Data",
                "scraping_browser_get_text_Bright_Data",
                "enhanced_web_scraper",
            ],
            "scrape_as_html_Bright_Data": [
                "scrape_as_markdown_Bright_Data",
                "scraping_browser_get_html_Bright_Data",
                "enhanced_web_scraper",
            ],
            "brave_web_search_Brave": [
                "enhanced_web_search",
                "search_engine_Bright_Data",
            ],
            "web_data_amazon_product_Bright_Data": [
                "product_comparison",
                "scrape_as_markdown_Bright_Data",
            ],
            "web_data_instagram_profiles_Bright_Data": [
                "social_media_analyzer",
                "scrape_as_markdown_Bright_Data",
            ],
            "web_data_facebook_posts_Bright_Data": [
                "social_media_analyzer",
                "scrape_as_markdown_Bright_Data",
            ],
            "web_data_x_posts_Bright_Data": [
                "social_media_analyzer",
                "scrape_as_markdown_Bright_Data",
            ],
            "enhanced_web_scraper": [
                "scrape_as_markdown_Bright_Data",
                "scrape_as_html_Bright_Data",
            ],
            "enhanced_web_search": [
                "brave_web_search_Brave",
                "search_engine_Bright_Data",
            ],
            "product_comparison": [
                "web_data_amazon_product_Bright_Data",
                "scrape_as_markdown_Bright_Data",
            ],
            "social_media_analyzer": [
                "web_data_instagram_profiles_Bright_Data",
                "web_data_facebook_posts_Bright_Data",
                "web_data_x_posts_Bright_Data",
                "scrape_as_markdown_Bright_Data",
            ],
        }

        # Filter to ensure all tools exist
        alternatives = [
            t for t in predefined_alternatives.get(failed_tool, []) if t in self.tool_map
        ]

        # If no alternatives found, return the most reliable tools
        if not alternatives:
            # Get tools with highest success rates
            success_rates = {}
            for tool_name in self.tool_map:
                executions = self.db.get_entities_by_prefix(
                    "error_recovery", f"execution_{tool_name}"
                )
                if executions:
                    successes = sum(1 for e in executions if e.get("success", False))
                    success_rates[tool_name] = successes / len(executions)

            # Sort by success rate and return top 2
            sorted_tools = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            alternatives = [t[0] for t in sorted_tools[:2] if t[0] != failed_tool]

        return alternatives

    async def try_with_fallbacks(
        self,
        primary_tool: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        max_fallbacks: int = 2,
    ) -> Tuple[Any, str, bool]:
        """Try a primary tool with fallbacks if it fails.

        Args:
            primary_tool: Name of the primary tool to try
            args: Arguments for the tool
            context: Context information about the operation
            max_fallbacks: Maximum number of fallbacks to try

        Returns:
            Tuple of (result, tool_used, success)
        """
        # Try the primary tool first
        if primary_tool in self.tool_map:
            try:
                result = await self.with_advanced_retry(
                    self.tool_map[primary_tool].invoke, args, tool_name=primary_tool
                )
                return result, primary_tool, True
            except Exception as e:
                # Primary tool failed, get alternatives
                logger.warning(f"Primary tool {primary_tool} failed: {str(e)}")
                alternatives = await self.get_alternative_tools(primary_tool, context)

                # Try alternatives
                for i, alt_tool in enumerate(alternatives[:max_fallbacks]):
                    if alt_tool in self.tool_map:
                        try:
                            logger.info(
                                f"Trying fallback tool {i + 1}/{min(max_fallbacks, len(alternatives))}: {alt_tool}"
                            )
                            result = await self.with_advanced_retry(
                                self.tool_map[alt_tool].invoke, args, tool_name=alt_tool
                            )
                            return result, alt_tool, True
                        except Exception as alt_e:
                            logger.warning(f"Fallback tool {alt_tool} failed: {str(alt_e)}")

                # All fallbacks failed
                raise Exception(
                    f"All tools failed for operation: {primary_tool} and {len(alternatives[:max_fallbacks])} fallbacks"
                )
        else:
            raise ValueError(f"Tool not found: {primary_tool}")

    async def learn_from_errors(self) -> Dict[str, Any]:
        """Learn from past errors to improve recovery strategies.

        Returns:
            Learning results with improvement suggestions
        """
        # Get error patterns from the database
        executions = self.db.get_entities_by_prefix("error_recovery", "execution_")
        analyses = self.db.get_entities_by_prefix("error_recovery", "analysis_")

        if not executions:
            return {
                "identified_patterns": [],
                "successful_strategies": {},
                "retry_improvements": [],
                "fallback_improvements": [],
                "self_healing_improvements": [],
            }

        # Format error patterns
        error_patterns = []
        for execution in executions:
            if not execution.get("success", True) and "error_message" in execution:
                error_patterns.append(
                    {
                        "tool": execution.get("tool_name", "unknown"),
                        "error": execution.get("error_message", "unknown error"),
                        "timestamp": execution.get("timestamp", 0),
                    }
                )

        # Format recovery successes and failures
        recovery_successes = []
        recovery_failures = []

        for analysis in analyses:
            tool_name = analysis.get("tool_name", "unknown")

            # Find executions after this analysis
            analysis_time = analysis.get("timestamp", 0)
            related_executions = [
                e
                for e in executions
                if e.get("tool_name") == tool_name and e.get("timestamp", 0) > analysis_time
            ]

            if related_executions:
                success_rate = sum(1 for e in related_executions if e.get("success", False)) / len(
                    related_executions
                )

                if success_rate > 0.5:
                    recovery_successes.append(
                        {
                            "tool": tool_name,
                            "strategy": analysis.get("retry_strategy", "unknown"),
                            "alternatives": analysis.get("alternative_tools", []),
                            "success_rate": success_rate,
                        }
                    )
                else:
                    recovery_failures.append(
                        {
                            "tool": tool_name,
                            "strategy": analysis.get("retry_strategy", "unknown"),
                            "alternatives": analysis.get("alternative_tools", []),
                            "success_rate": success_rate,
                        }
                    )

        # Format the input for the learning prompt
        input_values = {
            "error_patterns": json.dumps(error_patterns),
            "recovery_successes": json.dumps(recovery_successes),
            "recovery_failures": json.dumps(recovery_failures),
        }

        # Get the learning insights from the model
        messages = self.learning_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            learning = json.loads(json_str)

            # Save the learning to the database
            self.db.save_entity("error_recovery", f"learning_{int(time.time())}", learning)

            return learning
        except Exception as e:
            # If parsing fails, return a default learning
            default_learning = {
                "identified_patterns": ["Error parsing learning results"],
                "successful_strategies": {},
                "retry_improvements": ["Implement more sophisticated retry strategies"],
                "fallback_improvements": ["Improve fallback tool selection"],
                "self_healing_improvements": ["Enhance error pattern recognition"],
            }

            # Save the default learning to the database
            self.db.save_entity(
                "error_recovery",
                f"learning_{int(time.time())}",
                {**default_learning, "parsing_error": str(e)},
            )

            return default_learning
