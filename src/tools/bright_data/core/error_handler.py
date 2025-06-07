"""
Advanced error handling and recovery for Bright Data MCP Integration

This module provides comprehensive error handling with:
- Custom exception types
- Error categorization and analysis
- Automatic recovery strategies
- Error metrics and reporting
- Circuit breaker integration
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Error information container"""
    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any]
    traceback_str: str
    retry_count: int = 0
    recoverable: bool = True

class BrightDataException(Exception):
    """Base exception for Bright Data operations"""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class NetworkException(BrightDataException):
    """Network-related errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, context)

class AuthenticationException(BrightDataException):
    """Authentication-related errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, context)

class RateLimitException(BrightDataException):
    """Rate limiting errors"""

    def __init__(self, message: str, retry_after: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM, context)
        self.retry_after = retry_after

class ValidationException(BrightDataException):
    """Validation errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.LOW, context)

class ServerException(BrightDataException):
    """Server-side errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH, context)
        self.status_code = status_code

class TimeoutException(BrightDataException):
    """Timeout errors"""

    def __init__(self, message: str, timeout_duration: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, context)
        self.timeout_duration = timeout_duration

class BrightDataErrorHandler:
    """Advanced error handler with analytics and recovery strategies"""

    def __init__(self, max_error_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.error_callbacks: List[Callable[[ErrorInfo], None]] = []

        # Register default recovery strategies
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies"""
        self.recovery_strategies[ErrorCategory.RATE_LIMIT] = self._handle_rate_limit
        self.recovery_strategies[ErrorCategory.NETWORK] = self._handle_network_error
        self.recovery_strategies[ErrorCategory.TIMEOUT] = self._handle_timeout
        self.recovery_strategies[ErrorCategory.SERVER_ERROR] = self._handle_server_error

    def categorize_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """Categorize an exception and create ErrorInfo"""
        category = self._determine_category(exception)
        severity = self._determine_severity(exception, category)

        error_info = ErrorInfo(
            exception=exception,
            category=category,
            severity=severity,
            timestamp=time.time(),
            context=context or {},
            traceback_str=traceback.format_exc(),
            recoverable=self._is_recoverable(exception, category)
        )

        # Add to history and update counts
        self.error_history.append(error_info)
        self.error_counts[category] += 1

        return error_info

    def _determine_category(self, exception: Exception) -> ErrorCategory:
        """Determine error category based on exception type and message"""
        if isinstance(exception, BrightDataException):
            return exception.category

        exception_name = exception.__class__.__name__.lower()
        exception_message = str(exception).lower()

        # Network errors
        if any(keyword in exception_name for keyword in ['connection', 'network', 'socket']):
            return ErrorCategory.NETWORK

        # Timeout errors
        if any(keyword in exception_name for keyword in ['timeout', 'read']):
            return ErrorCategory.TIMEOUT

        # Authentication errors
        if any(keyword in exception_message for keyword in ['unauthorized', 'forbidden', 'authentication', 'api key']):
            return ErrorCategory.AUTHENTICATION

        # Rate limiting
        if any(keyword in exception_message for keyword in ['rate limit', 'too many requests', '429']):
            return ErrorCategory.RATE_LIMIT

        # Server errors
        if any(keyword in exception_message for keyword in ['500', '502', '503', '504', 'server error']):
            return ErrorCategory.SERVER_ERROR

        # Client errors
        if any(keyword in exception_message for keyword in ['400', '404', 'bad request', 'not found']):
            return ErrorCategory.CLIENT_ERROR

        # Validation errors
        if any(keyword in exception_name for keyword in ['validation', 'value', 'type']):
            return ErrorCategory.VALIDATION

        return ErrorCategory.UNKNOWN

    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(exception, BrightDataException):
            return exception.severity

        severity_map = {
            ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
            ErrorCategory.SERVER_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.CLIENT_ERROR: ErrorSeverity.LOW,
            ErrorCategory.VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
        }

        return severity_map.get(category, ErrorSeverity.MEDIUM)

    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable"""
        non_recoverable_categories = {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.VALIDATION,
            ErrorCategory.CLIENT_ERROR
        }

        return category not in non_recoverable_categories

    async def handle_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle an error with recovery strategies"""
        error_info = self.categorize_error(exception, context)

        # Log the error
        self._log_error(error_info)

        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")

        # Attempt recovery if possible
        if error_info.recoverable and error_info.category in self.recovery_strategies:
            try:
                return await self.recovery_strategies[error_info.category](error_info)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")

        # Re-raise if not handled
        raise exception

    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error information"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(error_info.severity, logging.ERROR)

        self.logger.log(
            log_level,
            f"Error [{error_info.category.value}]: {error_info.exception}",
            extra={
                'error_category': error_info.category.value,
                'error_severity': error_info.severity.value,
                'error_context': error_info.context,
                'error_traceback': error_info.traceback_str,
                'retry_count': error_info.retry_count,
                'recoverable': error_info.recoverable,
            }
        )

    async def _handle_rate_limit(self, error_info: ErrorInfo) -> None:
        """Handle rate limit errors"""
        if isinstance(error_info.exception, RateLimitException) and error_info.exception.retry_after:
            wait_time = error_info.exception.retry_after
        else:
            # Default exponential backoff
            wait_time = min(2 ** error_info.retry_count, 60)

        self.logger.info(f"Rate limited, waiting {wait_time} seconds")
        await asyncio.sleep(wait_time)

    async def _handle_network_error(self, error_info: ErrorInfo) -> None:
        """Handle network errors"""
        wait_time = min(2 ** error_info.retry_count, 30)
        self.logger.info(f"Network error, waiting {wait_time} seconds before retry")
        await asyncio.sleep(wait_time)

    async def _handle_timeout(self, error_info: ErrorInfo) -> None:
        """Handle timeout errors"""
        wait_time = min(1.5 ** error_info.retry_count, 15)
        self.logger.info(f"Timeout error, waiting {wait_time} seconds before retry")
        await asyncio.sleep(wait_time)

    async def _handle_server_error(self, error_info: ErrorInfo) -> None:
        """Handle server errors"""
        wait_time = min(3 ** error_info.retry_count, 120)
        self.logger.info(f"Server error, waiting {wait_time} seconds before retry")
        await asyncio.sleep(wait_time)

    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable) -> None:
        """Register a custom recovery strategy"""
        self.recovery_strategies[category] = strategy

    def add_error_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """Add an error callback"""
        self.error_callbacks.append(callback)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0}

        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour

        category_stats = {}
        for category, count in self.error_counts.items():
            category_stats[category.value] = {
                "total": count,
                "percentage": (count / total_errors) * 100
            }

        severity_counts = defaultdict(int)
        for error in self.error_history:
            severity_counts[error.severity] += 1

        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "category_breakdown": category_stats,
            "severity_breakdown": {
                severity.value: count for severity, count in severity_counts.items()
            },
            "error_rate_per_hour": len(recent_errors),
        }

    def clear_history(self) -> None:
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()

# Import asyncio at the end to avoid circular imports
import asyncio
