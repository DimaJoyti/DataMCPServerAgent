"""
Custom exceptions for the institutional trading system.
"""

from typing import Any, Dict, Optional


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class OrderValidationError(TradingSystemError):
    """Raised when order validation fails."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, validation_errors: Optional[Dict] = None):
        super().__init__(message, "ORDER_VALIDATION_ERROR")
        self.order_id = order_id
        self.validation_errors = validation_errors or {}


class RiskLimitExceededError(TradingSystemError):
    """Raised when risk limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str, current_value: float, limit_value: float):
        super().__init__(message, "RISK_LIMIT_EXCEEDED")
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class InsufficientFundsError(TradingSystemError):
    """Raised when there are insufficient funds for an order."""
    
    def __init__(self, message: str, required_amount: float, available_amount: float, currency: str):
        super().__init__(message, "INSUFFICIENT_FUNDS")
        self.required_amount = required_amount
        self.available_amount = available_amount
        self.currency = currency


class MarketDataError(TradingSystemError):
    """Raised when market data issues occur."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, exchange: Optional[str] = None):
        super().__init__(message, "MARKET_DATA_ERROR")
        self.symbol = symbol
        self.exchange = exchange


class ExecutionError(TradingSystemError):
    """Raised when order execution fails."""
    
    def __init__(self, message: str, order_id: str, execution_details: Optional[Dict] = None):
        super().__init__(message, "EXECUTION_ERROR")
        self.order_id = order_id
        self.execution_details = execution_details or {}


class StrategyError(TradingSystemError):
    """Raised when strategy execution fails."""
    
    def __init__(self, message: str, strategy_name: str, strategy_details: Optional[Dict] = None):
        super().__init__(message, "STRATEGY_ERROR")
        self.strategy_name = strategy_name
        self.strategy_details = strategy_details or {}


class ConnectivityError(TradingSystemError):
    """Raised when connectivity issues occur."""
    
    def __init__(self, message: str, endpoint: str, error_details: Optional[Dict] = None):
        super().__init__(message, "CONNECTIVITY_ERROR")
        self.endpoint = endpoint
        self.error_details = error_details or {}


class ConfigurationError(TradingSystemError):
    """Raised when configuration issues occur."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key


class PositionError(TradingSystemError):
    """Raised when position management issues occur."""
    
    def __init__(self, message: str, symbol: str, position_details: Optional[Dict] = None):
        super().__init__(message, "POSITION_ERROR")
        self.symbol = symbol
        self.position_details = position_details or {}


class SystemMaintenanceError(TradingSystemError):
    """Raised when system is under maintenance."""
    
    def __init__(self, message: str, maintenance_window: Optional[str] = None):
        super().__init__(message, "SYSTEM_MAINTENANCE")
        self.maintenance_window = maintenance_window


class RegulatoryError(TradingSystemError):
    """Raised when regulatory compliance issues occur."""
    
    def __init__(self, message: str, regulation: str, violation_details: Optional[Dict] = None):
        super().__init__(message, "REGULATORY_ERROR")
        self.regulation = regulation
        self.violation_details = violation_details or {}


class LatencyError(TradingSystemError):
    """Raised when latency thresholds are exceeded."""
    
    def __init__(self, message: str, operation: str, latency_ms: float, threshold_ms: float):
        super().__init__(message, "LATENCY_ERROR")
        self.operation = operation
        self.latency_ms = latency_ms
        self.threshold_ms = threshold_ms


class BacktestError(TradingSystemError):
    """Raised when backtesting issues occur."""
    
    def __init__(self, message: str, strategy_name: str, backtest_details: Optional[Dict] = None):
        super().__init__(message, "BACKTEST_ERROR")
        self.strategy_name = strategy_name
        self.backtest_details = backtest_details or {}


# Error severity levels
class ErrorSeverity:
    """Error severity levels for monitoring and alerting."""
    
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Error categories for classification
class ErrorCategory:
    """Error categories for classification and routing."""
    
    TECHNICAL = "TECHNICAL"
    BUSINESS = "BUSINESS"
    OPERATIONAL = "OPERATIONAL"
    REGULATORY = "REGULATORY"
    PERFORMANCE = "PERFORMANCE"


# Error handling utilities
def classify_error(error: Exception) -> Dict[str, str]:
    """Classify an error by type, severity, and category."""
    
    error_mapping = {
        OrderValidationError: {"severity": ErrorSeverity.MEDIUM, "category": ErrorCategory.BUSINESS},
        RiskLimitExceededError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.BUSINESS},
        InsufficientFundsError: {"severity": ErrorSeverity.MEDIUM, "category": ErrorCategory.BUSINESS},
        MarketDataError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.TECHNICAL},
        ExecutionError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.OPERATIONAL},
        StrategyError: {"severity": ErrorSeverity.MEDIUM, "category": ErrorCategory.BUSINESS},
        ConnectivityError: {"severity": ErrorSeverity.CRITICAL, "category": ErrorCategory.TECHNICAL},
        ConfigurationError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.TECHNICAL},
        PositionError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.BUSINESS},
        SystemMaintenanceError: {"severity": ErrorSeverity.LOW, "category": ErrorCategory.OPERATIONAL},
        RegulatoryError: {"severity": ErrorSeverity.CRITICAL, "category": ErrorCategory.REGULATORY},
        LatencyError: {"severity": ErrorSeverity.HIGH, "category": ErrorCategory.PERFORMANCE},
        BacktestError: {"severity": ErrorSeverity.LOW, "category": ErrorCategory.BUSINESS},
    }
    
    error_type = type(error)
    classification = error_mapping.get(error_type, {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TECHNICAL
    })
    
    return {
        "error_type": error_type.__name__,
        "severity": classification["severity"],
        "category": classification["category"],
        "message": str(error)
    }
