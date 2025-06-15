"""
Improved Exception System for DataMCPServerAgent.

This module provides a comprehensive exception hierarchy with:
- Structured error information
- Error codes and categories
- Contextual error details
- Recovery suggestions
- Logging integration
"""

import traceback
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(str, Enum):
    """Error category enumeration."""

    VALIDATION = "validation"
    BUSINESS_RULE = "business_rule"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    INFRASTRUCTURE = "infrastructure"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseError(Exception):
    """Base exception class for all application errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
        self.context = context or {}
        self.cause = cause
        self.error_id = str(uuid.uuid4())
        self.traceback_str = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": (
                self.traceback_str
                if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
                else None
            ),
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(error_code='{self.error_code}', message='{self.message}')"
        )


class ValidationError(BaseError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if expected_type:
            details["expected_type"] = expected_type

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check the input format and try again",
                "Refer to the API documentation for valid input formats",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "VALIDATION_ERROR"),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class BusinessRuleError(BaseError):
    """Raised when business rules are violated."""

    def __init__(self, message: str, rule_name: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if rule_name:
            details["rule_name"] = rule_name

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Review the business rules and constraints",
                "Contact support if you believe this is an error",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "BUSINESS_RULE_ERROR"),
            category=ErrorCategory.BUSINESS_RULE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class AuthenticationError(BaseError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check your credentials and try again",
                "Ensure your API key is valid and not expired",
                "Contact support if the problem persists",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "AUTHENTICATION_ERROR"),
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions,
            **kwargs,
        )


class AuthorizationError(BaseError):
    """Raised when authorization fails."""

    def __init__(
        self, message: str = "Access denied", required_permission: Optional[str] = None, **kwargs
    ):
        details = kwargs.get("details", {})
        if required_permission:
            details["required_permission"] = required_permission

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Contact an administrator to request access",
                "Verify you have the necessary permissions",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "AUTHORIZATION_ERROR"),
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class EntityNotFoundError(BaseError):
    """Raised when a requested entity is not found."""

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if entity_type:
            details["entity_type"] = entity_type
        if entity_id:
            details["entity_id"] = entity_id

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Verify the ID is correct",
                "Check if the entity was deleted",
                "Ensure you have access to this entity",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "ENTITY_NOT_FOUND"),
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.LOW,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class ConflictError(BaseError):
    """Raised when there's a conflict with the current state."""

    def __init__(self, message: str, conflicting_entity: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if conflicting_entity:
            details["conflicting_entity"] = conflicting_entity

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Refresh the data and try again",
                "Resolve the conflict manually",
                "Contact support if the issue persists",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "CONFLICT_ERROR"),
            category=ErrorCategory.CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class RateLimitError(BaseError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs
    ):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                (
                    f"Wait {retry_after} seconds before retrying"
                    if retry_after
                    else "Wait before retrying"
                ),
                "Reduce the frequency of requests",
                "Consider upgrading your plan for higher limits",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "RATE_LIMIT_ERROR"),
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class ExternalServiceError(BaseError):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Try again later",
                "Check the service status",
                "Contact support if the problem persists",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "EXTERNAL_SERVICE_ERROR"),
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class InfrastructureError(BaseError):
    """Raised when infrastructure components fail."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if component:
            details["component"] = component

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check system health",
                "Restart the affected component",
                "Contact system administrator",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "INFRASTRUCTURE_ERROR"),
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class ConcurrencyError(BaseError):
    """Raised when concurrency conflicts occur."""

    def __init__(self, message: str = "Concurrency conflict detected", **kwargs):
        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Retry the operation",
                "Refresh the data and try again",
                "Use optimistic locking",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "CONCURRENCY_ERROR"),
            category=ErrorCategory.CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions,
            **kwargs,
        )


class ConfigurationError(BaseError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check the configuration file",
                "Verify environment variables",
                "Refer to the configuration documentation",
            ]

        super().__init__(
            message=message,
            error_code=kwargs.get("error_code", "CONFIGURATION_ERROR"),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


def handle_exception(exc: Exception) -> BaseError:
    """Convert any exception to a BaseError."""
    if isinstance(exc, BaseError):
        return exc

    # Map common exceptions
    if isinstance(exc, ValueError):
        return ValidationError(message=str(exc), error_code="VALUE_ERROR", cause=exc)
    elif isinstance(exc, KeyError):
        return EntityNotFoundError(
            message=f"Key not found: {exc}", error_code="KEY_ERROR", cause=exc
        )
    elif isinstance(exc, PermissionError):
        return AuthorizationError(message=str(exc), error_code="PERMISSION_ERROR", cause=exc)
    else:
        return BaseError(
            message=str(exc),
            error_code="UNKNOWN_ERROR",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
            cause=exc,
        )
