"""
Improved Logging System for DataMCPServerAgent.

This module provides a comprehensive, structured logging system with:
- JSON and text formatting
- Context-aware logging
- Performance tracking
- Error tracking
- Correlation IDs
- Structured metadata
"""

import logging
import logging.config
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from app.core.config_improved import Settings

# Context variables for request tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
agent_id_var: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Global console for rich output
console = Console()


class ContextFilter(logging.Filter):
    """Add context variables to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to the log record."""
        record.correlation_id = correlation_id_var.get()
        record.user_id = user_id_var.get()
        record.agent_id = agent_id_var.get()
        record.request_id = request_id_var.get()
        return True


class PerformanceFilter(logging.Filter):
    """Add performance metrics to log records."""

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to the log record."""
        record.uptime = time.time() - self.start_time
        record.timestamp = time.time()
        return True


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to log events."""
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_user_context(logger, method_name, event_dict):
    """Add user context to log events."""
    user_id = user_id_var.get()
    agent_id = agent_id_var.get()
    request_id = request_id_var.get()

    if user_id:
        event_dict["user_id"] = user_id
    if agent_id:
        event_dict["agent_id"] = agent_id
    if request_id:
        event_dict["request_id"] = request_id

    return event_dict


def add_performance_metrics(logger, method_name, event_dict):
    """Add performance metrics to log events."""
    event_dict["timestamp"] = time.time()
    return event_dict


def setup_logging(settings: Settings) -> None:
    """Setup comprehensive logging system."""

    # Create logs directory
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_correlation_id,
        add_user_context,
        add_performance_metrics,
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.processors.ExceptionPrettyPrinter(),
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers = []

    # Console handler
    if settings.is_development:
        console_handler = RichHandler(
            console=console, show_time=True, show_path=True, markup=True, rich_tracebacks=True
        )
        console_handler.setLevel(settings.log_level.value)
        handlers.append(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(settings.log_level.value)

        if settings.log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s", '
                '"correlation_id": "%(correlation_id)s", "user_id": "%(user_id)s", '
                '"agent_id": "%(agent_id)s", "request_id": "%(request_id)s"}'
            )
        else:
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(logging.DEBUG)

        if settings.log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s", '
                '"correlation_id": "%(correlation_id)s", "user_id": "%(user_id)s", '
                '"agent_id": "%(agent_id)s", "request_id": "%(request_id)s", '
                '"uptime": %(uptime)f}'
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] [%(correlation_id)s] %(message)s"
            )

        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Add filters
    context_filter = ContextFilter()
    performance_filter = PerformanceFilter()

    for handler in handlers:
        handler.addFilter(context_filter)
        handler.addFilter(performance_filter)

    # Configure root logger
    logging.basicConfig(level=settings.log_level.value, handlers=handlers, force=True)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.database.echo_sql else logging.WARNING
    )

    # Suppress noisy loggers in production
    if settings.is_production:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from the current context."""
    return correlation_id_var.get()


def set_user_id(user_id: str) -> None:
    """Set user ID for the current context."""
    user_id_var.set(user_id)


def get_user_id() -> Optional[str]:
    """Get user ID from the current context."""
    return user_id_var.get()


def set_agent_id(agent_id: str) -> None:
    """Set agent ID for the current context."""
    agent_id_var.set(agent_id)


def get_agent_id() -> Optional[str]:
    """Get agent ID from the current context."""
    return agent_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID for the current context."""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """Get request ID from the current context."""
    return request_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


class LoggerMixin:
    """Mixin to add logging capabilities to classes."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed {self.operation}", duration_ms=round(duration * 1000, 2))
        else:
            self.logger.error(
                f"Failed {self.operation}",
                duration_ms=round(duration * 1000, 2),
                error=str(exc_val),
            )


def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        with PerformanceLogger(f"{func.__name__}", logger):
            return func(*args, **kwargs)

    return wrapper


async def log_async_function_call(func):
    """Decorator to log async function calls."""

    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        with PerformanceLogger(f"{func.__name__}", logger):
            return await func(*args, **kwargs)

    return wrapper
