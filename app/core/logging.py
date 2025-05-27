"""
Centralized logging configuration for DataMCPServerAgent.
Provides structured logging with correlation IDs, metrics, and different output formats.
"""

import json
import logging
import logging.config
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings

# Context variables for request correlation
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
agent_id: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID and context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to log record."""
        record.correlation_id = correlation_id.get()
        record.user_id = user_id.get()
        record.agent_id = agent_id.get()
        record.timestamp = datetime.now(timezone.utc).isoformat()
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": getattr(record, "timestamp", datetime.now(timezone.utc).isoformat()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if available
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id

        if hasattr(record, "user_id") and record.user_id:
            log_entry["user_id"] = record.user_id

        if hasattr(record, "agent_id") and record.agent_id:
            log_entry["agent_id"] = record.agent_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "timestamp",
                "correlation_id",
                "user_id",
                "agent_id",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, "")
        reset_color = self.COLORS["RESET"]

        # Create colored level name
        colored_level = f"{level_color}{record.levelname}{reset_color}"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Build log message
        parts = [f"[{timestamp}]", f"[{colored_level}]", f"[{record.name}]"]

        # Add context if available
        if hasattr(record, "correlation_id") and record.correlation_id:
            parts.append(f"[{record.correlation_id[:8]}]")

        if hasattr(record, "agent_id") and record.agent_id:
            parts.append(f"[{record.agent_id}]")

        # Add message
        parts.append(record.getMessage())

        log_line = " ".join(parts)

        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def setup_logging() -> None:
    """Setup logging configuration."""

    # Create directories if they don't exist
    if settings.logs_dir:
        Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)

    # Base logging config
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "colored": {
                "()": ColoredFormatter,
            },
            "standard": {
                "format": settings.log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "filters": {
            "correlation": {
                "()": CorrelationFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level.value,
                "formatter": "colored" if sys.stdout.isatty() else "json",
                "filters": ["correlation"],
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "app": {
                "level": settings.log_level.value,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.log_level.value,
            "handlers": ["console"],
        },
    }

    # Add file handler if log file is specified
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level.value,
            "formatter": "json",
            "filters": ["correlation"],
            "filename": str(log_file_path),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }

        # Add file handler to loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
        config["root"]["handlers"].append("file")

    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"app.{name}")


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(cid)


def set_user_id(uid: str) -> None:
    """Set user ID for current context."""
    user_id.set(uid)


def set_agent_id(aid: str) -> None:
    """Set agent ID for current context."""
    agent_id.set(aid)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    return user_id.get()


def get_agent_id() -> Optional[str]:
    """Get current agent ID."""
    return agent_id.get()


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {e}")
            raise

    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls."""

    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async function {func.__name__} failed with error: {e}")
            raise

    return wrapper


# Initialize logging on module import
setup_logging()
