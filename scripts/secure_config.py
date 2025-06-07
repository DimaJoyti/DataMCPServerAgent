"""
Secure configuration management with environment variables.
Enhanced with Pydantic validation and modern Python practices.
"""

import os
import secrets
from typing import Optional, List
from enum import Enum

from pydantic import Field, validator, SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__)


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class CloudflareConfig(BaseSettings):
    """Cloudflare configuration with validation."""
    account_id: str = Field(..., description="Cloudflare account ID")
    api_token: SecretStr = Field(..., description="Cloudflare API token")
    email: str = Field(..., description="Cloudflare account email")
    zone_id: Optional[str] = Field(None, description="Cloudflare zone ID")
    worker_subdomain: Optional[str] = Field(
        None, description="Worker subdomain"
    )
    worker_script_name: str = Field(
        "agent-worker", description="Worker script name"
    )

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v

    class Config:
        env_prefix = "CLOUDFLARE_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration with validation."""
    url: str = Field("sqlite:///./agents.db", description="Database URL")
    redis_url: str = Field("redis://localhost:6379", description="Redis URL")
    pool_size: int = Field(10, description="Database connection pool size")
    max_overflow: int = Field(20, description="Maximum connection overflow")
    pool_timeout: int = Field(30, description="Pool timeout in seconds")

    class Config:
        env_prefix = "DATABASE_"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security configuration with validation."""
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32))
    )
    jwt_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32))
    )
    encryption_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32))
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3002"],
        description="Allowed CORS origins"
    )
    rate_limit_requests: int = Field(
        100, description="Rate limit requests per window"
    )
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    jwt_expiry_minutes: int = Field(30, description="JWT token expiry in minutes")
    refresh_token_expiry_days: int = Field(
        7, description="Refresh token expiry in days"
    )
    password_min_length: int = Field(8, description="Minimum password length")

    @validator('rate_limit_requests')
    def validate_rate_limit(cls, v):
        """Validate rate limit is positive."""
        if v <= 0:
            raise ValueError('Rate limit requests must be positive')
        return v

    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class WorkflowConfig(BaseSettings):
    """Workflow configuration with validation."""
    namespace: str = Field("agent-workflows", description="Workflow namespace")
    timeout: int = Field(300000, description="Workflow timeout in milliseconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

    class Config:
        env_prefix = "WORKFLOW_"
        case_sensitive = False


class ObservabilityConfig(BaseSettings):
    """Observability configuration with validation."""
    log_level: str = Field("INFO", description="Logging level")
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    tracing_enabled: bool = Field(True, description="Enable distributed tracing")
    sentry_dsn: Optional[str] = Field(None, description="Sentry DSN for error tracking")

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    class Config:
        env_prefix = "OBSERVABILITY_"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Main application configuration."""
    environment: Environment = Field(
        Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(True, description="Enable debug mode")
    api_base_url: str = Field(
        "http://localhost:8002", description="API base URL"
    )

    class Config:
        env_prefix = "APP_"
        case_sensitive = False

def create_app_config() -> AppConfig:
    """Create application configuration with all sub-configs."""
    try:
        # Load individual configurations
        cloudflare_config = CloudflareConfig()
        database_config = DatabaseConfig()
        security_config = SecurityConfig()
        workflow_config = WorkflowConfig()
        observability_config = ObservabilityConfig()
        app_config = AppConfig()

        logger.info(
            f"Configuration loaded for environment: {app_config.environment}"
        )

        return app_config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_production_config(config: AppConfig) -> None:
    """Validate configuration for production environment."""
    errors = []

    if config.environment == Environment.PRODUCTION:
        if config.debug:
            errors.append("DEBUG should be False in production")

    if errors:
        error_msg = f"Production validation failed: {', '.join(errors)}"
        raise ValueError(error_msg)


def setup_structured_logging(config: AppConfig) -> None:
    """Setup structured logging with structlog."""
    import logging.config

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Global configuration instance
try:
    config = create_app_config()
    setup_structured_logging(config)

    # Validate production configuration
    if config.environment == Environment.PRODUCTION:
        validate_production_config(config)

    logger.info(f"Configuration loaded for environment: {config.environment}")

except Exception as e:
    print(f"Failed to load configuration: {e}")
    raise


# Export commonly used configurations
__all__ = [
    "Environment",
    "CloudflareConfig",
    "DatabaseConfig",
    "SecurityConfig",
    "WorkflowConfig",
    "ObservabilityConfig",
    "AppConfig",
    "config",
    "create_app_config",
    "validate_production_config",
    "setup_structured_logging"
]
