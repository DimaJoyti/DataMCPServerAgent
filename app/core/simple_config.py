"""
Simple Configuration for DataMCPServerAgent.

Simplified configuration without complex dependencies for initial testing.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SimpleSettings(BaseSettings):
    """Simple application settings without complex dependencies."""

    # Application metadata
    app_name: str = Field(default="DataMCPServerAgent", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    app_description: str = Field(
        default="Advanced AI Agent System with MCP Integration",
        description="Application description",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8003, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_format: str = Field(default="text", description="Log format (json/text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # Directories
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")

    # Feature flags (simplified)
    enable_api: bool = Field(default=True, description="Enable API server")
    enable_cli: bool = Field(default=True, description="Enable CLI interface")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING


# Global settings instance - create when needed to avoid import-time issues
def get_simple_settings() -> SimpleSettings:
    """Get global simple settings instance."""
    if not hasattr(get_simple_settings, "_instance"):
        get_simple_settings._instance = SimpleSettings()
    return get_simple_settings._instance
