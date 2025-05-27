"""
Improved Application Configuration for DataMCPServerAgent.

This module provides a comprehensive, type-safe configuration system using Pydantic Settings.
It supports multiple environments, validation, and hierarchical configuration loading.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
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


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    # Connection settings
    url: str = Field(
        default="sqlite+aiosqlite:///./datamcp.db", description="Database connection URL"
    )
    echo_sql: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")

    # Migration settings
    auto_migrate: bool = Field(default=True, description="Auto-run migrations")
    migration_timeout: int = Field(default=300, description="Migration timeout")

    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class CacheConfig(BaseSettings):
    """Cache configuration."""

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_timeout: int = Field(default=5, description="Redis timeout")

    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    max_connections: int = Field(default=10, description="Max Redis connections")

    model_config = SettingsConfigDict(env_prefix="CACHE_")


class SecurityConfig(BaseSettings):
    """Security configuration."""

    # JWT settings
    secret_key: str = Field(description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time")

    # API security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")

    # Security headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class CloudflareConfig(BaseSettings):
    """Cloudflare integration configuration."""

    # API settings
    api_token: str = Field(default="", description="Cloudflare API token")
    account_id: str = Field(default="", description="Cloudflare account ID")
    zone_id: str = Field(default="", description="Cloudflare zone ID")

    # Workers settings
    workers_subdomain: str = Field(default="", description="Workers subdomain")
    workers_script_name: str = Field(default="datamcp-agent", description="Workers script name")

    # KV settings
    kv_namespace_id: str = Field(default="", description="KV namespace ID")
    kv_preview_namespace_id: str = Field(default="", description="KV preview namespace ID")

    # R2 settings
    r2_bucket_name: str = Field(default="", description="R2 bucket name")
    r2_access_key_id: str = Field(default="", description="R2 access key ID")
    r2_secret_access_key: str = Field(default="", description="R2 secret access key")

    # D1 settings
    d1_database_id: str = Field(default="", description="D1 database ID")

    # Durable Objects settings
    durable_objects_namespace: str = Field(
        default="AGENT_OBJECTS", description="Durable Objects namespace"
    )

    model_config = SettingsConfigDict(env_prefix="CLOUDFLARE_")


class EmailConfig(BaseSettings):
    """Email configuration."""

    # SMTP settings
    smtp_host: str = Field(default="localhost", description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")

    # SendGrid settings
    sendgrid_api_key: str = Field(default="", description="SendGrid API key")

    # Mailgun settings
    mailgun_api_key: str = Field(default="", description="Mailgun API key")
    mailgun_domain: str = Field(default="", description="Mailgun domain")

    # Email settings
    default_from_email: str = Field(default="noreply@datamcp.com", description="Default from email")
    admin_email: str = Field(default="admin@datamcp.com", description="Admin email")
    template_directory: str = Field(
        default="./templates/email", description="Email templates directory"
    )

    model_config = SettingsConfigDict(env_prefix="EMAIL_")


class WebRTCConfig(BaseSettings):
    """WebRTC configuration."""

    # STUN/TURN servers
    stun_servers: List[str] = Field(
        default=["stun:stun.l.google.com:19302"], description="STUN servers"
    )
    turn_servers: List[Dict[str, Any]] = Field(
        default_factory=list, description="TURN servers configuration"
    )

    # Cloudflare Calls settings
    calls_app_id: str = Field(default="", description="Cloudflare Calls app ID")
    calls_app_secret: str = Field(default="", description="Cloudflare Calls app secret")

    # Recording settings
    enable_recording: bool = Field(default=False, description="Enable call recording")
    recording_bucket: str = Field(default="", description="Recording storage bucket")

    model_config = SettingsConfigDict(env_prefix="WEBRTC_")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")

    # Tracing
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    jaeger_endpoint: str = Field(default="", description="Jaeger endpoint")

    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval")
    health_check_timeout: int = Field(default=10, description="Health check timeout")

    # Alerting
    enable_alerting: bool = Field(default=False, description="Enable alerting")
    webhook_url: str = Field(default="", description="Alert webhook URL")

    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class Settings(BaseSettings):
    """Main application settings."""

    # Application metadata
    app_name: str = Field(default="DataMCPServerAgent", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    app_description: str = Field(
        default="Advanced AI Agent System with MCP Integration",
        description="Application description",
    )

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8002, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # Directories
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")

    # Feature flags
    enable_cloudflare: bool = Field(default=True, description="Enable Cloudflare integration")
    enable_email: bool = Field(default=True, description="Enable email integration")
    enable_webrtc: bool = Field(default=True, description="Enable WebRTC integration")
    enable_self_hosting: bool = Field(default=True, description="Enable self-hosting features")

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cloudflare: CloudflareConfig = Field(default_factory=CloudflareConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    webrtc: WebRTCConfig = Field(default_factory=WebRTCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @model_validator(mode="after")
    def set_debug_from_env(self):
        """Set debug mode based on environment."""
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
        return self

    @field_validator("data_dir", "temp_dir", "logs_dir", mode="before")
    @classmethod
    def validate_directories(cls, v):
        """Validate and create directories."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

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


# Global settings instance
settings = Settings()
