"""
Core configuration management for DataMCPServerAgent.
Handles environment variables, secrets, and application settings.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    # Primary database
    database_url: str = Field(
        default="sqlite:///./data/agent.db", description="Primary database connection URL"
    )

    # Connection pool settings
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Maximum connection overflow")
    pool_timeout: int = Field(default=30, description="Connection pool timeout in seconds")

    # Query settings
    echo_sql: bool = Field(default=False, description="Echo SQL queries to logs")

    class Config:
        env_prefix = "DATABASE_"


class CloudflareSettings(BaseSettings):
    """Cloudflare integration settings."""

    # API credentials
    api_key: str = Field(default="", description="Cloudflare API key")
    account_id: str = Field(default="", description="Cloudflare account ID")
    zone_id: Optional[str] = Field(default=None, description="Cloudflare zone ID")

    # Workers settings
    workers_subdomain: str = Field(default="", description="Workers subdomain")
    workers_script_name: str = Field(default="datamcp-agent", description="Main worker script name")

    # KV settings
    kv_namespace_id: str = Field(default="", description="KV namespace ID for agent state")
    kv_preview_namespace_id: Optional[str] = Field(
        default=None, description="KV preview namespace ID"
    )

    # R2 settings
    r2_bucket_name: str = Field(default="", description="R2 bucket for file storage")
    r2_access_key_id: str = Field(default="", description="R2 access key ID")
    r2_secret_access_key: str = Field(default="", description="R2 secret access key")

    # D1 settings
    d1_database_id: str = Field(default="", description="D1 database ID")

    # Durable Objects settings
    durable_objects_namespace: str = Field(
        default="AGENT_OBJECTS", description="Durable Objects namespace"
    )

    class Config:
        env_prefix = "CLOUDFLARE_"


class EmailSettings(BaseSettings):
    """Email integration settings."""

    # Default provider
    default_provider: str = Field(default="smtp", description="Default email provider")

    # SMTP settings
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")

    # SendGrid settings
    sendgrid_api_key: str = Field(default="", description="SendGrid API key")

    # Mailgun settings
    mailgun_api_key: str = Field(default="", description="Mailgun API key")
    mailgun_domain: str = Field(default="", description="Mailgun domain")

    # Email templates
    template_directory: str = Field(
        default="./templates/email", description="Email templates directory"
    )

    # From addresses
    default_from_email: str = Field(
        default="noreply@datamcp.com", description="Default from email address"
    )
    admin_email: str = Field(default="admin@datamcp.com", description="Admin email address")

    class Config:
        env_prefix = "EMAIL_"


class WebRTCSettings(BaseSettings):
    """WebRTC integration settings."""

    # Cloudflare Calls settings
    calls_app_id: str = Field(default="", description="Cloudflare Calls app ID")
    calls_app_secret: str = Field(default="", description="Cloudflare Calls app secret")

    # STUN/TURN servers
    stun_servers: List[str] = Field(
        default=["stun:stun.cloudflare.com:3478"], description="STUN servers for WebRTC"
    )
    turn_servers: List[Dict[str, Any]] = Field(default=[], description="TURN servers configuration")

    # Voice settings
    voice_to_text_provider: str = Field(default="cloudflare", description="Voice-to-text provider")
    text_to_speech_provider: str = Field(
        default="cloudflare", description="Text-to-speech provider"
    )
    default_voice: str = Field(default="en-US-Neural2-A", description="Default TTS voice")

    # Recording settings
    enable_recording: bool = Field(default=True, description="Enable call recording by default")
    recording_storage: str = Field(default="r2", description="Recording storage backend")

    class Config:
        env_prefix = "WEBRTC_"


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""

    # JWT settings
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production", description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")

    # API keys
    api_key_length: int = Field(default=32, description="API key length")
    api_key_prefix: str = Field(default="dmcp_", description="API key prefix")

    # Password settings
    password_min_length: int = Field(default=8, description="Minimum password length")
    password_require_special: bool = Field(
        default=True, description="Require special characters in passwords"
    )

    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")

    # Tracing
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    jaeger_endpoint: str = Field(default="", description="Jaeger collector endpoint")

    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")

    # Alerting
    enable_alerts: bool = Field(default=False, description="Enable alerting")
    alert_webhook_url: str = Field(default="", description="Alert webhook URL")

    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings."""

    # Application info
    app_name: str = Field(default="DataMCPServerAgent", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    app_description: str = Field(
        default="Enhanced AI Agent System with Cloudflare Integration",
        description="Application description",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # Data directories
    data_dir: str = Field(default="./data", description="Data directory")
    temp_dir: str = Field(default="./temp", description="Temporary files directory")
    logs_dir: str = Field(default="./logs", description="Logs directory")

    # Feature flags
    enable_cloudflare: bool = Field(default=True, description="Enable Cloudflare integration")
    enable_email: bool = Field(default=True, description="Enable email integration")
    enable_webrtc: bool = Field(default=True, description="Enable WebRTC integration")
    enable_self_hosting: bool = Field(default=True, description="Enable self-hosting features")

    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    cloudflare: CloudflareSettings = CloudflareSettings()
    email: EmailSettings = EmailSettings()
    webrtc: WebRTCSettings = WebRTCSettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()

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

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields from .env
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
