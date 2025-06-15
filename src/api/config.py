"""
Configuration for the API module.
Uses unified configuration system from app.core.config for consistency.
"""

import warnings
from typing import List

from pydantic import BaseModel, Field

# Import unified configuration
try:
    from app.core.config import get_settings

    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
    warnings.warn(
        "Unified configuration not available. Using legacy configuration.",
        DeprecationWarning,
        stacklevel=2
    )


class APIConfig(BaseModel):
    """Configuration for the API module (legacy, deprecated)."""

    # API settings
    title: str = Field(default="DataMCPServerAgent API", description="API title")
    description: str = Field(
        default="API for interacting with DataMCPServerAgent",
        description="API description"
    )
    version: str = Field(default="0.1.0", description="API version")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI URL")
    docs_url: str = Field(default="/docs", description="Docs URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc URL")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload")

    # Security settings (DEPRECATED - use unified config)
    enable_auth: bool = Field(default=False, description="Enable authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header")
    api_keys: List[str] = Field(default=[], description="Valid API keys")

    # CORS settings (DEPRECATED - use unified config)
    allow_origins: List[str] = Field(
        default=["http://localhost:3002", "http://localhost:3000"],
        description="CORS allowed origins (DEPRECATED)"
    )
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods (DEPRECATED)"
    )
    allow_headers: List[str] = Field(
        default=["Content-Type", "Authorization", "X-API-Key"],
        description="CORS allowed headers (DEPRECATED)"
    )

    # Rate limiting
    enable_rate_limiting: bool = False
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"

    # Agent settings
    default_agent_mode: str = "basic"
    available_agent_modes: List[str] = [
        "basic",
        "advanced",
        "enhanced",
        "advanced_enhanced",
        "multi_agent",
        "reinforcement_learning",
        "distributed_memory",
        "knowledge_graph",
        "error_recovery",
        "research_reports",
        "seo",
    ]

    # Memory settings
    memory_backend: str = "sqlite"  # sqlite, file, redis, mongodb

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "datamcp:"

    # MongoDB settings
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "datamcp"

    # Distributed settings
    enable_distributed: bool = False
    distributed_backend: str = "redis"  # redis, mongodb
    session_store: str = "redis"  # redis, mongodb, memory

    # Tool settings
    enable_all_tools: bool = True

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create a config from environment variables."""
        config_dict = {}

        # API settings
        if os.getenv("API_TITLE"):
            config_dict["title"] = os.getenv("API_TITLE")
        if os.getenv("API_DESCRIPTION"):
            config_dict["description"] = os.getenv("API_DESCRIPTION")
        if os.getenv("API_VERSION"):
            config_dict["version"] = os.getenv("API_VERSION")
        if os.getenv("API_OPENAPI_URL"):
            config_dict["openapi_url"] = os.getenv("API_OPENAPI_URL")
        if os.getenv("API_DOCS_URL"):
            config_dict["docs_url"] = os.getenv("API_DOCS_URL")
        if os.getenv("API_REDOC_URL"):
            config_dict["redoc_url"] = os.getenv("API_REDOC_URL")

        # Server settings
        if os.getenv("API_HOST"):
            config_dict["host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config_dict["port"] = int(os.getenv("API_PORT"))
        if os.getenv("API_DEBUG"):
            config_dict["debug"] = os.getenv("API_DEBUG").lower() == "true"
        if os.getenv("API_RELOAD"):
            config_dict["reload"] = os.getenv("API_RELOAD").lower() == "true"

        # Security settings
        if os.getenv("API_ENABLE_AUTH"):
            config_dict["enable_auth"] = os.getenv("API_ENABLE_AUTH").lower() == "true"
        if os.getenv("API_KEY_HEADER"):
            config_dict["api_key_header"] = os.getenv("API_KEY_HEADER")
        if os.getenv("API_KEYS"):
            config_dict["api_keys"] = os.getenv("API_KEYS").split(",")

        # CORS settings
        if os.getenv("API_ALLOW_ORIGINS"):
            config_dict["allow_origins"] = os.getenv("API_ALLOW_ORIGINS").split(",")
        if os.getenv("API_ALLOW_METHODS"):
            config_dict["allow_methods"] = os.getenv("API_ALLOW_METHODS").split(",")
        if os.getenv("API_ALLOW_HEADERS"):
            config_dict["allow_headers"] = os.getenv("API_ALLOW_HEADERS").split(",")

        # Rate limiting
        if os.getenv("API_ENABLE_RATE_LIMITING"):
            config_dict["enable_rate_limiting"] = (
                os.getenv("API_ENABLE_RATE_LIMITING").lower() == "true"
            )
        if os.getenv("API_RATE_LIMIT_PER_MINUTE"):
            config_dict["rate_limit_per_minute"] = int(os.getenv("API_RATE_LIMIT_PER_MINUTE"))

        # Logging
        if os.getenv("API_LOG_LEVEL"):
            config_dict["log_level"] = os.getenv("API_LOG_LEVEL")

        # Agent settings
        if os.getenv("API_DEFAULT_AGENT_MODE"):
            config_dict["default_agent_mode"] = os.getenv("API_DEFAULT_AGENT_MODE")

        # Memory settings
        if os.getenv("API_MEMORY_BACKEND"):
            config_dict["memory_backend"] = os.getenv("API_MEMORY_BACKEND")

        # Redis settings
        if os.getenv("API_REDIS_HOST"):
            config_dict["redis_host"] = os.getenv("API_REDIS_HOST")
        if os.getenv("API_REDIS_PORT"):
            config_dict["redis_port"] = int(os.getenv("API_REDIS_PORT"))
        if os.getenv("API_REDIS_DB"):
            config_dict["redis_db"] = int(os.getenv("API_REDIS_DB"))
        if os.getenv("API_REDIS_PASSWORD"):
            config_dict["redis_password"] = os.getenv("API_REDIS_PASSWORD")
        if os.getenv("API_REDIS_PREFIX"):
            config_dict["redis_prefix"] = os.getenv("API_REDIS_PREFIX")

        # MongoDB settings
        if os.getenv("API_MONGODB_URI"):
            config_dict["mongodb_uri"] = os.getenv("API_MONGODB_URI")
        if os.getenv("API_MONGODB_DB"):
            config_dict["mongodb_db"] = os.getenv("API_MONGODB_DB")

        # Distributed settings
        if os.getenv("API_ENABLE_DISTRIBUTED"):
            config_dict["enable_distributed"] = (
                os.getenv("API_ENABLE_DISTRIBUTED").lower() == "true"
            )
        if os.getenv("API_DISTRIBUTED_BACKEND"):
            config_dict["distributed_backend"] = os.getenv("API_DISTRIBUTED_BACKEND")
        if os.getenv("API_SESSION_STORE"):
            config_dict["session_store"] = os.getenv("API_SESSION_STORE")

        # Tool settings
        if os.getenv("API_ENABLE_ALL_TOOLS"):
            config_dict["enable_all_tools"] = os.getenv("API_ENABLE_ALL_TOOLS").lower() == "true"

        return cls(**config_dict)


# Unified configuration adapter
def get_api_config() -> APIConfig:
    """Get API configuration with unified settings when available."""
    if UNIFIED_CONFIG_AVAILABLE:
        try:
            settings = get_settings()
            # Create adapter using unified configuration
            return APIConfig(
                title=f"{settings.app_name} API",
                description=settings.app_description,
                version=settings.app_version,
                host=settings.api_host,
                port=settings.api_port,
                debug=settings.debug,
                allow_origins=settings.security.cors_origins,
                allow_methods=settings.security.cors_methods,
                allow_headers=settings.security.cors_headers,
                api_key_header=settings.security.api_key_header,
                rate_limit_per_minute=settings.security.rate_limit_per_minute,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to load unified configuration: {e}. Using legacy configuration.",
                RuntimeWarning,
                stacklevel=2
            )

    # Fallback to legacy configuration
    return APIConfig.from_env()


# Create a global config instance
config = get_api_config()
