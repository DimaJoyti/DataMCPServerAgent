"""
Configuration for the API module.
"""

import os
from typing import List, Optional

from pydantic import BaseModel


class APIConfig(BaseModel):
    """Configuration for the API."""

    # API settings
    title: str = "DataMCPServerAgent API"
    description: str = "API for interacting with DataMCPServerAgent"
    version: str = "0.1.0"
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False

    # Security settings
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = []

    # CORS settings
    allow_origins: List[str] = ["*"]
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]

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
            config_dict["rate_limit_per_minute"] = int(
                os.getenv("API_RATE_LIMIT_PER_MINUTE")
            )

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
            config_dict["enable_all_tools"] = (
                os.getenv("API_ENABLE_ALL_TOOLS").lower() == "true"
            )

        return cls(**config_dict)


# Create a global config instance
config = APIConfig.from_env()
