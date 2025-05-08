"""
Environment configuration module for DataMCPServerAgent.
This module provides centralized access to environment variables.
"""

import os
from typing import Dict, Optional, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env(key: str, default: Optional[Any] = None) -> Any:
    """Get an environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if the environment variable is not set
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_mcp_server_params() -> Dict[str, Any]:
    """Get MCP server parameters from environment variables.
    
    Returns:
        Dictionary of MCP server parameters
    """
    return {
        "command": "npx",
        "env": {
            "API_TOKEN": get_env("API_TOKEN"),
            "BROWSER_AUTH": get_env("BROWSER_AUTH"),
            "WEB_UNLOCKER_ZONE": get_env("WEB_UNLOCKER_ZONE"),
        },
        "args": ["@brightdata/mcp"],
    }


def get_model_config() -> Dict[str, str]:
    """Get model configuration from environment variables.
    
    Returns:
        Dictionary of model configuration
    """
    return {
        "model_name": get_env("MODEL_NAME", "claude-3-5-sonnet-20240620"),
        "model_provider": get_env("MODEL_PROVIDER", "anthropic"),
    }


def get_memory_config() -> Dict[str, Any]:
    """Get memory configuration from environment variables.
    
    Returns:
        Dictionary of memory configuration
    """
    memory_type = get_env("MEMORY_TYPE", "sqlite")
    
    config = {
        "memory_type": memory_type,
        "db_path": get_env("MEMORY_DB_PATH", "agent_memory.db"),
    }
    
    # Add Redis configuration if using Redis
    if memory_type == "redis":
        config["redis"] = {
            "host": get_env("REDIS_HOST", "localhost"),
            "port": int(get_env("REDIS_PORT", "6379")),
            "db": int(get_env("REDIS_DB", "0")),
            "password": get_env("REDIS_PASSWORD", ""),
        }
    
    # Add MongoDB configuration if using MongoDB
    if memory_type == "mongodb":
        config["mongodb"] = {
            "uri": get_env("MONGODB_URI", "mongodb://localhost:27017"),
            "db": get_env("MONGODB_DB", "agent_memory"),
        }
    
    return config


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration from environment variables.
    
    Returns:
        Dictionary of logging configuration
    """
    return {
        "level": get_env("LOG_LEVEL", "INFO"),
    }