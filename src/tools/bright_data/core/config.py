"""
Configuration management for Enhanced Bright Data MCP Integration

This module provides centralized configuration management with support for:
- Environment variables
- Configuration files
- Runtime configuration updates
- Validation and defaults
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    memory_cache_size: int = 1000
    default_ttl: int = 3600  # 1 hour
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10
    adaptive_throttling: bool = True
    backoff_factor: float = 1.5

@dataclass
class RetryConfig:
    """Retry configuration settings"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: tuple = (Exception,)

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_endpoint: str = "/health"
    log_level: str = "INFO"

@dataclass
class BrightDataConfig:
    """Main configuration class for Bright Data integration"""

    # API Configuration
    api_base_url: str = "https://api.brightdata.com"
    api_timeout: int = 30
    max_concurrent_requests: int = 10

    # Authentication
    api_key: Optional[str] = None
    user_agent: str = "DataMCPServerAgent/2.0.0"

    # Component configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Feature flags
    enable_compression: bool = True
    enable_connection_pooling: bool = True
    enable_request_logging: bool = True

    @classmethod
    def from_env(cls) -> 'BrightDataConfig':
        """Create configuration from environment variables"""
        config = cls()

        # API Configuration
        config.api_key = os.getenv('BRIGHT_DATA_API_KEY')
        config.api_base_url = os.getenv('BRIGHT_DATA_API_URL', config.api_base_url)
        config.api_timeout = int(os.getenv('BRIGHT_DATA_TIMEOUT', config.api_timeout))

        # Cache Configuration
        config.cache.redis_url = os.getenv('REDIS_URL', config.cache.redis_url)
        config.cache.enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        config.cache.default_ttl = int(os.getenv('CACHE_TTL', config.cache.default_ttl))

        # Rate Limiting
        config.rate_limit.requests_per_minute = int(
            os.getenv('RATE_LIMIT_RPM', config.rate_limit.requests_per_minute)
        )
        config.rate_limit.enabled = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'

        # Retry Configuration
        config.retry.max_retries = int(os.getenv('MAX_RETRIES', config.retry.max_retries))
        config.retry.base_delay = float(os.getenv('RETRY_BASE_DELAY', config.retry.base_delay))

        # Monitoring
        config.monitoring.log_level = os.getenv('LOG_LEVEL', config.monitoring.log_level)

        return config

    @classmethod
    def from_file(cls, config_path: str) -> 'BrightDataConfig':
        """Load configuration from JSON file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrightDataConfig':
        """Create configuration from dictionary"""
        config = cls()

        # Update main config
        for key, value in data.items():
            if hasattr(config, key) and not isinstance(getattr(config, key), (CacheConfig, RateLimitConfig, RetryConfig, CircuitBreakerConfig, MonitoringConfig)):
                setattr(config, key, value)

        # Update nested configs
        if 'cache' in data:
            for key, value in data['cache'].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)

        if 'rate_limit' in data:
            for key, value in data['rate_limit'].items():
                if hasattr(config.rate_limit, key):
                    setattr(config.rate_limit, key, value)

        if 'retry' in data:
            for key, value in data['retry'].items():
                if hasattr(config.retry, key):
                    setattr(config.retry, key, value)

        if 'circuit_breaker' in data:
            for key, value in data['circuit_breaker'].items():
                if hasattr(config.circuit_breaker, key):
                    setattr(config.circuit_breaker, key, value)

        if 'monitoring' in data:
            for key, value in data['monitoring'].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'api_base_url': self.api_base_url,
            'api_timeout': self.api_timeout,
            'max_concurrent_requests': self.max_concurrent_requests,
            'user_agent': self.user_agent,
            'enable_compression': self.enable_compression,
            'enable_connection_pooling': self.enable_connection_pooling,
            'enable_request_logging': self.enable_request_logging,
            'cache': {
                'enabled': self.cache.enabled,
                'redis_url': self.cache.redis_url,
                'memory_cache_size': self.cache.memory_cache_size,
                'default_ttl': self.cache.default_ttl,
                'compression_enabled': self.cache.compression_enabled,
                'compression_threshold': self.cache.compression_threshold,
            },
            'rate_limit': {
                'enabled': self.rate_limit.enabled,
                'requests_per_minute': self.rate_limit.requests_per_minute,
                'burst_size': self.rate_limit.burst_size,
                'adaptive_throttling': self.rate_limit.adaptive_throttling,
                'backoff_factor': self.rate_limit.backoff_factor,
            },
            'retry': {
                'max_retries': self.retry.max_retries,
                'base_delay': self.retry.base_delay,
                'max_delay': self.retry.max_delay,
                'exponential_base': self.retry.exponential_base,
                'jitter': self.retry.jitter,
            },
            'circuit_breaker': {
                'enabled': self.circuit_breaker.enabled,
                'failure_threshold': self.circuit_breaker.failure_threshold,
                'recovery_timeout': self.circuit_breaker.recovery_timeout,
            },
            'monitoring': {
                'enabled': self.monitoring.enabled,
                'metrics_endpoint': self.monitoring.metrics_endpoint,
                'health_check_endpoint': self.monitoring.health_check_endpoint,
                'log_level': self.monitoring.log_level,
            }
        }

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> None:
        """Validate configuration settings"""
        if not self.api_key:
            raise ValueError("API key is required")

        if self.api_timeout <= 0:
            raise ValueError("API timeout must be positive")

        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive")

        if self.cache.default_ttl <= 0:
            raise ValueError("Cache TTL must be positive")

        if self.rate_limit.requests_per_minute <= 0:
            raise ValueError("Rate limit requests per minute must be positive")

        if self.retry.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

# Global configuration instance
_config: Optional[BrightDataConfig] = None

def get_config() -> BrightDataConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = BrightDataConfig.from_env()
    return _config

def set_config(config: BrightDataConfig) -> None:
    """Set the global configuration instance"""
    global _config
    config.validate()
    _config = config

def reset_config() -> None:
    """Reset the global configuration instance"""
    global _config
    _config = None
