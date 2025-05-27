"""
Secure configuration management with environment variables.
"""

import os
import secrets
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class CloudflareConfig:
    """Cloudflare configuration."""
    account_id: str
    api_token: str
    email: str
    zone_id: Optional[str] = None
    worker_subdomain: Optional[str] = None
    worker_script_name: str = "agent-worker"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    redis_url: str

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    jwt_secret_key: str
    encryption_key: str
    allowed_origins: List[str]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    namespace: str
    timeout: int = 300000  # 5 minutes
    retry_attempts: int = 3

@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    log_level: str = "INFO"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    sentry_dsn: Optional[str] = None

@dataclass
class AppConfig:
    """Application configuration."""
    environment: str
    debug: bool
    api_base_url: str
    cloudflare: CloudflareConfig
    database: DatabaseConfig
    security: SecurityConfig
    workflow: WorkflowConfig
    observability: ObservabilityConfig

def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get environment variable with validation."""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_list(key: str, default: List[str] = None, separator: str = ",") -> List[str]:
    """Get list environment variable."""
    value = os.getenv(key)
    if not value:
        return default or []
    return [item.strip() for item in value.split(separator)]

def generate_secure_key() -> str:
    """Generate a secure random key."""
    return secrets.token_urlsafe(32)

def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    
    # Generate secure keys if not provided
    secret_key = get_env_var("SECRET_KEY", generate_secure_key(), required=False)
    jwt_secret_key = get_env_var("JWT_SECRET_KEY", generate_secure_key(), required=False)
    encryption_key = get_env_var("ENCRYPTION_KEY", generate_secure_key(), required=False)
    
    # Cloudflare configuration
    cloudflare = CloudflareConfig(
        account_id=get_env_var("CLOUDFLARE_ACCOUNT_ID"),
        api_token=get_env_var("CLOUDFLARE_API_TOKEN"),
        email=get_env_var("CLOUDFLARE_EMAIL"),
        zone_id=get_env_var("CLOUDFLARE_ZONE_ID", required=False),
        worker_subdomain=get_env_var("WORKER_SUBDOMAIN", required=False),
        worker_script_name=get_env_var("WORKER_SCRIPT_NAME", "agent-worker", required=False)
    )
    
    # Database configuration
    database = DatabaseConfig(
        url=get_env_var("DATABASE_URL", "sqlite:///./agents.db", required=False),
        redis_url=get_env_var("REDIS_URL", "redis://localhost:6379", required=False)
    )
    
    # Security configuration
    security = SecurityConfig(
        secret_key=secret_key,
        jwt_secret_key=jwt_secret_key,
        encryption_key=encryption_key,
        allowed_origins=get_env_list("ALLOWED_ORIGINS", ["http://localhost:3000", "http://localhost:3002"]),
        rate_limit_requests=get_env_int("RATE_LIMIT_REQUESTS", 100),
        rate_limit_window=get_env_int("RATE_LIMIT_WINDOW", 60)
    )
    
    # Workflow configuration
    workflow = WorkflowConfig(
        namespace=get_env_var("WORKFLOW_NAMESPACE", "agent-workflows", required=False),
        timeout=get_env_int("WORKFLOW_TIMEOUT", 300000),
        retry_attempts=get_env_int("WORKFLOW_RETRY_ATTEMPTS", 3)
    )
    
    # Observability configuration
    observability = ObservabilityConfig(
        log_level=get_env_var("LOG_LEVEL", "INFO", required=False),
        metrics_enabled=get_env_bool("METRICS_ENABLED", True),
        tracing_enabled=get_env_bool("TRACING_ENABLED", True),
        sentry_dsn=get_env_var("SENTRY_DSN", required=False)
    )
    
    return AppConfig(
        environment=get_env_var("ENVIRONMENT", "development", required=False),
        debug=get_env_bool("DEBUG", True),
        api_base_url=get_env_var("API_BASE_URL", "http://localhost:8002", required=False),
        cloudflare=cloudflare,
        database=database,
        security=security,
        workflow=workflow,
        observability=observability
    )

def setup_logging(config: AppConfig):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent_system.log')
        ]
    )
    
    # Disable debug logging for third-party libraries in production
    if not config.debug:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

def validate_config(config: AppConfig):
    """Validate configuration."""
    errors = []
    
    # Validate Cloudflare configuration
    if not config.cloudflare.account_id:
        errors.append("CLOUDFLARE_ACCOUNT_ID is required")
    
    if not config.cloudflare.api_token or config.cloudflare.api_token == "your_cloudflare_api_token_here":
        errors.append("CLOUDFLARE_API_TOKEN must be set to a valid token")
    
    # Validate security configuration
    if config.security.secret_key == "your_super_secret_key_here_change_in_production":
        errors.append("SECRET_KEY must be changed from default value")
    
    # Validate production settings
    if config.environment == "production":
        if config.debug:
            errors.append("DEBUG should be False in production")
        
        if not config.observability.sentry_dsn:
            errors.append("SENTRY_DSN should be set in production")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

# Global configuration instance
try:
    config = load_config()
    setup_logging(config)
    
    # Only validate in production or if explicitly requested
    if config.environment == "production" or get_env_bool("VALIDATE_CONFIG", False):
        validate_config(config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded for environment: {config.environment}")
    
except Exception as e:
    print(f"Failed to load configuration: {e}")
    raise
