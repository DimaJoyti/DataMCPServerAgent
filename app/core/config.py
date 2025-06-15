"""
Consolidated Application Configuration for DataMCPServerAgent.

This module provides a comprehensive, type-safe configuration system using Pydantic Settings.
It supports multiple environments, validation, hierarchical configuration loading, and
integration with semantic agents, Cloudflare services, and LLM-driven pipelines.

Features:
- Environment-specific configurations
- Type-safe settings with validation
- Cloudflare integration (Workers, KV, R2, D1)
- Email and WebRTC configurations
- Monitoring and observability settings
- Semantic agents configuration
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
        default="sqlite+aiosqlite:///./datamcp.db",
        description="Database connection URL",
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
    secret_key: str = Field(
        default="dev-secret-key-change-in-production-12345678901234567890",
        description="Secret key for JWT"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time")

    # API security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")

    # CORS settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3002",
            "http://localhost:3000",
            "http://127.0.0.1:3002",
        ],
        description="CORS allowed origins",
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods",
    )
    cors_headers: List[str] = Field(
        default=["Content-Type", "Authorization", "X-API-Key"],
        description="CORS allowed headers",
    )

    # Security headers
    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class CloudflareConfig(BaseSettings):
    """Cloudflare integration configuration."""

    # API settings
    api_token: str = Field(default="", description="Cloudflare API token")
    account_id: str = Field(default="", description="Cloudflare account ID")
    zone_id: str = Field(default="", description="Cloudflare zone ID")

    # Workers settings
    workers_subdomain: str = Field(default="", description="Workers subdomain")
    workers_script_name: str = Field(
        default="datamcp-agent", description="Workers script name"
    )

    # KV settings
    kv_namespace_id: str = Field(default="", description="KV namespace ID")
    kv_preview_namespace_id: str = Field(
        default="", description="KV preview namespace ID"
    )

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
    default_from_email: str = Field(
        default="noreply@datamcp.com", description="Default from email"
    )
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
    enable_tracing: bool = Field(
        default=False, description="Enable distributed tracing"
    )
    jaeger_endpoint: str = Field(default="", description="Jaeger endpoint")

    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval")
    health_check_timeout: int = Field(default=10, description="Health check timeout")

    # Alerting
    enable_alerting: bool = Field(default=False, description="Enable alerting")
    webhook_url: str = Field(default="", description="Alert webhook URL")

    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class SemanticAgentsConfig(BaseSettings):
    """Semantic agents configuration."""

    # Agent settings
    enable_semantic_agents: bool = Field(
        default=True, description="Enable semantic agents"
    )
    max_agents: int = Field(default=10, description="Maximum number of agents")
    agent_timeout: int = Field(default=300, description="Agent timeout in seconds")

    # Model settings
    default_model: str = Field(
        default="claude-3-sonnet-20240229", description="Default LLM model"
    )
    model_temperature: float = Field(default=0.1, description="Model temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")

    # Memory settings
    memory_enabled: bool = Field(default=True, description="Enable agent memory")
    memory_retention_days: int = Field(
        default=30, description="Memory retention in days"
    )
    knowledge_graph_enabled: bool = Field(
        default=True, description="Enable knowledge graph"
    )

    # Communication settings
    communication_enabled: bool = Field(
        default=True, description="Enable inter-agent communication"
    )
    message_queue_size: int = Field(default=1000, description="Message queue size")
    broadcast_timeout: int = Field(
        default=30, description="Broadcast timeout in seconds"
    )

    # Performance settings
    auto_scaling_enabled: bool = Field(default=True, description="Enable auto-scaling")
    cpu_threshold_high: float = Field(
        default=80.0, description="High CPU threshold for scaling"
    )
    cpu_threshold_low: float = Field(
        default=20.0, description="Low CPU threshold for scaling"
    )
    memory_threshold_high: float = Field(
        default=85.0, description="High memory threshold"
    )
    memory_threshold_low: float = Field(
        default=30.0, description="Low memory threshold"
    )

    # Caching settings
    cache_enabled: bool = Field(default=True, description="Enable agent caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")

    model_config = SettingsConfigDict(env_prefix="SEMANTIC_AGENTS_")


class LLMPipelineConfig(BaseSettings):
    """LLM-driven pipeline configuration."""

    # Pipeline settings
    enable_multimodal: bool = Field(
        default=True, description="Enable multimodal processing"
    )
    max_concurrent_pipelines: int = Field(
        default=5, description="Max concurrent pipelines"
    )
    pipeline_timeout: int = Field(
        default=600, description="Pipeline timeout in seconds"
    )

    # Text processing
    text_chunk_size: int = Field(default=1000, description="Text chunk size")
    text_overlap: int = Field(default=200, description="Text chunk overlap")
    enable_semantic_chunking: bool = Field(
        default=True, description="Enable semantic chunking"
    )

    # Image processing
    max_image_size: int = Field(
        default=10485760, description="Max image size in bytes (10MB)"
    )
    supported_image_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "webp"],
        description="Supported image formats",
    )

    # Audio processing
    max_audio_duration: int = Field(
        default=300, description="Max audio duration in seconds"
    )
    supported_audio_formats: List[str] = Field(
        default=["mp3", "wav", "m4a", "ogg"], description="Supported audio formats"
    )

    # Vector stores
    default_vector_store: str = Field(
        default="chromadb", description="Default vector store"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002", description="Embedding model"
    )
    vector_dimension: int = Field(default=1536, description="Vector dimension")

    # RAG settings
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search")
    retrieval_top_k: int = Field(default=5, description="Top K results for retrieval")
    rerank_enabled: bool = Field(default=True, description="Enable result reranking")

    model_config = SettingsConfigDict(env_prefix="LLM_PIPELINE_")


class ReinforcementLearningConfig(BaseSettings):
    """Reinforcement Learning system configuration."""

    # Core RL settings
    enabled: bool = Field(default=True, description="Enable RL system")
    mode: str = Field(default="modern_deep", description="RL mode")
    algorithm: str = Field(default="dqn", description="RL algorithm")
    state_representation: str = Field(default="contextual", description="State representation")

    # Training settings
    training_enabled: bool = Field(default=True, description="Enable training")
    evaluation_episodes: int = Field(default=10, description="Evaluation episodes")
    save_frequency: int = Field(default=100, description="Model save frequency")
    max_episodes: int = Field(default=10000, description="Maximum training episodes")
    episode_timeout: int = Field(default=300, description="Episode timeout in seconds")

    # Model settings
    state_dim: int = Field(default=128, description="State dimension")
    action_dim: int = Field(default=5, description="Action dimension")
    hidden_dim: int = Field(default=256, description="Hidden layer dimension")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    buffer_size: int = Field(default=10000, description="Experience buffer size")
    gamma: float = Field(default=0.99, description="Discount factor")

    # DQN specific settings
    epsilon: float = Field(default=1.0, description="Initial epsilon for exploration")
    epsilon_min: float = Field(default=0.01, description="Minimum epsilon")
    epsilon_decay: float = Field(default=0.995, description="Epsilon decay rate")
    target_update_freq: int = Field(default=1000, description="Target network update frequency")
    double_dqn: bool = Field(default=True, description="Enable Double DQN")
    dueling: bool = Field(default=True, description="Enable Dueling DQN")
    prioritized_replay: bool = Field(default=True, description="Enable prioritized replay")

    # PPO specific settings
    clip_epsilon: float = Field(default=0.2, description="PPO clip epsilon")
    ppo_epochs: int = Field(default=4, description="PPO training epochs")
    gae_lambda: float = Field(default=0.95, description="GAE lambda")
    value_coef: float = Field(default=0.5, description="Value loss coefficient")
    entropy_coef: float = Field(default=0.01, description="Entropy coefficient")

    # Safety settings
    safety_enabled: bool = Field(default=True, description="Enable safety constraints")
    max_resource_usage: float = Field(default=0.8, description="Max resource usage")
    max_response_time: float = Field(default=5.0, description="Max response time")
    safety_weight: float = Field(default=0.5, description="Safety weight in reward")
    constraint_violation_penalty: float = Field(default=-10.0, description="Violation penalty")

    # Explainability settings
    explanation_enabled: bool = Field(default=True, description="Enable explanations")
    explanation_methods: List[str] = Field(
        default=["gradient", "permutation"], description="Explanation methods"
    )
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    generate_natural_language: bool = Field(default=True, description="Generate NL explanations")

    # Distributed settings
    distributed_enabled: bool = Field(default=False, description="Enable distributed training")
    num_workers: int = Field(default=4, description="Number of distributed workers")
    parameter_server_host: str = Field(default="localhost", description="Parameter server host")
    parameter_server_port: int = Field(default=8000, description="Parameter server port")
    aggregation_method: str = Field(default="weighted_average", description="Gradient aggregation")
    sync_frequency: int = Field(default=10, description="Worker sync frequency")

    # Multi-agent settings
    multi_agent_enabled: bool = Field(default=False, description="Enable multi-agent RL")
    num_agents: int = Field(default=3, description="Number of agents")
    cooperation_mode: str = Field(default="cooperative", description="Cooperation mode")
    communication_enabled: bool = Field(default=True, description="Enable communication")
    message_dim: int = Field(default=64, description="Message dimension")
    attention_heads: int = Field(default=4, description="Attention heads")

    # Curriculum learning settings
    curriculum_enabled: bool = Field(default=False, description="Enable curriculum learning")
    difficulty_increment: float = Field(default=0.1, description="Difficulty increment")
    mastery_threshold: float = Field(default=0.8, description="Mastery threshold")
    curriculum_stages: int = Field(default=5, description="Number of curriculum stages")

    # Meta-learning settings
    meta_learning_enabled: bool = Field(default=False, description="Enable meta-learning")
    meta_lr: float = Field(default=1e-3, description="Meta learning rate")
    inner_lr: float = Field(default=1e-2, description="Inner learning rate")
    inner_steps: int = Field(default=5, description="Inner gradient steps")
    task_batch_size: int = Field(default=4, description="Task batch size")

    # Memory settings
    memory_enabled: bool = Field(default=True, description="Enable advanced memory")
    episodic_memory_size: int = Field(default=1000, description="Episodic memory size")
    working_memory_size: int = Field(default=100, description="Working memory size")
    memory_consolidation: bool = Field(default=True, description="Enable memory consolidation")
    memory_retrieval_k: int = Field(default=5, description="Memory retrieval top-k")

    # Monitoring settings
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    dashboard_enabled: bool = Field(default=True, description="Enable web dashboard")
    dashboard_update_interval: int = Field(default=30, description="Dashboard update interval")
    metrics_retention_days: int = Field(default=30, description="Metrics retention days")
    export_metrics: bool = Field(default=True, description="Export metrics to file")

    # Performance settings
    device: str = Field(default="auto", description="Device (cpu/cuda/auto)")
    num_threads: int = Field(default=4, description="Number of threads")
    mixed_precision: bool = Field(default=False, description="Enable mixed precision")
    gradient_clipping: float = Field(default=1.0, description="Gradient clipping norm")
    checkpoint_enabled: bool = Field(default=True, description="Enable checkpointing")

    # Database settings
    db_path: str = Field(default="rl_agent_memory.db", description="Database path")
    db_backup_enabled: bool = Field(default=True, description="Enable database backup")
    db_backup_interval: int = Field(default=3600, description="Backup interval in seconds")

    model_config = SettingsConfigDict(env_prefix="RL_")


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
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Environment"
    )
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
    enable_cloudflare: bool = Field(
        default=True, description="Enable Cloudflare integration"
    )
    enable_email: bool = Field(default=True, description="Enable email integration")
    enable_webrtc: bool = Field(default=True, description="Enable WebRTC integration")
    enable_self_hosting: bool = Field(
        default=True, description="Enable self-hosting features"
    )
    enable_semantic_agents: bool = Field(
        default=True, description="Enable semantic agents"
    )
    enable_llm_pipelines: bool = Field(
        default=True, description="Enable LLM-driven pipelines"
    )
    enable_reinforcement_learning: bool = Field(
        default=True, description="Enable reinforcement learning"
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cloudflare: CloudflareConfig = Field(default_factory=CloudflareConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    webrtc: WebRTCConfig = Field(default_factory=WebRTCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    semantic_agents: SemanticAgentsConfig = Field(default_factory=SemanticAgentsConfig)
    llm_pipeline: LLMPipelineConfig = Field(default_factory=LLMPipelineConfig)
    reinforcement_learning: ReinforcementLearningConfig = Field(default_factory=ReinforcementLearningConfig)

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
            object.__setattr__(self, "debug", True)

        # Validate CORS settings for security
        if self.environment == Environment.PRODUCTION:
            # In production, ensure CORS is not too permissive
            if "*" in self.security.cors_origins:
                raise ValueError(
                    "CORS cannot use wildcard (*) in production environment"
                )
        elif self.environment == Environment.DEVELOPMENT:
            # In development, allow wildcard if explicitly set
            if "*" in self.security.cors_origins:
                import warnings

                warnings.warn("Using wildcard CORS in development environment")

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


# Global settings instance - create when needed to avoid import-time issues
def get_settings() -> Settings:
    """Get global settings instance."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()
    return get_settings._instance
