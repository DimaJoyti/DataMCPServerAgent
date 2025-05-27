"""
Спрощена конфігурація для DataMCPServerAgent v2.0
Базова конфігурація без складних залежностей для початкового запуску.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Середовище додатку."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Рівень логування."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SimpleSettings(BaseSettings):
    """Спрощені налаштування додатку."""

    # Основні налаштування
    app_name: str = Field(default="DataMCPServerAgent", description="Назва додатку")
    app_version: str = Field(default="2.0.0", description="Версія додатку")
    app_description: str = Field(
        default="Advanced AI Agent System with MCP Integration", description="Опис додатку"
    )

    # Середовище
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Середовище")
    debug: bool = Field(default=False, description="Режим налагодження")

    # API налаштування
    api_host: str = Field(default="0.0.0.0", description="Хост API")
    api_port: int = Field(default=8002, description="Порт API")
    api_workers: int = Field(default=1, description="Кількість воркерів API")

    # Логування
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Рівень логування")
    log_format: str = Field(default="json", description="Формат логування")
    log_file: Optional[str] = Field(default=None, description="Файл логування")

    # Директорії
    data_dir: Path = Field(default=Path("./data"), description="Директорія даних")
    temp_dir: Path = Field(default=Path("./temp"), description="Тимчасова директорія")
    logs_dir: Path = Field(default=Path("./logs"), description="Директорія логів")

    # База даних
    database_url: str = Field(
        default="sqlite+aiosqlite:///./datamcp.db", description="URL бази даних"
    )
    database_echo_sql: bool = Field(default=False, description="Виводити SQL запити")

    # Кеш
    redis_url: str = Field(default="redis://localhost:6379/0", description="URL Redis")
    cache_default_ttl: int = Field(default=3600, description="TTL кешу за замовчуванням")

    # Безпека
    secret_key: str = Field(description="Секретний ключ")
    jwt_secret_key: str = Field(default="", description="JWT секретний ключ")
    jwt_algorithm: str = Field(default="HS256", description="JWT алгоритм")
    jwt_expire_minutes: int = Field(default=30, description="Час життя JWT")

    # CORS
    cors_origins: list[str] = Field(default=["*"], description="CORS origins")
    cors_methods: list[str] = Field(default=["*"], description="CORS methods")
    cors_headers: list[str] = Field(default=["*"], description="CORS headers")

    # Функціональність
    enable_cloudflare: bool = Field(default=False, description="Увімкнути Cloudflare")
    enable_email: bool = Field(default=False, description="Увімкнути email")
    enable_webrtc: bool = Field(default=False, description="Увімкнути WebRTC")
    enable_self_hosting: bool = Field(default=True, description="Увімкнути self-hosting")

    # Моніторинг
    enable_metrics: bool = Field(default=True, description="Увімкнути метрики")
    metrics_port: int = Field(default=9090, description="Порт метрик")
    enable_tracing: bool = Field(default=False, description="Увімкнути трейсинг")

    # Заголовки безпеки
    enable_security_headers: bool = Field(default=True, description="Увімкнути заголовки безпеки")
    rate_limit_per_minute: int = Field(default=60, description="Ліміт запитів на хвилину")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    def __init__(self, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)

        # Створення директорій
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Налаштування debug режиму
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True

        # Налаштування JWT ключа
        if not self.jwt_secret_key:
            self.jwt_secret_key = self.secret_key

    @property
    def is_development(self) -> bool:
        """Перевірка режиму розробки."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Перевірка продакшн режиму."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Перевірка тестового режиму."""
        return self.environment == Environment.TESTING


# Глобальний екземпляр налаштувань
def get_settings() -> SimpleSettings:
    """Отримати налаштування."""
    return SimpleSettings()


# Для зворотної сумісності
Settings = SimpleSettings
settings = get_settings()
