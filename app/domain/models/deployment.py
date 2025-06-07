"""
Deployment domain models.
Defines models for deployment configurations and environments.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseEntity, BaseValueObject, ValidationError

class Environment(str, Enum):
    """Deployment environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DeploymentType(str, Enum):
    """Deployment type enumeration."""

    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    SERVERLESS = "serverless"

class ServiceConfig(BaseValueObject):
    """Service configuration value object."""

    name: str = Field(description="Service name")
    image: str = Field(description="Container image")
    port: int = Field(description="Service port")
    environment_variables: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    volumes: List[str] = Field(default_factory=list, description="Volume mounts")
    dependencies: List[str] = Field(default_factory=list, description="Service dependencies")
    health_check: Optional[str] = Field(default=None, description="Health check endpoint")
    replicas: int = Field(default=1, description="Number of replicas")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Service name cannot be empty")
        return v.strip().lower()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v <= 0 or v > 65535:
            raise ValidationError("Port must be between 1 and 65535")
        return v

    @field_validator("replicas")
    @classmethod
    def validate_replicas(cls, v):
        if v < 0:
            raise ValidationError("Replicas cannot be negative")
        return v

class DatabaseConfig(BaseValueObject):
    """Database configuration value object."""

    type: str = Field(description="Database type")
    host: str = Field(description="Database host")
    port: int = Field(description="Database port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    ssl_enabled: bool = Field(default=False, description="Whether SSL is enabled")
    connection_pool_size: int = Field(default=10, description="Connection pool size")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        allowed_types = ["postgresql", "mysql", "sqlite", "mongodb", "redis"]
        if v.lower() not in allowed_types:
            raise ValidationError(f"Database type must be one of: {allowed_types}")
        return v.lower()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v <= 0 or v > 65535:
            raise ValidationError("Port must be between 1 and 65535")
        return v

    @field_validator("connection_pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        if v <= 0:
            raise ValidationError("Connection pool size must be positive")
        return v

class IngressConfig(BaseValueObject):
    """Ingress configuration value object."""

    enabled: bool = Field(default=False, description="Whether ingress is enabled")
    host: str = Field(description="Ingress host")
    tls_enabled: bool = Field(default=False, description="Whether TLS is enabled")
    cert_manager: bool = Field(default=False, description="Whether to use cert-manager")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Ingress annotations")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v):
        if not v or not v.strip():
            raise ValidationError("Ingress host cannot be empty")
        return v.strip().lower()

class MonitoringConfig(BaseValueObject):
    """Monitoring configuration value object."""

    prometheus: bool = Field(default=False, description="Enable Prometheus metrics")
    grafana: bool = Field(default=False, description="Enable Grafana dashboards")
    alerts: bool = Field(default=False, description="Enable alerting")
    tracing: bool = Field(default=False, description="Enable distributed tracing")
    logging: bool = Field(default=True, description="Enable centralized logging")
    health_checks: bool = Field(default=True, description="Enable health checks")

class DeploymentConfig(BaseEntity):
    """Deployment configuration entity."""

    name: str = Field(description="Deployment configuration name")
    environment: Environment = Field(description="Target environment")
    deployment_type: DeploymentType = Field(description="Type of deployment")
    services: List[ServiceConfig] = Field(
        default_factory=list, description="Service configurations"
    )
    databases: List[DatabaseConfig] = Field(
        default_factory=list, description="Database configurations"
    )
    ingress_config: IngressConfig = Field(
        default_factory=IngressConfig, description="Ingress configuration"
    )
    monitoring_config: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    secrets: Dict[str, str] = Field(default_factory=dict, description="Secret values")
    tags: Dict[str, str] = Field(default_factory=dict, description="Deployment tags")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Deployment name cannot be empty")
        return v.strip()

    def add_service(self, service: ServiceConfig) -> None:
        """Add a service to the deployment."""
        # Check if service already exists
        existing = next((s for s in self.services if s.name == service.name), None)
        if existing:
            # Update existing service
            self.services = [s if s.name != service.name else service for s in self.services]
        else:
            # Add new service
            self.services.append(service)

    def remove_service(self, service_name: str) -> bool:
        """Remove a service from the deployment."""
        original_count = len(self.services)
        self.services = [s for s in self.services if s.name != service_name]
        return len(self.services) < original_count

    def add_database(self, database: DatabaseConfig) -> None:
        """Add a database to the deployment."""
        # Check if database already exists
        existing = next((d for d in self.databases if d.database == database.database), None)
        if existing:
            # Update existing database
            self.databases = [
                d if d.database != database.database else database for d in self.databases
            ]
        else:
            # Add new database
            self.databases.append(database)

    def get_service(self, service_name: str) -> Optional[ServiceConfig]:
        """Get a service by name."""
        return next((s for s in self.services if s.name == service_name), None)

    def get_database(self, database_name: str) -> Optional[DatabaseConfig]:
        """Get a database by name."""
        return next((d for d in self.databases if d.database == database_name), None)

    @property
    def service_count(self) -> int:
        """Get number of services."""
        return len(self.services)

    @property
    def database_count(self) -> int:
        """Get number of databases."""
        return len(self.databases)

    @property
    def total_replicas(self) -> int:
        """Get total number of service replicas."""
        return sum(service.replicas for service in self.services)
