"""
Deployment domain services.
Contains business logic for deployment configuration and management.
"""


from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService
from app.domain.models.deployment import DeploymentConfig, Environment

logger = get_logger(__name__)


class DeploymentService(DomainService, LoggerMixin):
    """Deployment configuration service."""

    async def create_deployment_config(
        self, name: str, environment: Environment, deployment_type: str
    ) -> DeploymentConfig:
        """Create a new deployment configuration."""
        self.logger.info(f"Creating deployment config: {name} for {environment}")

        from app.domain.models.deployment import DeploymentType

        # Create deployment config
        config = DeploymentConfig(
            name=name, environment=environment, deployment_type=DeploymentType(deployment_type)
        )

        # Save config
        deployment_repo = self.get_repository("deployment")
        saved_config = await deployment_repo.save(config)

        self.logger.info(f"Deployment config created: {saved_config.id}")
        return saved_config
