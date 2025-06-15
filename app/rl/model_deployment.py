"""
Model Deployment and MLOps System for DataMCPServerAgent.
This module implements automated model deployment, versioning, and lifecycle management.
"""

import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging_improved import get_logger
from app.monitoring.rl_analytics import get_metrics_collector

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DeploymentStrategy(str, Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class ModelMetadata:
    """Model metadata information."""
    model_id: str
    name: str
    version: str
    algorithm: str
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: float
    trained_by: str
    model_size_mb: float
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    strategy: DeploymentStrategy
    traffic_percentage: float = 100.0
    rollback_threshold: float = 0.05  # 5% error rate threshold
    monitoring_duration: int = 3600  # 1 hour monitoring
    auto_promote: bool = False
    health_check_interval: int = 60  # 1 minute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["strategy"] = self.strategy.value
        return result


@dataclass
class ModelDeployment:
    """Represents a model deployment."""
    deployment_id: str
    model_id: str
    environment: str  # staging, production
    status: ModelStatus
    config: DeploymentConfig
    deployed_at: float
    health_status: str = "unknown"
    traffic_percentage: float = 0.0
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        result["config"] = self.config.to_dict()
        return result


class ModelRegistry:
    """Model registry for version control and metadata management."""

    def __init__(self, registry_path: str = "models/registry"):
        """Initialize model registry.
        
        Args:
            registry_path: Path to model registry
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelMetadata] = {}
        self.model_files: Dict[str, str] = {}  # model_id -> file_path

        # Load existing models
        self._load_registry()

    def _load_registry(self):
        """Load existing models from registry."""
        registry_file = self.registry_path / "registry.json"

        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)

                for model_data in data.get("models", []):
                    metadata = ModelMetadata(**model_data)
                    self.models[metadata.model_id] = metadata

                    # Check if model file exists
                    model_file = self.registry_path / f"{metadata.model_id}.pth"
                    if model_file.exists():
                        self.model_files[metadata.model_id] = str(model_file)

                logger.info(f"📚 Loaded {len(self.models)} models from registry")

            except Exception as e:
                logger.error(f"Error loading model registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"

        try:
            data = {
                "models": [model.to_dict() for model in self.models.values()],
                "last_updated": time.time(),
            }

            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def register_model(
        self,
        name: str,
        version: str,
        algorithm: str,
        model_path: str,
        training_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        trained_by: str = "system"
    ) -> str:
        """Register a new model in the registry.
        
        Args:
            name: Model name
            version: Model version
            algorithm: Algorithm used
            model_path: Path to model file
            training_config: Training configuration
            performance_metrics: Performance metrics
            trained_by: Who trained the model
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = hashlib.md5(f"{name}_{version}_{time.time()}".encode()).hexdigest()[:12]

        # Calculate model size and checksum
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_size_mb = model_file.stat().st_size / (1024 * 1024)

        # Calculate checksum
        with open(model_file, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Copy model to registry
        registry_model_path = self.registry_path / f"{model_id}.pth"
        shutil.copy2(model_path, registry_model_path)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            algorithm=algorithm,
            training_config=training_config,
            performance_metrics=performance_metrics,
            created_at=time.time(),
            trained_by=trained_by,
            model_size_mb=model_size_mb,
            checksum=checksum,
        )

        # Register model
        self.models[model_id] = metadata
        self.model_files[model_id] = str(registry_model_path)

        # Save registry
        self._save_registry()

        logger.info(f"📦 Registered model: {name} v{version} (ID: {model_id})")

        return model_id

    def get_model(self, model_id: str) -> Optional[Tuple[ModelMetadata, str]]:
        """Get model metadata and file path.
        
        Args:
            model_id: Model ID
            
        Returns:
            Tuple of (metadata, file_path) or None
        """
        if model_id not in self.models:
            return None

        metadata = self.models[model_id]
        file_path = self.model_files.get(model_id)

        return metadata, file_path

    def list_models(self, name_filter: Optional[str] = None) -> List[ModelMetadata]:
        """List all models in registry.
        
        Args:
            name_filter: Optional name filter
            
        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        if name_filter:
            models = [m for m in models if name_filter.lower() in m.name.lower()]

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if deleted successfully
        """
        if model_id not in self.models:
            return False

        try:
            # Remove model file
            if model_id in self.model_files:
                model_file = Path(self.model_files[model_id])
                if model_file.exists():
                    model_file.unlink()
                del self.model_files[model_id]

            # Remove from registry
            del self.models[model_id]

            # Save registry
            self._save_registry()

            logger.info(f"🗑️ Deleted model: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False


class ModelDeploymentManager:
    """Manages model deployments and lifecycle."""

    def __init__(self):
        """Initialize deployment manager."""
        self.settings = get_settings()
        self.metrics_collector = get_metrics_collector()
        self.registry = ModelRegistry()

        # Deployment tracking
        self.deployments: Dict[str, ModelDeployment] = {}
        self.active_deployments: Dict[str, str] = {}  # environment -> deployment_id

        # Health monitoring
        self.health_check_task = None
        self.is_monitoring = False

    async def deploy_model(
        self,
        model_id: str,
        environment: str,
        config: DeploymentConfig
    ) -> str:
        """Deploy a model to an environment.
        
        Args:
            model_id: Model ID to deploy
            environment: Target environment
            config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        # Validate model exists
        model_info = self.registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata, model_path = model_info

        # Generate deployment ID
        deployment_id = hashlib.md5(f"{model_id}_{environment}_{time.time()}".encode()).hexdigest()[:12]

        # Create deployment
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            environment=environment,
            status=ModelStatus.STAGING,
            config=config,
            deployed_at=time.time(),
        )

        # Execute deployment strategy
        success = await self._execute_deployment_strategy(deployment, metadata, model_path)

        if success:
            self.deployments[deployment_id] = deployment

            # Update active deployment for environment
            if config.strategy != DeploymentStrategy.SHADOW:
                self.active_deployments[environment] = deployment_id

            logger.info(f"🚀 Deployed model {model_id} to {environment} (Deployment: {deployment_id})")

            # Record deployment event
            self.metrics_collector.record_event(
                "model_deployed",
                {
                    "model_id": model_id,
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "strategy": config.strategy.value,
                },
                "info"
            )

            # Start health monitoring
            if not self.is_monitoring:
                await self._start_health_monitoring()
        else:
            deployment.status = ModelStatus.FAILED
            logger.error(f"❌ Failed to deploy model {model_id} to {environment}")

        return deployment_id

    async def _execute_deployment_strategy(
        self,
        deployment: ModelDeployment,
        metadata: ModelMetadata,
        model_path: str
    ) -> bool:
        """Execute deployment strategy.
        
        Args:
            deployment: Deployment configuration
            metadata: Model metadata
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        strategy = deployment.config.strategy

        try:
            if strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._blue_green_deployment(deployment, metadata, model_path)
            elif strategy == DeploymentStrategy.CANARY:
                return await self._canary_deployment(deployment, metadata, model_path)
            elif strategy == DeploymentStrategy.ROLLING:
                return await self._rolling_deployment(deployment, metadata, model_path)
            elif strategy == DeploymentStrategy.SHADOW:
                return await self._shadow_deployment(deployment, metadata, model_path)
            else:
                logger.error(f"Unknown deployment strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Error executing {strategy} deployment: {e}")
            return False

    async def _blue_green_deployment(
        self,
        deployment: ModelDeployment,
        metadata: ModelMetadata,
        model_path: str
    ) -> bool:
        """Execute blue-green deployment.
        
        Args:
            deployment: Deployment configuration
            metadata: Model metadata
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        logger.info(f"🔵🟢 Executing blue-green deployment for {deployment.model_id}")

        # Simulate model loading and validation
        await asyncio.sleep(2)

        # Switch traffic
        deployment.traffic_percentage = 100.0
        deployment.status = ModelStatus.PRODUCTION
        deployment.health_status = "healthy"

        return True

    async def _canary_deployment(
        self,
        deployment: ModelDeployment,
        metadata: ModelMetadata,
        model_path: str
    ) -> bool:
        """Execute canary deployment.
        
        Args:
            deployment: Deployment configuration
            metadata: Model metadata
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        logger.info(f"🐤 Executing canary deployment for {deployment.model_id}")

        # Start with small traffic percentage
        deployment.traffic_percentage = deployment.config.traffic_percentage
        deployment.status = ModelStatus.STAGING
        deployment.health_status = "healthy"

        # Monitor performance and gradually increase traffic
        # This would be handled by the health monitoring system

        return True

    async def _rolling_deployment(
        self,
        deployment: ModelDeployment,
        metadata: ModelMetadata,
        model_path: str
    ) -> bool:
        """Execute rolling deployment.
        
        Args:
            deployment: Deployment configuration
            metadata: Model metadata
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        logger.info(f"🔄 Executing rolling deployment for {deployment.model_id}")

        # Simulate gradual rollout
        for percentage in [25, 50, 75, 100]:
            deployment.traffic_percentage = percentage
            logger.info(f"   Rolling out to {percentage}% traffic")
            await asyncio.sleep(1)

        deployment.status = ModelStatus.PRODUCTION
        deployment.health_status = "healthy"

        return True

    async def _shadow_deployment(
        self,
        deployment: ModelDeployment,
        metadata: ModelMetadata,
        model_path: str
    ) -> bool:
        """Execute shadow deployment.
        
        Args:
            deployment: Deployment configuration
            metadata: Model metadata
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        logger.info(f"👥 Executing shadow deployment for {deployment.model_id}")

        # Shadow deployment receives traffic but doesn't serve responses
        deployment.traffic_percentage = 0.0  # No user-facing traffic
        deployment.status = ModelStatus.STAGING
        deployment.health_status = "healthy"

        return True

    async def _start_health_monitoring(self):
        """Start health monitoring for deployments."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("💓 Started deployment health monitoring")

    async def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_deployment_health()
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)

    async def _check_deployment_health(self):
        """Check health of all active deployments."""
        for deployment_id, deployment in self.deployments.items():
            if deployment.status not in [ModelStatus.STAGING, ModelStatus.PRODUCTION]:
                continue

            try:
                # Simulate health check
                health_score = np.random.uniform(0.8, 1.0)  # Mock health score

                if health_score > 0.95:
                    deployment.health_status = "healthy"
                elif health_score > 0.8:
                    deployment.health_status = "warning"
                else:
                    deployment.health_status = "unhealthy"

                # Record health metrics
                self.metrics_collector.record_metric(
                    f"deployment_health_{deployment_id}",
                    health_score,
                    {
                        "deployment_id": deployment_id,
                        "model_id": deployment.model_id,
                        "environment": deployment.environment,
                    }
                )

                # Check for auto-promotion (canary -> production)
                if (deployment.status == ModelStatus.STAGING and
                    deployment.config.auto_promote and
                    deployment.health_status == "healthy" and
                    time.time() - deployment.deployed_at > deployment.config.monitoring_duration):

                    await self._promote_deployment(deployment_id)

            except Exception as e:
                logger.error(f"Error checking health for deployment {deployment_id}: {e}")

    async def _promote_deployment(self, deployment_id: str):
        """Promote a staging deployment to production.
        
        Args:
            deployment_id: Deployment ID to promote
        """
        if deployment_id not in self.deployments:
            return

        deployment = self.deployments[deployment_id]

        if deployment.status != ModelStatus.STAGING:
            logger.warning(f"Cannot promote deployment {deployment_id}: not in staging")
            return

        logger.info(f"⬆️ Promoting deployment {deployment_id} to production")

        deployment.status = ModelStatus.PRODUCTION
        deployment.traffic_percentage = 100.0

        # Record promotion event
        self.metrics_collector.record_event(
            "deployment_promoted",
            {
                "deployment_id": deployment_id,
                "model_id": deployment.model_id,
                "environment": deployment.environment,
            },
            "info"
        )

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment.
        
        Args:
            deployment_id: Deployment ID to rollback
            
        Returns:
            True if rollback successful
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False

        deployment = self.deployments[deployment_id]

        logger.info(f"⏪ Rolling back deployment {deployment_id}")

        # Set deployment to deprecated
        deployment.status = ModelStatus.DEPRECATED
        deployment.traffic_percentage = 0.0

        # Remove from active deployments
        if deployment.environment in self.active_deployments:
            if self.active_deployments[deployment.environment] == deployment_id:
                del self.active_deployments[deployment.environment]

        # Record rollback event
        self.metrics_collector.record_event(
            "deployment_rolled_back",
            {
                "deployment_id": deployment_id,
                "model_id": deployment.model_id,
                "environment": deployment.environment,
            },
            "warning"
        )

        return True

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status or None
        """
        if deployment_id not in self.deployments:
            return None

        deployment = self.deployments[deployment_id]
        model_info = self.registry.get_model(deployment.model_id)

        status = deployment.to_dict()

        if model_info:
            metadata, _ = model_info
            status["model_metadata"] = metadata.to_dict()

        # Add runtime metrics
        status["uptime"] = time.time() - deployment.deployed_at
        status["is_active"] = (
            deployment.environment in self.active_deployments and
            self.active_deployments[deployment.environment] == deployment_id
        )

        return status

    def list_deployments(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments.
        
        Args:
            environment: Optional environment filter
            
        Returns:
            List of deployment statuses
        """
        deployments = []

        for deployment_id, deployment in self.deployments.items():
            if environment and deployment.environment != environment:
                continue

            status = self.get_deployment_status(deployment_id)
            if status:
                deployments.append(status)

        # Sort by deployment time (newest first)
        deployments.sort(key=lambda d: d["deployed_at"], reverse=True)

        return deployments


# Global deployment manager instance
_deployment_manager: Optional[ModelDeploymentManager] = None


def get_deployment_manager() -> ModelDeploymentManager:
    """Get global deployment manager."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = ModelDeploymentManager()
    return _deployment_manager
