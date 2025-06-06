"""
Self-hosting configuration and deployment utilities for DataMCPServerAgent.
Supports Docker containerization, Kubernetes deployment, and local development setup.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentType(Enum):
    """Deployment type enumeration."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"

class Environment(Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    image: str
    port: int
    environment_variables: Dict[str, str]
    volumes: List[str]
    dependencies: List[str]
    health_check: Optional[str]
    replicas: int = 1

@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str  # postgresql, mysql, sqlite, mongodb
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_enabled: bool = False
    connection_pool_size: int = 10

@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    environment: Environment
    deployment_type: DeploymentType
    services: List[ServiceConfig]
    databases: List[DatabaseConfig]
    ingress_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    secrets: Dict[str, str]

class SelfHostingManager:
    """Manager for self-hosting configurations and deployments."""

    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self._initialize_default_configs()

        logger.info("Self-hosting manager initialized")

    def _initialize_default_configs(self):
        """Initialize default deployment configurations."""

        # Development configuration
        dev_config = DeploymentConfig(
            environment=Environment.DEVELOPMENT,
            deployment_type=DeploymentType.LOCAL,
            services=[
                ServiceConfig(
                    name="agent-server",
                    image="datamcpserveragent:dev",
                    port=8000,
                    environment_variables={
                        "ENVIRONMENT": "development",
                        "DEBUG": "true",
                        "LOG_LEVEL": "DEBUG"
                    },
                    volumes=[
                        "./:/app",
                        "./data:/app/data"
                    ],
                    dependencies=[],
                    health_check="/health",
                    replicas=1
                ),
                ServiceConfig(
                    name="agent-ui",
                    image="datamcpserveragent-ui:dev",
                    port=3000,
                    environment_variables={
                        "NODE_ENV": "development",
                        "NEXT_PUBLIC_API_URL": "http://localhost:8000"
                    },
                    volumes=[
                        "./agent-ui:/app"
                    ],
                    dependencies=["agent-server"],
                    health_check="/",
                    replicas=1
                )
            ],
            databases=[
                DatabaseConfig(
                    type="sqlite",
                    host="localhost",
                    port=0,
                    database="./data/agent.db",
                    username="",
                    password="",
                    ssl_enabled=False
                )
            ],
            ingress_config={
                "enabled": False
            },
            monitoring_config={
                "prometheus": False,
                "grafana": False
            },
            secrets={}
        )

        self.deployment_configs["development"] = dev_config

        # Production configuration
        prod_config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            deployment_type=DeploymentType.KUBERNETES,
            services=[
                ServiceConfig(
                    name="agent-server",
                    image="datamcpserveragent:latest",
                    port=8000,
                    environment_variables={
                        "ENVIRONMENT": "production",
                        "DEBUG": "false",
                        "LOG_LEVEL": "INFO"
                    },
                    volumes=[
                        "/data:/app/data"
                    ],
                    dependencies=["postgresql"],
                    health_check="/health",
                    replicas=3
                ),
                ServiceConfig(
                    name="agent-ui",
                    image="datamcpserveragent-ui:latest",
                    port=3000,
                    environment_variables={
                        "NODE_ENV": "production",
                        "NEXT_PUBLIC_API_URL": "https://api.yourdomain.com"
                    },
                    volumes=[],
                    dependencies=["agent-server"],
                    health_check="/",
                    replicas=2
                )
            ],
            databases=[
                DatabaseConfig(
                    type="postgresql",
                    host="postgresql-service",
                    port=5432,
                    database="agent_db",
                    username="agent_user",
                    password="${POSTGRES_PASSWORD}",
                    ssl_enabled=True,
                    connection_pool_size=20
                )
            ],
            ingress_config={
                "enabled": True,
                "host": "yourdomain.com",
                "tls_enabled": True,
                "cert_manager": True
            },
            monitoring_config={
                "prometheus": True,
                "grafana": True,
                "alerts": True
            },
            secrets={
                "POSTGRES_PASSWORD": "your-secure-password",
                "JWT_SECRET": "your-jwt-secret",
                "CLOUDFLARE_API_KEY": "your-cloudflare-api-key"
            }
        )

        self.deployment_configs["production"] = prod_config

    # ==================== DOCKER CONFIGURATION ====================

    def generate_dockerfile(self, service_name: str, environment: str = "development") -> str:
        """Generate Dockerfile for a service."""

        if service_name == "agent-server":
            return """
# Multi-stage build for Python application
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "integrated_agent_server:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        elif service_name == "agent-ui":
            return """
# Multi-stage build for Next.js application
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY agent-ui/package*.json ./
RUN npm ci --only=production

# Copy source code and build
COPY agent-ui/ .
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:3000/ || exit 1

# Expose port
EXPOSE 3000

# Start application
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]
"""

        else:
            raise ValueError(f"Unknown service: {service_name}")

    def generate_docker_compose(self, environment: str = "development") -> str:
        """Generate docker-compose.yml file."""

        config = self.deployment_configs.get(environment)
        if not config:
            raise ValueError(f"Configuration not found for environment: {environment}")

        compose_config = {
            "version": "3.8",
            "services": {},
            "volumes": {},
            "networks": {
                "agent-network": {
                    "driver": "bridge"
                }
            }
        }

        # Add services
        for service in config.services:
            service_config = {
                "build": {
                    "context": ".",
                    "dockerfile": f"Dockerfile.{service.name}"
                },
                "ports": [f"{service.port}:{service.port}"],
                "environment": service.environment_variables,
                "volumes": service.volumes,
                "networks": ["agent-network"],
                "restart": "unless-stopped"
            }

            if service.dependencies:
                service_config["depends_on"] = service.dependencies

            if service.health_check:
                service_config["healthcheck"] = {
                    "test": f"curl -f http://localhost:{service.port}{service.health_check}",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }

            compose_config["services"][service.name] = service_config

        # Add databases
        for db in config.databases:
            if db.type == "postgresql":
                compose_config["services"]["postgresql"] = {
                    "image": "postgres:15-alpine",
                    "environment": {
                        "POSTGRES_DB": db.database,
                        "POSTGRES_USER": db.username,
                        "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}"
                    },
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data"
                    ],
                    "networks": ["agent-network"],
                    "restart": "unless-stopped"
                }
                compose_config["volumes"]["postgres_data"] = {}

        return yaml.dump(compose_config, default_flow_style=False)

    def generate_env_file(self, environment: str = "development") -> str:
        """Generate .env file for the environment."""

        config = self.deployment_configs.get(environment)
        if not config:
            raise ValueError(f"Configuration not found for environment: {environment}")

        env_vars = []

        # Add secrets
        for key, value in config.secrets.items():
            env_vars.append(f"{key}={value}")

        # Add common environment variables
        env_vars.extend([
            f"ENVIRONMENT={environment}",
            f"DEPLOYMENT_TYPE={config.deployment_type.value}",
            "# Database Configuration",
        ])

        # Add database configuration
        for db in config.databases:
            if db.type == "postgresql":
                env_vars.extend([
                    f"DATABASE_URL=postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}",
                    f"POSTGRES_PASSWORD={db.password}"
                ])

        return "\n".join(env_vars)

    # ==================== KUBERNETES CONFIGURATION ====================

    def generate_kubernetes_manifests(self, environment: str = "production") -> Dict[str, str]:
        """Generate Kubernetes manifests for deployment."""

        config = self.deployment_configs.get(environment)
        if not config:
            raise ValueError(f"Configuration not found for environment: {environment}")

        manifests = {}

        # Generate namespace
        manifests["namespace.yaml"] = self._generate_namespace_manifest()

        # Generate secrets
        manifests["secrets.yaml"] = self._generate_secrets_manifest(config)

        # Generate configmaps
        manifests["configmap.yaml"] = self._generate_configmap_manifest(config)

        # Generate services and deployments
        for service in config.services:
            manifests[f"{service.name}-deployment.yaml"] = self._generate_deployment_manifest(service, config)
            manifests[f"{service.name}-service.yaml"] = self._generate_service_manifest(service)

        # Generate ingress if enabled
        if config.ingress_config.get("enabled"):
            manifests["ingress.yaml"] = self._generate_ingress_manifest(config)

        # Generate monitoring if enabled
        if config.monitoring_config.get("prometheus"):
            manifests["monitoring.yaml"] = self._generate_monitoring_manifest(config)

        return manifests

    def _generate_namespace_manifest(self) -> str:
        """Generate namespace manifest."""
        return """
apiVersion: v1
kind: Namespace
metadata:
  name: datamcpserveragent
  labels:
    name: datamcpserveragent
"""

    def _generate_secrets_manifest(self, config: DeploymentConfig) -> str:
        """Generate secrets manifest."""
        secrets_data = {}
        for key, value in config.secrets.items():
            # In production, these should be base64 encoded
            secrets_data[key] = value

        return f"""
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: datamcpserveragent
type: Opaque
data:
{yaml.dump(secrets_data, default_flow_style=False, indent=2)}
"""

    def _generate_configmap_manifest(self, config: DeploymentConfig) -> str:
        """Generate configmap manifest."""
        return f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: datamcpserveragent
data:
  environment: "{config.environment.value}"
  deployment_type: "{config.deployment_type.value}"
  log_level: "INFO"
"""

    def _generate_deployment_manifest(self, service: ServiceConfig, config: DeploymentConfig) -> str:
        """Generate deployment manifest for a service."""

        env_vars = []
        for key, value in service.environment_variables.items():
            env_vars.append(f"""
        - name: {key}
          value: "{value}" """)

        # Add secrets as environment variables
        for secret_key in config.secrets.keys():
            env_vars.append(f"""
        - name: {secret_key}
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: {secret_key}""")

        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service.name}
  namespace: datamcpserveragent
  labels:
    app: {service.name}
spec:
  replicas: {service.replicas}
  selector:
    matchLabels:
      app: {service.name}
  template:
    metadata:
      labels:
        app: {service.name}
    spec:
      containers:
      - name: {service.name}
        image: {service.image}
        ports:
        - containerPort: {service.port}
        env:{''.join(env_vars)}
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: {service.health_check or '/health'}
            port: {service.port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {service.health_check or '/health'}
            port: {service.port}
          initialDelaySeconds: 5
          periodSeconds: 5
"""

    def _generate_service_manifest(self, service: ServiceConfig) -> str:
        """Generate service manifest."""
        return f"""
apiVersion: v1
kind: Service
metadata:
  name: {service.name}-service
  namespace: datamcpserveragent
  labels:
    app: {service.name}
spec:
  selector:
    app: {service.name}
  ports:
  - port: {service.port}
    targetPort: {service.port}
    protocol: TCP
  type: ClusterIP
"""

    def _generate_ingress_manifest(self, config: DeploymentConfig) -> str:
        """Generate ingress manifest."""
        host = config.ingress_config.get("host", "yourdomain.com")
        tls_enabled = config.ingress_config.get("tls_enabled", False)

        tls_section = ""
        if tls_enabled:
            tls_section = f"""
  tls:
  - hosts:
    - {host}
    secretName: {host}-tls"""

        return f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-ingress
  namespace: datamcpserveragent
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:{tls_section}
  rules:
  - host: {host}
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: agent-server-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agent-ui-service
            port:
              number: 3000
"""

    def _generate_monitoring_manifest(self, config: DeploymentConfig) -> str:
        """Generate monitoring manifest."""
        return """
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: agent-monitoring
  namespace: datamcpserveragent
  labels:
    app: agent-server
spec:
  selector:
    matchLabels:
      app: agent-server
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: datamcpserveragent
data:
  agent-dashboard.json: |
    {
      "dashboard": {
        "title": "DataMCPServerAgent Dashboard",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total[5m])"
              }
            ]
          }
        ]
      }
    }
"""

    # ==================== DEPLOYMENT UTILITIES ====================

    def save_deployment_files(self, environment: str, output_dir: str = "./deployment"):
        """Save all deployment files to disk."""
        import os

        config = self.deployment_configs.get(environment)
        if not config:
            raise ValueError(f"Configuration not found for environment: {environment}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save Docker files
        if config.deployment_type in [DeploymentType.DOCKER, DeploymentType.DOCKER_COMPOSE]:
            for service in config.services:
                dockerfile_content = self.generate_dockerfile(service.name, environment)
                with open(f"{output_dir}/Dockerfile.{service.name}", "w") as f:
                    f.write(dockerfile_content)

            # Save docker-compose.yml
            compose_content = self.generate_docker_compose(environment)
            with open(f"{output_dir}/docker-compose.yml", "w") as f:
                f.write(compose_content)

        # Save Kubernetes files
        if config.deployment_type == DeploymentType.KUBERNETES:
            k8s_dir = f"{output_dir}/kubernetes"
            os.makedirs(k8s_dir, exist_ok=True)

            manifests = self.generate_kubernetes_manifests(environment)
            for filename, content in manifests.items():
                with open(f"{k8s_dir}/{filename}", "w") as f:
                    f.write(content)

        # Save environment file
        env_content = self.generate_env_file(environment)
        with open(f"{output_dir}/.env.{environment}", "w") as f:
            f.write(env_content)

        logger.info(f"Deployment files saved to {output_dir}")

    def get_deployment_config(self, environment: str) -> Optional[DeploymentConfig]:
        """Get deployment configuration for environment."""
        return self.deployment_configs.get(environment)

    def list_environments(self) -> List[str]:
        """List available environments."""
        return list(self.deployment_configs.keys())

# Global instance
self_hosting_manager = SelfHostingManager()
