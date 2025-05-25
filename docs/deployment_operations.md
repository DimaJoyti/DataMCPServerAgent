# Deployment & Operations Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying, configuring, and operating the DataMCPServerAgent system in various environments, from development to production scale.

## 🏗️ Deployment Architecture

### 1. Environment Tiers

#### Development Environment
```yaml
# config/environments/development.yaml
environment:
  name: development
  tier: dev

resources:
  cpu_limit: "2"
  memory_limit: "4Gi"
  storage_limit: "50Gi"

scaling:
  min_replicas: 1
  max_replicas: 2

services:
  database:
    type: sqlite
    file: "data/dev.db"
  cache:
    type: memory
  message_broker:
    type: memory
```

#### Staging Environment
```yaml
# config/environments/staging.yaml
environment:
  name: staging
  tier: staging

resources:
  cpu_limit: "4"
  memory_limit: "8Gi"
  storage_limit: "200Gi"

scaling:
  min_replicas: 2
  max_replicas: 5

services:
  database:
    type: postgresql
    host: staging-db.internal
    database: datamcp_staging
  cache:
    type: redis
    host: staging-redis.internal
  message_broker:
    type: kafka
    brokers: ["staging-kafka-1:9092", "staging-kafka-2:9092"]
```

#### Production Environment
```yaml
# config/environments/production.yaml
environment:
  name: production
  tier: prod

resources:
  cpu_limit: "8"
  memory_limit: "16Gi"
  storage_limit: "1Ti"

scaling:
  min_replicas: 5
  max_replicas: 20
  auto_scaling: true

services:
  database:
    type: postgresql
    cluster: prod-postgres-cluster
    read_replicas: 3
    connection_pool_size: 50
  cache:
    type: redis
    cluster: prod-redis-cluster
    sentinel_enabled: true
  message_broker:
    type: kafka
    cluster: prod-kafka-cluster
    partitions: 12
    replication_factor: 3
```

### 2. Container Strategy

#### Multi-Stage Dockerfile
```dockerfile
# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "src.core.main"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY requirements.txt .

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Change ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "-m", "src.core.main"]
```

#### Docker Compose for Development
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  datamcp-agent:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
      - "8001:8001"  # Debug port
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://user:password@postgres:5432/datamcp_dev
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    networks:
      - datamcp-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: datamcp_dev
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - datamcp-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - datamcp-network

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    networks:
      - datamcp-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - datamcp-network

volumes:
  postgres_data:
  redis_data:

networks:
  datamcp-network:
    driver: bridge
```

### 3. Kubernetes Deployment

#### Namespace Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: datamcp-system
  labels:
    name: datamcp-system
    tier: production
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: datamcp-config
  namespace: datamcp-system
data:
  config.yaml: |
    environment:
      name: production
      log_level: INFO
      debug_mode: false

    database:
      host: postgres-service
      port: 5432
      database: datamcp_prod
      pool_size: 20
      max_overflow: 30

    cache:
      host: redis-service
      port: 6379
      db: 0

    agents:
      max_instances: 10
      timeout: 300

    pipelines:
      max_concurrent: 50
      default_timeout: 3600
```

#### Secret Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: datamcp-secrets
  namespace: datamcp-system
type: Opaque
data:
  database-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>
  encryption-key: <base64-encoded-encryption-key>
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datamcp-agent
  namespace: datamcp-system
  labels:
    app: datamcp-agent
    version: v1.0.0
spec:
  replicas: 5
  selector:
    matchLabels:
      app: datamcp-agent
  template:
    metadata:
      labels:
        app: datamcp-agent
        version: v1.0.0
    spec:
      containers:
      - name: datamcp-agent
        image: datamcp/agent:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: metrics
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: datamcp-secrets
              key: database-password
        - name: CONFIG_PATH
          value: "/etc/config/config.yaml"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: datamcp-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: datamcp-data-pvc
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: datamcp-service
  namespace: datamcp-system
  labels:
    app: datamcp-agent
spec:
  selector:
    app: datamcp-agent
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8001
    targetPort: 8001
    protocol: TCP
  type: ClusterIP
```

#### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: datamcp-hpa
  namespace: datamcp-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: datamcp-agent
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔧 Configuration Management

### 1. Environment-Specific Configuration

#### Configuration Hierarchy
```
config/
├── base/
│   ├── system.yaml
│   ├── agents.yaml
│   ├── pipelines.yaml
│   └── security.yaml
├── environments/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
└── secrets/
    ├── development.env
    ├── staging.env
    └── production.env
```

#### Configuration Loading Strategy
```python
class ConfigurationManager:
    """Manages configuration loading and merging"""

    def __init__(self, environment: str):
        self.environment = environment
        self.config_cache = {}

    def load_configuration(self) -> SystemConfig:
        """Load and merge configuration from multiple sources"""

        # Load base configuration
        base_config = self._load_base_config()

        # Load environment-specific configuration
        env_config = self._load_environment_config(self.environment)

        # Load secrets
        secrets = self._load_secrets(self.environment)

        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config, secrets)

        # Validate configuration
        self._validate_config(merged_config)

        return SystemConfig(**merged_config)

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration files"""
        config = {}

        base_files = [
            "config/base/system.yaml",
            "config/base/agents.yaml",
            "config/base/pipelines.yaml",
            "config/base/security.yaml"
        ]

        for file_path in base_files:
            with open(file_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config = self._deep_merge(config, file_config)

        return config
```

### 2. Secret Management

#### Kubernetes Secrets
```bash
# Create secrets from files
kubectl create secret generic datamcp-secrets \
  --from-file=database-password=secrets/db-password.txt \
  --from-file=api-key=secrets/api-key.txt \
  --from-file=encryption-key=secrets/encryption-key.txt \
  --namespace=datamcp-system

# Create secrets from literals
kubectl create secret generic datamcp-config-secrets \
  --from-literal=database-url="postgresql://user:password@host:5432/db" \
  --from-literal=redis-url="redis://host:6379/0" \
  --namespace=datamcp-system
```

#### External Secret Management
```yaml
# Using External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: datamcp-system
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "datamcp-role"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: datamcp-vault-secrets
  namespace: datamcp-system
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: datamcp-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-password
    remoteRef:
      key: datamcp/database
      property: password
  - secretKey: api-key
    remoteRef:
      key: datamcp/api
      property: key
```

## 📊 Monitoring & Observability

### 1. Prometheus Monitoring

#### ServiceMonitor
```yaml
# k8s/monitoring/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: datamcp-metrics
  namespace: datamcp-system
  labels:
    app: datamcp-agent
spec:
  selector:
    matchLabels:
      app: datamcp-agent
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### PrometheusRule
```yaml
# k8s/monitoring/prometheusrule.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: datamcp-alerts
  namespace: datamcp-system
spec:
  groups:
  - name: datamcp.rules
    rules:
    - alert: DataMCPHighErrorRate
      expr: rate(datamcp_errors_total[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors per second"

    - alert: DataMCPHighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"datamcp-.*"} / container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High memory usage"
        description: "Memory usage is above 90%"
```

### 2. Logging Configuration

#### Structured Logging
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage in application
logger = structlog.get_logger("datamcp.agent")

logger.info(
    "Agent request processed",
    agent_id="research-001",
    request_id="req-12345",
    duration=1.23,
    status="success"
)
```

#### Log Aggregation with Fluentd
```yaml
# k8s/logging/fluentd-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: datamcp-system
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/datamcp-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name datamcp-logs
      type_name _doc
    </match>
```

## 🚀 Deployment Procedures

### 1. CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy DataMCP Agent

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest tests/ --cov=src/ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        target: production
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3

    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

        # Update image tag
        kubectl set image deployment/datamcp-agent \
          datamcp-agent=ghcr.io/${{ github.repository }}:${{ github.sha }} \
          -n datamcp-system

        # Wait for rollout
        kubectl rollout status deployment/datamcp-agent -n datamcp-system
```

### 2. Blue-Green Deployment

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

NAMESPACE="datamcp-system"
NEW_VERSION=$1
CURRENT_COLOR=$(kubectl get service datamcp-service -n $NAMESPACE -o jsonpath='{.spec.selector.color}')

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "Current color: $CURRENT_COLOR"
echo "Deploying to: $NEW_COLOR"

# Deploy new version
kubectl set image deployment/datamcp-agent-$NEW_COLOR \
    datamcp-agent=datamcp/agent:$NEW_VERSION \
    -n $NAMESPACE

# Wait for deployment
kubectl rollout status deployment/datamcp-agent-$NEW_COLOR -n $NAMESPACE

# Health check
echo "Performing health check..."
kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- \
    curl -f http://datamcp-service-$NEW_COLOR.$NAMESPACE.svc.cluster.local/health

# Switch traffic
kubectl patch service datamcp-service -n $NAMESPACE \
    -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'

echo "Traffic switched to $NEW_COLOR"

# Optional: Scale down old deployment
read -p "Scale down $CURRENT_COLOR deployment? (y/n): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl scale deployment datamcp-agent-$CURRENT_COLOR --replicas=0 -n $NAMESPACE
fi
```

### 3. Database Migrations

```python
# scripts/migrate.py
import asyncio
import asyncpg
from alembic import command
from alembic.config import Config

async def run_migrations():
    """Run database migrations"""

    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    # Run custom data migrations
    await run_data_migrations()

async def run_data_migrations():
    """Run custom data migrations"""

    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # Example: Migrate old memory format
        await conn.execute("""
            UPDATE memories
            SET content = jsonb_set(content, '{version}', '"2.0"')
            WHERE content->>'version' = '1.0'
        """)

        print("Data migration completed")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run_migrations())
```

## 🔄 Operational Procedures

### 1. Backup & Recovery

#### Database Backup
```bash
#!/bin/bash
# scripts/backup-database.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
    --format=custom --compress=9 \
    --file=$BACKUP_DIR/datamcp_$(date +%Y%m%d_%H%M%S).backup

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://datamcp-backups/database/ --recursive
```

#### Memory Store Backup
```python
# scripts/backup-memory.py
async def backup_memory_stores():
    """Backup all memory stores"""

    memory_manager = DistributedMemoryManager()

    # Export all memories
    memories = await memory_manager.export_all_memories()

    # Create backup file
    backup_file = f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(backup_file, 'w') as f:
        json.dump(memories, f, indent=2, default=str)

    # Upload to backup storage
    await upload_to_backup_storage(backup_file)
```

### 2. Disaster Recovery

#### Recovery Procedures
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

echo "Starting disaster recovery procedure..."

# 1. Restore database
echo "Restoring database..."
pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME \
    --clean --if-exists $BACKUP_FILE

# 2. Restore memory stores
echo "Restoring memory stores..."
python scripts/restore-memory.py --backup-file $MEMORY_BACKUP

# 3. Restart services
echo "Restarting services..."
kubectl rollout restart deployment/datamcp-agent -n datamcp-system

# 4. Verify system health
echo "Verifying system health..."
python scripts/health-check.py --full-check

echo "Disaster recovery completed"
```

### 3. Performance Tuning

#### Database Optimization
```sql
-- Performance tuning queries
-- Index optimization
CREATE INDEX CONCURRENTLY idx_memories_agent_timestamp
ON memories(agent_id, created_at);

CREATE INDEX CONCURRENTLY idx_pipeline_runs_status
ON pipeline_runs(status, created_at);

-- Query optimization
ANALYZE memories;
ANALYZE pipeline_runs;
ANALYZE tasks;

-- Connection pooling optimization
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
```

This deployment and operations guide provides comprehensive instructions for deploying and managing the DataMCPServerAgent system across different environments, ensuring reliable, scalable, and maintainable operations.
