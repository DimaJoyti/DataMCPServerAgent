# Trading Infinite Loop System - Deployment Guide

## 🚀 Production Deployment

This guide covers deploying the Trading Infinite Loop system to production environments with high availability, scalability, and security.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 8 cores (16 threads)
- RAM: 32GB
- Storage: 500GB SSD
- Network: 1Gbps connection

**Recommended for Production:**
- CPU: 16 cores (32 threads)
- RAM: 64GB
- Storage: 1TB NVMe SSD
- Network: 10Gbps connection
- GPU: NVIDIA RTX 4090 (for ML models)

### Software Dependencies

```bash
# Operating System
Ubuntu 22.04 LTS or CentOS 8+

# Python
Python 3.11+

# Node.js
Node.js 18+

# Database
PostgreSQL 15+
Redis 7+

# Container Runtime
Docker 24+
Docker Compose 2.20+

# Reverse Proxy
Nginx 1.22+

# Process Manager
PM2 or systemd
```

## 🏗️ Architecture Overview

```mermaid
graph TB
    A[Load Balancer] --> B[Nginx Reverse Proxy]
    B --> C[Frontend - Next.js]
    B --> D[Backend API - FastAPI]
    
    D --> E[Trading Strategy Service]
    D --> F[Infinite Loop Orchestrator]
    D --> G[Trading System]
    
    E --> H[PostgreSQL Database]
    E --> I[Redis Cache]
    
    F --> J[ML Model Manager]
    F --> K[Data Source Manager]
    
    G --> L[Exchange APIs]
    G --> M[Market Data Feeds]
    
    N[Monitoring Stack] --> O[Prometheus]
    N --> P[Grafana]
    N --> Q[AlertManager]
```

## 🐳 Docker Deployment

### 1. Create Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_loop
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Cache
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Backend API
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    environment:
      - DATABASE_URL=postgresql://trading_user:${POSTGRES_PASSWORD}@postgres:5432/trading_loop
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - EXCHANGE_API_KEY=${EXCHANGE_API_KEY}
      - EXCHANGE_API_SECRET=${EXCHANGE_API_SECRET}
    depends_on:
      - postgres
      - redis
    ports:
      - "8001:8001"
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
      - ./generated_strategies:/app/generated_strategies

  # Frontend
  frontend:
    build:
      context: ./agent-ui
      dockerfile: Dockerfile
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8001
    depends_on:
      - backend
    ports:
      - "3002:3000"
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 2. Create Dockerfiles

**Backend Dockerfile:**
```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Create directories
RUN mkdir -p logs generated_strategies

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Frontend Dockerfile:**
```dockerfile
# agent-ui/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production image
FROM node:18-alpine AS runner

WORKDIR /app

# Copy built application
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

# Start application
CMD ["npm", "start"]
```

### 3. Environment Configuration

```bash
# .env.prod
# Database
POSTGRES_PASSWORD=your_secure_postgres_password

# Cache
REDIS_PASSWORD=your_secure_redis_password

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_API_SECRET=your_exchange_api_secret

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password

# Security
JWT_SECRET=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# Performance
MAX_WORKERS=16
REDIS_MAX_CONNECTIONS=100
DB_POOL_SIZE=20
```

### 4. Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8001;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ui:10m rate=30r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket routes
        location /ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # Frontend routes
        location / {
            limit_req zone=ui burst=50 nodelay;
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 🚀 Deployment Steps

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
sudo mkdir -p /opt/trading-loop
cd /opt/trading-loop
```

### 2. Deploy Application

```bash
# Clone repository
git clone https://github.com/your-repo/DataMCPServerAgent.git .

# Set up environment
cp .env.example .env.prod
# Edit .env.prod with your configuration

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
```

### 3. SSL Certificate Setup

```bash
# Using Let's Encrypt
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./ssl/key.pem

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 4. Database Initialization

```bash
# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend python -m alembic upgrade head

# Create initial data
docker-compose -f docker-compose.prod.yml exec backend python scripts/init_data.py
```

## 📊 Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'trading-loop-backend'
    static_configs:
      - targets: ['backend:8001']
    metrics_path: '/metrics'

  - job_name: 'trading-loop-frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/api/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Trading Loop Performance",
    "panels": [
      {
        "title": "Strategy Generation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(strategies_generated_total[5m])"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### 1. Firewall Setup

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Application Security

```python
# Security middleware configuration
SECURITY_CONFIG = {
    "cors_origins": ["https://your-domain.com"],
    "rate_limiting": {
        "requests_per_minute": 100,
        "burst_limit": 20
    },
    "authentication": {
        "jwt_algorithm": "HS256",
        "token_expiry": 3600
    },
    "encryption": {
        "algorithm": "AES-256-GCM",
        "key_rotation_interval": 86400
    }
}
```

## 📈 Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_strategies_performance ON strategies USING GIN (performance);
CREATE INDEX CONCURRENTLY idx_backtest_results_date ON backtest_results (created_at);
CREATE INDEX CONCURRENTLY idx_trades_symbol_timestamp ON trades (symbol, timestamp);

-- Configure PostgreSQL
-- postgresql.conf
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB
max_connections = 200
```

### 2. Application Optimization

```python
# Performance configuration
PERFORMANCE_CONFIG = {
    "async_workers": 16,
    "connection_pool_size": 20,
    "redis_max_connections": 100,
    "cache_ttl": 3600,
    "batch_size": 1000,
    "memory_limit": "16GB"
}
```

## 🔄 Backup and Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"

# Create backup
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U trading_user trading_loop > $BACKUP_DIR/db_backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/db_backup_$DATE.sql

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://your-backup-bucket/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +30 -delete
```

### 2. Application Data Backup

```bash
#!/bin/bash
# backup_app_data.sh
DATE=$(date +%Y%m%d_%H%M%S)

# Backup generated strategies
tar -czf /opt/backups/strategies_$DATE.tar.gz /opt/trading-loop/generated_strategies/

# Backup configuration
tar -czf /opt/backups/config_$DATE.tar.gz /opt/trading-loop/.env.prod /opt/trading-loop/nginx.conf
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Optimize configuration
   # Reduce batch sizes and worker counts
   ```

2. **Database Connection Issues**
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Verify connection
   docker-compose exec postgres psql -U trading_user -d trading_loop
   ```

3. **API Performance Issues**
   ```bash
   # Check API logs
   docker-compose logs backend
   
   # Monitor metrics
   curl http://localhost:8001/metrics
   ```

### Health Checks

```bash
#!/bin/bash
# health_check.sh

# Check all services
docker-compose -f docker-compose.prod.yml ps

# Check API health
curl -f http://localhost:8001/health

# Check frontend
curl -f http://localhost:3002/api/health

# Check database
docker-compose -f docker-compose.prod.yml exec postgres pg_isready
```

## 📞 Support

For deployment support:
- Documentation: [docs.your-domain.com](https://docs.your-domain.com)
- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Email: support@your-domain.com
