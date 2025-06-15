# Production Deployment Guide

## 🚀 DataMCPServerAgent Production Deployment

Этот гид описывает полное развертывание DataMCPServerAgent с продвинутой RL системой в production среде.

## 📋 Предварительные Требования

### Системные Требования
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+
- **Python**: 3.9+
- **RAM**: Минимум 8GB, рекомендуется 16GB+
- **CPU**: 4+ cores, рекомендуется 8+ cores
- **GPU**: Опционально для ускорения RL обучения
- **Диск**: 50GB+ свободного места

### Зависимости
```bash
# Основные зависимости
pip install torch torchvision torchaudio
pip install langchain-anthropic
pip install fastapi uvicorn
pip install optuna
pip install numpy pandas scikit-learn
pip install rich typer
pip install asyncio aiofiles
pip install websockets
pip install prometheus-client
```

## 🔧 Конфигурация

### Переменные Окружения

Создайте файл `.env` с необходимыми настройками:

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# RL Configuration
RL_MODE=modern_deep
RL_ALGORITHM=dqn
STATE_REPRESENTATION=contextual
RL_TRAINING_ENABLED=true
RL_EVALUATION_EPISODES=10
RL_SAVE_FREQUENCY=100

# Safety Configuration
RL_SAFETY_ENABLED=true
SAFE_MAX_RESOURCE_USAGE=0.8
SAFE_MAX_RESPONSE_TIME=5.0
SAFE_WEIGHT=0.5

# Explainability Configuration
RL_EXPLANATION_ENABLED=true
EXPLAINABLE_METHODS=gradient,permutation

# Distributed Configuration
DISTRIBUTED_WORKERS=4
PARAMETER_SERVER_HOST=localhost
PARAMETER_SERVER_PORT=8000

# Multi-Agent Configuration
MULTI_AGENT_COUNT=3
MULTI_AGENT_MODE=cooperative
MULTI_AGENT_COMMUNICATION=true

# Database Configuration
RL_DB_PATH=production_rl_memory.db

# Monitoring Configuration
METRICS_RETENTION_DAYS=30
DASHBOARD_UPDATE_INTERVAL=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8002
API_WORKERS=4

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/datamcp.log
```

### Docker Configuration

Создайте `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p logs models data

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run application
CMD ["python", "app/main_consolidated.py", "api", "--host", "0.0.0.0", "--port", "8002"]
```

Создайте `docker-compose.yml`:

```yaml
version: '3.8'

services:
  datamcp-agent:
    build: .
    ports:
      - "8002:8002"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - RL_MODE=modern_deep
      - RL_TRAINING_ENABLED=true
      - RL_SAFETY_ENABLED=true
      - RL_EXPLANATION_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

## 🚀 Развертывание

### 1. Локальное Развертывание

```bash
# Клонирование репозитория
git clone <repository-url>
cd DataMCPServerAgent

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Отредактируйте .env файл

# Инициализация системы
python app/main_consolidated.py migrate

# Запуск API сервера
python app/main_consolidated.py api --host 0.0.0.0 --port 8002

# Запуск CLI (в отдельном терминале)
python app/main_consolidated.py cli
```

### 2. Docker Развертывание

```bash
# Сборка и запуск
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f datamcp-agent

# Остановка
docker-compose down
```

### 3. Kubernetes Развертывание

Создайте `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datamcp-agent
  labels:
    app: datamcp-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: datamcp-agent
  template:
    metadata:
      labels:
        app: datamcp-agent
    spec:
      containers:
      - name: datamcp-agent
        image: datamcp-agent:latest
        ports:
        - containerPort: 8002
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: datamcp-secrets
              key: anthropic-api-key
        - name: RL_MODE
          value: "modern_deep"
        - name: RL_TRAINING_ENABLED
          value: "true"
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
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: datamcp-agent-service
spec:
  selector:
    app: datamcp-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8002
  type: LoadBalancer
```

Развертывание:

```bash
# Создание секретов
kubectl create secret generic datamcp-secrets \
  --from-literal=anthropic-api-key=your_key_here

# Развертывание
kubectl apply -f k8s/

# Проверка статуса
kubectl get pods -l app=datamcp-agent
kubectl get services
```

## 📊 Мониторинг и Логирование

### Prometheus Метрики

Создайте `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'datamcp-agent'
    static_configs:
      - targets: ['datamcp-agent:8002']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana Dashboard

Основные метрики для мониторинга:

- **System Metrics**:
  - CPU и Memory usage
  - Request rate и latency
  - Error rate
  - Uptime

- **RL Metrics**:
  - Training episodes
  - Reward trends
  - Loss trends
  - Action distribution

- **Safety Metrics**:
  - Constraint violations
  - Safety scores
  - Risk assessments

- **Performance Metrics**:
  - Response times (P50, P95, P99)
  - Throughput
  - SLA compliance

### Логирование

Настройка структурированного логирования:

```python
# app/core/logging_production.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, 'rl_mode'):
            log_entry["rl_mode"] = record.rl_mode
        
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)
```

## 🔒 Безопасность

### API Security

```python
# app/core/security.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.route("/api/rl/process")
@limiter.limit("10/minute")
async def process_request(request: Request):
    # Process request
    pass
```

## 🧪 Тестирование

### Unit Tests

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Тесты с покрытием
python -m pytest tests/ --cov=app --cov-report=html

# Тесты RL системы
python -m pytest tests/test_rl/ -v
```

### Integration Tests

```bash
# API тесты
python -m pytest tests/integration/test_api.py -v

# RL интеграционные тесты
python -m pytest tests/integration/test_rl_integration.py -v
```

### Load Testing

```bash
# Установка locust
pip install locust

# Запуск нагрузочного тестирования
locust -f tests/load/locustfile.py --host=http://localhost:8002
```

## 📈 Масштабирование

### Горизонтальное Масштабирование

1. **Load Balancer**: Nginx или HAProxy
2. **Multiple Instances**: Docker Swarm или Kubernetes
3. **Database Sharding**: Распределение данных
4. **Caching**: Redis для кэширования

### Вертикальное Масштабирование

1. **CPU**: Увеличение количества cores
2. **Memory**: Больше RAM для моделей
3. **GPU**: Для ускорения RL обучения
4. **Storage**: SSD для быстрого доступа к данным

## 🔧 Обслуживание

### Регулярные Задачи

```bash
# Ежедневные задачи
0 2 * * * /app/scripts/daily_maintenance.sh

# Еженедельные задачи
0 3 * * 0 /app/scripts/weekly_maintenance.sh

# Ежемесячные задачи
0 4 1 * * /app/scripts/monthly_maintenance.sh
```

### Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

# Backup database
cp data/production_rl_memory.db backups/rl_memory_$(date +%Y%m%d).db

# Backup models
tar -czf backups/models_$(date +%Y%m%d).tar.gz models/

# Backup logs
tar -czf backups/logs_$(date +%Y%m%d).tar.gz logs/

# Clean old backups (keep 30 days)
find backups/ -name "*.db" -mtime +30 -delete
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

## 🚨 Troubleshooting

### Общие Проблемы

1. **High Memory Usage**:
   - Проверьте размер моделей RL
   - Настройте garbage collection
   - Уменьшите batch size

2. **Slow Response Times**:
   - Проверьте CPU usage
   - Оптимизируйте RL inference
   - Добавьте кэширование

3. **Training Issues**:
   - Проверьте learning rate
   - Валидируйте данные
   - Мониторьте loss trends

### Логи и Диагностика

```bash
# Проверка статуса системы
python app/main_consolidated.py status

# Проверка RL системы
python app/main_consolidated.py rl status

# Просмотр логов
tail -f logs/datamcp.log

# Анализ производительности
python scripts/performance_analysis.py
```

## 🎯 Best Practices

1. **Мониторинг**: Настройте алерты для критических метрик
2. **Backup**: Регулярное резервное копирование
3. **Security**: Регулярные обновления безопасности
4. **Testing**: Автоматизированное тестирование
5. **Documentation**: Поддерживайте документацию в актуальном состоянии
6. **Capacity Planning**: Планируйте ресурсы заранее
7. **Disaster Recovery**: План восстановления после сбоев

## 📞 Поддержка

Для получения поддержки:

1. Проверьте документацию
2. Просмотрите известные проблемы
3. Создайте issue в репозитории
4. Обратитесь к команде разработки

---

**Система готова к production развертыванию! 🚀**
