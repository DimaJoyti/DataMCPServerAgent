# Enhanced Bright Data MCP Integration - Setup Guide

Цей документ містить детальні інструкції по встановленню та налаштуванню покращеної інтеграції з Bright Data MCP.

## 📋 Передумови

### Системні вимоги
- Python 3.8 або новіший
- Redis (опціонально, для distributed caching)
- Bright Data API ключ
- Мінімум 512MB RAM
- Інтернет з'єднання

### Необхідні Python пакети
```bash
# Основні залежності
pip install aiohttp>=3.8.0
pip install asyncio
pip install dataclasses-json

# Опціональні залежності для повної функціональності
pip install redis>=4.0.0          # Для Redis кешування
     # Для REST API
```

## 🚀 Швидке встановлення

### 1. Клонування та встановлення

```bash
# Перейдіть до директорії проекту
cd DataMCPServerAgent

# Встановіть залежності
pip install -r requirements.txt
pip install prometheus-client     # Для метрик
pip install websockets            # Для WebSocket API
pip install fastapi uvicorn  
# Або використовуйте uv (рекомендовано)
uv pip install -r requirements.txt
```

### 2. Налаштування змінних середовища

Створіть файл `.env` або встановіть змінні середовища:

```bash
# Основні налаштування
export BRIGHT_DATA_API_KEY="your_bright_data_api_key_here"
export BRIGHT_DATA_API_URL="https://api.brightdata.com"

# Кешування
export REDIS_URL="redis://localhost:6379/0"
export CACHE_ENABLED="true"
export CACHE_TTL="3600"

# Rate Limiting
export RATE_LIMIT_ENABLED="true"
export RATE_LIMIT_RPM="60"
export RATE_LIMIT_BURST="10"

# Логування
export LOG_LEVEL="INFO"
```

### 3. Перевірка встановлення

```bash
# Запустіть швидкий тест
python scripts/test_bright_data_enhanced.py

# Або запустіть повний приклад
python examples/enhanced_bright_data_example.py
```

## ⚙️ Детальне налаштування

### Конфігураційний файл

Створіть файл `configs/bright_data_config.json`:

```json
{
  "api_base_url": "https://api.brightdata.com",
  "api_timeout": 30,
  "max_concurrent_requests": 10,
  "user_agent": "DataMCPServerAgent/2.0.0",
  
  "cache": {
    "enabled": true,
    "redis_url": "redis://localhost:6379/0",
    "memory_cache_size": 1000,
    "default_ttl": 3600,
    "compression_enabled": true,
    "compression_threshold": 1024
  },
  
  "rate_limit": {
    "enabled": true,
    "requests_per_minute": 60,
    "burst_size": 10,
    "adaptive_throttling": true,
    "backoff_factor": 1.5
  },
  
  "retry": {
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
    "jitter": true
  },
  
  "circuit_breaker": {
    "enabled": true,
    "failure_threshold": 5,
    "recovery_timeout": 60
  },
  
  "monitoring": {
    "enabled": true,
    "log_level": "INFO"
  }
}
```

### Redis налаштування (опціонально)

Якщо ви хочете використовувати Redis для distributed caching:

```bash
# Встановлення Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Запуск Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Перевірка
redis-cli ping
# Повинно повернути: PONG
```

Для Docker:
```bash
# Запуск Redis в Docker
docker run -d --name redis -p 6379:6379 redis:alpine

# Перевірка
docker exec redis redis-cli ping
```

## 🔧 Програмне налаштування

### Базове використання

```python
import asyncio
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.config import BrightDataConfig

async def main():
    # Завантаження конфігурації
    config = BrightDataConfig.from_env()
    
    # Створення клієнта
    client = EnhancedBrightDataClient(config=config)
    
    try:
        # Використання клієнта
        result = await client.scrape_url("https://example.com")
        print(f"Scraped {len(str(result))} characters")
        
    finally:
        await client.close()

asyncio.run(main())
```

### Повне налаштування з усіма компонентами

```python
import asyncio
from src.tools.bright_data.core.config import BrightDataConfig
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.cache_manager import CacheManager, MemoryCache, RedisCache
from src.tools.bright_data.core.rate_limiter import RateLimiter, ThrottleStrategy
from src.tools.bright_data.core.error_handler import BrightDataErrorHandler

async def setup_enhanced_system():
    # 1. Конфігурація
    config = BrightDataConfig.from_file("configs/bright_data_config.json")
    
    # 2. Кешування
    memory_cache = MemoryCache(max_size=1000, default_ttl=3600)
    redis_cache = RedisCache(redis_url=config.cache.redis_url)
    cache_manager = CacheManager(memory_cache, redis_cache)
    
    # 3. Rate Limiting
    rate_limiter = RateLimiter(
        requests_per_minute=config.rate_limit.requests_per_minute,
        burst_size=config.rate_limit.burst_size,
        strategy=ThrottleStrategy.ADAPTIVE
    )
    
    # 4. Error Handling
    error_handler = BrightDataErrorHandler()
    
    # 5. Enhanced Client
    client = EnhancedBrightDataClient(
        config=config,
        cache_manager=cache_manager,
        rate_limiter=rate_limiter,
        error_handler=error_handler
    )
    
    return client

async def main():
    client = await setup_enhanced_system()
    
    try:
        # Ваш код тут
        pass
    finally:
        await client.close()

asyncio.run(main())
```

## 🧪 Тестування налаштування

### 1. Швидкий тест компонентів

```bash
# Запуск швидкого тесту (не потребує API ключа)
python scripts/test_bright_data_enhanced.py
```

Очікуваний вивід:
```
🚀 Starting Enhanced Bright Data Integration Tests
============================================================
✅ Configuration Management (0.001s)
✅ Memory Cache (0.005s)
✅ Rate Limiter (0.003s)
✅ Error Handler (0.002s)
✅ Cache Manager (0.004s)
✅ Enhanced Client Basic (0.001s)
✅ Performance Benchmark (0.156s)
============================================================
✅ Test suite completed in 0.17 seconds
📊 Results: 7 passed, 0 failed
```

### 2. Тест з реальним API

```python
# test_real_api.py
import asyncio
import os
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.config import BrightDataConfig

async def test_real_api():
    # Переконайтеся, що API ключ встановлений
    if not os.getenv('BRIGHT_DATA_API_KEY'):
        print("❌ BRIGHT_DATA_API_KEY not set")
        return
    
    config = BrightDataConfig.from_env()
    client = EnhancedBrightDataClient(config=config)
    
    try:
        # Тест health check
        health = await client.health_check()
        print(f"Health check: {health['status']}")
        
        # Тест простого скрапінгу
        result = await client.scrape_url("https://httpbin.org/json")
        print(f"✅ Scraping successful: {len(str(result))} characters")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await client.close()

asyncio.run(test_real_api())
```

### 3. Performance тест

```bash
# Запуск performance тесту
python -c "
import asyncio
from scripts.test_bright_data_enhanced import BrightDataTester

async def perf_test():
    tester = BrightDataTester()
    await tester.test_performance_benchmark()

asyncio.run(perf_test())
"
```

## 🔍 Діагностика проблем

### Поширені проблеми та рішення

#### 1. Redis connection failed
```
Error: ConnectionError: Error 111 connecting to localhost:6379
```

**Рішення:**
```bash
# Перевірте чи запущений Redis
sudo systemctl status redis-server

# Або запустіть Redis
sudo systemctl start redis-server

# Або вимкніть Redis кешування
export REDIS_URL=""
```

#### 2. Import errors
```
ImportError: No module named 'src.tools.bright_data'
```

**Рішення:**
```bash
# Переконайтеся, що ви в правильній директорії
pwd  # Повинно показати .../DataMCPServerAgent

# Додайте проект до PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. API key issues
```
AuthenticationException: Authentication failed
```

**Рішення:**
```bash
# Перевірте API ключ
echo $BRIGHT_DATA_API_KEY

# Встановіть правильний ключ
export BRIGHT_DATA_API_KEY="your_actual_api_key"
```

#### 4. Rate limiting issues
```
RateLimitException: Rate limit exceeded
```

**Рішення:**
```python
# Збільшіть ліміти в конфігурації
config.rate_limit.requests_per_minute = 120
config.rate_limit.burst_size = 20
```

### Логування для діагностики

```python
import logging

# Увімкніть детальне логування
logging.basicConfig(level=logging.DEBUG)

# Або тільки для Bright Data компонентів
logging.getLogger('src.tools.bright_data').setLevel(logging.DEBUG)
```

## 📊 Моніторинг та метрики

### Отримання метрик

```python
# Метрики клієнта
metrics = client.get_metrics()
print(f"Success rate: {metrics['success_rate']:.2f}%")
print(f"Total requests: {metrics['total_requests']}")

# Метрики кешування
cache_stats = cache_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate_percentage']:.2f}%")

# Метрики rate limiting
rate_stats = rate_limiter.get_global_stats()
print(f"Rejected requests: {rate_stats['rejected_requests']}")
```

### Health Check

```python
# Перевірка здоров'я системи
health = await client.health_check()
if health["status"] == "healthy":
    print("✅ System is healthy")
else:
    print(f"❌ System issue: {health.get('error', 'Unknown')}")
```

## 🔄 Оновлення та підтримка

### Оновлення компонентів

```bash
# Оновлення залежностей
pip install --upgrade aiohttp redis

# Або з uv
uv pip install --upgrade aiohttp redis
```

### Backup конфігурації

```bash
# Створіть backup поточної конфігурації
cp configs/bright_data_config.json configs/bright_data_config.backup.json

# Або експортуйте змінні середовища
env | grep BRIGHT_DATA > bright_data_env.backup
```

## 📞 Підтримка

Якщо у вас виникли проблеми:

1. Перевірте [Troubleshooting Guide](BRIGHT_DATA_TROUBLESHOOTING.md)
2. Запустіть діагностичний скрипт: `python scripts/test_bright_data_enhanced.py`
3. Перевірте логи з рівнем DEBUG
4. Створіть issue в GitHub репозиторії з детальним описом проблеми

## 🎯 Наступні кроки

Після успішного налаштування:

1. Ознайомтеся з [Advanced Features Guide](BRIGHT_DATA_ADVANCED.md)
2. Вивчіть [API Reference](BRIGHT_DATA_API.md)
3. Спробуйте [Examples](../examples/)
4. Налаштуйте [Monitoring Dashboard](BRIGHT_DATA_MONITORING.md)
