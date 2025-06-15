# Enhanced Bright Data MCP Integration - Setup Guide

This document contains detailed instructions for installing and configuring the enhanced Bright Data MCP integration.

## 📋 Prerequisites

### System Requirements
- Python 3.8 or newer
- Redis (optional, for distributed caching)
- Bright Data API key
- Minimum 512MB RAM
- Internet connection

### Required Python Packages
```bash
# Core dependencies
pip install aiohttp>=3.8.0
pip install asyncio
pip install dataclasses-json

# Optional dependencies for full functionality
pip install redis>=4.0.0          # For Redis caching
pip install fastapi uvicorn        # For REST API
```

## 🚀 Quick Installation

### 1. Clone and Install

```bash
# Navigate to project directory
cd DataMCPServerAgent

# Install dependencies
pip install -r requirements.txt
pip install prometheus-client     # For metrics
pip install websockets            # For WebSocket API
pip install fastapi uvicorn

# Or use uv (recommended)
uv pip install -r requirements.txt
```

### 2. Environment Variables Setup

Create a `.env` file or set environment variables:

```bash
# Basic settings
export BRIGHT_DATA_API_KEY="your_bright_data_api_key_here"
export BRIGHT_DATA_API_URL="https://api.brightdata.com"

# Caching
export REDIS_URL="redis://localhost:6379/0"
export CACHE_ENABLED="true"
export CACHE_TTL="3600"

# Rate Limiting
export RATE_LIMIT_ENABLED="true"
export RATE_LIMIT_RPM="60"
export RATE_LIMIT_BURST="10"

# Logging
export LOG_LEVEL="INFO"
```

### 3. Installation Verification

```bash
# Run quick test
python scripts/test_bright_data_enhanced.py

# Or run full example
python examples/enhanced_bright_data_example.py
```

## ⚙️ Detailed Configuration

### Configuration File

Create file `configs/bright_data_config.json`:

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

### Redis Setup (Optional)

If you want to use Redis for distributed caching:

```bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test
redis-cli ping
# Should return: PONG
```

For Docker:
```bash
# Run Redis in Docker
docker run -d --name redis -p 6379:6379 redis:alpine

# Test
docker exec redis redis-cli ping
```

## 🔧 Programmatic Configuration

### Basic Usage

```python
import asyncio
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.config import BrightDataConfig

async def main():
    # Load configuration
    config = BrightDataConfig.from_env()
    
    # Create client
    client = EnhancedBrightDataClient(config=config)
    
    try:
        # Use client
        result = await client.scrape_url("https://example.com")
        print(f"Scraped {len(str(result))} characters")
        
    finally:
        await client.close()

asyncio.run(main())
```

### Full Setup with All Components

```python
import asyncio
from src.tools.bright_data.core.config import BrightDataConfig
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.cache_manager import CacheManager, MemoryCache, RedisCache
from src.tools.bright_data.core.rate_limiter import RateLimiter, ThrottleStrategy
from src.tools.bright_data.core.error_handler import BrightDataErrorHandler

async def setup_enhanced_system():
    # 1. Configuration
    config = BrightDataConfig.from_file("configs/bright_data_config.json")
    
    # 2. Caching
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
        # Your code here
        pass
    finally:
        await client.close()

asyncio.run(main())
```

## 🧪 Setup Testing

### 1. Quick Component Test

```bash
# Run quick test (doesn't require API key)
python scripts/test_bright_data_enhanced.py
```

Expected output:
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

### 2. Real API Test

```python
# test_real_api.py
import asyncio
import os
from src.tools.bright_data.core.enhanced_client import EnhancedBrightDataClient
from src.tools.bright_data.core.config import BrightDataConfig

async def test_real_api():
    # Make sure API key is set
    if not os.getenv('BRIGHT_DATA_API_KEY'):
        print("❌ BRIGHT_DATA_API_KEY not set")
        return
    
    config = BrightDataConfig.from_env()
    client = EnhancedBrightDataClient(config=config)
    
    try:
        # Test health check
        health = await client.health_check()
        print(f"Health check: {health['status']}")
        
        # Test simple scraping
        result = await client.scrape_url("https://httpbin.org/json")
        print(f"✅ Scraping successful: {len(str(result))} characters")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await client.close()

asyncio.run(test_real_api())
```

### 3. Performance Test

```bash
# Run performance test
python -c "
import asyncio
from scripts.test_bright_data_enhanced import BrightDataTester

async def perf_test():
    tester = BrightDataTester()
    await tester.test_performance_benchmark()

asyncio.run(perf_test())
"
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Redis connection failed
```
Error: ConnectionError: Error 111 connecting to localhost:6379
```

**Solution:**
```bash
# Check if Redis is running
sudo systemctl status redis-server

# Or start Redis
sudo systemctl start redis-server

# Or disable Redis caching
export REDIS_URL=""
```

#### 2. Import errors
```
ImportError: No module named 'src.tools.bright_data'
```

**Solution:**
```bash
# Make sure you're in the correct directory
pwd  # Should show .../DataMCPServerAgent

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. API key issues
```
AuthenticationException: Authentication failed
```

**Solution:**
```bash
# Check API key
echo $BRIGHT_DATA_API_KEY

# Set correct key
export BRIGHT_DATA_API_KEY="your_actual_api_key"
```

#### 4. Rate limiting issues
```
RateLimitException: Rate limit exceeded
```

**Solution:**
```python
# Increase limits in configuration
config.rate_limit.requests_per_minute = 120
config.rate_limit.burst_size = 20
```

### Diagnostic Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Or only for Bright Data components
logging.getLogger('src.tools.bright_data').setLevel(logging.DEBUG)
```

## 📊 Monitoring and Metrics

### Getting Metrics

```python
# Client metrics
metrics = client.get_metrics()
print(f"Success rate: {metrics['success_rate']:.2f}%")
print(f"Total requests: {metrics['total_requests']}")

# Cache metrics
cache_stats = cache_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate_percentage']:.2f}%")

# Rate limiting metrics
rate_stats = rate_limiter.get_global_stats()
print(f"Rejected requests: {rate_stats['rejected_requests']}")
```

### Health Check

```python
# System health check
health = await client.health_check()
if health["status"] == "healthy":
    print("✅ System is healthy")
else:
    print(f"❌ System issue: {health.get('error', 'Unknown')}")
```

## 🔄 Updates and Maintenance

### Component Updates

```bash
# Update dependencies
pip install --upgrade aiohttp redis

# Or with uv
uv pip install --upgrade aiohttp redis
```

### Configuration Backup

```bash
# Create backup of current configuration
cp configs/bright_data_config.json configs/bright_data_config.backup.json

# Or export environment variables
env | grep BRIGHT_DATA > bright_data_env.backup
```

## 📞 Support

If you encounter issues:

1. Check [Troubleshooting Guide](BRIGHT_DATA_TROUBLESHOOTING.md)
2. Run diagnostic script: `python scripts/test_bright_data_enhanced.py`
3. Check logs with DEBUG level
4. Create issue in GitHub repository with detailed problem description

## 🎯 Next Steps

After successful setup:

1. Read [Advanced Features Guide](BRIGHT_DATA_ADVANCED.md)
2. Study [API Reference](BRIGHT_DATA_API.md)
3. Try [Examples](../examples/)
4. Set up [Monitoring Dashboard](BRIGHT_DATA_MONITORING.md)
