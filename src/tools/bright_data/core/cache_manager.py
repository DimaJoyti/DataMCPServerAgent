"""
Advanced caching system for Bright Data MCP Integration

This module provides multi-level caching with:
- Memory cache (LRU)
- Redis distributed cache
- Compression support
- TTL management
- Cache warming strategies
- Metrics and monitoring
"""

import asyncio
import json
import gzip
import hashlib
import time
import logging
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass
from collections import OrderedDict
from abc import ABC, abstractmethod

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: float
    ttl: Optional[int]
    access_count: int = 0
    last_access: float = 0
    compressed: bool = False
    size_bytes: int = 0

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0

class CacheBackend(ABC):
    """Abstract cache backend interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass

class MemoryCache(CacheBackend):
    """In-memory LRU cache implementation"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self.cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None

            # Update access info and move to end (LRU)
            entry.access_count += 1
            entry.last_access = time.time()
            self.cache.move_to_end(key)

            self.stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))

            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )

            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size -= old_entry.size_bytes
                del self.cache[key]

            # Evict entries if necessary
            while len(self.cache) >= self.max_size:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.stats.total_size -= oldest_entry.size_bytes
                self.stats.evictions += 1

            # Add new entry
            self.cache[key] = entry
            self.stats.total_size += size_bytes
            self.stats.sets += 1
            self.stats.entry_count = len(self.cache)

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size -= entry.size_bytes
                del self.cache[key]
                self.stats.deletes += 1
                self.stats.entry_count = len(self.cache)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        async with self._lock:
            if key not in self.cache:
                return False

            entry = self.cache[key]
            # Check TTL
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self.cache[key]
                self.stats.evictions += 1
                return False

            return True

    async def clear(self) -> bool:
        """Clear all entries from memory cache"""
        async with self._lock:
            self.cache.clear()
            self.stats = CacheStats()
            return True

    def get_stats(self) -> CacheStats:
        """Get memory cache statistics"""
        self.stats.entry_count = len(self.cache)
        return self.stats

class RedisCache(CacheBackend):
    """Redis distributed cache implementation"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 key_prefix: str = "bright_data:", default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed key"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)

            data = await r.get(prefixed_key)
            if data is None:
                self.stats.misses += 1
                return None

            # Deserialize
            value = json.loads(data.decode('utf-8'))
            self.stats.hits += 1
            return value

        except Exception as e:
            logging.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)

            # Serialize
            data = json.dumps(value).encode('utf-8')

            # Set with TTL
            ttl_seconds = ttl or self.default_ttl
            await r.setex(prefixed_key, ttl_seconds, data)

            self.stats.sets += 1
            return True

        except Exception as e:
            logging.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)

            result = await r.delete(prefixed_key)
            if result > 0:
                self.stats.deletes += 1
                return True
            return False

        except Exception as e:
            logging.error(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)

            result = await r.exists(prefixed_key)
            return result > 0

        except Exception as e:
            logging.error(f"Redis exists error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all entries with prefix from Redis cache"""
        try:
            r = await self._get_redis()
            pattern = f"{self.key_prefix}*"

            keys = await r.keys(pattern)
            if keys:
                await r.delete(*keys)

            self.stats = CacheStats()
            return True

        except Exception as e:
            logging.error(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Get Redis cache statistics"""
        return self.stats

    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()

class CacheManager:
    """Multi-level cache manager with compression and warming"""

    def __init__(self, memory_cache: Optional[MemoryCache] = None,
                 redis_cache: Optional[RedisCache] = None,
                 compression_threshold: int = 1024,
                 enable_compression: bool = True):

        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        self.logger = logging.getLogger(__name__)

        # Cache warming
        self.warming_functions: Dict[str, Callable] = {}
        self.warming_schedules: Dict[str, float] = {}

        # Metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _should_compress(self, data: str) -> bool:
        """Check if data should be compressed"""
        return (self.enable_compression and
                len(data.encode('utf-8')) > self.compression_threshold)

    def _compress_data(self, data: str) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data.encode('utf-8'))

    def _decompress_data(self, data: bytes) -> str:
        """Decompress data using gzip"""
        return gzip.decompress(data).decode('utf-8')

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)"""
        self.total_requests += 1

        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.cache_hits += 1
            return value

        # Try Redis cache if available
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                await self.memory_cache.set(key, value)
                self.cache_hits += 1
                return value

        self.cache_misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (both memory and Redis)"""
        success = True

        # Set in memory cache
        memory_success = await self.memory_cache.set(key, value, ttl)
        if not memory_success:
            success = False

        # Set in Redis cache if available
        if self.redis_cache:
            redis_success = await self.redis_cache.set(key, value, ttl)
            if not redis_success:
                success = False

        return success

    async def delete(self, key: str) -> bool:
        """Delete value from cache (both memory and Redis)"""
        success = True

        # Delete from memory cache
        memory_success = await self.memory_cache.delete(key)

        # Delete from Redis cache if available
        if self.redis_cache:
            redis_success = await self.redis_cache.delete(key)
            success = memory_success or redis_success
        else:
            success = memory_success

        return success

    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level"""
        # Check memory cache first
        if await self.memory_cache.exists(key):
            return True

        # Check Redis cache if available
        if self.redis_cache:
            return await self.redis_cache.exists(key)

        return False

    async def clear(self) -> bool:
        """Clear all cache levels"""
        success = True

        # Clear memory cache
        memory_success = await self.memory_cache.clear()
        if not memory_success:
            success = False

        # Clear Redis cache if available
        if self.redis_cache:
            redis_success = await self.redis_cache.clear()
            if not redis_success:
                success = False

        # Reset metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        return success

    def cache_result(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, *args, **kwargs)

                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)

                return result

            return wrapper
        return decorator

    def register_warming_function(self, name: str, func: Callable, interval: float = 3600):
        """Register a function for cache warming"""
        self.warming_functions[name] = func
        self.warming_schedules[name] = interval

    async def warm_cache(self, name: Optional[str] = None) -> None:
        """Warm cache using registered functions"""
        if name:
            if name in self.warming_functions:
                try:
                    await self.warming_functions[name]()
                    self.logger.info(f"Cache warming completed for {name}")
                except Exception as e:
                    self.logger.error(f"Cache warming failed for {name}: {e}")
        else:
            # Warm all registered functions
            for func_name, func in self.warming_functions.items():
                try:
                    await func()
                    self.logger.info(f"Cache warming completed for {func_name}")
                except Exception as e:
                    self.logger.error(f"Cache warming failed for {func_name}: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats() if self.redis_cache else CacheStats()

        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percentage": hit_rate,
            "memory_cache": {
                "hits": memory_stats.hits,
                "misses": memory_stats.misses,
                "sets": memory_stats.sets,
                "deletes": memory_stats.deletes,
                "evictions": memory_stats.evictions,
                "entry_count": memory_stats.entry_count,
                "total_size_bytes": memory_stats.total_size,
            },
            "redis_cache": {
                "hits": redis_stats.hits,
                "misses": redis_stats.misses,
                "sets": redis_stats.sets,
                "deletes": redis_stats.deletes,
                "available": self.redis_cache is not None,
            } if self.redis_cache else {"available": False}
        }

    async def close(self) -> None:
        """Close cache connections"""
        if self.redis_cache:
            await self.redis_cache.close()
