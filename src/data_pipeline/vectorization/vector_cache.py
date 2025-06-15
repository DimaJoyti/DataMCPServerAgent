"""
Vector caching system for embedding results.
"""

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from .embeddings.base_embedder import EmbeddingResult


class CacheConfig(BaseModel):
    """Configuration for vector caching."""

    # Cache backend
    backend: str = Field(default="memory", description="Cache backend (memory, file, redis)")

    # Cache settings
    ttl: int = Field(default=86400, description="Time to live in seconds (24 hours)")
    max_size: int = Field(default=10000, description="Maximum number of cached items")

    # File backend settings
    cache_dir: str = Field(default="cache/vectors", description="Cache directory for file backend")

    # Redis backend settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: Optional[str] = Field(None, description="Redis password")
    redis_prefix: str = Field(default="vector_cache:", description="Redis key prefix")

    # Serialization
    compression: bool = Field(default=True, description="Enable compression")

    # Performance
    enable_stats: bool = Field(default=True, description="Enable cache statistics")


class CacheStats(BaseModel):
    """Cache statistics."""

    hits: int = Field(default=0, description="Cache hits")
    misses: int = Field(default=0, description="Cache misses")
    sets: int = Field(default=0, description="Cache sets")
    deletes: int = Field(default=0, description="Cache deletes")
    evictions: int = Field(default=0, description="Cache evictions")

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""

    def __init__(self, config: CacheConfig):
        """Initialize cache backend."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = CacheStats() if config.enable_stats else None

    @abstractmethod
    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get embedding result from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: EmbeddingResult) -> None:
        """Set embedding result in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete embedding result from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of items in cache."""
        pass

    def get_stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        return self.stats


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend."""

    def __init__(self, config: CacheConfig):
        """Initialize memory cache."""
        super().__init__(config)
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)

    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get embedding result from memory cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]

            # Check if expired
            if time.time() - timestamp > self.config.ttl:
                del self._cache[key]
                if self.stats:
                    self.stats.misses += 1
                return None

            if self.stats:
                self.stats.hits += 1
            return value

        if self.stats:
            self.stats.misses += 1
        return None

    def set(self, key: str, value: EmbeddingResult) -> None:
        """Set embedding result in memory cache."""
        # Check size limit
        if len(self._cache) >= self.config.max_size:
            self._evict_oldest()

        self._cache[key] = (value, time.time())

        if self.stats:
            self.stats.sets += 1

    def delete(self, key: str) -> bool:
        """Delete embedding result from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if self.stats:
                self.stats.deletes += 1
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

        if self.stats:
            self.stats.evictions += 1


class FileCacheBackend(BaseCacheBackend):
    """File-based cache backend."""

    def __init__(self, config: CacheConfig):
        """Initialize file cache."""
        super().__init__(config)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file to track cache entries
        self.index_file = self.cache_dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    self._index = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use first 2 characters for subdirectory
        subdir = key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{key}.pkl"

    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get embedding result from file cache."""
        if key not in self._index:
            if self.stats:
                self.stats.misses += 1
            return None

        timestamp = self._index[key]

        # Check if expired
        if time.time() - timestamp > self.config.ttl:
            self.delete(key)
            if self.stats:
                self.stats.misses += 1
            return None

        # Load from file
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # Remove from index if file doesn't exist
            del self._index[key]
            self._save_index()
            if self.stats:
                self.stats.misses += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)

            if self.stats:
                self.stats.hits += 1
            return value
        except Exception as e:
            self.logger.error(f"Failed to load cache file {cache_path}: {e}")
            self.delete(key)
            if self.stats:
                self.stats.misses += 1
            return None

    def set(self, key: str, value: EmbeddingResult) -> None:
        """Set embedding result in file cache."""
        # Check size limit
        if len(self._index) >= self.config.max_size:
            self._evict_oldest()

        # Save to file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)

            # Update index
            self._index[key] = time.time()
            self._save_index()

            if self.stats:
                self.stats.sets += 1
        except Exception as e:
            self.logger.error(f"Failed to save cache file {cache_path}: {e}")

    def delete(self, key: str) -> bool:
        """Delete embedding result from file cache."""
        if key not in self._index:
            return False

        # Remove file
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            self.logger.error(f"Failed to delete cache file {cache_path}: {e}")

        # Remove from index
        del self._index[key]
        self._save_index()

        if self.stats:
            self.stats.deletes += 1
        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        # Remove all cache files
        for key in list(self._index.keys()):
            self.delete(key)

        self._index.clear()
        self._save_index()

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._index)

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._index:
            return

        # Find oldest entry
        oldest_key = min(self._index.keys(), key=lambda k: self._index[k])
        self.delete(oldest_key)

        if self.stats:
            self.stats.evictions += 1


class VectorCache:
    """Vector cache for embedding results."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize vector cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize backend
        self.backend = self._create_backend()

    def _create_backend(self) -> BaseCacheBackend:
        """Create cache backend based on configuration."""
        if self.config.backend == "memory":
            return MemoryCacheBackend(self.config)
        elif self.config.backend == "file":
            return FileCacheBackend(self.config)
        elif self.config.backend == "redis":
            try:
                from .redis_cache_backend import RedisCacheBackend

                return RedisCacheBackend(self.config)
            except ImportError:
                self.logger.warning("Redis not available, falling back to memory cache")
                return MemoryCacheBackend(self.config)
        else:
            raise ValueError(f"Unknown cache backend: {self.config.backend}")

    def get(self, text_hash: str) -> Optional[EmbeddingResult]:
        """
        Get embedding result from cache.

        Args:
            text_hash: Hash of the text

        Returns:
            Optional[EmbeddingResult]: Cached embedding result
        """
        try:
            result = self.backend.get(text_hash)
            if result:
                result.from_cache = True
            return result
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    def set(self, text_hash: str, result: EmbeddingResult) -> None:
        """
        Set embedding result in cache.

        Args:
            text_hash: Hash of the text
            result: Embedding result to cache
        """
        try:
            self.backend.set(text_hash, result)
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")

    def delete(self, text_hash: str) -> bool:
        """
        Delete embedding result from cache.

        Args:
            text_hash: Hash of the text

        Returns:
            bool: True if deleted
        """
        try:
            return self.backend.delete(text_hash)
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.backend.clear()
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")

    def size(self) -> int:
        """Get number of items in cache."""
        try:
            return self.backend.size()
        except Exception as e:
            self.logger.error(f"Cache size error: {e}")
            return 0

    def get_stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        return self.backend.get_stats()

    def health_check(self) -> bool:
        """Perform cache health check."""
        try:
            # Try to set and get a test value
            test_key = "health_check_test"
            test_result = EmbeddingResult(
                text="test",
                text_hash=test_key,
                embedding=[0.1, 0.2, 0.3],
                embedding_dimension=3,
                model_name="test",
                model_provider="test",
                processing_time=0.0,
            )

            self.set(test_key, test_result)
            retrieved = self.get(test_key)
            self.delete(test_key)

            return retrieved is not None
        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return False
