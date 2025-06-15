"""
Memory-efficient bounded collections for DataMCPServerAgent.
Provides data structures with automatic size limits and cleanup.
"""

import time
import weakref
from collections import OrderedDict, deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import threading
import logging

logger = logging.getLogger(__name__)


class BoundedDict:
    """Dictionary with automatic size limiting and LRU eviction."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 eviction_callback: Optional[Callable[[Any, Any], None]] = None,
                 ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.eviction_callback = eviction_callback
        self.ttl_seconds = ttl_seconds
        
        self._data = OrderedDict()
        self._access_times = {} if ttl_seconds else None
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def __setitem__(self, key: Any, value: Any):
        """Set item with automatic eviction if needed."""
        with self._lock:
            current_time = time.time() if self.ttl_seconds else None
            
            # Remove existing key to update order
            if key in self._data:
                del self._data[key]
            
            # Evict oldest items if at capacity
            while len(self._data) >= self.max_size:
                self._evict_oldest()
            
            # Add new item
            self._data[key] = value
            if self._access_times is not None:
                self._access_times[key] = current_time
    
    def __getitem__(self, key: Any) -> Any:
        """Get item and move to end (LRU)."""
        with self._lock:
            if key not in self._data:
                self._misses += 1
                raise KeyError(key)
            
            # Check TTL if enabled
            if self._access_times is not None:
                if self._is_expired(key):
                    del self[key]
                    self._misses += 1
                    raise KeyError(key)
                
                # Update access time
                self._access_times[key] = time.time()
            
            # Move to end (most recently used)
            value = self._data.pop(key)
            self._data[key] = value
            self._hits += 1
            return value
    
    def __delitem__(self, key: Any):
        """Delete item."""
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
                if self._access_times is not None:
                    self._access_times.pop(key, None)
                
                # Call eviction callback
                if self.eviction_callback:
                    try:
                        self.eviction_callback(key, value)
                    except Exception as e:
                        logger.warning(f"Eviction callback error: {e}")
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._data:
                return False
            
            if self._access_times is not None and self._is_expired(key):
                del self[key]
                return False
            
            return True
    
    def __len__(self) -> int:
        """Get number of items."""
        with self._lock:
            self._cleanup_expired()
            return len(self._data)
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys."""
        with self._lock:
            self._cleanup_expired()
            return iter(list(self._data.keys()))
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item with default value."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: Any, default: Any = None) -> Any:
        """Pop item with default value."""
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
                if self._access_times is not None:
                    self._access_times.pop(key, None)
                return value
            return default
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            # Call eviction callbacks
            if self.eviction_callback:
                for key, value in self._data.items():
                    try:
                        self.eviction_callback(key, value)
                    except Exception as e:
                        logger.warning(f"Eviction callback error: {e}")
            
            self._data.clear()
            if self._access_times is not None:
                self._access_times.clear()
            
            self._evictions += len(self._data)
    
    def keys(self):
        """Get keys view."""
        with self._lock:
            self._cleanup_expired()
            return self._data.keys()
    
    def values(self):
        """Get values view."""
        with self._lock:
            self._cleanup_expired()
            return self._data.values()
    
    def items(self):
        """Get items view."""
        with self._lock:
            self._cleanup_expired()
            return self._data.items()
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) item."""
        if not self._data:
            return
        
        key, value = self._data.popitem(last=False)
        if self._access_times is not None:
            self._access_times.pop(key, None)
        
        self._evictions += 1
        
        # Call eviction callback
        if self.eviction_callback:
            try:
                self.eviction_callback(key, value)
            except Exception as e:
                logger.warning(f"Eviction callback error: {e}")
    
    def _is_expired(self, key: Any) -> bool:
        """Check if key is expired."""
        if self._access_times is None or self.ttl_seconds is None:
            return False
        
        access_time = self._access_times.get(key, 0)
        return (time.time() - access_time) > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove all expired items."""
        if self._access_times is None:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if (current_time - access_time) > self.ttl_seconds
        ]
        
        for key in expired_keys:
            try:
                del self[key]
            except KeyError:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0
            
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "has_ttl": self.ttl_seconds is not None,
                "ttl_seconds": self.ttl_seconds
            }


class BoundedList:
    """List with automatic size limiting and configurable eviction strategy."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 eviction_strategy: str = "fifo",  # "fifo", "lifo", "random"
                 eviction_callback: Optional[Callable[[Any], None]] = None):
        self.max_size = max_size
        self.eviction_strategy = eviction_strategy
        self.eviction_callback = eviction_callback
        
        self._data = deque(maxlen=max_size if eviction_strategy == "fifo" else None)
        self._lock = threading.RLock()
        
        # Statistics
        self._evictions = 0
    
    def append(self, item: Any):
        """Append item with automatic eviction."""
        with self._lock:
            if self.eviction_strategy == "fifo":
                # deque handles this automatically with maxlen
                if len(self._data) == self.max_size:
                    evicted = self._data[0] if self._data else None
                    if evicted is not None and self.eviction_callback:
                        try:
                            self.eviction_callback(evicted)
                        except Exception as e:
                            logger.warning(f"Eviction callback error: {e}")
                    self._evictions += 1
                
                self._data.append(item)
            else:
                # Manual size management for other strategies
                while len(self._data) >= self.max_size:
                    self._evict_item()
                
                self._data.append(item)
    
    def extend(self, items: List[Any]):
        """Extend with multiple items."""
        for item in items:
            self.append(item)
    
    def pop(self, index: int = -1) -> Any:
        """Pop item at index."""
        with self._lock:
            if not self._data:
                raise IndexError("pop from empty list")
            
            if self.eviction_strategy == "fifo":
                return self._data.pop() if index == -1 else self._data.popleft()
            else:
                return self._data.pop(index)
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            if self.eviction_callback:
                for item in self._data:
                    try:
                        self.eviction_callback(item)
                    except Exception as e:
                        logger.warning(f"Eviction callback error: {e}")
            
            self._evictions += len(self._data)
            self._data.clear()
    
    def __len__(self) -> int:
        """Get length."""
        return len(self._data)
    
    def __getitem__(self, index: Union[int, slice]) -> Any:
        """Get item by index."""
        return self._data[index]
    
    def __setitem__(self, index: int, value: Any):
        """Set item by index."""
        with self._lock:
            self._data[index] = value
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over items."""
        return iter(list(self._data))
    
    def __contains__(self, item: Any) -> bool:
        """Check if item is in list."""
        return item in self._data
    
    def _evict_item(self):
        """Evict item based on strategy."""
        if not self._data:
            return
        
        if self.eviction_strategy == "lifo":
            evicted = self._data.pop()
        elif self.eviction_strategy == "fifo":
            evicted = self._data.popleft()
        elif self.eviction_strategy == "random":
            import random
            index = random.randint(0, len(self._data) - 1)
            evicted = self._data[index]
            del self._data[index]
        else:
            evicted = self._data.popleft()  # Default to FIFO
        
        self._evictions += 1
        
        if self.eviction_callback:
            try:
                self.eviction_callback(evicted)
            except Exception as e:
                logger.warning(f"Eviction callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "evictions": self._evictions,
            "eviction_strategy": self.eviction_strategy
        }


class BoundedSet:
    """Set with automatic size limiting."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 eviction_callback: Optional[Callable[[Any], None]] = None):
        self.max_size = max_size
        self.eviction_callback = eviction_callback
        
        self._data = OrderedDict()  # Use OrderedDict to maintain insertion order
        self._lock = threading.RLock()
        
        # Statistics
        self._evictions = 0
    
    def add(self, item: Any):
        """Add item to set."""
        with self._lock:
            if item in self._data:
                # Move to end (most recently added)
                del self._data[item]
            
            # Evict oldest items if at capacity
            while len(self._data) >= self.max_size:
                self._evict_oldest()
            
            self._data[item] = None
    
    def remove(self, item: Any):
        """Remove item from set."""
        with self._lock:
            if item not in self._data:
                raise KeyError(item)
            del self._data[item]
    
    def discard(self, item: Any):
        """Remove item if present."""
        with self._lock:
            self._data.pop(item, None)
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            if self.eviction_callback:
                for item in self._data:
                    try:
                        self.eviction_callback(item)
                    except Exception as e:
                        logger.warning(f"Eviction callback error: {e}")
            
            self._evictions += len(self._data)
            self._data.clear()
    
    def __contains__(self, item: Any) -> bool:
        """Check if item is in set."""
        return item in self._data
    
    def __len__(self) -> int:
        """Get size."""
        return len(self._data)
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over items."""
        return iter(list(self._data.keys()))
    
    def _evict_oldest(self):
        """Evict the oldest item."""
        if not self._data:
            return
        
        item = next(iter(self._data))
        del self._data[item]
        self._evictions += 1
        
        if self.eviction_callback:
            try:
                self.eviction_callback(item)
            except Exception as e:
                logger.warning(f"Eviction callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "evictions": self._evictions
        }


class WeakBoundedDict:
    """Bounded dictionary with weak references to values."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._data = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._weak_cleanups = 0
    
    def __setitem__(self, key: Any, value: Any):
        """Set item with weak reference."""
        with self._lock:
            # Clean up dead references
            self._cleanup_dead_refs()
            
            # Remove existing key
            if key in self._data:
                del self._data[key]
            
            # Evict if needed
            while len(self._data) >= self.max_size:
                self._evict_oldest()
            
            # Create weak reference
            def cleanup_callback(ref):
                with self._lock:
                    if key in self._data and self._data[key] is ref:
                        del self._data[key]
                        self._weak_cleanups += 1
            
            weak_ref = weakref.ref(value, cleanup_callback)
            self._data[key] = weak_ref
    
    def __getitem__(self, key: Any) -> Any:
        """Get item and check if reference is still alive."""
        with self._lock:
            if key not in self._data:
                self._misses += 1
                raise KeyError(key)
            
            weak_ref = self._data[key]
            value = weak_ref()
            
            if value is None:
                # Reference died
                del self._data[key]
                self._misses += 1
                self._weak_cleanups += 1
                raise KeyError(key)
            
            # Move to end (LRU)
            del self._data[key]
            self._data[key] = weak_ref
            self._hits += 1
            return value
    
    def __delitem__(self, key: Any):
        """Delete item."""
        with self._lock:
            if key in self._data:
                del self._data[key]
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists and reference is alive."""
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def __len__(self) -> int:
        """Get number of live references."""
        with self._lock:
            self._cleanup_dead_refs()
            return len(self._data)
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item with default."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def _evict_oldest(self):
        """Evict oldest item."""
        if self._data:
            key, _ = self._data.popitem(last=False)
            self._evictions += 1
    
    def _cleanup_dead_refs(self):
        """Remove dead weak references."""
        dead_keys = [
            key for key, weak_ref in self._data.items()
            if weak_ref() is None
        ]
        
        for key in dead_keys:
            del self._data[key]
            self._weak_cleanups += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        with self._lock:
            self._cleanup_dead_refs()
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0
            
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "weak_cleanups": self._weak_cleanups
            }


# Factory functions for easy creation
def create_lru_cache(max_size: int = 1000, ttl_seconds: Optional[float] = None) -> BoundedDict:
    """Create an LRU cache with optional TTL."""
    return BoundedDict(max_size=max_size, ttl_seconds=ttl_seconds)

def create_bounded_list(max_size: int = 1000, strategy: str = "fifo") -> BoundedList:
    """Create a bounded list with specified eviction strategy."""
    return BoundedList(max_size=max_size, eviction_strategy=strategy)

def create_bounded_set(max_size: int = 1000) -> BoundedSet:
    """Create a bounded set."""
    return BoundedSet(max_size=max_size)

def create_weak_cache(max_size: int = 1000) -> WeakBoundedDict:
    """Create a weak reference cache."""
    return WeakBoundedDict(max_size=max_size)


# Example usage and testing
if __name__ == "__main__":
    print("Testing bounded collections...")
    
    # Test BoundedDict
    cache = BoundedDict(max_size=3)
    cache["a"] = "value_a"
    cache["b"] = "value_b"
    cache["c"] = "value_c"
    cache["d"] = "value_d"  # Should evict "a"
    
    print(f"Cache size: {len(cache)}")
    print(f"Cache keys: {list(cache.keys())}")
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test BoundedList
    blist = BoundedList(max_size=3, eviction_strategy="fifo")
    blist.extend([1, 2, 3, 4, 5])  # Should keep only [3, 4, 5]
    
    print(f"List contents: {list(blist)}")
    print(f"List stats: {blist.get_stats()}")
    
    print("Bounded collections test completed.")