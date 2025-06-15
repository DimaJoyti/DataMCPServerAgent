"""
Memory monitoring utilities for DataMCPServerAgent.
Provides real-time memory usage tracking, optimization suggestions, and automatic cleanup.
"""

import gc
import logging
import os
import sys
import time
import weakref
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import functools

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False

logger = logging.getLogger(__name__)


class MemoryStats:
    """Container for memory statistics."""
    
    def __init__(self):
        self.rss_mb: float = 0.0
        self.vms_mb: float = 0.0
        self.percent: float = 0.0
        self.available_mb: float = 0.0
        self.gc_count: Tuple[int, int, int] = (0, 0, 0)
        self.object_count: int = 0
        self.timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rss_mb": self.rss_mb,
            "vms_mb": self.vms_mb,
            "percent": self.percent,
            "available_mb": self.available_mb,
            "gc_count": self.gc_count,
            "object_count": self.object_count,
            "timestamp": self.timestamp
        }


class MemoryMonitor:
    """Real-time memory usage monitor with optimization features."""
    
    def __init__(self, 
                 threshold_mb: int = 1000,
                 critical_threshold_mb: int = 2000,
                 check_interval: float = 10.0,
                 history_size: int = 100):
        self.threshold_mb = threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.check_interval = check_interval
        self.history_size = history_size
        
        self.logger = logging.getLogger(f"{__name__}.MemoryMonitor")
        self._history: deque = deque(maxlen=history_size)
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks: List[Callable[[MemoryStats], None]] = []
        self._large_objects: weakref.WeakSet = weakref.WeakSet()
        
        # Statistics
        self._gc_forced_count = 0
        self._cleanup_actions = 0
        self._peak_memory = 0.0
        
        if not HAS_PSUTIL:
            self.logger.warning("psutil not available. Limited memory monitoring.")
    
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        stats = MemoryStats()
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                stats.rss_mb = memory_info.rss / 1024 / 1024
                stats.vms_mb = memory_info.vms / 1024 / 1024
                stats.percent = memory_percent
                
                # System memory info
                system_memory = psutil.virtual_memory()
                stats.available_mb = system_memory.available / 1024 / 1024
                
                # Update peak memory
                if stats.rss_mb > self._peak_memory:
                    self._peak_memory = stats.rss_mb
                    
            except Exception as e:
                self.logger.warning(f"Error getting process memory info: {e}")
        
        # GC statistics
        stats.gc_count = gc.get_count()
        stats.object_count = len(gc.get_objects())
        
        return stats
    
    def add_callback(self, callback: Callable[[MemoryStats], None]):
        """Add callback to be called when memory stats are updated."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemoryStats], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self.logger.info(f"Started memory monitoring (threshold: {self.threshold_mb}MB)")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped memory monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_current_stats()
                self._history.append(stats)
                
                # Check thresholds and trigger actions
                if stats.rss_mb > self.critical_threshold_mb:
                    self._handle_critical_memory(stats)
                elif stats.rss_mb > self.threshold_mb:
                    self._handle_high_memory(stats)
                
                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        self.logger.warning(f"Memory callback error: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _handle_high_memory(self, stats: MemoryStats):
        """Handle high memory usage."""
        self.logger.warning(
            f"High memory usage detected: {stats.rss_mb:.2f}MB "
            f"(threshold: {self.threshold_mb}MB)"
        )
        
        # Trigger garbage collection
        self.force_garbage_collection()
    
    def _handle_critical_memory(self, stats: MemoryStats):
        """Handle critical memory usage."""
        self.logger.error(
            f"CRITICAL memory usage: {stats.rss_mb:.2f}MB "
            f"(critical threshold: {self.critical_threshold_mb}MB)"
        )
        
        # Aggressive cleanup
        self.aggressive_cleanup()
        
        # Log largest objects
        self._log_large_objects()
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_count = gc.get_count()
        
        # Collect all generations
        collected = {}
        for generation in range(3):
            collected[f"gen_{generation}"] = gc.collect(generation)
        
        after_count = gc.get_count()
        self._gc_forced_count += 1
        
        self.logger.debug(
            f"Forced GC: collected {sum(collected.values())} objects, "
            f"counts: {before_count} -> {after_count}"
        )
        
        return collected
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        self._cleanup_actions += 1
        
        # Multiple GC passes
        for _ in range(3):
            self.force_garbage_collection()
        
        # Clear module-level caches if available
        try:
            # Clear LRU caches
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                    try:
                        obj.cache_clear()
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Error during aggressive cleanup: {e}")
        
        self.logger.info("Performed aggressive memory cleanup")
    
    def _log_large_objects(self):
        """Log information about large objects."""
        if not HAS_TRACEMALLOC:
            return
        
        try:
            # Get memory usage by object type
            object_types = defaultdict(int)
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_types[obj_type] += 1
            
            # Log top 10 most common object types
            top_types = sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.warning(f"Top object types: {top_types}")
            
        except Exception as e:
            self.logger.warning(f"Error logging large objects: {e}")
    
    def get_memory_history(self, minutes: int = 10) -> List[MemoryStats]:
        """Get memory history for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [stats for stats in self._history if stats.timestamp >= cutoff_time]
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend."""
        if len(self._history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_stats = list(self._history)[-10:]  # Last 10 measurements
        first_memory = recent_stats[0].rss_mb
        last_memory = recent_stats[-1].rss_mb
        
        trend = "stable"
        if last_memory > first_memory * 1.1:
            trend = "increasing"
        elif last_memory < first_memory * 0.9:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "first_memory_mb": first_memory,
            "last_memory_mb": last_memory,
            "change_mb": last_memory - first_memory,
            "change_percent": ((last_memory - first_memory) / first_memory) * 100,
            "peak_memory_mb": self._peak_memory,
            "measurements": len(recent_stats)
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get memory optimization suggestions."""
        suggestions = []
        current_stats = self.get_current_stats()
        
        # High memory usage
        if current_stats.rss_mb > self.threshold_mb:
            suggestions.append(f"Memory usage ({current_stats.rss_mb:.1f}MB) exceeds threshold ({self.threshold_mb}MB)")
        
        # Too many objects
        if current_stats.object_count > 100000:
            suggestions.append(f"High object count ({current_stats.object_count:,}). Consider object pooling or cleanup.")
        
        # GC pressure
        total_gc = sum(current_stats.gc_count)
        if total_gc > 1000:
            suggestions.append(f"High GC pressure ({total_gc} objects). Consider reducing object creation.")
        
        # Memory trend
        trend = self.get_memory_trend()
        if trend["trend"] == "increasing" and trend["change_percent"] > 20:
            suggestions.append(f"Memory usage increasing rapidly (+{trend['change_percent']:.1f}%). Check for memory leaks.")
        
        # Available system memory
        if current_stats.available_mb < 500:
            suggestions.append(f"Low system memory available ({current_stats.available_mb:.1f}MB). Consider reducing memory usage.")
        
        return suggestions
    
    def register_large_object(self, obj: Any, name: str = None):
        """Register a large object for tracking."""
        self._large_objects.add(obj)
        if name:
            setattr(obj, '_memory_monitor_name', name)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive memory summary report."""
        current_stats = self.get_current_stats()
        trend = self.get_memory_trend()
        suggestions = self.get_optimization_suggestions()
        
        return {
            "current_memory": current_stats.to_dict(),
            "trend_analysis": trend,
            "optimization_suggestions": suggestions,
            "monitoring_stats": {
                "gc_forced_count": self._gc_forced_count,
                "cleanup_actions": self._cleanup_actions,
                "peak_memory_mb": self._peak_memory,
                "history_length": len(self._history),
                "large_objects_tracked": len(self._large_objects)
            },
            "system_info": {
                "has_psutil": HAS_PSUTIL,
                "has_tracemalloc": HAS_TRACEMALLOC,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }


# Decorators for memory monitoring
def memory_profile(threshold_mb: float = 50.0, log_level: int = logging.INFO):
    """Decorator to profile memory usage of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not HAS_PSUTIL:
                return func(*args, **kwargs)
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_diff = memory_after - memory_before
                
                if abs(memory_diff) > threshold_mb:
                    logger.log(
                        log_level,
                        f"Function {func.__name__} memory usage: "
                        f"{memory_diff:+.2f}MB (before: {memory_before:.2f}MB, after: {memory_after:.2f}MB)"
                    )
        
        return wrapper
    return decorator


def memory_limit(max_mb: float):
    """Decorator to enforce memory limits on functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not HAS_PSUTIL:
                return func(*args, **kwargs)
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024 / 1024
            if memory_after > max_mb:
                logger.warning(
                    f"Function {func.__name__} exceeded memory limit: "
                    f"{memory_after:.2f}MB > {max_mb}MB"
                )
                # Force garbage collection
                gc.collect()
            
            return result
        
        return wrapper
    return decorator


# Global memory monitor instance
_global_monitor: Optional[MemoryMonitor] = None

def get_global_monitor(auto_start: bool = True) -> MemoryMonitor:
    """Get or create the global memory monitor."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
        if auto_start:
            _global_monitor.start_monitoring()
    
    return _global_monitor

def log_memory_usage(operation: str, level: int = logging.INFO):
    """Log current memory usage for an operation."""
    monitor = get_global_monitor(auto_start=False)
    stats = monitor.get_current_stats()
    
    logger.log(
        level,
        f"{operation}: Memory usage: {stats.rss_mb:.2f}MB "
        f"({stats.percent:.1f}% of system), {stats.object_count:,} objects"
    )

def check_memory_health() -> Dict[str, Any]:
    """Quick memory health check."""
    monitor = get_global_monitor(auto_start=False)
    return monitor.get_summary_report()

# Context manager for temporary memory monitoring
class MemoryContext:
    """Context manager for monitoring memory usage in a block of code."""
    
    def __init__(self, name: str, threshold_mb: float = 50.0):
        self.name = name
        self.threshold_mb = threshold_mb
        self.memory_before = 0.0
        self.memory_after = 0.0
    
    def __enter__(self):
        if HAS_PSUTIL:
            process = psutil.Process()
            self.memory_before = process.memory_info().rss / 1024 / 1024
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if HAS_PSUTIL:
            process = psutil.Process()
            self.memory_after = process.memory_info().rss / 1024 / 1024
            memory_diff = self.memory_after - self.memory_before
            
            if abs(memory_diff) > self.threshold_mb:
                logger.info(
                    f"Memory usage for {self.name}: "
                    f"{memory_diff:+.2f}MB (before: {self.memory_before:.2f}MB, after: {self.memory_after:.2f}MB)"
                )
    
    @property
    def memory_delta(self) -> float:
        """Get memory usage delta."""
        return self.memory_after - self.memory_before


# Example usage
if __name__ == "__main__":
    # Test memory monitoring
    print("Testing memory monitoring utilities...")
    
    monitor = MemoryMonitor(threshold_mb=100)
    monitor.start_monitoring()
    
    # Get current stats
    stats = monitor.get_current_stats()
    print(f"Current memory: {stats.rss_mb:.2f}MB")
    
    # Test memory context
    with MemoryContext("test_operation") as ctx:
        # Simulate memory usage
        data = [i for i in range(100000)]
        del data
    
    print(f"Memory delta: {ctx.memory_delta:.2f}MB")
    
    # Get optimization suggestions
    suggestions = monitor.get_optimization_suggestions()
    print(f"Optimization suggestions: {suggestions}")
    
    monitor.stop_monitoring()
    print("Memory monitoring test completed.")