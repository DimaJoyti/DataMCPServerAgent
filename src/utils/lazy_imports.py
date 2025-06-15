"""
Lazy import utilities for DataMCPServerAgent.
Provides lazy loading of heavy libraries to reduce memory usage and startup time.
"""

import importlib
import logging
import sys
from typing import Any, Optional, Dict, Set
import weakref

logger = logging.getLogger(__name__)


class LazyLoader:
    """Lazy loader for heavy Python modules."""
    
    def __init__(self, module_name: str, error_msg: Optional[str] = None):
        self.module_name = module_name
        self.error_msg = error_msg or f"Module '{module_name}' not available. Install with: pip install {module_name}"
        self._module = None
        self._loading = False
    
    def __getattr__(self, name: str) -> Any:
        """Load module on first attribute access."""
        if self._module is None and not self._loading:
            self._loading = True
            try:
                logger.debug(f"Lazy loading module: {self.module_name}")
                self._module = importlib.import_module(self.module_name)
                logger.debug(f"Successfully loaded module: {self.module_name}")
            except ImportError as e:
                logger.warning(f"Failed to load module {self.module_name}: {e}")
                raise ImportError(self.error_msg) from e
            finally:
                self._loading = False
        
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs):
        """Support calling the module directly if it's callable."""
        if self._module is None:
            # Trigger loading
            self.__getattr__("__name__")
        return self._module(*args, **kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Check if module is already loaded."""
        return self._module is not None
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage of loaded module in MB."""
        if not self.is_loaded:
            return 0.0
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


class LazyRegistry:
    """Registry for managing lazy loaded modules."""
    
    def __init__(self):
        self._registry: Dict[str, LazyLoader] = {}
        self._loaded_modules: Set[str] = set()
    
    def register(self, name: str, module_name: str, error_msg: Optional[str] = None) -> LazyLoader:
        """Register a module for lazy loading."""
        if name not in self._registry:
            self._registry[name] = LazyLoader(module_name, error_msg)
        return self._registry[name]
    
    def get(self, name: str) -> Optional[LazyLoader]:
        """Get a registered lazy loader."""
        return self._registry.get(name)
    
    def get_loaded_modules(self) -> Set[str]:
        """Get list of currently loaded modules."""
        return {name for name, loader in self._registry.items() if loader.is_loaded}
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get memory usage report for all registered modules."""
        report = {
            "total_registered": len(self._registry),
            "total_loaded": len(self.get_loaded_modules()),
            "modules": {}
        }
        
        for name, loader in self._registry.items():
            report["modules"][name] = {
                "loaded": loader.is_loaded,
                "memory_mb": loader.get_memory_usage() if loader.is_loaded else 0.0
            }
        
        return report


# Global registry instance
_registry = LazyRegistry()

# Register commonly used heavy libraries
def _register_common_libraries():
    """Register commonly used heavy libraries for lazy loading."""
    
    # Data science libraries
    _registry.register(
        "pandas", 
        "pandas", 
        "pandas not available. Install with: pip install pandas"
    )
    
    _registry.register(
        "numpy", 
        "numpy", 
        "numpy not available. Install with: pip install numpy"
    )
    
    _registry.register(
        "scipy", 
        "scipy", 
        "scipy not available. Install with: pip install scipy"
    )
    
    _registry.register(
        "sklearn", 
        "sklearn", 
        "scikit-learn not available. Install with: pip install scikit-learn"
    )
    
    # ML/AI frameworks
    _registry.register(
        "torch", 
        "torch", 
        "PyTorch not available. Install with: pip install torch"
    )
    
    _registry.register(
        "tensorflow", 
        "tensorflow", 
        "TensorFlow not available. Install with: pip install tensorflow"
    )
    
    _registry.register(
        "transformers", 
        "transformers", 
        "Transformers not available. Install with: pip install transformers"
    )
    
    # LangChain libraries
    _registry.register(
        "langchain_anthropic", 
        "langchain_anthropic", 
        "LangChain Anthropic not available. Install with: pip install langchain-anthropic"
    )
    
    _registry.register(
        "langchain_core", 
        "langchain_core", 
        "LangChain Core not available. Install with: pip install langchain-core"
    )
    
    _registry.register(
        "langchain_community", 
        "langchain_community", 
        "LangChain Community not available. Install with: pip install langchain-community"
    )
    
    # Database libraries
    _registry.register(
        "sqlalchemy", 
        "sqlalchemy", 
        "SQLAlchemy not available. Install with: pip install sqlalchemy"
    )
    
    _registry.register(
        "pymongo", 
        "pymongo", 
        "PyMongo not available. Install with: pip install pymongo"
    )
    
    # Other heavy libraries
    _registry.register(
        "cv2", 
        "cv2", 
        "OpenCV not available. Install with: pip install opencv-python"
    )
    
    _registry.register(
        "PIL", 
        "PIL", 
        "Pillow not available. Install with: pip install Pillow"
    )


# Initialize common libraries
_register_common_libraries()

# Expose lazy loaders as module attributes
pandas = _registry.get("pandas")
numpy = _registry.get("numpy")
scipy = _registry.get("scipy")
sklearn = _registry.get("sklearn")
torch = _registry.get("torch")
tensorflow = _registry.get("tensorflow")
transformers = _registry.get("transformers")
langchain_anthropic = _registry.get("langchain_anthropic")
langchain_core = _registry.get("langchain_core")
langchain_community = _registry.get("langchain_community")
sqlalchemy = _registry.get("sqlalchemy")
pymongo = _registry.get("pymongo")
cv2 = _registry.get("cv2")
PIL = _registry.get("PIL")

# Convenience functions
def register_lazy_module(name: str, module_name: str, error_msg: Optional[str] = None) -> LazyLoader:
    """Register a custom module for lazy loading."""
    return _registry.register(name, module_name, error_msg)

def get_lazy_module(name: str) -> Optional[LazyLoader]:
    """Get a registered lazy loader by name."""
    return _registry.get(name)

def get_memory_report() -> Dict[str, Any]:
    """Get memory usage report for all lazy loaded modules."""
    return _registry.get_memory_report()

def get_loaded_modules() -> Set[str]:
    """Get set of currently loaded module names."""
    return _registry.get_loaded_modules()

def force_load_module(name: str) -> bool:
    """Force load a registered module."""
    loader = _registry.get(name)
    if loader:
        try:
            # Access an attribute to trigger loading
            loader.__name__
            return True
        except ImportError:
            return False
    return False

def preload_essential_modules():
    """Preload essential modules for better performance."""
    essential = ["numpy", "pandas"]  # Add modules that are always needed
    
    for module_name in essential:
        try:
            force_load_module(module_name)
            logger.info(f"Preloaded essential module: {module_name}")
        except Exception as e:
            logger.warning(f"Failed to preload essential module {module_name}: {e}")


# Memory optimization utilities
def get_import_memory_impact() -> Dict[str, float]:
    """Get memory impact of importing heavy modules."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        impact = {}
        for name, loader in _registry._registry.items():
            if not loader.is_loaded:
                memory_before = process.memory_info().rss / 1024 / 1024
                try:
                    force_load_module(name)
                    memory_after = process.memory_info().rss / 1024 / 1024
                    impact[name] = memory_after - memory_before
                except ImportError:
                    impact[name] = 0.0
        
        return impact
    except ImportError:
        logger.warning("psutil not available for memory impact analysis")
        return {}


# Context manager for temporary module loading
class TemporaryModuleLoader:
    """Context manager for temporarily loading heavy modules."""
    
    def __init__(self, *module_names: str):
        self.module_names = module_names
        self.loaded_modules = []
    
    def __enter__(self):
        for name in self.module_names:
            if force_load_module(name):
                self.loaded_modules.append(name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Note: Python doesn't support unloading modules reliably
        # This is mainly for tracking purposes
        pass
    
    def get_modules(self) -> Dict[str, Any]:
        """Get the loaded modules."""
        modules = {}
        for name in self.loaded_modules:
            loader = _registry.get(name)
            if loader and loader.is_loaded:
                modules[name] = loader
        return modules


# Example usage and testing
if __name__ == "__main__":
    # Test lazy loading
    print("Testing lazy import utilities...")
    
    # Check initial state
    print(f"Loaded modules: {get_loaded_modules()}")
    
    # Test lazy loading
    try:
        np = numpy
        print(f"Accessing numpy: {np.version.version}")
        print(f"Loaded modules after numpy: {get_loaded_modules()}")
    except ImportError as e:
        print(f"Numpy not available: {e}")
    
    # Get memory report
    report = get_memory_report()
    print(f"Memory report: {report}")
    
    print("Lazy import utilities test completed.")