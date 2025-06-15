"""
Dependency injection container for DataMCPServerAgent.
Provides centralized dependency management following Clean Architecture patterns.
"""

import asyncio
import functools
import inspect
import logging
import threading
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifetime(Enum):
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes how a service should be registered and created."""
    
    def __init__(self,
                 service_type: Type[T],
                 implementation_type: Optional[Type[T]] = None,
                 factory: Optional[Callable[..., T]] = None,
                 instance: Optional[T] = None,
                 lifetime: Lifetime = Lifetime.TRANSIENT):
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        
        # Validation
        if not any([implementation_type, factory, instance]):
            raise ValueError("Must provide implementation_type, factory, or instance")
        
        if sum(x is not None for x in [implementation_type, factory, instance]) > 1:
            raise ValueError("Can only provide one of: implementation_type, factory, or instance")


class ServiceContainer:
    """Dependency injection container with async support."""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[int, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._building: Set[Type] = set()  # Circular dependency detection
        
        # Lifecycle hooks
        self._initialization_hooks: List[Callable[[Any], None]] = []
        self._disposal_hooks: List[Callable[[Any], None]] = []
        
        # Register self
        self.register_instance(ServiceContainer, self)
    
    def register_singleton(self, 
                         service_type: Type[T], 
                         implementation_type: Optional[Type[T]] = None,
                         factory: Optional[Callable[..., T]] = None,
                         instance: Optional[T] = None) -> 'ServiceContainer':
        """Register a singleton service."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=Lifetime.SINGLETON
        )
        
        with self._lock:
            self._services[service_type] = descriptor
            
            # If instance provided, store it directly
            if instance is not None:
                self._singletons[service_type] = instance
                self._run_initialization_hooks(instance)
        
        logger.debug(f"Registered singleton service: {service_type.__name__}")
        return self
    
    def register_transient(self,
                         service_type: Type[T],
                         implementation_type: Optional[Type[T]] = None,
                         factory: Optional[Callable[..., T]] = None) -> 'ServiceContainer':
        """Register a transient service (new instance every time)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=Lifetime.TRANSIENT
        )
        
        with self._lock:
            self._services[service_type] = descriptor
        
        logger.debug(f"Registered transient service: {service_type.__name__}")
        return self
    
    def register_scoped(self,
                       service_type: Type[T],
                       implementation_type: Optional[Type[T]] = None,
                       factory: Optional[Callable[..., T]] = None) -> 'ServiceContainer':
        """Register a scoped service (one instance per scope)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=Lifetime.SCOPED
        )
        
        with self._lock:
            self._services[service_type] = descriptor
        
        logger.debug(f"Registered scoped service: {service_type.__name__}")
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainer':
        """Register a specific instance as a singleton."""
        return self.register_singleton(service_type, instance=instance)
    
    def resolve(self, service_type: Type[T], scope_id: Optional[int] = None) -> T:
        """Resolve a service instance."""
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"Service {service_type.__name__} not registered")
            
            descriptor = self._services[service_type]
            
            # Check for circular dependencies
            if service_type in self._building:
                raise ValueError(f"Circular dependency detected for {service_type.__name__}")
            
            try:
                self._building.add(service_type)
                return self._create_instance(descriptor, scope_id)
            finally:
                self._building.discard(service_type)
    
    async def resolve_async(self, service_type: Type[T], scope_id: Optional[int] = None) -> T:
        """Resolve a service instance asynchronously."""
        # For now, delegate to sync resolve
        # In future, could support async factories
        return self.resolve(service_type, scope_id)
    
    def _create_instance(self, descriptor: ServiceDescriptor, scope_id: Optional[int] = None) -> Any:
        """Create an instance based on the service descriptor."""
        service_type = descriptor.service_type
        
        # Handle different lifetimes
        if descriptor.lifetime == Lifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._build_instance(descriptor)
            self._singletons[service_type] = instance
            self._run_initialization_hooks(instance)
            return instance
        
        elif descriptor.lifetime == Lifetime.SCOPED:
            if scope_id is None:
                scope_id = id(threading.current_thread())
            
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}
            
            scope_instances = self._scoped_instances[scope_id]
            if service_type in scope_instances:
                return scope_instances[service_type]
            
            instance = self._build_instance(descriptor)
            scope_instances[service_type] = instance
            self._run_initialization_hooks(instance)
            return instance
        
        elif descriptor.lifetime == Lifetime.TRANSIENT:
            instance = self._build_instance(descriptor)
            self._run_initialization_hooks(instance)
            return instance
        
        else:
            raise ValueError(f"Unknown lifetime: {descriptor.lifetime}")
    
    def _build_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Build an instance using the descriptor."""
        # Use provided instance
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Use factory
        if descriptor.factory is not None:
            return self._invoke_with_dependencies(descriptor.factory)
        
        # Use implementation type
        if descriptor.implementation_type is not None:
            return self._invoke_with_dependencies(descriptor.implementation_type)
        
        raise ValueError("No way to create instance")
    
    def _invoke_with_dependencies(self, callable_obj: Callable) -> Any:
        """Invoke a callable with dependency injection."""
        # Get signature
        sig = inspect.signature(callable_obj)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
                
                # Skip basic types
                if param_type in (str, int, float, bool, list, dict):
                    continue
                
                # Try to resolve the dependency
                try:
                    kwargs[param_name] = self.resolve(param_type)
                except ValueError:
                    # If dependency not found and parameter has default, skip it
                    if param.default == inspect.Parameter.empty:
                        logger.warning(f"Could not resolve dependency {param_type.__name__} for parameter {param_name}")
        
        return callable_obj(**kwargs)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        with self._lock:
            return service_type in self._services
    
    def clear_scope(self, scope_id: int):
        """Clear all scoped instances for a given scope."""
        with self._lock:
            if scope_id in self._scoped_instances:
                scope_instances = self._scoped_instances[scope_id]
                
                # Run disposal hooks
                for instance in scope_instances.values():
                    self._run_disposal_hooks(instance)
                
                del self._scoped_instances[scope_id]
                logger.debug(f"Cleared scope {scope_id}")
    
    def dispose(self):
        """Dispose of all services and clean up resources."""
        with self._lock:
            # Dispose singletons
            for instance in self._singletons.values():
                self._run_disposal_hooks(instance)
            
            # Dispose scoped instances
            for scope_instances in self._scoped_instances.values():
                for instance in scope_instances.values():
                    self._run_disposal_hooks(instance)
            
            # Clear all
            self._singletons.clear()
            self._scoped_instances.clear()
            self._services.clear()
            
            logger.info("Service container disposed")
    
    def add_initialization_hook(self, hook: Callable[[Any], None]):
        """Add a hook to run when services are initialized."""
        self._initialization_hooks.append(hook)
    
    def add_disposal_hook(self, hook: Callable[[Any], None]):
        """Add a hook to run when services are disposed."""
        self._disposal_hooks.append(hook)
    
    def _run_initialization_hooks(self, instance: Any):
        """Run initialization hooks for an instance."""
        for hook in self._initialization_hooks:
            try:
                hook(instance)
            except Exception as e:
                logger.warning(f"Initialization hook failed: {e}")
    
    def _run_disposal_hooks(self, instance: Any):
        """Run disposal hooks for an instance."""
        for hook in self._disposal_hooks:
            try:
                hook(instance)
            except Exception as e:
                logger.warning(f"Disposal hook failed: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about registered services."""
        with self._lock:
            info = {
                "registered_services": len(self._services),
                "singleton_instances": len(self._singletons),
                "scoped_instances": sum(len(scope) for scope in self._scoped_instances.values()),
                "active_scopes": len(self._scoped_instances),
                "services": {}
            }
            
            for service_type, descriptor in self._services.items():
                info["services"][service_type.__name__] = {
                    "lifetime": descriptor.lifetime.value,
                    "has_instance": service_type in self._singletons,
                    "implementation": descriptor.implementation_type.__name__ if descriptor.implementation_type else None,
                    "has_factory": descriptor.factory is not None
                }
            
            return info


# Scope management
class ServiceScope:
    """Context manager for scoped services."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.scope_id = id(self)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.clear_scope(self.scope_id)
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        return self.container.resolve(service_type, self.scope_id)


@asynccontextmanager
async def async_service_scope(container: ServiceContainer):
    """Async context manager for scoped services."""
    scope = ServiceScope(container)
    try:
        yield scope
    finally:
        container.clear_scope(scope.scope_id)


# Decorators
def inject(*dependencies: Type) -> Callable:
    """Decorator to inject dependencies into function parameters."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get container from kwargs or use global
            container = kwargs.pop('_container', get_container())
            
            # Resolve dependencies
            sig = inspect.signature(func)
            for i, (param_name, param) in enumerate(sig.parameters.items()):
                if i < len(args):
                    continue  # Already provided as positional arg
                
                if param_name in kwargs:
                    continue  # Already provided as keyword arg
                
                if param.annotation in dependencies:
                    kwargs[param_name] = container.resolve(param.annotation)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def injectable(lifetime: Lifetime = Lifetime.TRANSIENT):
    """Class decorator to mark a class as injectable."""
    def decorator(cls: Type[T]) -> Type[T]:
        # Add metadata
        cls._injectable_lifetime = lifetime
        return cls
    
    return decorator


# Global container
_global_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """Get the global service container."""
    global _global_container
    
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = ServiceContainer()
    
    return _global_container


def set_container(container: ServiceContainer):
    """Set the global service container."""
    global _global_container
    _global_container = container


def configure_services(configuration_func: Callable[[ServiceContainer], None]):
    """Configure services using a configuration function."""
    container = get_container()
    configuration_func(container)
    return container


# Service locator pattern (use sparingly)
class ServiceLocator:
    """Service locator for accessing dependencies (anti-pattern, use DI instead)."""
    
    @staticmethod
    def get_service(service_type: Type[T]) -> T:
        """Get a service instance (discouraged, use DI instead)."""
        container = get_container()
        return container.resolve(service_type)
    
    @staticmethod
    async def get_service_async(service_type: Type[T]) -> T:
        """Get a service instance asynchronously."""
        container = get_container()
        return await container.resolve_async(service_type)


# Example base interfaces for common services
class ILogger(ABC):
    """Abstract logger interface."""
    
    @abstractmethod
    def info(self, message: str, **kwargs): pass
    
    @abstractmethod
    def error(self, message: str, **kwargs): pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs): pass


class IConfiguration(ABC):
    """Abstract configuration interface."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]: pass


class IRepository(ABC, Generic[T]):
    """Abstract repository interface."""
    
    @abstractmethod
    async def get_by_id(self, id: Any) -> Optional[T]: pass
    
    @abstractmethod
    async def create(self, entity: T) -> T: pass
    
    @abstractmethod
    async def update(self, entity: T) -> T: pass
    
    @abstractmethod
    async def delete(self, id: Any) -> bool: pass


# Example implementations
@injectable(Lifetime.SINGLETON)
class ConsoleLogger(ILogger):
    """Console logger implementation."""
    
    def info(self, message: str, **kwargs):
        print(f"INFO: {message}")
    
    def error(self, message: str, **kwargs):
        print(f"ERROR: {message}")
    
    def warning(self, message: str, **kwargs):
        print(f"WARNING: {message}")


# Configuration helper
def auto_register_services(container: ServiceContainer, module_or_package):
    """Automatically register services from a module or package."""
    import importlib
    import pkgutil
    
    if isinstance(module_or_package, str):
        module_or_package = importlib.import_module(module_or_package)
    
    # Walk through module and find injectable classes
    for importer, modname, ispkg in pkgutil.iter_modules(module_or_package.__path__):
        try:
            module = importlib.import_module(f"{module_or_package.__name__}.{modname}")
            
            for name in dir(module):
                obj = getattr(module, name)
                
                if (inspect.isclass(obj) and 
                    hasattr(obj, '_injectable_lifetime') and
                    obj.__module__ == module.__name__):
                    
                    lifetime = obj._injectable_lifetime
                    
                    # Register based on lifetime
                    if lifetime == Lifetime.SINGLETON:
                        container.register_singleton(obj, obj)
                    elif lifetime == Lifetime.TRANSIENT:
                        container.register_transient(obj, obj)
                    elif lifetime == Lifetime.SCOPED:
                        container.register_scoped(obj, obj)
                        
                    logger.info(f"Auto-registered {obj.__name__} as {lifetime.value}")
                    
        except Exception as e:
            logger.warning(f"Failed to auto-register from {modname}: {e}")


# Testing utilities
def create_test_container() -> ServiceContainer:
    """Create a clean container for testing."""
    return ServiceContainer()


# Example usage and testing
if __name__ == "__main__":
    print("Testing dependency injection container...")
    
    # Create container
    container = ServiceContainer()
    
    # Register services
    container.register_singleton(ILogger, ConsoleLogger)
    
    # Test resolution
    logger_service = container.resolve(ILogger)
    logger_service.info("Dependency injection working!")
    
    # Test scoped services
    with ServiceScope(container) as scope:
        scoped_logger = scope.resolve(ILogger)
        scoped_logger.info("Scoped service working!")
    
    # Show service info
    info = container.get_service_info()
    print(f"Container info: {info}")
    
    print("Dependency injection test completed.")