"""
FastAPI dependency injection integration for DataMCPServerAgent.
Provides FastAPI-compatible dependency injection using the core DI container.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Type, TypeVar, Generator, AsyncGenerator

from fastapi import Depends, HTTPException, status

from src.core.dependency_injection import (
    ServiceContainer, 
    get_container, 
    ServiceScope,
    Lifetime,
    ILogger,
    IConfiguration
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FastAPIServiceProvider:
    """FastAPI-compatible service provider."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    def get_service(self, service_type: Type[T]) -> Callable[[], T]:
        """Create a FastAPI dependency that resolves a service."""
        def dependency() -> T:
            try:
                return self.container.resolve(service_type)
            except Exception as e:
                logger.error(f"Failed to resolve service {service_type.__name__}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Service resolution failed: {service_type.__name__}"
                )
        
        # Set dependency name for FastAPI docs
        dependency.__name__ = f"get_{service_type.__name__.lower()}"
        return dependency
    
    def get_scoped_service(self, service_type: Type[T]) -> Callable[[], Generator[T, None, None]]:
        """Create a FastAPI dependency that resolves a scoped service."""
        def dependency() -> Generator[T, None, None]:
            with ServiceScope(self.container) as scope:
                try:
                    service = scope.resolve(service_type)
                    yield service
                except Exception as e:
                    logger.error(f"Failed to resolve scoped service {service_type.__name__}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Scoped service resolution failed: {service_type.__name__}"
                    )
        
        dependency.__name__ = f"get_scoped_{service_type.__name__.lower()}"
        return dependency
    
    async def get_async_service(self, service_type: Type[T]) -> T:
        """Async service resolution."""
        try:
            return await self.container.resolve_async(service_type)
        except Exception as e:
            logger.error(f"Failed to resolve async service {service_type.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Async service resolution failed: {service_type.__name__}"
            )


# Global service provider
_service_provider: FastAPIServiceProvider = None


def get_service_provider() -> FastAPIServiceProvider:
    """Get the global FastAPI service provider."""
    global _service_provider
    if _service_provider is None:
        container = get_container()
        _service_provider = FastAPIServiceProvider(container)
    return _service_provider


def set_service_provider(provider: FastAPIServiceProvider):
    """Set the global FastAPI service provider."""
    global _service_provider
    _service_provider = provider


# Common FastAPI dependencies
def get_logger() -> ILogger:
    """FastAPI dependency to get logger service."""
    provider = get_service_provider()
    return provider.get_service(ILogger)()


def get_config() -> IConfiguration:
    """FastAPI dependency to get configuration service."""
    provider = get_service_provider()
    return provider.get_service(IConfiguration)()


def get_container_dependency() -> ServiceContainer:
    """FastAPI dependency to get the DI container."""
    return get_container()


# Dependency decorators for FastAPI routes
def inject_service(service_type: Type[T]) -> Callable:
    """Decorator to inject a service into a FastAPI route."""
    def decorator(func: Callable) -> Callable:
        provider = get_service_provider()
        dependency = provider.get_service(service_type)
        
        # Add the dependency to the function
        func.__annotations__[f'{service_type.__name__.lower()}_service'] = service_type
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Inject the service
            service = dependency()
            kwargs[f'{service_type.__name__.lower()}_service'] = service
            
            # Call the original function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def inject_scoped_service(service_type: Type[T]) -> Callable:
    """Decorator to inject a scoped service into a FastAPI route."""
    def decorator(func: Callable) -> Callable:
        provider = get_service_provider()
        dependency = provider.get_scoped_service(service_type)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Use scoped dependency
            with ServiceScope(get_container()) as scope:
                service = scope.resolve(service_type)
                kwargs[f'{service_type.__name__.lower()}_service'] = service
                
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Service configuration for FastAPI app
def configure_fastapi_services(container: ServiceContainer):
    """Configure services for FastAPI application."""
    try:
        from app.core.config import Settings
    except ImportError:
        # Fallback configuration
        class Settings:
            app_name: str = "DataMCPServerAgent"
            debug: bool = False
            
            def model_dump(self):
                return {"app_name": self.app_name, "debug": self.debug}
    
    try:
        from app.core.logging import get_logger as get_app_logger
    except ImportError:
        # Fallback logger
        def get_app_logger(name):
            import logging
            return logging.getLogger(name)
    
    # Configuration service
    class FastAPIConfiguration(IConfiguration):
        def __init__(self):
            self.settings = Settings()
        
        def get(self, key: str, default: Any = None) -> Any:
            return getattr(self.settings, key, default)
        
        def get_section(self, section: str) -> dict:
            # Return all settings as dict
            return self.settings.model_dump()
    
    # Logger service  
    class FastAPILogger(ILogger):
        def __init__(self):
            self.logger = get_app_logger(__name__)
        
        def info(self, message: str, **kwargs):
            self.logger.info(message, extra=kwargs)
        
        def error(self, message: str, **kwargs):
            self.logger.error(message, extra=kwargs)
        
        def warning(self, message: str, **kwargs):
            self.logger.warning(message, extra=kwargs)
    
    # Register services
    container.register_singleton(IConfiguration, FastAPIConfiguration)
    container.register_singleton(ILogger, FastAPILogger)
    
    logger.info("FastAPI services configured")


# Lifespan integration
async def setup_services():
    """Setup services for FastAPI lifespan."""
    container = get_container()
    configure_fastapi_services(container)
    
    # Initialize service provider
    global _service_provider
    _service_provider = FastAPIServiceProvider(container)
    
    logger.info("Services setup completed")


async def cleanup_services():
    """Cleanup services for FastAPI lifespan."""
    container = get_container()
    container.dispose()
    
    global _service_provider
    _service_provider = None
    
    logger.info("Services cleanup completed")


# Request scoped dependencies
class RequestScope:
    """Request-scoped dependency manager."""
    
    def __init__(self):
        self.scope_id = id(self)
        self.container = get_container()
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service in request scope."""
        return self.container.resolve(service_type, self.scope_id)
    
    def cleanup(self):
        """Cleanup request scope."""
        self.container.clear_scope(self.scope_id)


def get_request_scope() -> RequestScope:
    """FastAPI dependency to get request scope."""
    scope = RequestScope()
    try:
        yield scope
    finally:
        scope.cleanup()


# Health check dependencies
def get_service_health() -> dict:
    """FastAPI dependency to get service health information."""
    container = get_container()
    return container.get_service_info()


# Example route dependencies
def create_service_dependencies():
    """Create common service dependencies for routes."""
    provider = get_service_provider()
    
    return {
        'logger': Depends(provider.get_service(ILogger)),
        'config': Depends(provider.get_service(IConfiguration)),
        'container': Depends(get_container_dependency),
        'request_scope': Depends(get_request_scope),
        'service_health': Depends(get_service_health)
    }


# Middleware integration
class DIMiddleware:
    """Middleware to set up dependency injection for each request."""
    
    def __init__(self, app, container: ServiceContainer):
        self.app = app
        self.container = container
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Create request scope
            request_scope = RequestScope()
            scope["di_scope"] = request_scope
            
            try:
                await self.app(scope, receive, send)
            finally:
                request_scope.cleanup()
        else:
            await self.app(scope, receive, send)


# Testing utilities
def create_test_dependencies():
    """Create dependencies for testing."""
    from src.core.dependency_injection import create_test_container
    
    container = create_test_container()
    configure_fastapi_services(container)
    
    return FastAPIServiceProvider(container)


# Repository pattern integration
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

EntityT = TypeVar('EntityT')
IdT = TypeVar('IdT')


class IRepository(ABC, Generic[EntityT, IdT]):
    """Abstract repository interface for dependency injection."""
    
    @abstractmethod
    async def get_by_id(self, id: IdT) -> Optional[EntityT]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[EntityT]:
        """Get all entities."""
        pass
    
    @abstractmethod
    async def create(self, entity: EntityT) -> EntityT:
        """Create new entity."""
        pass
    
    @abstractmethod
    async def update(self, entity: EntityT) -> EntityT:
        """Update existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: IdT) -> bool:
        """Delete entity by ID."""
        pass


def register_repository(container: ServiceContainer, 
                       interface: Type[IRepository], 
                       implementation: Type[IRepository],
                       lifetime: Lifetime = Lifetime.SCOPED):
    """Register a repository with the container."""
    if lifetime == Lifetime.SINGLETON:
        container.register_singleton(interface, implementation)
    elif lifetime == Lifetime.TRANSIENT:
        container.register_transient(interface, implementation)
    elif lifetime == Lifetime.SCOPED:
        container.register_scoped(interface, implementation)
    else:
        raise ValueError(f"Unknown lifetime: {lifetime}")


# Example usage in FastAPI routes
"""
from fastapi import APIRouter, Depends
from app.core.dependencies import get_logger, get_config, inject_service

router = APIRouter()

@router.get("/example")
async def example_route(
    logger: ILogger = Depends(get_logger),
    config: IConfiguration = Depends(get_config)
):
    logger.info("Example route called")
    app_name = config.get("app_name", "Unknown")
    return {"message": f"Hello from {app_name}"}

# Or using decorator
@router.get("/example2")
@inject_service(ILogger)
async def example_route2(logger_service: ILogger):
    logger_service.info("Example route 2 called")
    return {"message": "Hello with injected service"}
"""


# Example testing
if __name__ == "__main__":
    print("Testing FastAPI dependency injection...")
    
    # Setup
    container = get_container()
    configure_fastapi_services(container)
    
    provider = FastAPIServiceProvider(container)
    
    # Test service resolution
    logger_dep = provider.get_service(ILogger)
    logger_service = logger_dep()
    logger_service.info("FastAPI DI test successful!")
    
    # Test health check
    health = get_service_health()
    print(f"Service health: {health}")
    
    print("FastAPI dependency injection test completed.")