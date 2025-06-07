"""
Base domain models and abstractions.
Provides common functionality for all domain entities and value objects.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, OptionalVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")

class BaseValueObject(BaseModel):
    """Base class for value objects."""

    class Config:
        frozen = True  # Value objects are immutable
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v),
        }

class BaseEntity(BaseModel):
    """Base class for domain entities."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )
    version: int = Field(default=1, description="Entity version for optimistic locking")

    # Domain events
    _domain_events: List["DomainEvent"] = field(default_factory=list, init=False, repr=False)

    class Config:
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v),
        }

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to update timestamp on changes."""
        if name != "updated_at" and hasattr(self, name):
            super().__setattr__("updated_at", datetime.now(timezone.utc))
        super().__setattr__(name, value)

    def add_domain_event(self, event: "DomainEvent") -> None:
        """Add a domain event to this entity."""
        self._domain_events.append(event)

    def clear_domain_events(self) -> List["DomainEvent"]:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    def increment_version(self) -> None:
        """Increment entity version."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)

class DomainEvent(BaseValueObject):
    """Base class for domain events."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Event unique identifier"
    )
    event_type: str = Field(description="Type of the event")
    aggregate_id: str = Field(description="ID of the aggregate that generated the event")
    aggregate_type: str = Field(description="Type of the aggregate")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the event occurred"
    )
    version: int = Field(description="Version of the aggregate when event occurred")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")

    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v):
        """Set event type from class name if not provided."""
        if not v:
            return cls.__name__
        return v

class AggregateRoot(BaseEntity):
    """Base class for aggregate roots."""

    def apply_event(self, event: DomainEvent) -> None:
        """Apply a domain event to this aggregate."""
        self.add_domain_event(event)
        self.increment_version()

        # Try to find and call event handler method
        handler_name = f"_handle_{event.event_type.lower()}"
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            handler(event)

class Repository(ABC, Generic[T]):
    """Abstract base repository interface."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0, **filters) -> List[T]:
        """List entities with pagination and filters."""
        pass

    @abstractmethod
    async def count(self, **filters) -> int:
        """Count entities with filters."""
        pass

class DomainService(ABC):
    """Base class for domain services."""

    def __init__(self):
        self._repositories: Dict[str, Repository] = {}

    def register_repository(self, name: str, repository: Repository) -> None:
        """Register a repository."""
        self._repositories[name] = repository

    def get_repository(self, name: str) -> Repository:
        """Get a registered repository."""
        if name not in self._repositories:
            raise ValueError(f"Repository '{name}' not registered")
        return self._repositories[name]

class Specification(ABC, Generic[T]):
    """Base specification pattern implementation."""

    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies this specification."""
        pass

    def and_(self, other: "Specification[T]") -> "AndSpecification[T]":
        """Combine with another specification using AND."""
        return AndSpecification(self, other)

    def or_(self, other: "Specification[T]") -> "OrSpecification[T]":
        """Combine with another specification using OR."""
        return OrSpecification(self, other)

    def not_(self) -> "NotSpecification[T]":
        """Negate this specification."""
        return NotSpecification(self)

class AndSpecification(Specification[T]):
    """AND combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)

class OrSpecification(Specification[T]):
    """OR combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)

class NotSpecification(Specification[T]):
    """NOT negation of specification."""

    def __init__(self, spec: Specification[T]):
        self.spec = spec

    def is_satisfied_by(self, entity: T) -> bool:
        return not self.spec.is_satisfied_by(entity)

class DomainError(Exception):
    """Base class for domain errors."""

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

class ValidationError(DomainError):
    """Domain validation error."""

    pass

class BusinessRuleError(DomainError):
    """Business rule violation error."""

    pass

class ConcurrencyError(DomainError):
    """Concurrency/optimistic locking error."""

    pass

class EntityNotFoundError(DomainError):
    """Entity not found error."""

    pass
