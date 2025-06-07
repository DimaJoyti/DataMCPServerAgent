"""
State domain models.
Defines models for persistent state management and synchronization.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from .base import BaseEntity, BaseValueObject, ValidationError

class StateType(str, Enum):
    """Type of persistent state."""

    AGENT_STATE = "agent_state"
    TASK_STATE = "task_state"
    SESSION_STATE = "session_state"
    CONFIGURATION_STATE = "configuration_state"
    CACHE_STATE = "cache_state"

class StateStatus(str, Enum):
    """Status of state synchronization."""

    SYNCED = "synced"
    PENDING = "pending"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"

class StateMetadata(BaseValueObject):
    """State metadata value object."""

    checksum: str = Field(description="State data checksum")
    size_bytes: int = Field(description="State size in bytes")
    compression: Optional[str] = Field(default=None, description="Compression algorithm used")
    encryption: Optional[str] = Field(default=None, description="Encryption algorithm used")
    tags: Dict[str, str] = Field(default_factory=dict, description="State tags")

    @field_validator("size_bytes")
    @classmethod
    def validate_size(cls, v):
        if v < 0:
            raise ValidationError("State size cannot be negative")
        return v

class StateVersion(BaseValueObject):
    """State version information."""

    version_number: int = Field(description="Version number")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Version creation time"
    )
    created_by: str = Field(description="Who created this version")
    description: Optional[str] = Field(default=None, description="Version description")
    parent_version: Optional[int] = Field(default=None, description="Parent version number")

    @field_validator("version_number")
    @classmethod
    def validate_version_number(cls, v):
        if v < 1:
            raise ValidationError("Version number must be positive")
        return v

class PersistentState(BaseEntity):
    """Persistent state entity."""

    entity_id: str = Field(description="ID of the entity this state belongs to")
    entity_type: str = Field(description="Type of entity (agent, task, etc.)")
    state_type: StateType = Field(description="Type of state")
    state_data: Dict[str, Any] = Field(description="The actual state data")
    status: StateStatus = Field(default=StateStatus.SYNCED, description="Synchronization status")

    # Version information
    current_version: StateVersion = Field(description="Current version information")
    versions: list[StateVersion] = Field(default_factory=list, description="Version history")

    # Metadata
    metadata: StateMetadata = Field(description="State metadata")

    # Synchronization
    last_synced_at: Optional[datetime] = Field(
        default=None, description="Last synchronization time"
    )
    sync_source: Optional[str] = Field(default=None, description="Source of last sync")
    conflict_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Conflicting state data"
    )

    # Storage
    storage_backend: str = Field(default="cloudflare_kv", description="Storage backend used")
    storage_key: str = Field(description="Key used in storage backend")

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v):
        if not v or not v.strip():
            raise ValidationError("Entity ID cannot be empty")
        return v.strip()

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v):
        if not v or not v.strip():
            raise ValidationError("Entity type cannot be empty")
        return v.strip().lower()

    @field_validator("storage_key")
    @classmethod
    def validate_storage_key(cls, v):
        if not v or not v.strip():
            raise ValidationError("Storage key cannot be empty")
        return v.strip()

    def create_new_version(self, created_by: str, description: str = None) -> StateVersion:
        """Create a new version of the state."""
        new_version_number = self.current_version.version_number + 1

        new_version = StateVersion(
            version_number=new_version_number,
            created_by=created_by,
            description=description,
            parent_version=self.current_version.version_number,
        )

        # Add current version to history
        self.versions.append(self.current_version)

        # Set new version as current
        self.current_version = new_version

        return new_version

    def rollback_to_version(self, version_number: int) -> bool:
        """Rollback to a specific version."""
        target_version = next(
            (v for v in self.versions if v.version_number == version_number), None
        )

        if not target_version:
            return False

        # Move current version to history
        self.versions.append(self.current_version)

        # Set target version as current
        self.current_version = target_version

        # Remove target version from history
        self.versions = [v for v in self.versions if v.version_number != version_number]

        return True

    def mark_as_conflicted(self, conflict_data: Dict[str, Any]) -> None:
        """Mark state as having conflicts."""
        self.status = StateStatus.CONFLICT
        self.conflict_data = conflict_data

    def resolve_conflict(self, resolution_data: Dict[str, Any], resolved_by: str) -> None:
        """Resolve state conflict."""
        self.state_data = resolution_data
        self.conflict_data = None
        self.status = StateStatus.SYNCED
        self.create_new_version(resolved_by, "Conflict resolution")

    def update_state_data(self, new_data: Dict[str, Any], updated_by: str) -> None:
        """Update state data and create new version."""
        self.state_data = new_data
        self.status = StateStatus.PENDING
        self.create_new_version(updated_by, "State data update")

    def mark_as_synced(self, sync_source: str = None) -> None:
        """Mark state as successfully synced."""
        self.status = StateStatus.SYNCED
        self.last_synced_at = datetime.now(timezone.utc)
        if sync_source:
            self.sync_source = sync_source

    @property
    def is_synced(self) -> bool:
        """Check if state is synced."""
        return self.status == StateStatus.SYNCED

    @property
    def has_conflicts(self) -> bool:
        """Check if state has conflicts."""
        return self.status == StateStatus.CONFLICT

    @property
    def version_count(self) -> int:
        """Get number of versions."""
        return len(self.versions) + 1  # +1 for current version

    @property
    def latest_version_number(self) -> int:
        """Get latest version number."""
        return self.current_version.version_number
