"""
User domain models.
Defines models for users, roles, permissions, and sessions.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator

from .base import BaseEntity, BaseValueObject, ValidationError

class Role(str, Enum):
    """User roles enumeration."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    AGENT = "agent"
    API_USER = "api_user"

class Permission(str, Enum):
    """User permissions enumeration."""

    # Agent permissions
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_SCALE = "agent:scale"

    # Task permissions
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    TASK_EXECUTE = "task:execute"

    # State permissions
    STATE_READ = "state:read"
    STATE_WRITE = "state:write"
    STATE_DELETE = "state:delete"

    # Communication permissions
    EMAIL_SEND = "email:send"
    EMAIL_APPROVE = "email:approve"
    WEBRTC_CALL = "webrtc:call"

    # Deployment permissions
    DEPLOY_CREATE = "deploy:create"
    DEPLOY_UPDATE = "deploy:update"
    DEPLOY_DELETE = "deploy:delete"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"

class UserStatus(str, Enum):
    """User status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class ApiKey(BaseValueObject):
    """API key value object."""

    key_id: str = Field(description="API key identifier")
    key_hash: str = Field(description="Hashed API key")
    name: str = Field(description="API key name")
    permissions: Set[Permission] = Field(default_factory=set, description="API key permissions")
    expires_at: Optional[datetime] = Field(default=None, description="API key expiration")
    last_used_at: Optional[datetime] = Field(default=None, description="Last usage time")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation time"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("API key name cannot be empty")
        return v.strip()

    @property
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        return not self.is_expired

class Session(BaseValueObject):
    """User session value object."""

    session_id: str = Field(description="Session identifier")
    user_id: str = Field(description="User identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Session creation time"
    )
    expires_at: datetime = Field(description="Session expiration time")
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last activity time"
    )
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    @field_validator("session_id", "user_id")
    @classmethod
    def validate_ids(cls, v):
        if not v or not v.strip():
            raise ValidationError("ID cannot be empty")
        return v.strip()

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return not self.is_expired

    def extend_session(self, hours: int = 24) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.last_activity_at = datetime.now(timezone.utc)

class User(BaseEntity):
    """User entity."""

    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    full_name: str = Field(description="Full name")
    password_hash: str = Field(description="Hashed password")
    role: Role = Field(default=Role.VIEWER, description="User role")
    permissions: Set[Permission] = Field(default_factory=set, description="User permissions")
    status: UserStatus = Field(default=UserStatus.PENDING, description="User status")

    # API keys
    api_keys: List[ApiKey] = Field(default_factory=list, description="User API keys")

    # Profile information
    avatar_url: Optional[str] = Field(default=None, description="Avatar URL")
    timezone: str = Field(default="UTC", description="User timezone")
    language: str = Field(default="en", description="Preferred language")

    # Security
    last_login_at: Optional[datetime] = Field(default=None, description="Last login time")
    failed_login_attempts: int = Field(default=0, description="Failed login attempts")
    locked_until: Optional[datetime] = Field(default=None, description="Account lock expiration")

    # Preferences
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if not v or not v.strip():
            raise ValidationError("Username cannot be empty")
        if len(v.strip()) < 3:
            raise ValidationError("Username must be at least 3 characters")
        return v.strip().lower()

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if not v or "@" not in v:
            raise ValidationError("Invalid email address")
        return v.lower().strip()

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Full name cannot be empty")
        return v.strip()

    def add_permission(self, permission: Permission) -> None:
        """Add permission to user."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from user."""
        self.permissions.discard(permission)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        # Admin has all permissions
        if self.role == Role.ADMIN:
            return True

        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(perm) for perm in permissions)

    def create_api_key(
        self, name: str, permissions: Set[Permission] = None, expires_in_days: int = None
    ) -> ApiKey:
        """Create a new API key for the user."""
        import hashlib
        import secrets

        # Generate API key
        key_value = f"dmcp_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        # Create API key
        api_key = ApiKey(
            key_id=f"key_{secrets.token_hex(8)}",
            key_hash=key_hash,
            name=name,
            permissions=permissions or set(),
            expires_at=expires_at,
        )

        self.api_keys.append(api_key)
        return api_key

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        original_count = len(self.api_keys)
        self.api_keys = [key for key in self.api_keys if key.key_id != key_id]
        return len(self.api_keys) < original_count

    def get_api_key(self, key_id: str) -> Optional[ApiKey]:
        """Get API key by ID."""
        return next((key for key in self.api_keys if key.key_id == key_id), None)

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def lock_account(self, hours: int = 1) -> None:
        """Lock user account for specified hours."""
        self.locked_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.status = UserStatus.SUSPENDED

    def unlock_account(self) -> None:
        """Unlock user account."""
        self.locked_until = None
        self.failed_login_attempts = 0
        if self.status == UserStatus.SUSPENDED:
            self.status = UserStatus.ACTIVE

    def record_login_attempt(self, success: bool) -> None:
        """Record login attempt."""
        if success:
            self.last_login_at = datetime.now(timezone.utc)
            self.failed_login_attempts = 0
            self.unlock_account()
        else:
            self.failed_login_attempts += 1
            # Lock account after 5 failed attempts
            if self.failed_login_attempts >= 5:
                self.lock_account()

    @property
    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE and not self.is_locked()

    @property
    def active_api_keys(self) -> List[ApiKey]:
        """Get active (non-expired) API keys."""
        return [key for key in self.api_keys if key.is_valid]
