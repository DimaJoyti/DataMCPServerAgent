"""
Authentication and Authorization system for MCP Agents.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from cryptography.fernet import Fernet
import base64

from secure_config import config

logger = logging.getLogger(__name__)

class Permission(Enum):
    READ_WORKERS = "read:workers"
    WRITE_WORKERS = "write:workers"
    READ_KV = "read:kv"
    WRITE_KV = "write:kv"
    READ_R2 = "read:r2"
    WRITE_R2 = "write:r2"
    READ_D1 = "read:d1"
    WRITE_D1 = "write:d1"
    READ_ANALYTICS = "read:analytics"
    ADMIN_ACCESS = "admin:access"

class Role(Enum):
    GUEST = "guest"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    role: Role
    permissions: Set[Permission]
    api_key: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

class AuthSystem:
    """Authentication and Authorization system."""

    def __init__(self):
        self.secret_key = config.security.secret_key
        self.jwt_secret = config.security.jwt_secret_key
        self.encryption_key = config.security.encryption_key
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id

        # Initialize encryption
        self.cipher = Fernet(base64.urlsafe_b64encode(self.encryption_key.encode()[:32].ljust(32, b'0')))

        logger.info("AuthSystem initialized with secure configuration")

        # Role-based permissions
        self.role_permissions = {
            Role.GUEST: {Permission.READ_ANALYTICS},
            Role.USER: {
                Permission.READ_WORKERS, Permission.READ_KV,
                Permission.READ_R2, Permission.READ_D1, Permission.READ_ANALYTICS
            },
            Role.DEVELOPER: {
                Permission.READ_WORKERS, Permission.WRITE_WORKERS,
                Permission.READ_KV, Permission.WRITE_KV,
                Permission.READ_R2, Permission.WRITE_R2,
                Permission.READ_D1, Permission.WRITE_D1,
                Permission.READ_ANALYTICS
            },
            Role.ADMIN: set(Permission)  # All permissions
        }

        # Initialize default users
        self._create_default_users()

    def _create_default_users(self):
        """Create default users for testing."""
        # Admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@cloudflare.com",
            role=Role.ADMIN,
            permissions=self.role_permissions[Role.ADMIN],
            api_key=self._generate_api_key(),
            created_at=datetime.utcnow()
        )
        self.users[admin_user.user_id] = admin_user
        self.api_keys[admin_user.api_key] = admin_user.user_id

        # Developer user
        dev_user = User(
            user_id="dev_001",
            username="developer",
            email="dev@cloudflare.com",
            role=Role.DEVELOPER,
            permissions=self.role_permissions[Role.DEVELOPER],
            api_key=self._generate_api_key(),
            created_at=datetime.utcnow()
        )
        self.users[dev_user.user_id] = dev_user
        self.api_keys[dev_user.api_key] = dev_user.user_id

        # Regular user
        user = User(
            user_id="user_001",
            username="user",
            email="user@cloudflare.com",
            role=Role.USER,
            permissions=self.role_permissions[Role.USER],
            api_key=self._generate_api_key(),
            created_at=datetime.utcnow()
        )
        self.users[user.user_id] = user
        self.api_keys[user.api_key] = user.user_id

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"cf_agent_{secrets.token_urlsafe(32)}"

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str, role: Role = Role.USER) -> User:
        """Create a new user."""
        user_id = f"user_{secrets.token_urlsafe(8)}"

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions[role],
            api_key=self._generate_api_key(),
            created_at=datetime.utcnow()
        )

        self.users[user_id] = user
        self.api_keys[user.api_key] = user_id

        return user

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        user_id = self.api_keys.get(api_key)
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.is_active:
                user.last_login = datetime.utcnow()
                return user
        return None

    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> Optional[Session]:
        """Create a new session."""
        if user_id not in self.users:
            return None

        session_id = f"session_{secrets.token_urlsafe(16)}"
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session and session.is_active and session.expires_at > datetime.utcnow():
            return session
        elif session:
            # Session expired
            session.is_active = False
        return None

    def get_user_by_session(self, session_id: str) -> Optional[User]:
        """Get user by session ID."""
        session = self.get_session(session_id)
        if session:
            return self.users.get(session.user_id)
        return None

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions

    def check_tool_access(self, user: User, tool_name: str) -> bool:
        """Check if user has access to specific tool."""
        # Tool-specific permission mapping
        tool_permissions = {
            "workers_list": Permission.READ_WORKERS,
            "workers_get_worker": Permission.READ_WORKERS,
            "workers_get_worker_code": Permission.READ_WORKERS,
            "kv_namespaces_list": Permission.READ_KV,
            "kv_namespace_create": Permission.WRITE_KV,
            "kv_namespace_delete": Permission.WRITE_KV,
            "r2_buckets_list": Permission.READ_R2,
            "r2_bucket_create": Permission.WRITE_R2,
            "r2_bucket_delete": Permission.WRITE_R2,
            "d1_databases_list": Permission.READ_D1,
            "d1_database_create": Permission.WRITE_D1,
            "d1_database_delete": Permission.WRITE_D1,
            "d1_database_query": Permission.WRITE_D1,
            "query_worker_observability": Permission.READ_ANALYTICS,
        }

        required_permission = tool_permissions.get(tool_name)
        if required_permission:
            return self.check_permission(user, required_permission)

        # Default: allow if no specific permission required
        return True

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and session.is_active
        ]

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at <= now
        ]

        for session_id in expired_sessions:
            self.sessions[session_id].is_active = False

    def get_users_summary(self) -> Dict[str, Any]:
        """Get summary of all users."""
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "users_by_role": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in Role
            },
            "active_sessions": len([s for s in self.sessions.values() if s.is_active])
        }

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information (safe for API responses)."""
        user = self.users.get(user_id)
        if user:
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active
            }
        return None

# Global auth system instance
auth_system = AuthSystem()

# Authentication decorator
def require_auth(permission: Permission = None):
    """Decorator to require authentication and optional permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            from mcp_inspector import mcp_inspector

            # Extract auth info from kwargs
            api_key = kwargs.get('api_key')
            session_id = kwargs.get('session_id', 'unknown')

            if not api_key:
                mcp_inspector.log_auth_check(
                    session_id=session_id,
                    user_id="unknown",
                    tool_name=func.__name__,
                    success=False,
                    error="No API key provided"
                )
                raise PermissionError("Authentication required")

            # Authenticate user
            user = auth_system.authenticate_api_key(api_key)
            if not user:
                mcp_inspector.log_auth_check(
                    session_id=session_id,
                    user_id="unknown",
                    tool_name=func.__name__,
                    success=False,
                    error="Invalid API key"
                )
                raise PermissionError("Invalid API key")

            # Check permission if required
            if permission and not auth_system.check_permission(user, permission):
                mcp_inspector.log_auth_check(
                    session_id=session_id,
                    user_id=user.user_id,
                    tool_name=func.__name__,
                    success=False,
                    error=f"Missing permission: {permission.value}"
                )
                raise PermissionError(f"Missing permission: {permission.value}")

            # Check tool-specific access
            if not auth_system.check_tool_access(user, func.__name__):
                mcp_inspector.log_auth_check(
                    session_id=session_id,
                    user_id=user.user_id,
                    tool_name=func.__name__,
                    success=False,
                    error="Tool access denied"
                )
                raise PermissionError("Tool access denied")

            # Log successful auth
            mcp_inspector.log_auth_check(
                session_id=session_id,
                user_id=user.user_id,
                tool_name=func.__name__,
                success=True
            )

            # Add user to kwargs
            kwargs['authenticated_user'] = user

            return await func(*args, **kwargs)

        return wrapper
    return decorator
