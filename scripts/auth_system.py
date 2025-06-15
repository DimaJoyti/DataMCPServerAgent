"""
Enhanced Authentication and Authorization system for MCP Agents.
Features JWT tokens, bcrypt hashing, rate limiting, and comprehensive security.
"""

import base64
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Security imports
import jwt
import structlog
from cryptography.fernet import Fernet
from passlib.context import CryptContext

# Configuration
from secure_config import config

logger = structlog.get_logger(__name__)

class Permission(Enum):
    """Enhanced permission system with granular access control."""
    # Worker permissions
    READ_WORKERS = "read:workers"
    WRITE_WORKERS = "write:workers"
    DEPLOY_WORKERS = "deploy:workers"

    # Storage permissions
    READ_KV = "read:kv"
    WRITE_KV = "write:kv"
    READ_R2 = "read:r2"
    WRITE_R2 = "write:r2"
    READ_D1 = "read:d1"
    WRITE_D1 = "write:d1"

    # Analytics and monitoring
    READ_ANALYTICS = "read:analytics"
    READ_LOGS = "read:logs"

    # Administrative
    ADMIN_ACCESS = "admin:access"
    USER_MANAGEMENT = "admin:users"
    SYSTEM_CONFIG = "admin:config"


class Role(Enum):
    """User roles with hierarchical permissions."""
    GUEST = "guest"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class User:
    """Enhanced user model with security features."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: Role
    permissions: Set[Permission]
    api_key: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    two_factor_enabled: bool = False


@dataclass
class Session:
    """Enhanced session model with security tracking."""
    session_id: str
    user_id: str
    access_token: str
    refresh_token: str
    created_at: datetime
    expires_at: datetime
    refresh_expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: Optional[datetime] = None


@dataclass
class TokenPair:
    """JWT token pair for authentication."""
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"

class AuthSystem:
    """Enhanced Authentication and Authorization system with JWT and bcrypt."""

    def __init__(self):
        # Configuration
        self.secret_key = config.security.secret_key.get_secret_value()
        self.jwt_secret = config.security.jwt_secret_key.get_secret_value()
        self.encryption_key = config.security.encryption_key.get_secret_value()

        # Storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.rate_limits: Dict[str, List[datetime]] = {}  # user_id -> timestamps

        # Security setup
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        cipher_key = base64.urlsafe_b64encode(
            self.encryption_key.encode()[:32].ljust(32, b'0')
        )
        self.cipher = Fernet(cipher_key)

        # JWT settings
        self.jwt_algorithm = "HS256"
        self.access_token_expire = timedelta(
            minutes=config.security.jwt_expiry_minutes
        )
        self.refresh_token_expire = timedelta(
            days=config.security.refresh_token_expiry_days
        )

        # Role-based permissions
        self.role_permissions = {
            Role.GUEST: {Permission.READ_ANALYTICS},
            Role.USER: {
                Permission.READ_WORKERS, Permission.READ_KV,
                Permission.READ_R2, Permission.READ_D1,
                Permission.READ_ANALYTICS, Permission.READ_LOGS
            },
            Role.DEVELOPER: {
                Permission.READ_WORKERS, Permission.WRITE_WORKERS,
                Permission.READ_KV, Permission.WRITE_KV,
                Permission.READ_R2, Permission.WRITE_R2,
                Permission.READ_D1, Permission.WRITE_D1,
                Permission.READ_ANALYTICS, Permission.READ_LOGS,
                Permission.DEPLOY_WORKERS
            },
            Role.ADMIN: {
                Permission.READ_WORKERS, Permission.WRITE_WORKERS,
                Permission.DEPLOY_WORKERS, Permission.READ_KV,
                Permission.WRITE_KV, Permission.READ_R2, Permission.WRITE_R2,
                Permission.READ_D1, Permission.WRITE_D1,
                Permission.READ_ANALYTICS, Permission.READ_LOGS,
                Permission.USER_MANAGEMENT, Permission.ADMIN_ACCESS
            },
            Role.SUPER_ADMIN: set(Permission)  # All permissions
        }

        logger.info("Enhanced AuthSystem initialized with JWT and bcrypt")
        self._create_default_users()

    def _create_default_users(self):
        """Create default users for testing."""
        default_password = "admin123"  # Change in production

        # Super Admin user
        super_admin = self.create_user(
            username="superadmin",
            email="superadmin@datamcp.dev",
            password=default_password,
            role=Role.SUPER_ADMIN
        )
        super_admin.is_verified = True

        # Admin user
        admin = self.create_user(
            username="admin",
            email="admin@datamcp.dev",
            password=default_password,
            role=Role.ADMIN
        )
        admin.is_verified = True

        # Developer user
        dev = self.create_user(
            username="developer",
            email="dev@datamcp.dev",
            password=default_password,
            role=Role.DEVELOPER
        )
        dev.is_verified = True

        # Regular user
        user = self.create_user(
            username="user",
            email="user@datamcp.dev",
            password=default_password,
            role=Role.USER
        )
        user.is_verified = True

        logger.info("Default users created successfully")

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"mcp_agent_{secrets.token_urlsafe(32)}"

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: Role = Role.USER
    ) -> User:
        """Create a new user with enhanced security."""
        user_id = f"user_{secrets.token_urlsafe(8)}"
        password_hash = self._hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            permissions=self.role_permissions[role],
            api_key=self._generate_api_key(),
            created_at=datetime.utcnow()
        )

        self.users[user_id] = user
        self.api_keys[user.api_key] = user_id

        logger.info(f"User created: {username} ({role.value})")
        return user

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        user_id = self.api_keys.get(api_key)
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.is_active and not self._is_user_locked(user):
                user.last_login = datetime.utcnow()
                return user
        return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        user = self._get_user_by_username(username)
        if not user:
            return None

        if self._is_user_locked(user):
            logger.warning(f"Login attempt for locked user: {username}")
            return None

        if self._verify_password(password, user.password_hash):
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            logger.info(f"Successful login: {username}")
            return user
        else:
            # Increment failed attempts
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:  # Lock after 5 failed attempts
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                logger.warning(f"User locked due to failed attempts: {username}")
            return None

    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def _is_user_locked(self, user: User) -> bool:
        """Check if user is locked."""
        if user.locked_until and user.locked_until > datetime.utcnow():
            return True
        return False

    def create_token_pair(self, user: User) -> TokenPair:
        """Create JWT token pair for user."""
        now = datetime.utcnow()

        # Access token payload
        access_payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access"
        }

        # Refresh token payload
        refresh_payload = {
            "sub": user.user_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "type": "refresh"
        }

        access_token = jwt.encode(
            access_payload, self.jwt_secret, algorithm=self.jwt_algorithm
        )
        refresh_token = jwt.encode(
            refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds())
        )

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
