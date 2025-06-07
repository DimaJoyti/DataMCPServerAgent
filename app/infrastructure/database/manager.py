"""
Database manager for handling connections and sessions.
"""

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None

    async def initialize(self) -> None:
        """Initialize database connection."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                settings.database.database_url,
                echo=settings.database.echo_sql,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    def get_session(self) -> AsyncSession:
        """Get database session."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")

        return self.session_factory()

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.engine:
                return False

            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")

            return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
