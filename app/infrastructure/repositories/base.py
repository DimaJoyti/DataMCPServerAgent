"""
Base repository implementations.
Provides common functionality for all repositories.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import and_, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import BaseEntity, ConcurrencyError, Repository

T = TypeVar("T", bound=BaseEntity)
logger = get_logger(__name__)

class SQLAlchemyRepository(Repository[T], LoggerMixin, Generic[T]):
    """Base SQLAlchemy repository implementation."""

    def __init__(self, session: AsyncSession, model_class: Type[T], entity_class: Type[T]):
        self.session = session
        self.model_class = model_class  # SQLAlchemy model
        self.entity_class = entity_class  # Domain entity

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        try:
            stmt = select(self.model_class).where(self.model_class.id == entity_id)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()

            if model is None:
                return None

            return self._model_to_entity(model)

        except Exception as e:
            self.logger.error(f"Error getting entity by ID {entity_id}: {e}")
            raise

    async def save(self, entity: T) -> T:
        """Save entity."""
        try:
            # Check if entity exists
            existing_model = None
            if hasattr(entity, "id") and entity.id:
                stmt = select(self.model_class).where(self.model_class.id == entity.id)
                result = await self.session.execute(stmt)
                existing_model = result.scalar_one_or_none()

            if existing_model:
                # Update existing
                await self._update_model_from_entity(existing_model, entity)
            else:
                # Create new
                model = self._entity_to_model(entity)
                self.session.add(model)

            await self.session.commit()

            # Refresh entity from database
            return await self.get_by_id(entity.id)

        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"Integrity error saving entity: {e}")
            raise ConcurrencyError("Entity was modified by another process")
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error saving entity: {e}")
            raise

    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        try:
            stmt = delete(self.model_class).where(self.model_class.id == entity_id)
            result = await self.session.execute(stmt)
            await self.session.commit()

            return result.rowcount > 0

        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error deleting entity {entity_id}: {e}")
            raise

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True,
        **filters,
    ) -> List[T]:
        """List entities with pagination and filters."""
        try:
            stmt = select(self.model_class)

            # Apply filters
            if filters:
                conditions = []
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        column = getattr(self.model_class, key)

                        # Handle different filter types
                        if isinstance(value, list):
                            conditions.append(column.in_(value))
                        elif isinstance(value, dict):
                            # Range filters like {"gte": value, "lte": value}
                            if "gte" in value:
                                conditions.append(column >= value["gte"])
                            if "lte" in value:
                                conditions.append(column <= value["lte"])
                            if "gt" in value:
                                conditions.append(column > value["gt"])
                            if "lt" in value:
                                conditions.append(column < value["lt"])
                        else:
                            conditions.append(column == value)

                if conditions:
                    stmt = stmt.where(and_(*conditions))

            # Apply ordering
            if hasattr(self.model_class, order_by):
                order_column = getattr(self.model_class, order_by)
                if order_desc:
                    stmt = stmt.order_by(order_column.desc())
                else:
                    stmt = stmt.order_by(order_column.asc())

            # Apply pagination
            stmt = stmt.offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

        except Exception as e:
            self.logger.error(f"Error listing entities: {e}")
            raise

    async def count(self, **filters) -> int:
        """Count entities with filters."""
        try:
            stmt = select(func.count(self.model_class.id))

            # Apply filters (same logic as list method)
            if filters:
                conditions = []
                for key, value in filters.items():
                    if hasattr(self.model_class, key):
                        column = getattr(self.model_class, key)

                        if isinstance(value, list):
                            conditions.append(column.in_(value))
                        elif isinstance(value, dict):
                            if "gte" in value:
                                conditions.append(column >= value["gte"])
                            if "lte" in value:
                                conditions.append(column <= value["lte"])
                            if "gt" in value:
                                conditions.append(column > value["gt"])
                            if "lt" in value:
                                conditions.append(column < value["lt"])
                        else:
                            conditions.append(column == value)

                if conditions:
                    stmt = stmt.where(and_(*conditions))

            result = await self.session.execute(stmt)
            return result.scalar()

        except Exception as e:
            self.logger.error(f"Error counting entities: {e}")
            raise

    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists."""
        try:
            stmt = select(func.count(self.model_class.id)).where(self.model_class.id == entity_id)
            result = await self.session.execute(stmt)
            count = result.scalar()
            return count > 0

        except Exception as e:
            self.logger.error(f"Error checking entity existence {entity_id}: {e}")
            raise

    async def find_by(self, **criteria) -> List[T]:
        """Find entities by criteria."""
        return await self.list(**criteria)

    async def find_one_by(self, **criteria) -> Optional[T]:
        """Find single entity by criteria."""
        entities = await self.list(limit=1, **criteria)
        return entities[0] if entities else None

    @abstractmethod
    def _entity_to_model(self, entity: T) -> Any:
        """Convert domain entity to SQLAlchemy model."""
        pass

    @abstractmethod
    def _model_to_entity(self, model: Any) -> T:
        """Convert SQLAlchemy model to domain entity."""
        pass

    @abstractmethod
    async def _update_model_from_entity(self, model: Any, entity: T) -> None:
        """Update SQLAlchemy model from domain entity."""
        pass

class InMemoryRepository(Repository[T], LoggerMixin, Generic[T]):
    """In-memory repository implementation for testing."""

    def __init__(self):
        self._entities: Dict[str, T] = {}
        self._next_id = 1

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    async def save(self, entity: T) -> T:
        """Save entity."""
        if not entity.id:
            entity.id = str(self._next_id)
            self._next_id += 1

        # Simulate optimistic locking
        existing = self._entities.get(entity.id)
        if existing and existing.version != entity.version - 1:
            raise ConcurrencyError("Entity was modified by another process")

        self._entities[entity.id] = entity
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    async def list(self, limit: int = 100, offset: int = 0, **filters) -> List[T]:
        """List entities with pagination and filters."""
        entities = list(self._entities.values())

        # Apply filters
        if filters:
            filtered_entities = []
            for entity in entities:
                match = True
                for key, value in filters.items():
                    if hasattr(entity, key):
                        entity_value = getattr(entity, key)
                        if isinstance(value, dict):
                            # Range filters
                            if "gte" in value and entity_value < value["gte"]:
                                match = False
                                break
                            if "lte" in value and entity_value > value["lte"]:
                                match = False
                                break
                        elif entity_value != value:
                            match = False
                            break
                    else:
                        match = False
                        break

                if match:
                    filtered_entities.append(entity)

            entities = filtered_entities

        # Apply pagination
        start = offset
        end = offset + limit
        return entities[start:end]

    async def count(self, **filters) -> int:
        """Count entities with filters."""
        entities = await self.list(**filters)
        return len(entities)

    def clear(self) -> None:
        """Clear all entities (for testing)."""
        self._entities.clear()
        self._next_id = 1
