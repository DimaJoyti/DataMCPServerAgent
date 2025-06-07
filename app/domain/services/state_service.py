"""
State domain services.
Contains business logic for state management and synchronization.
"""

from typing import Any, Dict, Optional

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService, ValidationError
from app.domain.models.state import PersistentState, StateMetadata, StateType, StateVersion

logger = get_logger(__name__)

class StateService(DomainService, LoggerMixin):
    """Core state management service."""

    async def save_state(
        self,
        entity_id: str,
        entity_type: str,
        state_type: StateType,
        state_data: Dict[str, Any],
        created_by: str,
    ) -> PersistentState:
        """Save entity state."""
        self.logger.info(f"Saving state for {entity_type} {entity_id}")

        # Create metadata
        import hashlib
        import json

        data_json = json.dumps(state_data, sort_keys=True)
        checksum = hashlib.sha256(data_json.encode()).hexdigest()

        metadata = StateMetadata(
            checksum=checksum, size_bytes=len(data_json.encode()), compression=None, encryption=None
        )

        # Create version
        version = StateVersion(version_number=1, created_by=created_by, description="Initial state")

        # Create state
        state = PersistentState(
            entity_id=entity_id,
            entity_type=entity_type,
            state_type=state_type,
            state_data=state_data,
            current_version=version,
            metadata=metadata,
            storage_key=f"{entity_type}:{entity_id}:{state_type.value}",
        )

        # Save state
        state_repo = self.get_repository("state")
        saved_state = await state_repo.save(state)

        self.logger.info(f"State saved successfully: {saved_state.id}")
        return saved_state

    async def load_state(
        self, entity_id: str, entity_type: str, state_type: StateType
    ) -> Optional[PersistentState]:
        """Load entity state."""
        state_repo = self.get_repository("state")
        states = await state_repo.list(
            entity_id=entity_id, entity_type=entity_type, state_type=state_type
        )

        return states[0] if states else None

class StateSynchronizationService(DomainService, LoggerMixin):
    """Service for state synchronization operations."""

    async def sync_state(self, state_id: str) -> bool:
        """Synchronize state with storage backend."""
        state_repo = self.get_repository("state")
        state = await state_repo.get_by_id(state_id)

        if not state:
            raise ValidationError(f"State not found: {state_id}")

        # Mark as synced
        state.mark_as_synced("local_sync")
        await state_repo.save(state)

        return True
