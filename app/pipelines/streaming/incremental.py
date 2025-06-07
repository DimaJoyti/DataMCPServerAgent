"""
Incremental Processing for Streaming Pipeline.

This module provides incremental update capabilities for maintaining
vector stores, indexes, and other data structures in real-time:
- Incremental vector updates
- Index maintenance
- Change detection
- Conflict resolution
- Batch optimization
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

class UpdateStrategy(str, Enum):
    """Strategies for incremental updates."""
    IMMEDIATE = "immediate"  # Apply updates immediately
    BATCHED = "batched"     # Batch updates for efficiency
    SCHEDULED = "scheduled"  # Apply updates on schedule
    ADAPTIVE = "adaptive"   # Adapt strategy based on load

class UpdateType(str, Enum):
    """Types of incremental updates."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"
    BATCH_INSERT = "batch_insert"
    BATCH_UPDATE = "batch_update"
    BATCH_DELETE = "batch_delete"

class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    MANUAL = "manual"
    VERSION_BASED = "version_based"

@dataclass
class IncrementalUpdate:
    """Represents an incremental update operation."""

    # Update identification
    update_id: str
    document_id: str
    update_type: UpdateType
    timestamp: float

    # Update content
    data: Any
    metadata: Dict[str, Any]

    # Versioning
    version: Optional[int] = None
    previous_version: Optional[int] = None

    # Processing info
    priority: int = 0  # Higher number = higher priority
    dependencies: List[str] = None  # List of update_ids this depends on

    # Status
    applied: bool = False
    failed: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class IndexManager:
    """Manages incremental updates to various indexes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize index manager."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Index state
        self.indexes: Dict[str, Dict[str, Any]] = {}
        self.pending_updates: Dict[str, IncrementalUpdate] = {}
        self.applied_updates: Dict[str, IncrementalUpdate] = {}

        # Configuration
        self.batch_size = self.config.get("batch_size", 100)
        self.batch_timeout = self.config.get("batch_timeout", 5.0)
        self.max_retries = self.config.get("max_retries", 3)

        self.logger.info("IndexManager initialized")

    async def create_index(self, index_name: str, schema: Dict[str, Any]) -> bool:
        """Create a new index."""
        try:
            self.indexes[index_name] = {
                "schema": schema,
                "documents": {},
                "metadata": {
                    "created_at": time.time(),
                    "document_count": 0,
                    "last_updated": time.time()
                }
            }

            self.logger.info(f"Created index: {index_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")
            return False

    async def apply_update(self, update: IncrementalUpdate, index_name: str) -> bool:
        """Apply an incremental update to an index."""
        try:
            if index_name not in self.indexes:
                self.logger.error(f"Index {index_name} does not exist")
                return False

            index = self.indexes[index_name]

            # Apply update based on type
            if update.update_type == UpdateType.INSERT:
                success = await self._apply_insert(update, index)
            elif update.update_type == UpdateType.UPDATE:
                success = await self._apply_update(update, index)
            elif update.update_type == UpdateType.DELETE:
                success = await self._apply_delete(update, index)
            elif update.update_type == UpdateType.UPSERT:
                success = await self._apply_upsert(update, index)
            else:
                self.logger.error(f"Unsupported update type: {update.update_type}")
                return False

            if success:
                update.applied = True
                self.applied_updates[update.update_id] = update
                index["metadata"]["last_updated"] = time.time()

                self.logger.debug(f"Applied update {update.update_id} to index {index_name}")
            else:
                update.failed = True

            return success

        except Exception as e:
            error_msg = f"Failed to apply update {update.update_id}: {e}"
            self.logger.error(error_msg)
            update.failed = True
            update.error_message = error_msg
            return False

    async def _apply_insert(self, update: IncrementalUpdate, index: Dict[str, Any]) -> bool:
        """Apply insert update."""
        document_id = update.document_id

        if document_id in index["documents"]:
            self.logger.warning(f"Document {document_id} already exists, skipping insert")
            return False

        index["documents"][document_id] = {
            "data": update.data,
            "metadata": update.metadata,
            "version": update.version or 1,
            "created_at": update.timestamp,
            "updated_at": update.timestamp
        }

        index["metadata"]["document_count"] += 1
        return True

    async def _apply_update(self, update: IncrementalUpdate, index: Dict[str, Any]) -> bool:
        """Apply update operation."""
        document_id = update.document_id

        if document_id not in index["documents"]:
            self.logger.warning(f"Document {document_id} does not exist, cannot update")
            return False

        existing_doc = index["documents"][document_id]

        # Version check
        if update.version and existing_doc.get("version", 0) >= update.version:
            self.logger.warning(f"Version conflict for document {document_id}")
            return False

        # Update document
        existing_doc["data"] = update.data
        existing_doc["metadata"].update(update.metadata)
        existing_doc["version"] = update.version or (existing_doc.get("version", 0) + 1)
        existing_doc["updated_at"] = update.timestamp

        return True

    async def _apply_delete(self, update: IncrementalUpdate, index: Dict[str, Any]) -> bool:
        """Apply delete operation."""
        document_id = update.document_id

        if document_id not in index["documents"]:
            self.logger.warning(f"Document {document_id} does not exist, cannot delete")
            return False

        del index["documents"][document_id]
        index["metadata"]["document_count"] -= 1
        return True

    async def _apply_upsert(self, update: IncrementalUpdate, index: Dict[str, Any]) -> bool:
        """Apply upsert operation."""
        document_id = update.document_id

        if document_id in index["documents"]:
            # Update existing
            return await self._apply_update(update, index)
        else:
            # Insert new
            return await self._apply_insert(update, index)

    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an index."""
        if index_name not in self.indexes:
            return None

        index = self.indexes[index_name]
        return {
            "name": index_name,
            "document_count": index["metadata"]["document_count"],
            "created_at": index["metadata"]["created_at"],
            "last_updated": index["metadata"]["last_updated"],
            "schema": index["schema"]
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all indexes."""
        return {
            "indexes": {name: self.get_index_stats(name) for name in self.indexes.keys()},
            "pending_updates": len(self.pending_updates),
            "applied_updates": len(self.applied_updates)
        }

class IncrementalProcessor:
    """Processes incremental updates with batching and optimization."""

    def __init__(self, index_manager: IndexManager, config: Optional[Dict[str, Any]] = None):
        """Initialize incremental processor."""
        self.index_manager = index_manager
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Processing configuration
        self.strategy = UpdateStrategy(self.config.get("strategy", UpdateStrategy.BATCHED))
        self.batch_size = self.config.get("batch_size", 50)
        self.batch_timeout = self.config.get("batch_timeout", 5.0)
        self.conflict_resolution = ConflictResolution(
            self.config.get("conflict_resolution", ConflictResolution.LAST_WRITE_WINS)
        )

        # Processing state
        self.is_running = False
        self.update_queue: asyncio.Queue[IncrementalUpdate] = asyncio.Queue()
        self.batch_buffer: List[IncrementalUpdate] = []
        self.last_batch_time = time.time()

        # Tasks
        self.processor_task: Optional[asyncio.Task] = None

        self.logger.info(f"IncrementalProcessor initialized with strategy: {self.strategy}")

    async def start(self) -> None:
        """Start the incremental processor."""
        if self.is_running:
            return

        self.is_running = True
        self.processor_task = asyncio.create_task(self._processor_loop())
        self.logger.info("IncrementalProcessor started")

    async def stop(self) -> None:
        """Stop the incremental processor."""
        if not self.is_running:
            return

        self.is_running = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        # Process remaining updates
        await self._flush_batch()

        self.logger.info("IncrementalProcessor stopped")

    async def submit_update(self, update: IncrementalUpdate) -> bool:
        """Submit an update for processing."""
        if not self.is_running:
            self.logger.warning("Processor is not running")
            return False

        try:
            await self.update_queue.put(update)
            self.logger.debug(f"Submitted update {update.update_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to submit update: {e}")
            return False

    async def _processor_loop(self) -> None:
        """Main processor loop."""
        while self.is_running:
            try:
                if self.strategy == UpdateStrategy.IMMEDIATE:
                    await self._process_immediate()
                elif self.strategy == UpdateStrategy.BATCHED:
                    await self._process_batched()
                elif self.strategy == UpdateStrategy.SCHEDULED:
                    await self._process_scheduled()
                elif self.strategy == UpdateStrategy.ADAPTIVE:
                    await self._process_adaptive()

            except Exception as e:
                self.logger.error(f"Error in processor loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_immediate(self) -> None:
        """Process updates immediately."""
        try:
            update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
            await self._apply_single_update(update)
        except asyncio.TimeoutError:
            pass

    async def _process_batched(self) -> None:
        """Process updates in batches."""
        try:
            # Try to get an update with timeout
            update = await asyncio.wait_for(self.update_queue.get(), timeout=0.1)
            self.batch_buffer.append(update)

            # Check if we should flush the batch
            should_flush = (
                len(self.batch_buffer) >= self.batch_size or
                time.time() - self.last_batch_time >= self.batch_timeout
            )

            if should_flush:
                await self._flush_batch()

        except asyncio.TimeoutError:
            # Check timeout-based flush
            if (self.batch_buffer and
                time.time() - self.last_batch_time >= self.batch_timeout):
                await self._flush_batch()

    async def _process_scheduled(self) -> None:
        """Process updates on schedule."""
        # Simple scheduled processing - flush every batch_timeout seconds
        await asyncio.sleep(self.batch_timeout)

        # Collect all pending updates
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                self.batch_buffer.append(update)
            except asyncio.QueueEmpty:
                break

        if self.batch_buffer:
            await self._flush_batch()

    async def _process_adaptive(self) -> None:
        """Adaptive processing based on load."""
        queue_size = self.update_queue.qsize()

        if queue_size > self.batch_size * 2:
            # High load - use batched processing
            await self._process_batched()
        elif queue_size > 0:
            # Medium load - use immediate processing
            await self._process_immediate()
        else:
            # Low load - wait a bit
            await asyncio.sleep(0.1)

    async def _flush_batch(self) -> None:
        """Flush the current batch of updates."""
        if not self.batch_buffer:
            return

        self.logger.debug(f"Flushing batch of {len(self.batch_buffer)} updates")

        # Sort updates by priority and dependencies
        sorted_updates = self._sort_updates(self.batch_buffer)

        # Apply updates
        for update in sorted_updates:
            # Determine target index (simplified - in production, this would be more sophisticated)
            index_name = update.metadata.get("index_name", "default")

            # Ensure index exists
            if index_name not in self.index_manager.indexes:
                await self.index_manager.create_index(index_name, {"type": "document"})

            # Apply update
            await self.index_manager.apply_update(update, index_name)

        # Clear batch
        self.batch_buffer.clear()
        self.last_batch_time = time.time()

    def _sort_updates(self, updates: List[IncrementalUpdate]) -> List[IncrementalUpdate]:
        """Sort updates by priority and dependencies."""
        # Simple sorting by priority (higher first) and timestamp (older first)
        return sorted(updates, key=lambda u: (-u.priority, u.timestamp))

    async def _apply_single_update(self, update: IncrementalUpdate) -> None:
        """Apply a single update."""
        index_name = update.metadata.get("index_name", "default")

        # Ensure index exists
        if index_name not in self.index_manager.indexes:
            await self.index_manager.create_index(index_name, {"type": "document"})

        # Apply update
        await self.index_manager.apply_update(update, index_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "is_running": self.is_running,
            "strategy": self.strategy,
            "queue_size": self.update_queue.qsize(),
            "batch_buffer_size": len(self.batch_buffer),
            "last_batch_time": self.last_batch_time,
            "index_manager_stats": self.index_manager.get_all_stats()
        }
