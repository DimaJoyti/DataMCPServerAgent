"""
Base vector store interface.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..schemas.base_schema import VectorRecord, VectorStoreConfig
from ..schemas.search_models import SearchQuery, SearchResults

class VectorStoreStats(BaseModel):
    """Vector store statistics."""

    total_vectors: int = Field(default=0, description="Total number of vectors")
    index_size: Optional[int] = Field(None, description="Index size in bytes")
    memory_usage: Optional[int] = Field(None, description="Memory usage in bytes")

    # Performance metrics
    avg_insert_time: Optional[float] = Field(None, description="Average insert time in seconds")
    avg_search_time: Optional[float] = Field(None, description="Average search time in seconds")

    # Index information
    index_type: Optional[str] = Field(None, description="Type of index used")
    is_trained: Optional[bool] = Field(None, description="Whether index is trained")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Store creation time")
    last_updated: Optional[datetime] = Field(None, description="Last update time")

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the vector store connection."""
        pass

    @abstractmethod
    async def create_collection(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new collection/index.

        Args:
            schema: Optional schema definition

        Returns:
            bool: True if created successfully
        """
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        """
        Delete the collection/index.

        Returns:
            bool: True if deleted successfully
        """
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """
        Check if collection exists.

        Returns:
            bool: True if collection exists
        """
        pass

    @abstractmethod
    async def insert_vectors(self, records: List[VectorRecord]) -> List[str]:
        """
        Insert vector records.

        Args:
            records: List of vector records to insert

        Returns:
            List[str]: List of inserted record IDs
        """
        pass

    @abstractmethod
    async def update_vectors(self, records: List[VectorRecord]) -> List[str]:
        """
        Update existing vector records.

        Args:
            records: List of vector records to update

        Returns:
            List[str]: List of updated record IDs
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            int: Number of vectors deleted
        """
        pass

    @abstractmethod
    async def get_vector(self, id: str) -> Optional[VectorRecord]:
        """
        Get a vector by ID.

        Args:
            id: Vector ID

        Returns:
            Optional[VectorRecord]: Vector record if found
        """
        pass

    @abstractmethod
    async def search_vectors(self, query: SearchQuery) -> SearchResults:
        """
        Search for similar vectors.

        Args:
            query: Search query

        Returns:
            SearchResults: Search results
        """
        pass

    @abstractmethod
    async def get_stats(self) -> VectorStoreStats:
        """
        Get vector store statistics.

        Returns:
            VectorStoreStats: Store statistics
        """
        pass

    # Utility methods

    async def upsert_vectors(self, records: List[VectorRecord]) -> List[str]:
        """
        Insert or update vector records.

        Args:
            records: List of vector records

        Returns:
            List[str]: List of record IDs
        """
        # Check which records exist
        existing_ids = []
        new_records = []
        update_records = []

        for record in records:
            existing = await self.get_vector(record.id)
            if existing:
                update_records.append(record)
                existing_ids.append(record.id)
            else:
                new_records.append(record)

        # Insert new records
        inserted_ids = []
        if new_records:
            inserted_ids = await self.insert_vectors(new_records)

        # Update existing records
        updated_ids = []
        if update_records:
            updated_ids = await self.update_vectors(update_records)

        return inserted_ids + updated_ids

    async def batch_insert(
        self,
        records: List[VectorRecord],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Insert records in batches.

        Args:
            records: List of records to insert
            batch_size: Batch size (uses config default if None)

        Returns:
            List[str]: List of inserted IDs
        """
        if not records:
            return []

        batch_size = batch_size or self.config.batch_size
        all_ids = []

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_ids = await self.insert_vectors(batch)
            all_ids.extend(batch_ids)

            self.logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch_ids)} records")

        return all_ids

    async def count_vectors(self) -> int:
        """
        Count total number of vectors.

        Returns:
            int: Number of vectors
        """
        stats = await self.get_stats()
        return stats.total_vectors

    async def health_check(self) -> bool:
        """
        Perform health check on the vector store.

        Returns:
            bool: True if healthy
        """
        try:
            # Check if initialized
            if not self._is_initialized:
                return False

            # Check if collection exists
            if not await self.collection_exists():
                return False

            # Try to get stats
            await self.get_stats()

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def _validate_records(self, records: List[VectorRecord]) -> None:
        """
        Validate vector records.

        Args:
            records: Records to validate

        Raises:
            ValueError: If validation fails
        """
        if not records:
            raise ValueError("Records list cannot be empty")

        expected_dim = self.config.embedding_dimension

        for i, record in enumerate(records):
            if len(record.vector) != expected_dim:
                raise ValueError(
                    f"Record {i} has wrong vector dimension: "
                    f"{len(record.vector)} != {expected_dim}"
                )

            if not record.id:
                raise ValueError(f"Record {i} missing ID")

            if not record.text:
                raise ValueError(f"Record {i} missing text")

    def _prepare_records_for_storage(self, records: List[VectorRecord]) -> List[Dict[str, Any]]:
        """
        Prepare records for storage in the specific backend.

        Args:
            records: Records to prepare

        Returns:
            List[Dict[str, Any]]: Prepared records
        """
        prepared = []

        for record in records:
            # Convert to storage format
            storage_record = {
                "id": record.id,
                "vector": record.vector,
                "text": record.text,
                "metadata": record.metadata.copy(),
                "created_at": record.created_at.isoformat(),
                "source": record.source,
                "source_type": record.source_type
            }

            if record.updated_at:
                storage_record["metadata"]["updated_at"] = record.updated_at.isoformat()

            prepared.append(storage_record)

        return prepared

    def _restore_records_from_storage(self, storage_records: List[Dict[str, Any]]) -> List[VectorRecord]:
        """
        Restore records from storage format.

        Args:
            storage_records: Records from storage

        Returns:
            List[VectorRecord]: Restored records
        """
        restored = []

        for storage_record in storage_records:
            # Parse timestamps
            created_at = datetime.fromisoformat(storage_record["created_at"])
            updated_at = None

            metadata = storage_record.get("metadata", {}).copy()
            if "updated_at" in metadata:
                updated_at = datetime.fromisoformat(metadata["updated_at"])
                del metadata["updated_at"]

            record = VectorRecord(
                id=storage_record["id"],
                vector=storage_record["vector"],
                text=storage_record["text"],
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                source=storage_record.get("source"),
                source_type=storage_record.get("source_type")
            )

            restored.append(record)

        return restored

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(collection={self.config.collection_name})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"collection={self.config.collection_name}, "
            f"dimension={self.config.embedding_dimension}, "
            f"metric={self.config.distance_metric}"
            f")"
        )
