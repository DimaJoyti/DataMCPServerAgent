"""
Base schema definitions for vector stores.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class VectorStoreType(str, Enum):
    """Supported vector store types."""

    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class IndexType(str, Enum):
    """Vector index types."""

    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    LSH = "lsh"


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""

    # Store identification
    store_type: VectorStoreType = Field(..., description="Type of vector store")
    collection_name: str = Field(..., description="Name of the collection/index")

    # Vector configuration
    embedding_dimension: int = Field(..., description="Dimension of embedding vectors")
    distance_metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE, description="Distance metric"
    )
    index_type: IndexType = Field(default=IndexType.HNSW, description="Index type")

    # Connection settings
    host: Optional[str] = Field(None, description="Host address")
    port: Optional[int] = Field(None, description="Port number")
    api_key: Optional[str] = Field(None, description="API key for cloud services")

    # Performance settings
    batch_size: int = Field(default=100, description="Batch size for operations")
    max_connections: int = Field(default=10, description="Maximum connections")
    timeout: float = Field(default=30.0, description="Operation timeout in seconds")

    # Index settings
    index_params: Dict[str, Any] = Field(
        default_factory=dict, description="Index-specific parameters"
    )

    # Storage settings
    persist_directory: Optional[str] = Field(None, description="Directory for persistent storage")

    # Custom settings
    custom_config: Dict[str, Any] = Field(
        default_factory=dict, description="Store-specific configuration"
    )

    @validator("embedding_dimension")
    def validate_embedding_dimension(cls, v):
        """Validate embedding dimension."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class VectorRecord(BaseModel):
    """Base vector record for storage."""

    # Identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique record ID")

    # Vector data
    vector: List[float] = Field(..., description="Embedding vector")

    # Content
    text: str = Field(..., description="Original text content")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    # Source information
    source: Optional[str] = Field(None, description="Source identifier")
    source_type: Optional[str] = Field(None, description="Type of source")

    @validator("vector")
    def validate_vector(cls, v):
        """Validate vector."""
        if not v:
            raise ValueError("Vector cannot be empty")
        return v

    def get_vector_dimension(self) -> int:
        """Get vector dimension."""
        return len(self.vector)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata field."""
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field."""
        return self.metadata.get(key, default)


class BaseVectorSchema(ABC):
    """Abstract base class for vector store schemas."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize schema.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def create_record(self, **kwargs) -> VectorRecord:
        """
        Create a vector record.

        Args:
            **kwargs: Record data

        Returns:
            VectorRecord: Created record
        """
        pass

    @abstractmethod
    def validate_record(self, record: VectorRecord) -> bool:
        """
        Validate a vector record.

        Args:
            record: Record to validate

        Returns:
            bool: True if valid
        """
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields for this schema.

        Returns:
            List[str]: Required field names
        """
        pass

    @abstractmethod
    def get_searchable_fields(self) -> List[str]:
        """
        Get list of searchable metadata fields.

        Returns:
            List[str]: Searchable field names
        """
        pass

    def prepare_for_storage(self, record: VectorRecord) -> Dict[str, Any]:
        """
        Prepare record for storage in vector store.

        Args:
            record: Record to prepare

        Returns:
            Dict[str, Any]: Prepared data
        """
        # Validate record
        if not self.validate_record(record):
            raise ValueError("Invalid record")

        # Convert to storage format
        storage_data = {
            "id": record.id,
            "vector": record.vector,
            "text": record.text,
            "metadata": record.metadata.copy(),
            "created_at": record.created_at.isoformat(),
            "source": record.source,
            "source_type": record.source_type,
        }

        if record.updated_at:
            storage_data["metadata"]["updated_at"] = record.updated_at.isoformat()

        return storage_data

    def restore_from_storage(self, storage_data: Dict[str, Any]) -> VectorRecord:
        """
        Restore record from storage format.

        Args:
            storage_data: Data from storage

        Returns:
            VectorRecord: Restored record
        """
        # Parse timestamps
        created_at = datetime.fromisoformat(storage_data["created_at"])
        updated_at = None
        if "updated_at" in storage_data.get("metadata", {}):
            updated_at = datetime.fromisoformat(storage_data["metadata"]["updated_at"])
            # Remove from metadata to avoid duplication
            storage_data["metadata"] = storage_data["metadata"].copy()
            del storage_data["metadata"]["updated_at"]

        return VectorRecord(
            id=storage_data["id"],
            vector=storage_data["vector"],
            text=storage_data["text"],
            metadata=storage_data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            source=storage_data.get("source"),
            source_type=storage_data.get("source_type"),
        )

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information.

        Returns:
            Dict[str, Any]: Schema information
        """
        return {
            "schema_type": self.__class__.__name__,
            "embedding_dimension": self.config.embedding_dimension,
            "distance_metric": self.config.distance_metric,
            "required_fields": self.get_required_fields(),
            "searchable_fields": self.get_searchable_fields(),
        }
