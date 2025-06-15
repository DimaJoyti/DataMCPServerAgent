"""
Multi-Vector Store for RAG Architecture.

This module provides multi-vector storage capabilities:
- Multiple embedding models
- Specialized indexes
- Cross-modal search
- Unified interface
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.logging import LoggerMixin, get_logger


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    OPENAI_ADA = "openai_ada"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    CLOUDFLARE = "cloudflare"


@dataclass
class VectorIndex:
    """Vector index configuration."""

    index_name: str
    embedding_model: EmbeddingModel
    dimension: int
    metric: str = "cosine"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    store_type: str = Field(default="memory", description="Type of vector store")
    indexes: List[VectorIndex] = Field(default_factory=list, description="Vector indexes")
    max_vectors: int = Field(default=100000, description="Maximum number of vectors")

    class Config:
        arbitrary_types_allowed = True


class MultiVectorStore(LoggerMixin):
    """Multi-vector store with multiple embedding models."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize multi-vector store."""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Storage
        self.indexes: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, List[List[float]]] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize indexes
        for index_config in config.indexes:
            self._create_index(index_config)

        self.logger.info(f"MultiVectorStore initialized with {len(self.indexes)} indexes")

    def _create_index(self, index_config: VectorIndex) -> None:
        """Create a vector index."""
        self.indexes[index_config.index_name] = {
            "config": index_config,
            "vector_count": 0,
            "created_at": 0.0,
        }
        self.vectors[index_config.index_name] = []
        self.metadata[index_config.index_name] = []

        self.logger.info(f"Created index: {index_config.index_name}")

    async def add_vectors(
        self, index_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]
    ) -> bool:
        """Add vectors to an index."""
        if index_name not in self.indexes:
            self.logger.error(f"Index {index_name} does not exist")
            return False

        if len(vectors) != len(metadata):
            self.logger.error("Vectors and metadata must have the same length")
            return False

        # Add vectors
        self.vectors[index_name].extend(vectors)
        self.metadata[index_name].extend(metadata)

        # Update count
        self.indexes[index_name]["vector_count"] += len(vectors)

        self.logger.debug(f"Added {len(vectors)} vectors to index {index_name}")
        return True

    async def search(
        self, index_name: str, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if index_name not in self.indexes:
            self.logger.error(f"Index {index_name} does not exist")
            return []

        vectors = self.vectors[index_name]
        metadata = self.metadata[index_name]

        if not vectors:
            return []

        # Simple cosine similarity search (placeholder)
        similarities = []
        for i, vector in enumerate(vectors):
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((similarity, metadata[i]))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an index."""
        if index_name not in self.indexes:
            return None

        index_info = self.indexes[index_name]
        return {
            "index_name": index_name,
            "vector_count": index_info["vector_count"],
            "embedding_model": index_info["config"].embedding_model,
            "dimension": index_info["config"].dimension,
            "metric": index_info["config"].metric,
        }

    def list_indexes(self) -> List[str]:
        """List all available indexes."""
        return list(self.indexes.keys())
