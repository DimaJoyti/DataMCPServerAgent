"""
Vector stores module for storing and searching embeddings.

This module provides comprehensive vector storage capabilities including:
- Multiple vector store backends (Chroma, FAISS, Pinecone, Weaviate)
- Flexible schema definitions for document metadata
- Hybrid search combining vector and keyword search
- Advanced filtering and querying capabilities
- Scalable indexing and retrieval
"""

from .backends import (
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    PineconeVectorStore,
    WeaviateVectorStore,
)
from .schemas import (
    BaseVectorSchema,
    DocumentVectorSchema,
    SearchQuery,
    SearchResult,
    VectorStoreConfig,
)
from .search import HybridSearchEngine, SearchFilters, VectorSearchEngine
from .vector_store_manager import VectorStoreFactory, VectorStoreManager

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    # Schemas
    "BaseVectorSchema",
    "DocumentVectorSchema",
    "VectorStoreConfig",
    "SearchQuery",
    "SearchResult",
    # Backends
    "BaseVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    # Search
    "VectorSearchEngine",
    "HybridSearchEngine",
    "SearchFilters",
    # Management
    "VectorStoreManager",
    "VectorStoreFactory",
]
