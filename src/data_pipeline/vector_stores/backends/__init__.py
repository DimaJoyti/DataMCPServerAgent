"""
Vector store backends for different storage systems.
"""

from .base_store import BaseVectorStore, VectorStoreStats
from .chroma_store import ChromaVectorStore
from .faiss_store import FAISSVectorStore
from .memory_store import MemoryVectorStore

# Optional backends (require additional dependencies)
try:
    from .pinecone_store import PineconeVectorStore

    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from .weaviate_store import WeaviateVectorStore

    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False

try:
    from .qdrant_store import QdrantVectorStore

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

__all__ = [
    "BaseVectorStore",
    "VectorStoreStats",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "MemoryVectorStore",
]

# Add optional backends to exports if available
if HAS_PINECONE:
    __all__.append("PineconeVectorStore")

if HAS_WEAVIATE:
    __all__.append("WeaviateVectorStore")

if HAS_QDRANT:
    __all__.append("QdrantVectorStore")
