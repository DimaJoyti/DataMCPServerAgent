"""
RAG (Retrieval-Augmented Generation) Architecture.

This module provides comprehensive RAG capabilities including:
- Hybrid search (vector + keyword + semantic)
- Adaptive chunking strategies
- Multi-vector stores with different embedding models
- Advanced reranking and result fusion
- Context-aware retrieval optimization
"""

from .hybrid_search import (
    HybridSearchEngine,
    SearchQuery,
    SearchResult,
    SearchFilters,
    RankedResults
)
from .adaptive_chunking import (
    AdaptiveChunker,
    ChunkingStrategy,
    ChunkMetadata,
    ChunkedDocument
)
from .multi_vector import (
    MultiVectorStore,
    VectorStoreConfig,
    EmbeddingModel,
    VectorIndex
)
from .reranking import (
    ReRanker,
    RerankingStrategy,
    ScoredResult,
    RerankingMetrics
)

__all__ = [
    # Hybrid Search
    "HybridSearchEngine",
    "SearchQuery",
    "SearchResult",
    "SearchFilters",
    "RankedResults",

    # Adaptive Chunking
    "AdaptiveChunker",
    "ChunkingStrategy",
    "ChunkMetadata",
    "ChunkedDocument",

    # Multi-Vector Store
    "MultiVectorStore",
    "VectorStoreConfig",
    "EmbeddingModel",
    "VectorIndex",

    # Reranking
    "ReRanker",
    "RerankingStrategy",
    "ScoredResult",
    "RerankingMetrics",
]
