"""
RAG (Retrieval-Augmented Generation) Architecture.

This module provides comprehensive RAG capabilities including:
- Hybrid search (vector + keyword + semantic)
- Adaptive chunking strategies
- Multi-vector stores with different embedding models
- Advanced reranking and result fusion
- Context-aware retrieval optimization
"""

from .adaptive_chunking import AdaptiveChunker, ChunkedDocument, ChunkingStrategy, ChunkMetadata
from .hybrid_search import (
    HybridSearchEngine,
    RankedResults,
    SearchFilters,
    SearchQuery,
    SearchResult,
)
from .multi_vector import EmbeddingModel, MultiVectorStore, VectorIndex, VectorStoreConfig
from .reranking import ReRanker, RerankingMetrics, RerankingStrategy, ScoredResult

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
