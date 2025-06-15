"""
Vectorization module for converting text to embeddings.

This module provides comprehensive vectorization capabilities including:
- Text embedding generation using various models
- Batch processing for large datasets
- Caching for improved performance
- Integration with multiple embedding providers
"""

from .batch_processor import BatchProcessingConfig, BatchVectorProcessor
from .embeddings import (
    BaseEmbedder,
    CloudflareEmbedder,
    EmbeddingConfig,
    EmbeddingResult,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
)
from .vector_cache import CacheConfig, VectorCache

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    # Embeddings
    "BaseEmbedder",
    "EmbeddingConfig",
    "EmbeddingResult",
    "OpenAIEmbedder",
    "HuggingFaceEmbedder",
    "CloudflareEmbedder",
    # Batch processing
    "BatchVectorProcessor",
    "BatchProcessingConfig",
    # Caching
    "VectorCache",
    "CacheConfig",
]
