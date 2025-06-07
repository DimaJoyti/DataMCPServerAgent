"""
Vectorization module for converting text to embeddings.

This module provides comprehensive vectorization capabilities including:
- Text embedding generation using various models
- Batch processing for large datasets
- Caching for improved performance
- Integration with multiple embedding providers
"""

from .embeddings import (
    BaseEmbedder,
    EmbeddingConfig,
    EmbeddingResult,
    OpenAIEmbedder,
    HuggingFaceEmbedder,
    CloudflareEmbedder
)
from .batch_processor import BatchVectorProcessor, BatchProcessingConfig
from .vector_cache import VectorCache, CacheConfig

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
