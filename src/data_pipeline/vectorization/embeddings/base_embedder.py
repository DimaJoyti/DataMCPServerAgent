"""
Base embedder interface for text vectorization.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    # Model configuration
    model_name: str = Field(..., description="Name of the embedding model")
    model_provider: str = Field(..., description="Embedding provider (openai, huggingface, etc.)")

    # Embedding parameters
    embedding_dimension: Optional[int] = Field(None, description="Embedding vector dimension")
    max_input_length: int = Field(default=8192, description="Maximum input text length")

    # Processing options
    normalize_embeddings: bool = Field(default=True, description="Normalize embedding vectors")
    batch_size: int = Field(default=100, description="Batch size for processing")

    # Performance options
    enable_caching: bool = Field(default=True, description="Enable embedding caching")
    cache_ttl: int = Field(default=86400, description="Cache TTL in seconds (24 hours)")

    # Error handling
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    # Custom options
    custom_options: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific options")

class EmbeddingResult(BaseModel):
    """Result of embedding generation."""

    # Input information
    text: str = Field(..., description="Input text")
    text_hash: str = Field(..., description="Hash of input text")

    # Embedding data
    embedding: List[float] = Field(..., description="Embedding vector")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")

    # Model information
    model_name: str = Field(..., description="Model used for embedding")
    model_provider: str = Field(..., description="Provider used for embedding")

    # Processing metadata
    processing_time: float = Field(..., description="Time taken to generate embedding (seconds)")
    token_count: Optional[int] = Field(None, description="Number of tokens processed")

    # Quality metrics
    embedding_norm: Optional[float] = Field(None, description="L2 norm of embedding vector")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    # Caching information
    from_cache: bool = Field(default=False, description="Whether result was retrieved from cache")

    def __len__(self) -> int:
        """Return embedding dimension."""
        return len(self.embedding)

    def get_vector(self) -> List[float]:
        """Get embedding vector."""
        return self.embedding

    def calculate_norm(self) -> float:
        """Calculate and return L2 norm of embedding."""
        import math
        norm = math.sqrt(sum(x * x for x in self.embedding))
        self.embedding_norm = norm
        return norm

    def normalize(self) -> "EmbeddingResult":
        """Normalize embedding vector to unit length."""
        norm = self.calculate_norm()
        if norm > 0:
            self.embedding = [x / norm for x in self.embedding]
            self.embedding_norm = 1.0
        return self

    def similarity(self, other: "EmbeddingResult") -> float:
        """
        Calculate cosine similarity with another embedding.

        Args:
            other: Another embedding result

        Returns:
            float: Cosine similarity (-1 to 1)
        """
        if len(self.embedding) != len(other.embedding):
            raise ValueError("Embeddings must have the same dimension")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))

        # Calculate norms
        norm_a = self.embedding_norm or self.calculate_norm()
        norm_b = other.embedding_norm or other.calculate_norm()

        # Calculate cosine similarity
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedder.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            EmbeddingResult: Embedding result
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List[EmbeddingResult]: List of embedding results
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            int: Embedding dimension
        """
        pass

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.model_name:
            raise ValueError("Model name is required")

        if not self.config.model_provider:
            raise ValueError("Model provider is required")

        if self.config.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.config.max_input_length <= 0:
            raise ValueError("Max input length must be positive")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.

        Args:
            text: Input text

        Returns:
            str: Preprocessed text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Truncate if too long
        if len(text) > self.config.max_input_length:
            self.logger.warning(f"Text truncated from {len(text)} to {self.config.max_input_length} characters")
            text = text[:self.config.max_input_length]

        # Basic cleaning
        text = text.strip()

        return text

    def _create_text_hash(self, text: str) -> str:
        """
        Create hash for text (for caching).

        Args:
            text: Input text

        Returns:
            str: Text hash
        """
        # Include model info in hash to avoid conflicts between models
        hash_input = f"{self.config.model_name}:{self.config.model_provider}:{text}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def _post_process_embedding(self, embedding: List[float], text: str) -> List[float]:
        """
        Post-process embedding vector.

        Args:
            embedding: Raw embedding vector
            text: Original text

        Returns:
            List[float]: Processed embedding vector
        """
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        # Normalize if configured
        if self.config.normalize_embeddings:
            import math
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

        return embedding

    def _create_embedding_result(
        self,
        text: str,
        embedding: List[float],
        processing_time: float,
        token_count: Optional[int] = None,
        from_cache: bool = False
    ) -> EmbeddingResult:
        """
        Create embedding result object.

        Args:
            text: Input text
            embedding: Embedding vector
            processing_time: Processing time
            token_count: Number of tokens
            from_cache: Whether from cache

        Returns:
            EmbeddingResult: Embedding result
        """
        text_hash = self._create_text_hash(text)

        result = EmbeddingResult(
            text=text,
            text_hash=text_hash,
            embedding=embedding,
            embedding_dimension=len(embedding),
            model_name=self.config.model_name,
            model_provider=self.config.model_provider,
            processing_time=processing_time,
            token_count=token_count,
            from_cache=from_cache
        )

        # Calculate norm
        result.calculate_norm()

        return result

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with retry and exponential backoff.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        import time

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_retries + 1} attempts failed")

        raise last_exception

    def health_check(self) -> bool:
        """
        Perform health check on the embedder.

        Returns:
            bool: True if healthy
        """
        try:
            # Try to embed a simple test text
            test_text = "This is a test."
            result = self.embed_text(test_text)
            return len(result.embedding) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
