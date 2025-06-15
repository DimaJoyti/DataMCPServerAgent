"""
Batch processor for vectorizing large amounts of text efficiently.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..document_processing.chunking.base_chunker import TextChunk
from .embeddings.base_embedder import BaseEmbedder, EmbeddingResult
from .vector_cache import CacheConfig, VectorCache


class BatchProcessingConfig(BaseModel):
    """Configuration for batch vector processing."""

    # Processing parameters
    batch_size: int = Field(default=100, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")

    # Performance options
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_config: Optional[CacheConfig] = Field(None, description="Cache configuration")

    # Progress tracking
    show_progress: bool = Field(default=True, description="Show progress information")
    progress_interval: int = Field(default=100, description="Progress update interval")

    # Error handling
    continue_on_error: bool = Field(default=True, description="Continue processing on errors")
    max_retries: int = Field(default=3, description="Maximum retry attempts per batch")

    # Rate limiting
    rate_limit: Optional[float] = Field(None, description="Rate limit (requests per second)")

    # Memory management
    clear_cache_interval: int = Field(default=1000, description="Clear cache every N items")


class BatchProcessingResult(BaseModel):
    """Result of batch vector processing."""

    # Processing statistics
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successfully processed items")
    failed_items: int = Field(..., description="Number of failed items")
    cached_items: int = Field(default=0, description="Number of items retrieved from cache")

    # Timing information
    total_time: float = Field(..., description="Total processing time in seconds")
    average_time_per_item: float = Field(..., description="Average time per item in seconds")

    # Results
    results: List[EmbeddingResult] = Field(..., description="Embedding results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Processing errors")

    # Cache statistics
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_items / self.total_items if self.total_items > 0 else 0.0

    def get_successful_results(self) -> List[EmbeddingResult]:
        """Get only successful results."""
        return [r for r in self.results if r is not None]


class BatchVectorProcessor:
    """Batch processor for vectorizing large amounts of text efficiently."""

    def __init__(self, embedder: BaseEmbedder, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize batch vector processor.

        Args:
            embedder: Embedder instance
            config: Processing configuration
        """
        self.embedder = embedder
        self.config = config or BatchProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize cache if enabled
        self.cache = None
        if self.config.enable_caching:
            cache_config = self.config.cache_config or CacheConfig()
            self.cache = VectorCache(cache_config)

        # Rate limiting
        self._last_request_time = 0.0

    def process_texts(self, texts: List[str]) -> BatchProcessingResult:
        """
        Process a list of texts to generate embeddings.

        Args:
            texts: List of texts to vectorize

        Returns:
            BatchProcessingResult: Processing result
        """
        start_time = time.time()

        self.logger.info(f"Starting batch processing of {len(texts)} texts")

        # Initialize result tracking
        results = [None] * len(texts)
        errors = []
        cached_count = 0

        # Check cache first if enabled
        if self.cache:
            results, cached_indices = self._check_cache(texts, results)
            cached_count = len(cached_indices)

            if cached_count > 0:
                self.logger.info(f"Found {cached_count} cached results")

        # Find texts that need processing
        pending_indices = [i for i, result in enumerate(results) if result is None]
        pending_texts = [texts[i] for i in pending_indices]

        if pending_texts:
            # Process pending texts
            processed_results = self._process_pending_texts(pending_texts, pending_indices)

            # Update results
            for i, result in zip(pending_indices, processed_results):
                if result is not None:
                    results[i] = result

                    # Cache result if enabled
                    if self.cache and result is not None:
                        self.cache.set(result.text_hash, result)
                else:
                    errors.append(
                        {
                            "index": i,
                            "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                            "error": "Processing failed",
                        }
                    )

        # Calculate statistics
        total_time = time.time() - start_time
        successful_results = [r for r in results if r is not None]
        successful_count = len(successful_results)
        failed_count = len(texts) - successful_count

        # Get cache statistics
        cache_hit_rate = None
        if self.cache:
            cache_stats = self.cache.get_stats()
            if cache_stats:
                cache_hit_rate = cache_stats.hit_rate

        result = BatchProcessingResult(
            total_items=len(texts),
            successful_items=successful_count,
            failed_items=failed_count,
            cached_items=cached_count,
            total_time=total_time,
            average_time_per_item=total_time / len(texts) if texts else 0.0,
            results=results,
            errors=errors,
            cache_hit_rate=cache_hit_rate,
        )

        self.logger.info(
            f"Batch processing completed: {successful_count}/{len(texts)} successful, "
            f"{cached_count} cached, {total_time:.2f}s total"
        )

        return result

    def process_chunks(self, chunks: List[TextChunk]) -> BatchProcessingResult:
        """
        Process a list of text chunks to generate embeddings.

        Args:
            chunks: List of text chunks to vectorize

        Returns:
            BatchProcessingResult: Processing result
        """
        texts = [chunk.text for chunk in chunks]
        result = self.process_texts(texts)

        # Update chunk metadata with embedding results
        for i, (chunk, embedding_result) in enumerate(zip(chunks, result.results)):
            if embedding_result is not None:
                chunk.metadata.add_custom_field("embedding_model", embedding_result.model_name)
                chunk.metadata.add_custom_field(
                    "embedding_dimension", embedding_result.embedding_dimension
                )
                chunk.metadata.add_custom_field(
                    "embedding_processing_time", embedding_result.processing_time
                )

        return result

    def _check_cache(
        self, texts: List[str], results: List[Optional[EmbeddingResult]]
    ) -> Tuple[List[Optional[EmbeddingResult]], List[int]]:
        """
        Check cache for existing embeddings.

        Args:
            texts: List of texts
            results: Current results list

        Returns:
            Tuple of updated results and cached indices
        """
        cached_indices = []

        for i, text in enumerate(texts):
            text_hash = self.embedder._create_text_hash(text)
            cached_result = self.cache.get(text_hash)

            if cached_result is not None:
                results[i] = cached_result
                cached_indices.append(i)

        return results, cached_indices

    def _process_pending_texts(
        self, texts: List[str], indices: List[int]
    ) -> List[Optional[EmbeddingResult]]:
        """
        Process texts that are not in cache.

        Args:
            texts: List of texts to process
            indices: Original indices of texts

        Returns:
            List of embedding results
        """
        if not texts:
            return []

        results = []

        # Process in batches
        batch_size = min(self.config.batch_size, len(texts))

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_indices = indices[i : i + batch_size]

            # Apply rate limiting
            self._apply_rate_limit()

            # Process batch
            batch_results = self._process_batch(batch_texts, batch_indices)
            results.extend(batch_results)

            # Show progress
            if self.config.show_progress and (i + batch_size) % self.config.progress_interval == 0:
                processed = min(i + batch_size, len(texts))
                self.logger.info(f"Processed {processed}/{len(texts)} texts")

            # Clear cache periodically to manage memory
            if (
                self.cache
                and self.config.clear_cache_interval > 0
                and (i + batch_size) % self.config.clear_cache_interval == 0
            ):
                # This would be a partial clear in a real implementation
                pass

        return results

    def _process_batch(
        self, texts: List[str], indices: List[int]
    ) -> List[Optional[EmbeddingResult]]:
        """
        Process a single batch of texts.

        Args:
            texts: Batch of texts
            indices: Original indices

        Returns:
            List of embedding results
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                # Use embedder's batch processing
                results = self.embedder.embed_batch(texts)
                return results

            except Exception as e:
                self.logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    # Wait before retry
                    time.sleep(2**attempt)
                elif self.config.continue_on_error:
                    # Return None results for failed batch
                    return [None] * len(texts)
                else:
                    raise

        return [None] * len(texts)

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self.config.rate_limit is None:
            return

        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.config.rate_limit

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    async def process_texts_async(self, texts: List[str]) -> BatchProcessingResult:
        """
        Process texts asynchronously.

        Args:
            texts: List of texts to vectorize

        Returns:
            BatchProcessingResult: Processing result
        """
        # Run synchronous processing in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            result = await loop.run_in_executor(executor, self.process_texts, texts)

        return result

    def estimate_processing_time(self, num_texts: int) -> float:
        """
        Estimate processing time for a number of texts.

        Args:
            num_texts: Number of texts to process

        Returns:
            float: Estimated time in seconds
        """
        # Base estimate: assume 0.1 seconds per text
        base_time_per_text = 0.1

        # Account for batching efficiency
        num_batches = (num_texts + self.config.batch_size - 1) // self.config.batch_size
        batch_overhead = num_batches * 0.05  # 50ms overhead per batch

        # Account for rate limiting
        rate_limit_time = 0.0
        if self.config.rate_limit:
            rate_limit_time = num_texts / self.config.rate_limit

        total_time = (num_texts * base_time_per_text) + batch_overhead + rate_limit_time

        return total_time

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if not self.cache:
            return None

        stats = self.cache.get_stats()
        if not stats:
            return None

        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "size": self.cache.size(),
        }
