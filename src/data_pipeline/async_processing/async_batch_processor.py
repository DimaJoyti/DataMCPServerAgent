"""
Asynchronous batch processor for vectorization and document processing.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..document_processing.chunking.models import TextChunk
from ..vectorization.batch_processor import BatchVectorProcessor

@dataclass
class AsyncBatchResult:
    """Result of async batch processing."""

    results: List[Any]
    successful_count: int
    failed_count: int
    total_time: float
    average_time_per_item: float
    errors: List[str]

    def get_successful_results(self) -> List[Any]:
        """Get only successful results (non-None)."""
        return [r for r in self.results if r is not None]

class AsyncBatchProcessor:
    """Asynchronous batch processor for high-throughput processing."""

    def __init__(
        self,
        batch_processor: BatchVectorProcessor,
        max_workers: int = 4,
        max_concurrent_batches: int = 2
    ):
        """
        Initialize async batch processor.

        Args:
            batch_processor: Underlying batch processor
            max_workers: Maximum number of worker threads
            max_concurrent_batches: Maximum number of concurrent batches
        """
        self.batch_processor = batch_processor
        self.max_workers = max_workers
        self.max_concurrent_batches = max_concurrent_batches

        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Semaphore to limit concurrent batches
        self.batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_texts_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> AsyncBatchResult:
        """
        Process texts asynchronously.

        Args:
            texts: List of texts to process
            batch_size: Batch size (uses processor default if None)
            progress_callback: Optional progress callback

        Returns:
            AsyncBatchResult: Processing results
        """
        start_time = time.time()
        batch_size = batch_size or self.batch_processor.config.batch_size

        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches):
            task = self._process_batch_async(batch, i, len(batches), progress_callback)
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        all_errors = []
        successful_count = 0
        failed_count = 0

        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch failed: {batch_result}")
                all_errors.append(str(batch_result))
                failed_count += batch_size  # Assume all items in batch failed
            else:
                results, errors = batch_result
                all_results.extend(results)
                all_errors.extend(errors)

                # Count successes and failures
                for result in results:
                    if result is not None:
                        successful_count += 1
                    else:
                        failed_count += 1

        total_time = time.time() - start_time
        average_time = total_time / len(texts) if texts else 0

        return AsyncBatchResult(
            results=all_results,
            successful_count=successful_count,
            failed_count=failed_count,
            total_time=total_time,
            average_time_per_item=average_time,
            errors=all_errors
        )

    async def process_chunks_async(
        self,
        chunks: List[TextChunk],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> AsyncBatchResult:
        """
        Process text chunks asynchronously.

        Args:
            chunks: List of text chunks to process
            batch_size: Batch size (uses processor default if None)
            progress_callback: Optional progress callback

        Returns:
            AsyncBatchResult: Processing results
        """
        start_time = time.time()
        batch_size = batch_size or self.batch_processor.config.batch_size

        # Split into batches
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches):
            task = self._process_chunk_batch_async(batch, i, len(batches), progress_callback)
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        all_errors = []
        successful_count = 0
        failed_count = 0

        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Chunk batch failed: {batch_result}")
                all_errors.append(str(batch_result))
                failed_count += len(batch)
            else:
                results, errors = batch_result
                all_results.extend(results)
                all_errors.extend(errors)

                # Count successes and failures
                for result in results:
                    if result is not None:
                        successful_count += 1
                    else:
                        failed_count += 1

        total_time = time.time() - start_time
        average_time = total_time / len(chunks) if chunks else 0

        return AsyncBatchResult(
            results=all_results,
            successful_count=successful_count,
            failed_count=failed_count,
            total_time=total_time,
            average_time_per_item=average_time,
            errors=all_errors
        )

    async def _process_batch_async(
        self,
        batch: List[str],
        batch_index: int,
        total_batches: int,
        progress_callback: Optional[callable] = None
    ) -> tuple[List[Any], List[str]]:
        """Process a single batch of texts asynchronously."""
        async with self.batch_semaphore:
            loop = asyncio.get_event_loop()

            try:
                # Run batch processing in executor
                result = await loop.run_in_executor(
                    self.executor,
                    self.batch_processor.process_texts,
                    batch
                )

                # Call progress callback
                if progress_callback:
                    progress = ((batch_index + 1) / total_batches) * 100
                    await self._safe_callback(
                        progress_callback,
                        batch_index + 1,
                        total_batches,
                        progress
                    )

                return result.results, result.errors

            except Exception as e:
                self.logger.error(f"Batch {batch_index} failed: {e}")
                return [None] * len(batch), [str(e)]

    async def _process_chunk_batch_async(
        self,
        batch: List[TextChunk],
        batch_index: int,
        total_batches: int,
        progress_callback: Optional[callable] = None
    ) -> tuple[List[Any], List[str]]:
        """Process a single batch of chunks asynchronously."""
        async with self.batch_semaphore:
            loop = asyncio.get_event_loop()

            try:
                # Run batch processing in executor
                result = await loop.run_in_executor(
                    self.executor,
                    self.batch_processor.process_chunks,
                    batch
                )

                # Call progress callback
                if progress_callback:
                    progress = ((batch_index + 1) / total_batches) * 100
                    await self._safe_callback(
                        progress_callback,
                        batch_index + 1,
                        total_batches,
                        progress
                    )

                return result.results, result.errors

            except Exception as e:
                self.logger.error(f"Chunk batch {batch_index} failed: {e}")
                return [None] * len(batch), [str(e)]

    async def process_with_retry(
        self,
        items: List[Union[str, TextChunk]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        progress_callback: Optional[callable] = None
    ) -> AsyncBatchResult:
        """
        Process items with retry logic.

        Args:
            items: Items to process
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            progress_callback: Optional progress callback

        Returns:
            AsyncBatchResult: Processing results
        """
        for attempt in range(max_retries + 1):
            try:
                if isinstance(items[0], str):
                    return await self.process_texts_async(items, progress_callback=progress_callback)
                else:
                    return await self.process_chunks_async(items, progress_callback=progress_callback)

            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"All retry attempts failed: {e}")
                    raise

                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    async def get_cache_stats_async(self) -> Dict[str, Any]:
        """Get cache statistics asynchronously."""
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            self.executor,
            self.batch_processor.get_cache_stats
        )

        return result

    async def clear_cache_async(self):
        """Clear cache asynchronously."""
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(
            self.executor,
            self.batch_processor.clear_cache
        )

    async def health_check_async(self) -> bool:
        """Perform health check asynchronously."""
        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self.executor,
                self.batch_processor.health_check
            )

            return result
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def _safe_callback(self, callback: callable, *args, **kwargs):
        """Safely call callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Callback failed: {e}")

    async def close(self):
        """Close the async batch processor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("Async batch processor closed")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
