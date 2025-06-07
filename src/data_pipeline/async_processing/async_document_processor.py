"""
Asynchronous document processor for high-performance document processing.
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..document_processing.document_processor import DocumentProcessingConfig, DocumentProcessor
from ..document_processing.metadata.models import DocumentMetadata
from ..document_processing.parsers.base_parser import ParsedDocument

class AsyncDocumentProcessor:
    """Asynchronous document processor with parallel processing capabilities."""

    def __init__(
        self,
        config: Optional[DocumentProcessingConfig] = None,
        max_workers: int = 4,
        use_process_pool: bool = False,
        chunk_size: int = 10
    ):
        """
        Initialize async document processor.

        Args:
            config: Document processing configuration
            max_workers: Maximum number of worker threads/processes
            use_process_pool: Whether to use process pool instead of thread pool
            chunk_size: Number of documents to process in each batch
        """
        self.config = config or DocumentProcessingConfig()
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        self.chunk_size = chunk_size

        self.logger = logging.getLogger(self.__class__.__name__)

        # Create executor
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Create sync processor for worker functions
        self._sync_processor = DocumentProcessor(config)

    async def process_file_async(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Process a single file asynchronously.

        Args:
            file_path: Path to file to process

        Returns:
            ParsedDocument: Processed document
        """
        loop = asyncio.get_event_loop()

        # Run in executor to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            self._sync_processor.process_file,
            file_path
        )

        return result

    async def process_files_async(
        self,
        file_paths: List[Union[str, Path]],
        progress_callback: Optional[callable] = None
    ) -> List[ParsedDocument]:
        """
        Process multiple files asynchronously.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates

        Returns:
            List[ParsedDocument]: List of processed documents
        """
        start_time = time.time()

        # Create tasks for all files
        tasks = []
        for file_path in file_paths:
            task = self.process_file_async(file_path)
            tasks.append(task)

        # Process with progress tracking
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1

                # Call progress callback if provided
                if progress_callback:
                    progress = (completed / len(file_paths)) * 100
                    await self._safe_callback(progress_callback, completed, len(file_paths), progress)

            except Exception as e:
                self.logger.error(f"Failed to process file: {e}")
                # Add None for failed files to maintain order
                results.append(None)
                completed += 1

        processing_time = time.time() - start_time
        self.logger.info(f"Processed {len(file_paths)} files in {processing_time:.2f}s")

        # Filter out None results (failed files)
        return [r for r in results if r is not None]

    async def process_files_in_batches(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ParsedDocument]:
        """
        Process files in batches to control memory usage.

        Args:
            file_paths: List of file paths to process
            batch_size: Size of each batch (uses instance chunk_size if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List[ParsedDocument]: List of processed documents
        """
        batch_size = batch_size or self.chunk_size
        all_results = []
        total_files = len(file_paths)
        processed_files = 0

        # Process in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]

            self.logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")

            # Process current batch
            batch_results = await self.process_files_async(batch)
            all_results.extend(batch_results)

            processed_files += len(batch)

            # Call progress callback
            if progress_callback:
                progress = (processed_files / total_files) * 100
                await self._safe_callback(progress_callback, processed_files, total_files, progress)

        return all_results

    async def process_content_async(
        self,
        content: str,
        document_id: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Process text content asynchronously.

        Args:
            content: Text content to process
            document_id: Document identifier
            document_type: Document type hint
            metadata: Additional metadata

        Returns:
            ParsedDocument: Processed document
        """
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._sync_processor.process_content,
            content,
            document_id,
            document_type,
            metadata
        )

        return result

    async def extract_metadata_async(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from file asynchronously.

        Args:
            file_path: Path to file

        Returns:
            DocumentMetadata: Extracted metadata
        """
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._sync_processor.extract_metadata,
            file_path
        )

        return result

    async def get_supported_formats_async(self) -> List[str]:
        """
        Get supported file formats asynchronously.

        Returns:
            List[str]: Supported file extensions
        """
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._sync_processor.get_supported_formats
        )

        return result

    async def health_check_async(self) -> bool:
        """
        Perform health check asynchronously.

        Returns:
            bool: True if healthy
        """
        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self.executor,
                self._sync_processor.health_check
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
        """Close the async processor and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("Async document processor closed")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

class AsyncDocumentProcessorManager:
    """Manager for multiple async document processors."""

    def __init__(self, num_processors: int = 2):
        """
        Initialize manager.

        Args:
            num_processors: Number of processors to create
        """
        self.num_processors = num_processors
        self.processors: List[AsyncDocumentProcessor] = []
        self.current_processor = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self, config: Optional[DocumentProcessingConfig] = None):
        """Initialize all processors."""
        for i in range(self.num_processors):
            processor = AsyncDocumentProcessor(
                config=config,
                max_workers=2,  # Fewer workers per processor
                use_process_pool=False
            )
            self.processors.append(processor)

        self.logger.info(f"Initialized {self.num_processors} async processors")

    def get_next_processor(self) -> AsyncDocumentProcessor:
        """Get next processor using round-robin."""
        processor = self.processors[self.current_processor]
        self.current_processor = (self.current_processor + 1) % self.num_processors
        return processor

    async def process_files_distributed(
        self,
        file_paths: List[Union[str, Path]],
        progress_callback: Optional[callable] = None
    ) -> List[ParsedDocument]:
        """
        Process files distributed across multiple processors.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional progress callback

        Returns:
            List[ParsedDocument]: Processed documents
        """
        if not self.processors:
            raise RuntimeError("Processors not initialized")

        # Distribute files across processors
        chunks = [[] for _ in range(self.num_processors)]
        for i, file_path in enumerate(file_paths):
            chunks[i % self.num_processors].append(file_path)

        # Create tasks for each processor
        tasks = []
        for i, chunk in enumerate(chunks):
            if chunk:  # Only create task if chunk has files
                processor = self.processors[i]
                task = processor.process_files_async(chunk, progress_callback)
                tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_documents = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Processor failed: {result}")
            elif isinstance(result, list):
                all_documents.extend(result)

        return all_documents

    async def close_all(self):
        """Close all processors."""
        for processor in self.processors:
            await processor.close()

        self.processors.clear()
        self.logger.info("All async processors closed")
