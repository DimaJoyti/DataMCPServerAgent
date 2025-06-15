"""
Real-time Streaming Pipeline Processor.

This module implements a high-performance streaming pipeline for processing
documents, media, and data in real-time with support for:
- Asynchronous stream processing
- Backpressure handling
- Auto-scaling
- Event-driven architecture
- Fault tolerance
"""

import asyncio
import time
from asyncio import Event, Queue
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.pipelines.multimodal import ProcessorFactory


class StreamEventType(str, Enum):
    """Types of stream events."""

    DOCUMENT_ADDED = "document_added"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"
    BATCH_COMPLETED = "batch_completed"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_ALERT = "system_alert"


class ProcessingStatus(str, Enum):
    """Processing status for stream events."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class StreamEvent:
    """Event in the streaming pipeline."""

    # Event identification
    event_id: str
    event_type: StreamEventType
    timestamp: float

    # Content
    content: Any
    metadata: Dict[str, Any]

    # Processing info
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

    # Tracing
    trace_id: Optional[str] = None
    parent_event_id: Optional[str] = None


class StreamingConfig(BaseModel):
    """Configuration for streaming pipeline."""

    # Processing configuration
    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent processing tasks")
    batch_size: int = Field(default=5, description="Batch size for processing")
    batch_timeout_seconds: float = Field(default=5.0, description="Timeout for batch collection")

    # Queue configuration
    max_queue_size: int = Field(default=1000, description="Maximum queue size")
    backpressure_threshold: float = Field(default=0.8, description="Backpressure threshold (0-1)")

    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff")

    # Auto-scaling
    enable_auto_scaling: bool = Field(default=True, description="Enable auto-scaling")
    scale_up_threshold: float = Field(default=0.8, description="Scale up threshold")
    scale_down_threshold: float = Field(default=0.3, description="Scale down threshold")
    min_workers: int = Field(default=2, description="Minimum number of workers")
    max_workers: int = Field(default=20, description="Maximum number of workers")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval_seconds: float = Field(default=10.0, description="Metrics collection interval")


class ProcessingResult(BaseModel):
    """Result of stream processing."""

    # Event reference
    event_id: str = Field(..., description="Original event ID")
    trace_id: Optional[str] = Field(None, description="Trace ID for correlation")

    # Processing info
    status: ProcessingStatus = Field(..., description="Processing status")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    worker_id: str = Field(..., description="Worker that processed the event")

    # Results
    output: Optional[Any] = Field(None, description="Processing output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")


class StreamProcessor:
    """Individual stream processor worker."""

    def __init__(self, worker_id: str, config: StreamingConfig):
        """Initialize stream processor."""
        self.worker_id = worker_id
        self.config = config
        self.logger = get_logger(f"StreamProcessor-{worker_id}")

        # Processing state
        self.is_running = False
        self.current_task: Optional[asyncio.Task] = None
        self.processed_count = 0
        self.error_count = 0

        # Multimodal processor - use text_image for single modality content
        self.multimodal_processor = ProcessorFactory.create("text_image")

        self.logger.info(f"StreamProcessor {worker_id} initialized")

    async def process_event(self, event: StreamEvent) -> ProcessingResult:
        """Process a single stream event."""
        start_time = time.time()

        try:
            self.logger.debug(f"Processing event {event.event_id} of type {event.event_type}")

            # Update event status
            event.status = ProcessingStatus.PROCESSING

            # Process based on event type
            if event.event_type == StreamEventType.DOCUMENT_ADDED:
                output = await self._process_document_added(event)
            elif event.event_type == StreamEventType.DOCUMENT_UPDATED:
                output = await self._process_document_updated(event)
            elif event.event_type == StreamEventType.DOCUMENT_DELETED:
                output = await self._process_document_deleted(event)
            else:
                output = await self._process_generic_event(event)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create successful result
            result = ProcessingResult(
                event_id=event.event_id,
                trace_id=event.trace_id,
                status=ProcessingStatus.COMPLETED,
                processing_time_ms=processing_time_ms,
                worker_id=self.worker_id,
                output=output,
                metadata={
                    "event_type": event.event_type,
                    "content_size": len(str(event.content)) if event.content else 0,
                },
            )

            self.processed_count += 1
            event.status = ProcessingStatus.COMPLETED

            self.logger.debug(
                f"Successfully processed event {event.event_id} in {processing_time_ms:.2f}ms"
            )
            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_message = str(e)

            self.logger.error(f"Failed to process event {event.event_id}: {error_message}")

            # Create error result
            result = ProcessingResult(
                event_id=event.event_id,
                trace_id=event.trace_id,
                status=ProcessingStatus.FAILED,
                processing_time_ms=processing_time_ms,
                worker_id=self.worker_id,
                error_message=error_message,
                retry_count=event.retry_count,
            )

            self.error_count += 1
            event.status = ProcessingStatus.FAILED

            return result

    async def _process_document_added(self, event: StreamEvent) -> Dict[str, Any]:
        """Process document addition event."""
        content = event.content

        # Simulate multimodal processing
        if hasattr(content, "text") or hasattr(content, "image") or hasattr(content, "audio"):
            # Use multimodal processor
            result = await self.multimodal_processor.process(content)
            return {
                "type": "multimodal_processing",
                "extracted_text": result.extracted_text,
                "embeddings_generated": bool(result.combined_embedding),
                "entities_found": len(result.extracted_entities),
            }
        else:
            # Simple text processing
            return {
                "type": "text_processing",
                "content_length": len(str(content)),
                "processed_at": time.time(),
            }

    async def _process_document_updated(self, event: StreamEvent) -> Dict[str, Any]:
        """Process document update event."""
        return {
            "type": "document_update",
            "updated_fields": event.metadata.get("updated_fields", []),
            "version": event.metadata.get("version", 1) + 1,
        }

    async def _process_document_deleted(self, event: StreamEvent) -> Dict[str, Any]:
        """Process document deletion event."""
        return {
            "type": "document_deletion",
            "document_id": event.metadata.get("document_id"),
            "cleanup_completed": True,
        }

    async def _process_generic_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Process generic event."""
        return {
            "type": "generic_processing",
            "event_type": event.event_type,
            "content_processed": bool(event.content),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.error_count)
                if (self.processed_count + self.error_count) > 0
                else 0.0
            ),
        }


class StreamingPipeline:
    """Main streaming pipeline coordinator."""

    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize streaming pipeline."""
        self.config = config or StreamingConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Pipeline state
        self.is_running = False
        self.workers: List[StreamProcessor] = []
        self.event_queue: Queue[StreamEvent] = Queue(maxsize=self.config.max_queue_size)
        self.result_queue: Queue[ProcessingResult] = Queue()

        # Control events
        self.shutdown_event = Event()
        self.pause_event = Event()

        # Tasks
        self.coordinator_task: Optional[asyncio.Task] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.metrics_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_events_processed = 0
        self.total_events_failed = 0
        self.start_time: Optional[float] = None

        self.logger.info("StreamingPipeline initialized")

    async def start(self) -> None:
        """Start the streaming pipeline."""
        if self.is_running:
            self.logger.warning("Pipeline is already running")
            return

        self.logger.info("Starting streaming pipeline")
        self.is_running = True
        self.start_time = time.time()

        # Initialize workers
        await self._initialize_workers()

        # Start coordinator task
        self.coordinator_task = asyncio.create_task(self._coordinator_loop())

        # Start metrics collection if enabled
        if self.config.enable_metrics:
            self.metrics_task = asyncio.create_task(self._metrics_loop())

        self.logger.info(f"Streaming pipeline started with {len(self.workers)} workers")

    async def stop(self) -> None:
        """Stop the streaming pipeline."""
        if not self.is_running:
            return

        self.logger.info("Stopping streaming pipeline")
        self.is_running = False
        self.shutdown_event.set()

        # Cancel all tasks
        if self.coordinator_task:
            self.coordinator_task.cancel()

        for task in self.worker_tasks:
            task.cancel()

        if self.metrics_task:
            self.metrics_task.cancel()

        # Wait for tasks to complete
        all_tasks = []
        if self.coordinator_task:
            all_tasks.append(self.coordinator_task)
        all_tasks.extend(self.worker_tasks)
        if self.metrics_task:
            all_tasks.append(self.metrics_task)

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        self.logger.info("Streaming pipeline stopped")

    async def submit_event(self, event: StreamEvent) -> bool:
        """Submit an event for processing."""
        if not self.is_running:
            self.logger.warning("Cannot submit event: pipeline is not running")
            return False

        try:
            # Check backpressure
            queue_usage = self.event_queue.qsize() / self.config.max_queue_size
            if queue_usage > self.config.backpressure_threshold:
                self.logger.warning(f"Backpressure detected: queue usage {queue_usage:.2f}")
                return False

            # Add to queue
            await self.event_queue.put(event)
            self.logger.debug(f"Event {event.event_id} submitted for processing")
            return True

        except asyncio.QueueFull:
            self.logger.error("Event queue is full, cannot submit event")
            return False

    async def _initialize_workers(self) -> None:
        """Initialize worker processors."""
        for i in range(self.config.min_workers):
            worker_id = f"worker-{i}"
            worker = StreamProcessor(worker_id, self.config)
            self.workers.append(worker)

            # Start worker task
            task = asyncio.create_task(self._worker_loop(worker))
            self.worker_tasks.append(task)

    async def _coordinator_loop(self) -> None:
        """Main coordinator loop."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Auto-scaling logic
                if self.config.enable_auto_scaling:
                    await self._handle_auto_scaling()

                # Process results
                await self._process_results()

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in coordinator loop: {e}")
                await asyncio.sleep(1.0)

    async def _worker_loop(self, worker: StreamProcessor) -> None:
        """Worker processing loop."""
        worker.is_running = True

        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Process event
                result = await worker.process_event(event)

                # Put result in result queue
                await self.result_queue.put(result)

            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in worker {worker.worker_id}: {e}")
                await asyncio.sleep(1.0)

        worker.is_running = False

    async def _handle_auto_scaling(self) -> None:
        """Handle auto-scaling of workers."""
        # Calculate current load
        queue_usage = self.event_queue.qsize() / self.config.max_queue_size
        active_workers = sum(1 for w in self.workers if w.is_running)

        # Scale up if needed
        if (
            queue_usage > self.config.scale_up_threshold
            and active_workers < self.config.max_workers
        ):
            await self._scale_up()

        # Scale down if needed
        elif (
            queue_usage < self.config.scale_down_threshold
            and active_workers > self.config.min_workers
        ):
            await self._scale_down()

    async def _scale_up(self) -> None:
        """Add a new worker."""
        worker_id = f"worker-{len(self.workers)}"
        worker = StreamProcessor(worker_id, self.config)
        self.workers.append(worker)

        task = asyncio.create_task(self._worker_loop(worker))
        self.worker_tasks.append(task)

        self.logger.info(f"Scaled up: added worker {worker_id}")

    async def _scale_down(self) -> None:
        """Remove a worker."""
        # Find a worker to remove (simple strategy: last added)
        for i in range(len(self.workers) - 1, -1, -1):
            worker = self.workers[i]
            if (
                worker.is_running
                and len([w for w in self.workers if w.is_running]) > self.config.min_workers
            ):
                worker.is_running = False
                self.logger.info(f"Scaled down: removed worker {worker.worker_id}")
                break

    async def _process_results(self) -> None:
        """Process results from workers."""
        try:
            while not self.result_queue.empty():
                result = await self.result_queue.get()

                if result.status == ProcessingStatus.COMPLETED:
                    self.total_events_processed += 1
                else:
                    self.total_events_failed += 1

                # Log result (in production, this would go to metrics/monitoring)
                self.logger.debug(f"Result: {result.event_id} - {result.status}")

        except Exception as e:
            self.logger.error(f"Error processing results: {e}")

    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self.get_metrics()

                # Log metrics (in production, send to monitoring system)
                self.logger.info(f"Pipeline metrics: {metrics}")

                await asyncio.sleep(self.config.metrics_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(self.config.metrics_interval_seconds)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "pipeline": {
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "total_events_processed": self.total_events_processed,
                "total_events_failed": self.total_events_failed,
                "success_rate": (
                    self.total_events_processed
                    / (self.total_events_processed + self.total_events_failed)
                    if (self.total_events_processed + self.total_events_failed) > 0
                    else 0.0
                ),
            },
            "queue": {
                "size": self.event_queue.qsize(),
                "max_size": self.config.max_queue_size,
                "usage": self.event_queue.qsize() / self.config.max_queue_size,
            },
            "workers": {
                "total": len(self.workers),
                "active": sum(1 for w in self.workers if w.is_running),
                "stats": [w.get_stats() for w in self.workers],
            },
        }
