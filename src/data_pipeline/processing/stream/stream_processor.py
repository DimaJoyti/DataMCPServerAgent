"""
Stream Processor for real-time data processing.

This module provides stream processing capabilities for handling
continuous data streams with low-latency processing.
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field


class StreamProcessingConfig(BaseModel):
    """Configuration for stream processing."""

    # Processing options
    window_size: int = Field(default=1000, description="Window size for processing")
    window_type: str = Field(
        default="tumbling", description="Window type (tumbling, sliding, session)"
    )
    window_duration: float = Field(default=60.0, description="Window duration in seconds")

    # Performance options
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    buffer_size: int = Field(default=10000, description="Buffer size for incoming data")
    batch_size: int = Field(default=100, description="Batch size for processing")

    # Latency options
    max_latency: float = Field(default=1.0, description="Maximum processing latency in seconds")
    enable_backpressure: bool = Field(default=True, description="Enable backpressure handling")

    # Error handling
    continue_on_error: bool = Field(default=True, description="Continue processing on errors")
    max_error_rate: float = Field(default=0.1, description="Maximum acceptable error rate")


class StreamMetrics(BaseModel):
    """Metrics for stream processing."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    windows_processed: int = 0

    # Performance metrics
    throughput_messages_per_second: float = 0.0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Buffer metrics
    buffer_utilization: float = 0.0
    backpressure_events: int = 0

    # Error metrics
    error_rate: float = 0.0

    # Timestamps
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


class StreamWindow(BaseModel):
    """Represents a processing window."""

    window_id: str
    window_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages: List[Any] = Field(default_factory=list)
    is_closed: bool = False


class StreamProcessor:
    """
    Stream processor for real-time data processing.

    Provides low-latency processing capabilities for continuous
    data streams with windowing and aggregation support.
    """

    def __init__(
        self,
        config: Optional[StreamProcessingConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the stream processor.

        Args:
            config: Processing configuration
            logger: Logger instance
        """
        self.config = config or StreamProcessingConfig()
        self.logger = logger or structlog.get_logger("stream_processor")

        # Processing state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.message_buffer = deque(maxlen=self.config.buffer_size)
        self.metrics = StreamMetrics(start_time=datetime.now(timezone.utc))

        # Windows
        self.active_windows: Dict[str, StreamWindow] = {}
        self.window_counter = 0

        # Processing handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.window_handlers: Dict[str, Callable] = {}

        # Latency tracking
        self.latency_samples = deque(maxlen=1000)

        self.logger.info("Stream processor initialized")

    def register_message_handler(self, message_type: str, handler: Callable[[Any], Any]) -> None:
        """
        Register a message handler.

        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        self.logger.info("Message handler registered", message_type=message_type)

    def register_window_handler(
        self, window_type: str, handler: Callable[[StreamWindow], Any]
    ) -> None:
        """
        Register a window handler.

        Args:
            window_type: Type of window to handle
            handler: Handler function
        """
        self.window_handlers[window_type] = handler
        self.logger.info("Window handler registered", window_type=window_type)

    async def start(self) -> None:
        """Start the stream processor."""
        if self.is_running:
            self.logger.warning("Stream processor is already running")
            return

        self.is_running = True
        self.shutdown_event.clear()
        self.metrics.start_time = datetime.now(timezone.utc)

        self.logger.info("Starting stream processor")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._message_processing_loop()),
            asyncio.create_task(self._window_management_loop()),
            asyncio.create_task(self._metrics_loop()),
        ]

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        finally:
            # Cancel background tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info("Stream processor stopped")

    async def stop(self) -> None:
        """Stop the stream processor."""
        if not self.is_running:
            return

        self.logger.info("Stopping stream processor")
        self.is_running = False
        self.shutdown_event.set()

    async def process_message(self, message: Any, message_type: str = "default") -> None:
        """
        Process a single message.

        Args:
            message: Message to process
            message_type: Type of message
        """
        try:
            # Add timestamp for latency tracking
            message_with_timestamp = {
                "data": message,
                "type": message_type,
                "timestamp": time.time(),
                "received_at": datetime.now(timezone.utc),
            }

            # Add to buffer
            if len(self.message_buffer) >= self.config.buffer_size:
                if self.config.enable_backpressure:
                    self.metrics.backpressure_events += 1
                    # Drop oldest message
                    self.message_buffer.popleft()
                else:
                    # Skip this message
                    return

            self.message_buffer.append(message_with_timestamp)
            self.metrics.messages_received += 1

        except Exception as e:
            self.logger.error("Failed to process message", error=str(e))
            self.metrics.messages_failed += 1

    async def _message_processing_loop(self) -> None:
        """Main message processing loop."""
        while self.is_running:
            try:
                # Process messages in batches
                batch = []

                # Collect batch
                while len(batch) < self.config.batch_size and self.message_buffer:
                    batch.append(self.message_buffer.popleft())

                if batch:
                    await self._process_message_batch(batch)

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Message processing loop error", error=str(e))
                await asyncio.sleep(1)

    async def _process_message_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of messages."""
        for message_data in batch:
            try:
                start_time = time.time()

                message = message_data["data"]
                message_type = message_data["type"]

                # Get handler
                handler = self.message_handlers.get(message_type)
                if handler:
                    # Process message
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)

                # Add to window if windowing is enabled
                await self._add_to_window(message_data)

                # Track latency
                processing_time = time.time() - start_time
                self.latency_samples.append(processing_time)

                self.metrics.messages_processed += 1

            except Exception as e:
                self.logger.error("Message processing failed", error=str(e))
                self.metrics.messages_failed += 1

    async def _add_to_window(self, message_data: Dict[str, Any]) -> None:
        """Add message to appropriate window."""
        current_time = datetime.now(timezone.utc)

        if self.config.window_type == "tumbling":
            # Create new window if needed
            if not self.active_windows:
                window_id = f"window_{self.window_counter}"
                self.window_counter += 1

                window = StreamWindow(
                    window_id=window_id, window_type="tumbling", start_time=current_time
                )
                self.active_windows[window_id] = window

            # Add to current window
            for window in self.active_windows.values():
                if not window.is_closed:
                    window.messages.append(message_data)
                    break

        elif self.config.window_type == "sliding":
            # Sliding window implementation
            # For simplicity, create overlapping windows
            pass

        elif self.config.window_type == "session":
            # Session window implementation
            # Group messages by session
            pass

    async def _window_management_loop(self) -> None:
        """Window management loop."""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)

                # Check for windows to close
                windows_to_close = []
                for window_id, window in self.active_windows.items():
                    if not window.is_closed:
                        # Check if window should be closed
                        window_age = (current_time - window.start_time).total_seconds()

                        if (
                            window_age >= self.config.window_duration
                            or len(window.messages) >= self.config.window_size
                        ):
                            windows_to_close.append(window_id)

                # Close windows and process them
                for window_id in windows_to_close:
                    window = self.active_windows[window_id]
                    window.is_closed = True
                    window.end_time = current_time

                    await self._process_window(window)

                    # Remove closed window
                    del self.active_windows[window_id]

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Window management loop error", error=str(e))
                await asyncio.sleep(1)

    async def _process_window(self, window: StreamWindow) -> None:
        """Process a completed window."""
        try:
            # Get window handler
            handler = self.window_handlers.get(window.window_type)
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(window)
                else:
                    handler(window)

            self.metrics.windows_processed += 1

            self.logger.debug(
                "Window processed",
                window_id=window.window_id,
                message_count=len(window.messages),
                duration=(window.end_time - window.start_time).total_seconds(),
            )

        except Exception as e:
            self.logger.error("Window processing failed", window_id=window.window_id, error=str(e))

    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running:
            try:
                await self._update_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics loop error", error=str(e))
                await asyncio.sleep(10)

    async def _update_metrics(self) -> None:
        """Update stream processing metrics."""
        current_time = datetime.now(timezone.utc)

        if self.metrics.last_update_time:
            time_diff = (current_time - self.metrics.last_update_time).total_seconds()

            if time_diff > 0:
                # Calculate throughput
                self.metrics.throughput_messages_per_second = (
                    self.metrics.messages_processed / time_diff
                )

        # Calculate latency metrics
        if self.latency_samples:
            sorted_samples = sorted(self.latency_samples)
            self.metrics.average_latency = sum(sorted_samples) / len(sorted_samples)

            # Calculate percentiles
            p95_index = int(0.95 * len(sorted_samples))
            p99_index = int(0.99 * len(sorted_samples))

            self.metrics.p95_latency = (
                sorted_samples[p95_index] if p95_index < len(sorted_samples) else 0
            )
            self.metrics.p99_latency = (
                sorted_samples[p99_index] if p99_index < len(sorted_samples) else 0
            )

        # Calculate buffer utilization
        self.metrics.buffer_utilization = len(self.message_buffer) / self.config.buffer_size

        # Calculate error rate
        if self.metrics.messages_received > 0:
            self.metrics.error_rate = self.metrics.messages_failed / self.metrics.messages_received

        self.metrics.last_update_time = current_time

        self.logger.debug(
            "Stream metrics updated",
            messages_received=self.metrics.messages_received,
            messages_processed=self.metrics.messages_processed,
            throughput_mps=self.metrics.throughput_messages_per_second,
            average_latency=self.metrics.average_latency,
            buffer_utilization=self.metrics.buffer_utilization,
        )

    async def get_metrics(self) -> StreamMetrics:
        """Get current stream processing metrics."""
        await self._update_metrics()
        return self.metrics

    async def get_status(self) -> Dict[str, Any]:
        """Get current stream processor status."""
        return {
            "is_running": self.is_running,
            "config": self.config.model_dump(),
            "metrics": self.metrics.model_dump(),
            "active_windows": len(self.active_windows),
            "buffer_size": len(self.message_buffer),
            "registered_handlers": {
                "message_handlers": list(self.message_handlers.keys()),
                "window_handlers": list(self.window_handlers.keys()),
            },
            "processor": "stream_processor",
        }
