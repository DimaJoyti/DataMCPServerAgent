"""
Streaming Data Ingestion Engine.

This module provides real-time streaming data ingestion capabilities
for processing continuous data streams from various sources.
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis
import structlog
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel, Field


class StreamIngestionConfig(BaseModel):
    """Configuration for streaming ingestion."""

    # Stream processing
    buffer_size: int = Field(default=1000, description="Buffer size for batching")
    batch_timeout: float = Field(default=5.0, description="Batch timeout in seconds")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")

    # Kafka configuration
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], description="Kafka bootstrap servers"
    )
    kafka_consumer_group: str = Field(default="data_pipeline", description="Kafka consumer group")
    kafka_auto_offset_reset: str = Field(default="latest", description="Kafka auto offset reset")

    # Redis configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")

    # Processing configuration
    enable_deduplication: bool = Field(default=True, description="Enable message deduplication")
    enable_ordering: bool = Field(default=False, description="Enable message ordering")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")

    # Quality and monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: float = Field(default=30.0, description="Metrics collection interval")


class StreamMetrics(BaseModel):
    """Metrics for streaming ingestion."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_processed: int = 0

    # Performance metrics
    throughput_messages_per_second: float = 0.0
    throughput_bytes_per_second: float = 0.0
    average_processing_time: float = 0.0

    # Quality metrics
    duplicate_messages: int = 0
    out_of_order_messages: int = 0
    error_rate: float = 0.0

    # Timestamps
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


class StreamMessage(BaseModel):
    """Represents a streaming message."""

    id: str
    topic: str
    partition: Optional[int] = None
    offset: Optional[int] = None
    timestamp: datetime
    headers: Dict[str, str] = Field(default_factory=dict)
    payload: Any

    # Processing metadata
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    retry_count: int = 0


class StreamIngestionEngine:
    """
    Engine for streaming data ingestion.

    Supports real-time data ingestion from Kafka, Redis Streams,
    and other streaming platforms.
    """

    def __init__(
        self,
        config: Optional[StreamIngestionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the streaming ingestion engine.

        Args:
            config: Streaming ingestion configuration
            logger: Logger instance
        """
        self.config = config or StreamIngestionConfig()
        self.logger = logger or structlog.get_logger("stream_ingestion")

        # Runtime state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.message_buffer = deque(maxlen=self.config.buffer_size)
        self.metrics = StreamMetrics(start_time=datetime.now(timezone.utc))

        # Message processing
        self.message_handlers: Dict[str, Callable] = {}
        self.processed_message_ids = set()  # For deduplication

        # Connections
        self.kafka_consumer: Optional[KafkaConsumer] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        self.redis_client: Optional[redis.Redis] = None

        self.logger.info("Streaming ingestion engine initialized")

    def register_message_handler(self, topic: str, handler: Callable[[StreamMessage], Any]) -> None:
        """
        Register a message handler for a specific topic.

        Args:
            topic: Topic name
            handler: Message handler function
        """
        self.message_handlers[topic] = handler
        self.logger.info("Message handler registered", topic=topic)

    async def start(self) -> None:
        """Start the streaming ingestion engine."""
        if self.is_running:
            self.logger.warning("Streaming engine is already running")
            return

        self.is_running = True
        self.shutdown_event.clear()
        self.metrics.start_time = datetime.now(timezone.utc)

        self.logger.info("Starting streaming ingestion engine")

        # Initialize connections
        await self._initialize_connections()

        # Start background tasks
        tasks = [
            asyncio.create_task(self._kafka_consumer_loop()),
            asyncio.create_task(self._redis_consumer_loop()),
            asyncio.create_task(self._message_processor_loop()),
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

            # Close connections
            await self._close_connections()

            self.logger.info("Streaming ingestion engine stopped")

    async def stop(self) -> None:
        """Stop the streaming ingestion engine."""
        if not self.is_running:
            return

        self.logger.info("Stopping streaming ingestion engine")
        self.is_running = False
        self.shutdown_event.set()

    async def send_message(
        self, topic: str, message: Any, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send a message to a topic.

        Args:
            topic: Topic name
            message: Message payload
            headers: Optional message headers

        Returns:
            True if message was sent successfully
        """
        try:
            if self.kafka_producer:
                # Send to Kafka
                message_bytes = json.dumps(message).encode("utf-8")
                future = self.kafka_producer.send(topic, value=message_bytes, headers=headers or {})

                # Wait for send to complete
                record_metadata = future.get(timeout=10)

                self.logger.debug(
                    "Message sent to Kafka",
                    topic=topic,
                    partition=record_metadata.partition,
                    offset=record_metadata.offset,
                )

                return True

            return False

        except Exception as e:
            self.logger.error("Failed to send message", topic=topic, error=str(e))
            return False

    async def _initialize_connections(self) -> None:
        """Initialize connections to streaming platforms."""
        try:
            # Initialize Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_consumer_group,
                auto_offset_reset=self.config.kafka_auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )

            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            )

            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )

            # Test Redis connection
            await self.redis_client.ping()

            self.logger.info("Streaming connections initialized")

        except Exception as e:
            self.logger.error("Failed to initialize streaming connections", error=str(e))
            raise e

    async def _close_connections(self) -> None:
        """Close streaming connections."""
        try:
            if self.kafka_consumer:
                self.kafka_consumer.close()

            if self.kafka_producer:
                self.kafka_producer.close()

            if self.redis_client:
                await self.redis_client.close()

            self.logger.info("Streaming connections closed")

        except Exception as e:
            self.logger.error("Error closing streaming connections", error=str(e))

    async def _kafka_consumer_loop(self) -> None:
        """Kafka consumer loop."""
        while self.is_running:
            try:
                if not self.kafka_consumer:
                    await asyncio.sleep(1)
                    continue

                # Poll for messages
                message_batch = self.kafka_consumer.poll(timeout_ms=1000)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        stream_message = StreamMessage(
                            id=f"{topic_partition.topic}:{topic_partition.partition}:{message.offset}",
                            topic=topic_partition.topic,
                            partition=topic_partition.partition,
                            offset=message.offset,
                            timestamp=datetime.fromtimestamp(
                                message.timestamp / 1000, tz=timezone.utc
                            ),
                            headers={k: v.decode() for k, v in message.headers},
                            payload=message.value,
                        )

                        await self._enqueue_message(stream_message)

            except Exception as e:
                self.logger.error("Kafka consumer error", error=str(e))
                await asyncio.sleep(1)

    async def _redis_consumer_loop(self) -> None:
        """Redis streams consumer loop."""
        while self.is_running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(1)
                    continue

                # Read from Redis streams
                # This is a placeholder - implement based on your Redis streams setup
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error("Redis consumer error", error=str(e))
                await asyncio.sleep(1)

    async def _message_processor_loop(self) -> None:
        """Message processor loop."""
        while self.is_running:
            try:
                # Process buffered messages
                if self.message_buffer:
                    messages_to_process = []

                    # Collect messages for batch processing
                    while (
                        self.message_buffer and len(messages_to_process) < self.config.buffer_size
                    ):
                        messages_to_process.append(self.message_buffer.popleft())

                    # Process batch
                    if messages_to_process:
                        await self._process_message_batch(messages_to_process)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error("Message processor error", error=str(e))
                await asyncio.sleep(1)

    async def _enqueue_message(self, message: StreamMessage) -> None:
        """Enqueue a message for processing."""
        # Deduplication check
        if self.config.enable_deduplication:
            if message.id in self.processed_message_ids:
                self.metrics.duplicate_messages += 1
                return

        # Add to buffer
        self.message_buffer.append(message)
        self.metrics.messages_received += 1

    async def _process_message_batch(self, messages: List[StreamMessage]) -> None:
        """Process a batch of messages."""
        for message in messages:
            try:
                start_time = time.time()

                # Get handler for topic
                handler = self.message_handlers.get(message.topic)
                if handler:
                    # Process message
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)

                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.messages_processed += 1
                self.metrics.bytes_processed += len(str(message.payload))

                # Update average processing time
                if self.metrics.messages_processed > 0:
                    self.metrics.average_processing_time = (
                        self.metrics.average_processing_time * (self.metrics.messages_processed - 1)
                        + processing_time
                    ) / self.metrics.messages_processed

                # Mark as processed for deduplication
                if self.config.enable_deduplication:
                    self.processed_message_ids.add(message.id)

                    # Limit size of processed IDs set
                    if len(self.processed_message_ids) > 10000:
                        # Remove oldest half
                        ids_to_remove = list(self.processed_message_ids)[:5000]
                        for id_to_remove in ids_to_remove:
                            self.processed_message_ids.discard(id_to_remove)

                message.processed_at = datetime.now(timezone.utc)

            except Exception as e:
                self.metrics.messages_failed += 1
                self.logger.error(
                    "Message processing failed",
                    message_id=message.id,
                    topic=message.topic,
                    error=str(e),
                )

                # Retry logic
                if message.retry_count < self.config.max_retries:
                    message.retry_count += 1
                    await asyncio.sleep(self.config.retry_delay)
                    self.message_buffer.append(message)

    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running:
            try:
                if self.config.enable_metrics:
                    await self._update_metrics()

                await asyncio.sleep(self.config.metrics_interval)

            except Exception as e:
                self.logger.error("Metrics loop error", error=str(e))
                await asyncio.sleep(self.config.metrics_interval)

    async def _update_metrics(self) -> None:
        """Update streaming metrics."""
        current_time = datetime.now(timezone.utc)

        if self.metrics.last_update_time:
            time_diff = (current_time - self.metrics.last_update_time).total_seconds()

            if time_diff > 0:
                # Calculate throughput
                self.metrics.throughput_messages_per_second = (
                    self.metrics.messages_processed / time_diff
                )
                self.metrics.throughput_bytes_per_second = self.metrics.bytes_processed / time_diff

        # Calculate error rate
        if self.metrics.messages_received > 0:
            self.metrics.error_rate = self.metrics.messages_failed / self.metrics.messages_received

        self.metrics.last_update_time = current_time

        self.logger.debug(
            "Streaming metrics updated",
            messages_received=self.metrics.messages_received,
            messages_processed=self.metrics.messages_processed,
            messages_failed=self.metrics.messages_failed,
            throughput_mps=self.metrics.throughput_messages_per_second,
            error_rate=self.metrics.error_rate,
        )

    async def get_metrics(self) -> StreamMetrics:
        """Get current streaming metrics."""
        await self._update_metrics()
        return self.metrics

    async def get_status(self) -> Dict[str, Any]:
        """Get current streaming engine status."""
        return {
            "engine": "stream_ingestion",
            "is_running": self.is_running,
            "config": self.config.model_dump(),
            "metrics": self.metrics.model_dump(),
            "buffer_size": len(self.message_buffer),
            "registered_handlers": list(self.message_handlers.keys()),
        }
