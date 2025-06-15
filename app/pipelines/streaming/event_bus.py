"""
Event Bus for Streaming Pipeline Coordination.

This module provides an event-driven architecture for coordinating
streaming pipeline components:
- Event publishing and subscription
- Event filtering and routing
- Asynchronous event handling
- Event persistence and replay
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.logging import get_logger


class EventPriority(int, Enum):
    """Event priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class EventFilter:
    """Filter for event subscriptions."""

    # Event type filtering
    event_types: Optional[Set[str]] = None

    # Source filtering
    sources: Optional[Set[str]] = None

    # Priority filtering
    min_priority: Optional[EventPriority] = None
    max_priority: Optional[EventPriority] = None

    # Metadata filtering
    metadata_filters: Optional[Dict[str, Any]] = None

    def matches(self, event: "BusEvent") -> bool:
        """Check if event matches this filter."""
        # Event type check
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Source check
        if self.sources and event.source not in self.sources:
            return False

        # Priority check
        if self.min_priority and event.priority < self.min_priority:
            return False
        if self.max_priority and event.priority > self.max_priority:
            return False

        # Metadata check
        if self.metadata_filters:
            for key, expected_value in self.metadata_filters.items():
                if key not in event.metadata or event.metadata[key] != expected_value:
                    return False

        return True


class BusEvent(BaseModel):
    """Event in the event bus."""

    # Event identification
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event ID")
    event_type: str = Field(..., description="Type of event")
    timestamp: float = Field(default_factory=time.time, description="Event timestamp")

    # Event content
    data: Any = Field(..., description="Event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")

    # Event properties
    source: str = Field(..., description="Event source")
    priority: EventPriority = Field(default=EventPriority.NORMAL, description="Event priority")

    # Correlation
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related events")
    parent_event_id: Optional[str] = Field(None, description="Parent event ID")

    # Processing
    processed: bool = Field(default=False, description="Whether event has been processed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    class Config:
        arbitrary_types_allowed = True


@dataclass
class EventSubscription:
    """Subscription to events."""

    subscription_id: str
    handler: Callable[[BusEvent], Any]
    event_filter: Optional[EventFilter] = None
    active: bool = True
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class EventHandler:
    """Base class for event handlers."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"EventHandler-{name}")

    async def handle(self, event: BusEvent) -> Any:
        """Handle an event."""
        raise NotImplementedError

    def can_handle(self, event: BusEvent) -> bool:
        """Check if this handler can handle the event."""
        return True


class EventBus:
    """Event bus for coordinating streaming pipeline components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize event bus."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.max_queue_size = self.config.get("max_queue_size", 10000)
        self.enable_persistence = self.config.get("enable_persistence", False)
        self.max_event_history = self.config.get("max_event_history", 1000)
        self.processing_timeout = self.config.get("processing_timeout", 30.0)

        # Event storage
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
        self.event_history: List[BusEvent] = []
        self.failed_events: List[BusEvent] = []

        # Subscriptions
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.handlers_by_type: Dict[str, List[EventSubscription]] = defaultdict(list)

        # Processing state
        self.is_running = False
        self.processor_tasks: List[asyncio.Task] = []
        self.num_processors = self.config.get("num_processors", 3)

        # Statistics
        self.events_published = 0
        self.events_processed = 0
        self.events_failed = 0
        self.start_time: Optional[float] = None

        self.logger.info("EventBus initialized")

    async def start(self) -> None:
        """Start the event bus."""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()

        # Start processor tasks
        for i in range(self.num_processors):
            task = asyncio.create_task(self._processor_loop(f"processor-{i}"))
            self.processor_tasks.append(task)

        self.logger.info(f"EventBus started with {self.num_processors} processors")

    async def stop(self) -> None:
        """Stop the event bus."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel processor tasks
        for task in self.processor_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processor_tasks, return_exceptions=True)

        self.logger.info("EventBus stopped")

    async def publish(self, event: BusEvent) -> bool:
        """Publish an event to the bus."""
        if not self.is_running:
            self.logger.warning("Cannot publish event: bus is not running")
            return False

        try:
            # Add to queue with priority (negative for max-heap behavior)
            priority = -event.priority.value
            await self.event_queue.put((priority, event.timestamp, event))

            self.events_published += 1
            self.logger.debug(f"Published event {event.event_id} of type {event.event_type}")
            return True

        except asyncio.QueueFull:
            self.logger.error("Event queue is full, cannot publish event")
            return False
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            return False

    def subscribe(
        self, handler: Callable[[BusEvent], Any], event_filter: Optional[EventFilter] = None
    ) -> str:
        """Subscribe to events."""
        subscription_id = str(uuid4())

        subscription = EventSubscription(
            subscription_id=subscription_id, handler=handler, event_filter=event_filter
        )

        self.subscriptions[subscription_id] = subscription

        # Index by event types if filter specifies them
        if event_filter and event_filter.event_types:
            for event_type in event_filter.event_types:
                self.handlers_by_type[event_type].append(subscription)
        else:
            # Subscribe to all event types
            self.handlers_by_type["*"].append(subscription)

        self.logger.info(f"Added subscription {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscription.active = False

        # Remove from indexes
        for event_type, subs in self.handlers_by_type.items():
            self.handlers_by_type[event_type] = [
                s for s in subs if s.subscription_id != subscription_id
            ]

        del self.subscriptions[subscription_id]

        self.logger.info(f"Removed subscription {subscription_id}")
        return True

    async def _processor_loop(self, processor_name: str) -> None:
        """Event processor loop."""
        self.logger.debug(f"Started processor: {processor_name}")

        while self.is_running:
            try:
                # Get event from queue with timeout
                priority, timestamp, event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )

                # Process event
                await self._process_event(event, processor_name)

            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in processor {processor_name}: {e}")
                await asyncio.sleep(1.0)

        self.logger.debug(f"Stopped processor: {processor_name}")

    async def _process_event(self, event: BusEvent, processor_name: str) -> None:
        """Process a single event."""
        try:
            self.logger.debug(f"Processing event {event.event_id} in {processor_name}")

            # Find matching subscriptions
            matching_subscriptions = self._find_matching_subscriptions(event)

            if not matching_subscriptions:
                self.logger.debug(f"No handlers for event {event.event_id}")
                return

            # Process with all matching handlers
            handler_tasks = []
            for subscription in matching_subscriptions:
                if subscription.active:
                    task = asyncio.create_task(self._handle_event_with_timeout(event, subscription))
                    handler_tasks.append(task)

            # Wait for all handlers to complete
            if handler_tasks:
                results = await asyncio.gather(*handler_tasks, return_exceptions=True)

                # Check for failures
                failures = [r for r in results if isinstance(r, Exception)]
                if failures:
                    self.logger.warning(
                        f"Event {event.event_id} had {len(failures)} handler failures"
                    )
                    for failure in failures:
                        self.logger.error(f"Handler failure: {failure}")

            # Mark as processed
            event.processed = True
            self.events_processed += 1

            # Add to history
            self._add_to_history(event)

        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_id}: {e}")
            event.retry_count += 1

            # Retry if under limit
            if event.retry_count < event.max_retries:
                self.logger.info(f"Retrying event {event.event_id} (attempt {event.retry_count})")
                await self.publish(event)
            else:
                self.logger.error(f"Event {event.event_id} exceeded max retries")
                self.failed_events.append(event)
                self.events_failed += 1

    def _find_matching_subscriptions(self, event: BusEvent) -> List[EventSubscription]:
        """Find subscriptions that match the event."""
        matching = []

        # Check specific event type handlers
        for subscription in self.handlers_by_type.get(event.event_type, []):
            if subscription.active and self._subscription_matches(subscription, event):
                matching.append(subscription)

        # Check wildcard handlers
        for subscription in self.handlers_by_type.get("*", []):
            if subscription.active and self._subscription_matches(subscription, event):
                matching.append(subscription)

        return matching

    def _subscription_matches(self, subscription: EventSubscription, event: BusEvent) -> bool:
        """Check if subscription matches event."""
        if subscription.event_filter:
            return subscription.event_filter.matches(event)
        return True

    async def _handle_event_with_timeout(
        self, event: BusEvent, subscription: EventSubscription
    ) -> Any:
        """Handle event with timeout."""
        try:
            if asyncio.iscoroutinefunction(subscription.handler):
                result = await asyncio.wait_for(
                    subscription.handler(event), timeout=self.processing_timeout
                )
            else:
                result = subscription.handler(event)

            return result

        except asyncio.TimeoutError:
            raise Exception(f"Handler {subscription.subscription_id} timed out")
        except Exception as e:
            raise Exception(f"Handler {subscription.subscription_id} failed: {e}")

    def _add_to_history(self, event: BusEvent) -> None:
        """Add event to history."""
        self.event_history.append(event)

        # Trim history if too large
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history :]

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "queue_size": self.event_queue.qsize(),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "total_subscriptions": len(self.subscriptions),
            "processing_rate": self.events_processed / max(uptime, 1),
            "success_rate": self.events_processed / max(self.events_published, 1),
        }

    def get_event_history(
        self, event_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[BusEvent]:
        """Get event history."""
        history = self.event_history

        # Filter by event type
        if event_type:
            history = [e for e in history if e.event_type == event_type]

        # Apply limit
        if limit:
            history = history[-limit:]

        return history

    def get_failed_events(self) -> List[BusEvent]:
        """Get failed events."""
        return self.failed_events.copy()

    async def replay_events(self, events: List[BusEvent]) -> int:
        """Replay a list of events."""
        replayed = 0

        for event in events:
            # Reset processing state
            event.processed = False
            event.retry_count = 0
            event.timestamp = time.time()

            # Republish
            if await self.publish(event):
                replayed += 1

        self.logger.info(f"Replayed {replayed} events")
        return replayed
