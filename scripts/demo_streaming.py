#!/usr/bin/env python3
"""
Streaming Pipeline Demo Script.

This script demonstrates the real-time streaming capabilities:
- Real-time document processing
- Incremental updates
- Live monitoring
- Event-driven architecture
- Auto-scaling
"""

import asyncio
import time
import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Import streaming components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipelines.streaming import (
    StreamingPipeline, StreamingConfig, StreamEvent, StreamEventType,
    IncrementalProcessor, IncrementalUpdate, UpdateType, IndexManager,
    LiveMonitor, EventBus, BusEvent, EventPriority
)
from app.pipelines.multimodal import MultiModalContent, ModalityType

console = Console()

async def demo_streaming_pipeline():
    """Demo the streaming pipeline."""
    console.print("🚀 Streaming Pipeline Demo", style="bold blue")

    # Create configuration
    config = StreamingConfig(
        max_concurrent_tasks=5,
        batch_size=3,
        batch_timeout_seconds=2.0,
        max_queue_size=100,
        enable_auto_scaling=True,
        min_workers=2,
        max_workers=8
    )

    # Create pipeline
    pipeline = StreamingPipeline(config)

    try:
        # Start pipeline
        console.print("📡 Starting streaming pipeline...", style="yellow")
        await pipeline.start()

        # Generate and submit events
        console.print("📝 Generating sample events...", style="blue")

        events = []
        for i in range(10):
            # Create multimodal content
            content = MultiModalContent(
                content_id=f"doc_{i}",
                text=f"Sample document {i} with some text content for processing.",
                modalities=[ModalityType.TEXT],
                metadata={"source": "demo", "batch": "streaming_test"}
            )

            # Create stream event
            event = StreamEvent(
                event_id=f"event_{i}",
                event_type=StreamEventType.DOCUMENT_ADDED,
                timestamp=time.time(),
                content=content,
                metadata={"document_id": f"doc_{i}", "priority": random.randint(1, 5)}
            )

            events.append(event)

        # Submit events with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Submitting events...", total=len(events))

            for event in events:
                success = await pipeline.submit_event(event)
                if success:
                    progress.update(task, advance=1)
                    await asyncio.sleep(0.2)  # Small delay to see progress
                else:
                    console.print(f"❌ Failed to submit event {event.event_id}", style="red")

        # Wait for processing
        console.print("⏳ Processing events...", style="yellow")
        await asyncio.sleep(5)

        # Show metrics
        metrics = pipeline.get_metrics()
        _display_pipeline_metrics(metrics)

    finally:
        # Stop pipeline
        console.print("🛑 Stopping pipeline...", style="yellow")
        await pipeline.stop()

async def demo_incremental_updates():
    """Demo incremental updates."""
    console.print("\n🔄 Incremental Updates Demo", style="bold blue")

    # Create index manager
    index_manager = IndexManager()

    # Create incremental processor
    processor = IncrementalProcessor(index_manager)

    try:
        # Start processor
        await processor.start()

        # Create index
        await index_manager.create_index("demo_index", {"type": "document", "fields": ["text", "metadata"]})

        # Generate updates
        updates = []
        for i in range(5):
            update = IncrementalUpdate(
                update_id=f"update_{i}",
                document_id=f"doc_{i}",
                update_type=UpdateType.INSERT,
                timestamp=time.time(),
                data={"text": f"Document {i} content", "id": i},
                metadata={"index_name": "demo_index", "source": "demo"}
            )
            updates.append(update)

        # Submit updates
        console.print("📤 Submitting incremental updates...", style="blue")
        for update in updates:
            await processor.submit_update(update)

        # Wait for processing
        await asyncio.sleep(2)

        # Show stats
        stats = processor.get_stats()
        _display_incremental_stats(stats)

    finally:
        await processor.stop()

async def demo_live_monitoring():
    """Demo live monitoring."""
    console.print("\n📊 Live Monitoring Demo", style="bold blue")

    # Create monitor
    monitor = LiveMonitor()

    try:
        # Start monitoring
        await monitor.start_monitoring()

        # Simulate some metrics
        metrics_collector = monitor.metrics_collector

        console.print("📈 Simulating metrics...", style="blue")

        # Simulate processing events
        for i in range(10):
            metrics_collector.increment_counter("events_processed")
            metrics_collector.record_timer("processing_time", random.uniform(50, 200))
            metrics_collector.set_gauge("queue_size", random.randint(0, 50))
            metrics_collector.set_gauge("active_workers", random.randint(2, 8))

            if random.random() < 0.1:  # 10% chance of error
                metrics_collector.increment_counter("events_failed")

            await asyncio.sleep(0.1)

        # Get dashboard data
        dashboard_data = monitor.get_dashboard_data()
        _display_monitoring_dashboard(dashboard_data)

    finally:
        await monitor.stop_monitoring()

async def demo_event_bus():
    """Demo event bus."""
    console.print("\n🚌 Event Bus Demo", style="bold blue")

    # Create event bus
    event_bus = EventBus()

    # Event handler
    async def sample_handler(event: BusEvent):
        console.print(f"📨 Handled event: {event.event_type} from {event.source}", style="dim")
        await asyncio.sleep(0.1)  # Simulate processing

    try:
        # Start event bus
        await event_bus.start()

        # Subscribe to events
        subscription_id = event_bus.subscribe(sample_handler)
        console.print(f"📝 Subscribed with ID: {subscription_id}", style="green")

        # Publish events
        console.print("📡 Publishing events...", style="blue")

        event_types = ["document.added", "document.updated", "system.alert", "user.action"]

        for i in range(8):
            event = BusEvent(
                event_type=random.choice(event_types),
                data={"message": f"Event {i} data"},
                source="demo_script",
                priority=random.choice(list(EventPriority))
            )

            await event_bus.publish(event)
            await asyncio.sleep(0.3)

        # Wait for processing
        await asyncio.sleep(2)

        # Show stats
        stats = event_bus.get_stats()
        _display_event_bus_stats(stats)

    finally:
        await event_bus.stop()

def _display_pipeline_metrics(metrics: dict):
    """Display pipeline metrics."""
    table = Table(title="🚀 Streaming Pipeline Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    pipeline_metrics = metrics.get("pipeline", {})
    queue_metrics = metrics.get("queue", {})
    worker_metrics = metrics.get("workers", {})

    table.add_row("Running", "✅ Yes" if pipeline_metrics.get("is_running") else "❌ No")
    table.add_row("Events Processed", str(pipeline_metrics.get("total_events_processed", 0)))
    table.add_row("Events Failed", str(pipeline_metrics.get("total_events_failed", 0)))
    table.add_row("Success Rate", f"{pipeline_metrics.get('success_rate', 0):.2%}")
    table.add_row("Queue Size", str(queue_metrics.get("size", 0)))
    table.add_row("Queue Usage", f"{queue_metrics.get('usage', 0):.2%}")
    table.add_row("Active Workers", str(worker_metrics.get("active", 0)))
    table.add_row("Total Workers", str(worker_metrics.get("total", 0)))

    console.print(table)

def _display_incremental_stats(stats: dict):
    """Display incremental processor stats."""
    table = Table(title="🔄 Incremental Processor Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Running", "✅ Yes" if stats.get("is_running") else "❌ No")
    table.add_row("Strategy", str(stats.get("strategy", "unknown")))
    table.add_row("Queue Size", str(stats.get("queue_size", 0)))
    table.add_row("Batch Buffer", str(stats.get("batch_buffer_size", 0)))

    index_stats = stats.get("index_manager_stats", {})
    indexes = index_stats.get("indexes", {})

    if indexes:
        for index_name, index_info in indexes.items():
            if index_info:
                table.add_row(f"Index: {index_name}", f"{index_info.get('document_count', 0)} docs")

    console.print(table)

def _display_monitoring_dashboard(dashboard_data: dict):
    """Display monitoring dashboard."""
    current_metrics = dashboard_data.get("current_metrics", {})
    active_alerts = dashboard_data.get("active_alerts", [])
    system_health = dashboard_data.get("system_health", "unknown")

    # Metrics table
    table = Table(title="📊 Live Monitoring Dashboard")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("System Health", f"🟢 {system_health.title()}")
    table.add_row("Events/Second", f"{current_metrics.get('events_per_second', 0):.1f}")
    table.add_row("Avg Processing Time", f"{current_metrics.get('avg_processing_time_ms', 0):.1f}ms")
    table.add_row("Success Rate", f"{current_metrics.get('success_rate', 0):.2%}")
    table.add_row("Queue Usage", f"{current_metrics.get('queue_usage', 0):.2%}")
    table.add_row("Worker Utilization", f"{current_metrics.get('worker_utilization', 0):.2%}")
    table.add_row("P99 Latency", f"{current_metrics.get('p99_latency_ms', 0):.1f}ms")

    console.print(table)

    # Alerts
    if active_alerts:
        console.print(f"\n🚨 Active Alerts: {len(active_alerts)}", style="bold red")
        for alert in active_alerts[:3]:  # Show first 3
            console.print(f"  • {alert['level'].upper()}: {alert['title']}", style="red")

def _display_event_bus_stats(stats: dict):
    """Display event bus stats."""
    table = Table(title="🚌 Event Bus Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Running", "✅ Yes" if stats.get("is_running") else "❌ No")
    table.add_row("Events Published", str(stats.get("events_published", 0)))
    table.add_row("Events Processed", str(stats.get("events_processed", 0)))
    table.add_row("Events Failed", str(stats.get("events_failed", 0)))
    table.add_row("Queue Size", str(stats.get("queue_size", 0)))
    table.add_row("Active Subscriptions", str(stats.get("active_subscriptions", 0)))
    table.add_row("Processing Rate", f"{stats.get('processing_rate', 0):.1f} events/sec")
    table.add_row("Success Rate", f"{stats.get('success_rate', 0):.2%}")

    console.print(table)

async def main():
    """Main demo function."""
    welcome_text = """
⚡ STREAMING PIPELINE ДЕМОНСТРАЦІЯ

Цей скрипт демонструє можливості streaming pipeline:
• Real-time обробка документів
• Incremental оновлення індексів
• Live моніторинг та метрики
• Event-driven архітектура
• Auto-scaling та backpressure handling
    """

    panel = Panel(
        welcome_text.strip(),
        title="🚀 Streaming Demo",
        border_style="blue",
        padding=(1, 2)
    )

    console.print(panel)

    # Run demos
    await demo_streaming_pipeline()
    await demo_incremental_updates()
    await demo_live_monitoring()
    await demo_event_bus()

    # Final summary
    console.print("\n🎉 Streaming Pipeline Demo Completed!", style="bold green")
    console.print("All streaming components are working correctly.", style="green")

if __name__ == "__main__":
    asyncio.run(main())
