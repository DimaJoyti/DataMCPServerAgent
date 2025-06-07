"""
Example script demonstrating the Data Pipeline system.

This example shows how to:
1. Create and configure data pipelines
2. Set up data ingestion from various sources
3. Apply transformations and validations
4. Monitor pipeline execution
5. Handle errors and retries
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.core.orchestrator import PipelineOrchestrator, OrchestratorConfig
from src.data_pipeline.core.pipeline_models import (
    PipelineConfig,
    TaskConfig,
    TaskType,
    DataSourceType,
)
from src.data_pipeline.ingestion.batch.batch_ingestion import BatchIngestionEngine
from src.data_pipeline.ingestion.streaming.stream_ingestion import StreamIngestionEngine

async def create_sample_csv_data():
    """Create sample CSV data for testing."""
    import pandas as pd

    # Create sample data
    data = {
        'id': range(1, 1001),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'email': [f'user_{i}@example.com' for i in range(1, 1001)],
        'age': [20 + (i % 50) for i in range(1, 1001)],
        'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'] * 200,
        'created_at': [datetime.now(timezone.utc) for _ in range(1000)]
    }

    df = pd.DataFrame(data)

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    df.to_csv('data/sample_users.csv', index=False)
    print("Sample CSV data created: data/sample_users.csv")

    return 'data/sample_users.csv'

async def example_batch_ingestion():
    """Example of batch data ingestion."""
    print("\n=== Batch Ingestion Example ===")

    # Create sample data
    csv_file = await create_sample_csv_data()

    # Initialize batch ingestion engine
    ingestion_engine = BatchIngestionEngine()

    # Configure source (CSV file)
    source_config = {
        "type": "file",
        "file_path": csv_file,
        "file_format": "csv",
        "encoding": "utf-8"
    }

    # Configure destination (another CSV file)
    destination_config = {
        "type": "file",
        "file_path": "data/processed_users.csv",
        "file_format": "csv",
        "encoding": "utf-8"
    }

    # Run ingestion
    try:
        metrics = await ingestion_engine.ingest_data(
            source_config=source_config,
            destination_config=destination_config
        )

        print(f"Ingestion completed successfully!")
        print(f"Total records: {metrics.total_records}")
        print(f"Processed records: {metrics.processed_records}")
        print(f"Failed records: {metrics.failed_records}")
        print(f"Processing time: {metrics.processing_time:.2f} seconds")
        print(f"Throughput: {metrics.throughput_records_per_second:.2f} records/sec")

    except Exception as e:
        print(f"Ingestion failed: {e}")

async def example_pipeline_creation():
    """Example of creating and running a data pipeline."""
    print("\n=== Pipeline Creation Example ===")

    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        pipeline_id="user_data_pipeline",
        name="User Data Processing Pipeline",
        description="Pipeline to process user data from CSV files",
        schedule="0 */6 * * *",  # Run every 6 hours
        max_parallel_tasks=3,
        tasks=[
            TaskConfig(
                task_id="ingest_users",
                task_type=TaskType.INGESTION,
                name="Ingest User Data",
                description="Ingest user data from CSV file",
                parameters={
                    "source_config": {
                        "type": "file",
                        "file_path": "data/sample_users.csv",
                        "file_format": "csv"
                    },
                    "destination_config": {
                        "type": "file",
                        "file_path": "data/ingested_users.csv",
                        "file_format": "csv"
                    }
                },
                retry_count=2,
                timeout=300
            ),
            TaskConfig(
                task_id="validate_users",
                task_type=TaskType.VALIDATION,
                name="Validate User Data",
                description="Validate user data quality",
                depends_on=["ingest_users"],
                parameters={
                    "validation_rules": [
                        {
                            "rule_id": "email_format",
                            "name": "Email Format Check",
                            "rule_type": "regex",
                            "column": "email",
                            "condition": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                        },
                        {
                            "rule_id": "age_range",
                            "name": "Age Range Check",
                            "rule_type": "range",
                            "column": "age",
                            "condition": "18 <= value <= 100"
                        }
                    ]
                },
                retry_count=1
            ),
            TaskConfig(
                task_id="transform_users",
                task_type=TaskType.TRANSFORMATION,
                name="Transform User Data",
                description="Apply transformations to user data",
                depends_on=["validate_users"],
                parameters={
                    "transformation_config": {
                        "operations": [
                            {
                                "type": "add_column",
                                "column": "age_group",
                                "expression": "case when age < 30 then 'Young' when age < 50 then 'Middle' else 'Senior' end"
                            },
                            {
                                "type": "uppercase",
                                "column": "city"
                            }
                        ]
                    }
                },
                retry_count=2
            ),
            TaskConfig(
                task_id="export_users",
                task_type=TaskType.EXPORT,
                name="Export Processed Data",
                description="Export processed user data",
                depends_on=["transform_users"],
                parameters={
                    "export_config": {
                        "destination": "data/final_users.csv",
                        "format": "csv"
                    }
                }
            )
        ]
    )

    # Initialize orchestrator
    orchestrator_config = OrchestratorConfig(
        max_concurrent_pipelines=2,
        max_concurrent_tasks=5,
        enable_metrics=True
    )

    orchestrator = PipelineOrchestrator(config=orchestrator_config)

    try:
        # Start orchestrator
        print("Starting pipeline orchestrator...")
        orchestrator_task = asyncio.create_task(orchestrator.start())

        # Wait a moment for orchestrator to start
        await asyncio.sleep(1)

        # Register pipeline
        print("Registering pipeline...")
        pipeline = await orchestrator.register_pipeline(pipeline_config)
        print(f"Pipeline registered: {pipeline.pipeline_id}")

        # Trigger pipeline execution
        print("Triggering pipeline execution...")
        run_id = await orchestrator.trigger_pipeline(
            pipeline_id=pipeline.pipeline_id,
            triggered_by="manual_example"
        )
        print(f"Pipeline triggered with run ID: {run_id}")

        # Monitor pipeline execution
        print("Monitoring pipeline execution...")
        for i in range(30):  # Monitor for up to 30 seconds
            pipeline_run = await orchestrator.get_pipeline_status(run_id)
            if pipeline_run:
                print(f"Pipeline status: {pipeline_run.status}")

                if pipeline_run.status.value in ["success", "failed", "cancelled"]:
                    print(f"Pipeline completed with status: {pipeline_run.status}")
                    if pipeline_run.error_message:
                        print(f"Error: {pipeline_run.error_message}")
                    break

            await asyncio.sleep(1)

        # Get final status
        final_status = await orchestrator.get_pipeline_status(run_id)
        if final_status:
            print(f"\nFinal Pipeline Status:")
            print(f"Status: {final_status.status}")
            print(f"Duration: {final_status.duration:.2f} seconds" if final_status.duration else "Duration: N/A")
            print(f"Tasks completed: {len([t for t in final_status.tasks if t.status.value == 'success'])}/{len(final_status.tasks)}")

        # Stop orchestrator
        print("\nStopping orchestrator...")
        await orchestrator.stop()

        # Cancel the orchestrator task
        orchestrator_task.cancel()
        try:
            await orchestrator_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        await orchestrator.stop()

async def example_streaming_ingestion():
    """Example of streaming data ingestion."""
    print("\n=== Streaming Ingestion Example ===")

    # Initialize streaming engine
    streaming_engine = StreamIngestionEngine()

    # Register a simple message handler
    def handle_user_events(message):
        print(f"Received user event: {message.payload}")
        # Process the message here
        return {"processed": True, "message_id": message.id}

    streaming_engine.register_message_handler("user_events", handle_user_events)

    try:
        # Start streaming engine
        print("Starting streaming engine...")
        streaming_task = asyncio.create_task(streaming_engine.start())

        # Wait a moment for engine to start
        await asyncio.sleep(2)

        # Send some test messages
        print("Sending test messages...")
        for i in range(5):
            await streaming_engine.send_message(
                topic="user_events",
                message={
                    "user_id": f"user_{i}",
                    "event": "login",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            await asyncio.sleep(0.5)

        # Let it process for a few seconds
        await asyncio.sleep(3)

        # Get metrics
        metrics = await streaming_engine.get_metrics()
        print(f"\nStreaming Metrics:")
        print(f"Messages received: {metrics.messages_received}")
        print(f"Messages processed: {metrics.messages_processed}")
        print(f"Messages failed: {metrics.messages_failed}")
        print(f"Throughput: {metrics.throughput_messages_per_second:.2f} msg/sec")

        # Stop streaming engine
        print("\nStopping streaming engine...")
        await streaming_engine.stop()

        # Cancel the streaming task
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        print(f"Streaming ingestion failed: {e}")

async def main():
    """Main example function."""
    print("Data Pipeline System Examples")
    print("=" * 50)

    try:
        # Run batch ingestion example
        await example_batch_ingestion()

        # Run pipeline creation example
        await example_pipeline_creation()

        # Run streaming ingestion example (commented out as it requires Kafka/Redis)
        # await example_streaming_ingestion()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
