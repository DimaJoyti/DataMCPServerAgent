"""
Data Pipeline Main Entry Point.

This module provides the main entry point for the data pipeline system,
allowing users to interact with pipelines, monitor execution, and manage
data processing workflows.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
import structlog

from ..data_pipeline.core.orchestrator import PipelineOrchestrator, OrchestratorConfig
from ..data_pipeline.core.pipeline_models import PipelineConfig, TaskConfig, TaskType
from ..data_pipeline.ingestion.batch.batch_ingestion import BatchIngestionEngine
from ..data_pipeline.ingestion.streaming.stream_ingestion import StreamIngestionEngine
from ..utils.error_handlers import format_error_for_user

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("data_pipeline_main")


class DataPipelineManager:
    """
    Main manager for the data pipeline system.
    
    Provides a high-level interface for managing pipelines, ingestion,
    and monitoring data processing workflows.
    """
    
    def __init__(self):
        """Initialize the data pipeline manager."""
        self.orchestrator: Optional[PipelineOrchestrator] = None
        self.batch_engine: Optional[BatchIngestionEngine] = None
        self.stream_engine: Optional[StreamIngestionEngine] = None
        self.is_running = False
        
        logger.info("Data pipeline manager initialized")
    
    async def start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Start the data pipeline system.
        
        Args:
            config: Optional configuration dictionary
        """
        if self.is_running:
            logger.warning("Data pipeline system is already running")
            return
        
        try:
            logger.info("Starting data pipeline system")
            
            # Initialize orchestrator
            orchestrator_config = OrchestratorConfig(
                max_concurrent_pipelines=config.get("max_concurrent_pipelines", 10) if config else 10,
                max_concurrent_tasks=config.get("max_concurrent_tasks", 50) if config else 50,
                enable_metrics=config.get("enable_metrics", True) if config else True,
                enable_logging=config.get("enable_logging", True) if config else True
            )
            
            self.orchestrator = PipelineOrchestrator(config=orchestrator_config)
            
            # Initialize batch ingestion engine
            self.batch_engine = BatchIngestionEngine()
            
            # Initialize streaming ingestion engine
            self.stream_engine = StreamIngestionEngine()
            
            # Start orchestrator
            await self.orchestrator.start()
            
            self.is_running = True
            logger.info("Data pipeline system started successfully")
            
        except Exception as e:
            logger.error("Failed to start data pipeline system", error=str(e))
            raise e
    
    async def stop(self) -> None:
        """Stop the data pipeline system."""
        if not self.is_running:
            return
        
        try:
            logger.info("Stopping data pipeline system")
            
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
            
            # Stop streaming engine
            if self.stream_engine:
                await self.stream_engine.stop()
            
            self.is_running = False
            logger.info("Data pipeline system stopped")
            
        except Exception as e:
            logger.error("Error stopping data pipeline system", error=str(e))
    
    async def create_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """
        Create a new data pipeline.
        
        Args:
            pipeline_config: Pipeline configuration
            
        Returns:
            Pipeline ID
        """
        if not self.orchestrator:
            raise RuntimeError("Data pipeline system not started")
        
        try:
            # Convert dict to PipelineConfig
            config = PipelineConfig(**pipeline_config)
            
            # Register pipeline
            pipeline = await self.orchestrator.register_pipeline(config)
            
            logger.info("Pipeline created", pipeline_id=pipeline.pipeline_id)
            return pipeline.pipeline_id
            
        except Exception as e:
            logger.error("Failed to create pipeline", error=str(e))
            raise e
    
    async def trigger_pipeline(
        self,
        pipeline_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger a pipeline execution.
        
        Args:
            pipeline_id: Pipeline identifier
            parameters: Runtime parameters
            
        Returns:
            Run ID
        """
        if not self.orchestrator:
            raise RuntimeError("Data pipeline system not started")
        
        try:
            run_id = await self.orchestrator.trigger_pipeline(
                pipeline_id=pipeline_id,
                parameters=parameters,
                triggered_by="user"
            )
            
            logger.info("Pipeline triggered", pipeline_id=pipeline_id, run_id=run_id)
            return run_id
            
        except Exception as e:
            logger.error("Failed to trigger pipeline", pipeline_id=pipeline_id, error=str(e))
            raise e
    
    async def get_pipeline_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pipeline execution status.
        
        Args:
            run_id: Pipeline run identifier
            
        Returns:
            Pipeline status information
        """
        if not self.orchestrator:
            raise RuntimeError("Data pipeline system not started")
        
        try:
            pipeline_run = await self.orchestrator.get_pipeline_status(run_id)
            
            if pipeline_run:
                return {
                    "run_id": pipeline_run.run_id,
                    "pipeline_id": pipeline_run.pipeline_id,
                    "status": pipeline_run.status.value,
                    "start_time": pipeline_run.start_time.isoformat() if pipeline_run.start_time else None,
                    "end_time": pipeline_run.end_time.isoformat() if pipeline_run.end_time else None,
                    "duration": pipeline_run.duration,
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "status": task.status.value,
                            "start_time": task.start_time.isoformat() if task.start_time else None,
                            "end_time": task.end_time.isoformat() if task.end_time else None,
                            "duration": task.duration,
                            "error_message": task.error_message
                        }
                        for task in pipeline_run.tasks
                    ],
                    "error_message": pipeline_run.error_message
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get pipeline status", run_id=run_id, error=str(e))
            raise e
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all registered pipelines.
        
        Returns:
            List of pipeline information
        """
        if not self.orchestrator:
            raise RuntimeError("Data pipeline system not started")
        
        try:
            pipelines = await self.orchestrator.list_registered_pipelines()
            
            return [
                {
                    "pipeline_id": pipeline.pipeline_id,
                    "name": pipeline.config.name,
                    "description": pipeline.config.description,
                    "is_active": pipeline.is_active,
                    "last_run_status": pipeline.last_run_status.value if pipeline.last_run_status else None,
                    "last_run_time": pipeline.last_run_time.isoformat() if pipeline.last_run_time else None,
                    "total_runs": pipeline.total_runs,
                    "successful_runs": pipeline.successful_runs,
                    "failed_runs": pipeline.failed_runs
                }
                for pipeline in pipelines
            ]
            
        except Exception as e:
            logger.error("Failed to list pipelines", error=str(e))
            raise e
    
    async def run_batch_ingestion(
        self,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run batch data ingestion.
        
        Args:
            source_config: Source configuration
            destination_config: Destination configuration
            transformation_config: Optional transformation configuration
            
        Returns:
            Ingestion metrics
        """
        if not self.batch_engine:
            raise RuntimeError("Data pipeline system not started")
        
        try:
            logger.info("Starting batch ingestion")
            
            metrics = await self.batch_engine.ingest_data(
                source_config=source_config,
                destination_config=destination_config,
                transformation_config=transformation_config
            )
            
            result = {
                "total_records": metrics.total_records,
                "processed_records": metrics.processed_records,
                "failed_records": metrics.failed_records,
                "bytes_processed": metrics.bytes_processed,
                "processing_time": metrics.processing_time,
                "throughput_records_per_second": metrics.throughput_records_per_second,
                "throughput_bytes_per_second": metrics.throughput_bytes_per_second,
                "error_rate": metrics.error_rate
            }
            
            logger.info("Batch ingestion completed", **result)
            return result
            
        except Exception as e:
            logger.error("Batch ingestion failed", error=str(e))
            raise e
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            System status information
        """
        try:
            status = {
                "is_running": self.is_running,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "orchestrator": self.orchestrator is not None,
                    "batch_engine": self.batch_engine is not None,
                    "stream_engine": self.stream_engine is not None
                }
            }
            
            if self.orchestrator:
                active_pipelines = await self.orchestrator.list_active_pipelines()
                registered_pipelines = await self.orchestrator.list_registered_pipelines()
                
                status["orchestrator_stats"] = {
                    "active_pipelines": len(active_pipelines),
                    "registered_pipelines": len(registered_pipelines)
                }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            raise e


# Global manager instance
pipeline_manager = DataPipelineManager()


async def chat_with_data_pipeline_system(config: Optional[Dict[str, Any]] = None):
    """
    Interactive chat interface for the data pipeline system.
    
    Args:
        config: Optional configuration
    """
    print("Data Pipeline System")
    print("=" * 50)
    print("Available commands:")
    print("- 'status' - Show system status")
    print("- 'list' - List all pipelines")
    print("- 'create <pipeline_config>' - Create a new pipeline")
    print("- 'trigger <pipeline_id>' - Trigger a pipeline")
    print("- 'check <run_id>' - Check pipeline run status")
    print("- 'ingest' - Run batch ingestion example")
    print("- 'help' - Show this help message")
    print("- 'exit' - Exit the system")
    print("=" * 50)
    
    try:
        # Start the system
        await pipeline_manager.start(config)
        
        while True:
            try:
                user_input = input("\nData Pipeline> ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'help':
                    print("Available commands:")
                    print("- status, list, create, trigger, check, ingest, help, exit")
                elif user_input.lower() == 'status':
                    status = await pipeline_manager.get_system_status()
                    print(f"System Status: {status}")
                elif user_input.lower() == 'list':
                    pipelines = await pipeline_manager.list_pipelines()
                    print(f"Registered Pipelines: {len(pipelines)}")
                    for pipeline in pipelines:
                        print(f"  - {pipeline['pipeline_id']}: {pipeline['name']} ({pipeline['last_run_status']})")
                elif user_input.lower() == 'ingest':
                    # Example batch ingestion
                    print("Running example batch ingestion...")
                    # This would need actual source and destination configs
                    print("Please provide source and destination configurations")
                else:
                    print(f"Unknown command: {user_input}")
                    print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_msg = format_error_for_user(e)
                print(f"Error: {error_msg}")
    
    finally:
        # Stop the system
        await pipeline_manager.stop()
        print("\nData pipeline system stopped. Goodbye!")


if __name__ == "__main__":
    asyncio.run(chat_with_data_pipeline_system())
