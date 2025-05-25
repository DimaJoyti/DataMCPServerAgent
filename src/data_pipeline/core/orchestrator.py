"""
Pipeline Orchestrator for managing data pipeline execution.

This module provides the main orchestration engine for data pipelines,
handling pipeline lifecycle, task coordination, and execution management.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

import structlog
from pydantic import BaseModel

from .pipeline_models import (
    Pipeline,
    PipelineConfig,
    PipelineRun,
    PipelineStatus,
    PipelineTask,
    TaskStatus,
)
from .scheduler import PipelineScheduler
from .executor import PipelineExecutor
from ..storage.unified_access.data_access_layer import DataAccessLayer
from ..monitoring.metrics.pipeline_metrics import PipelineMetrics


class OrchestratorConfig(BaseModel):
    """Configuration for the pipeline orchestrator."""
    max_concurrent_pipelines: int = 10
    max_concurrent_tasks: int = 50
    default_timeout: int = 3600  # 1 hour
    heartbeat_interval: int = 30  # seconds
    cleanup_interval: int = 300  # 5 minutes
    max_retry_attempts: int = 3
    enable_metrics: bool = True
    enable_logging: bool = True


class PipelineOrchestrator:
    """
    Main orchestrator for data pipeline execution.

    Manages pipeline lifecycle, task coordination, dependency resolution,
    and execution monitoring.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        data_access_layer: Optional[DataAccessLayer] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Orchestrator configuration
            data_access_layer: Data access layer for persistence
            logger: Logger instance
        """
        self.config = config or OrchestratorConfig()
        self.data_access_layer = data_access_layer
        self.logger = logger or structlog.get_logger("pipeline_orchestrator")

        # Initialize components
        self.scheduler = PipelineScheduler(logger=self.logger)
        self.executor = PipelineExecutor(logger=self.logger)
        self.metrics = PipelineMetrics() if self.config.enable_metrics else None

        # Runtime state
        self.active_pipelines: Dict[str, PipelineRun] = {}
        self.active_tasks: Dict[str, PipelineTask] = {}
        self.pipeline_registry: Dict[str, Pipeline] = {}

        # Execution control
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.pipeline_semaphore = asyncio.Semaphore(self.config.max_concurrent_pipelines)
        self.task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)

        self.logger.info("Pipeline orchestrator initialized", config=self.config.dict())

    async def start(self) -> None:
        """Start the orchestrator."""
        if self.is_running:
            self.logger.warning("Orchestrator is already running")
            return

        self.is_running = True
        self.shutdown_event.clear()

        self.logger.info("Starting pipeline orchestrator")

        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self.scheduler.start()),
        ]

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        finally:
            # Cancel background tasks
            for task in background_tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*background_tasks, return_exceptions=True)

            self.logger.info("Pipeline orchestrator stopped")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self.is_running:
            return

        self.logger.info("Stopping pipeline orchestrator")
        self.is_running = False
        self.shutdown_event.set()

        # Stop scheduler
        await self.scheduler.stop()

        # Cancel active pipelines
        for pipeline_run in self.active_pipelines.values():
            await self._cancel_pipeline(pipeline_run.run_id)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

    async def register_pipeline(self, pipeline_config: PipelineConfig) -> Pipeline:
        """
        Register a new pipeline.

        Args:
            pipeline_config: Pipeline configuration

        Returns:
            Registered pipeline
        """
        pipeline = Pipeline(
            pipeline_id=pipeline_config.pipeline_id,
            config=pipeline_config,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Validate pipeline configuration
        await self._validate_pipeline_config(pipeline_config)

        # Store in registry
        self.pipeline_registry[pipeline.pipeline_id] = pipeline

        # Schedule if needed
        if pipeline_config.schedule:
            await self.scheduler.schedule_pipeline(pipeline)

        # Persist if data access layer is available
        if self.data_access_layer:
            await self.data_access_layer.save_pipeline(pipeline)

        self.logger.info(
            "Pipeline registered",
            pipeline_id=pipeline.pipeline_id,
            name=pipeline.config.name
        )

        return pipeline

    async def unregister_pipeline(self, pipeline_id: str) -> bool:
        """
        Unregister a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if successfully unregistered
        """
        if pipeline_id not in self.pipeline_registry:
            self.logger.warning("Pipeline not found", pipeline_id=pipeline_id)
            return False

        # Cancel if running
        if pipeline_id in self.active_pipelines:
            await self._cancel_pipeline(pipeline_id)

        # Remove from scheduler
        await self.scheduler.unschedule_pipeline(pipeline_id)

        # Remove from registry
        pipeline = self.pipeline_registry.pop(pipeline_id)

        # Remove from persistence
        if self.data_access_layer:
            await self.data_access_layer.delete_pipeline(pipeline_id)

        self.logger.info("Pipeline unregistered", pipeline_id=pipeline_id)
        return True

    async def trigger_pipeline(
        self,
        pipeline_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        triggered_by: Optional[str] = None
    ) -> str:
        """
        Trigger a pipeline execution.

        Args:
            pipeline_id: Pipeline identifier
            parameters: Runtime parameters
            triggered_by: What triggered this execution

        Returns:
            Run ID
        """
        if pipeline_id not in self.pipeline_registry:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipeline_registry[pipeline_id]

        # Create pipeline run
        run_id = str(uuid.uuid4())
        pipeline_run = PipelineRun(
            run_id=run_id,
            pipeline_id=pipeline_id,
            config=pipeline.config,
            triggered_by=triggered_by or "manual",
            runtime_parameters=parameters or {},
        )

        # Add to active pipelines
        self.active_pipelines[run_id] = pipeline_run

        # Start execution
        asyncio.create_task(self._execute_pipeline(pipeline_run))

        self.logger.info(
            "Pipeline triggered",
            pipeline_id=pipeline_id,
            run_id=run_id,
            triggered_by=triggered_by
        )

        return run_id

    async def get_pipeline_status(self, run_id: str) -> Optional[PipelineRun]:
        """
        Get pipeline run status.

        Args:
            run_id: Pipeline run identifier

        Returns:
            Pipeline run or None if not found
        """
        if run_id in self.active_pipelines:
            return self.active_pipelines[run_id]

        # Check persistence layer
        if self.data_access_layer:
            return await self.data_access_layer.get_pipeline_run(run_id)

        return None

    async def cancel_pipeline(self, run_id: str) -> bool:
        """
        Cancel a pipeline execution.

        Args:
            run_id: Pipeline run identifier

        Returns:
            True if successfully cancelled
        """
        return await self._cancel_pipeline(run_id)

    async def list_active_pipelines(self) -> List[PipelineRun]:
        """
        List all active pipeline runs.

        Returns:
            List of active pipeline runs
        """
        return list(self.active_pipelines.values())

    async def list_registered_pipelines(self) -> List[Pipeline]:
        """
        List all registered pipelines.

        Returns:
            List of registered pipelines
        """
        return list(self.pipeline_registry.values())

    async def _execute_pipeline(self, pipeline_run: PipelineRun) -> None:
        """Execute a pipeline run."""
        async with self.pipeline_semaphore:
            try:
                await self._run_pipeline_internal(pipeline_run)
            except Exception as e:
                self.logger.error(
                    "Pipeline execution failed",
                    run_id=pipeline_run.run_id,
                    error=str(e),
                    exc_info=True
                )
                pipeline_run.status = PipelineStatus.FAILED
                pipeline_run.error_message = str(e)
            finally:
                # Clean up
                pipeline_run.end_time = datetime.now(timezone.utc)
                if pipeline_run.start_time:
                    pipeline_run.duration = (
                        pipeline_run.end_time - pipeline_run.start_time
                    ).total_seconds()

                # Remove from active pipelines
                self.active_pipelines.pop(pipeline_run.run_id, None)

                # Persist final state
                if self.data_access_layer:
                    await self.data_access_layer.save_pipeline_run(pipeline_run)

                # Update metrics
                if self.metrics:
                    await self.metrics.record_pipeline_completion(pipeline_run)

    async def _run_pipeline_internal(self, pipeline_run: PipelineRun) -> None:
        """Internal pipeline execution logic."""
        self.logger.info("Starting pipeline execution", run_id=pipeline_run.run_id)

        pipeline_run.status = PipelineStatus.RUNNING
        pipeline_run.start_time = datetime.now(timezone.utc)

        # Create tasks from configuration
        tasks = []
        for task_config in pipeline_run.config.tasks:
            task = PipelineTask(
                task_id=task_config.task_id,
                pipeline_id=pipeline_run.pipeline_id,
                run_id=pipeline_run.run_id,
                config=task_config,
            )
            tasks.append(task)
            self.active_tasks[f"{pipeline_run.run_id}:{task.task_id}"] = task

        pipeline_run.tasks = tasks

        # Execute tasks based on dependencies
        await self._execute_tasks_with_dependencies(pipeline_run)

        # Determine final status
        failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.error_message = f"Tasks failed: {[t.task_id for t in failed_tasks]}"
        else:
            pipeline_run.status = PipelineStatus.SUCCESS

        self.logger.info(
            "Pipeline execution completed",
            run_id=pipeline_run.run_id,
            status=pipeline_run.status
        )

    async def _execute_tasks_with_dependencies(self, pipeline_run: PipelineRun) -> None:
        """Execute tasks respecting dependencies."""
        tasks = {task.task_id: task for task in pipeline_run.tasks}
        completed_tasks: Set[str] = set()
        running_tasks: Set[str] = set()

        while len(completed_tasks) < len(tasks):
            # Find ready tasks
            ready_tasks = []
            for task_id, task in tasks.items():
                if (task_id not in completed_tasks and
                    task_id not in running_tasks and
                    all(dep in completed_tasks for dep in task.config.depends_on)):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Check if we're deadlocked
                if not running_tasks:
                    remaining_tasks = set(tasks.keys()) - completed_tasks
                    self.logger.error(
                        "Pipeline deadlock detected",
                        run_id=pipeline_run.run_id,
                        remaining_tasks=list(remaining_tasks)
                    )
                    break

                # Wait for running tasks to complete
                await asyncio.sleep(1)
                continue

            # Start ready tasks
            for task in ready_tasks:
                if len(running_tasks) < pipeline_run.config.max_parallel_tasks:
                    running_tasks.add(task.task_id)
                    asyncio.create_task(self._execute_task(task, completed_tasks, running_tasks))
                else:
                    break

            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    async def _execute_task(
        self,
        task: PipelineTask,
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ) -> None:
        """Execute a single task."""
        async with self.task_semaphore:
            try:
                await self.executor.execute_task(task)
                completed_tasks.add(task.task_id)
            except Exception as e:
                self.logger.error(
                    "Task execution failed",
                    task_id=task.task_id,
                    run_id=task.run_id,
                    error=str(e)
                )
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                completed_tasks.add(task.task_id)
            finally:
                running_tasks.discard(task.task_id)

                # Clean up task from active tasks
                task_key = f"{task.run_id}:{task.task_id}"
                self.active_tasks.pop(task_key, None)

    async def _cancel_pipeline(self, run_id: str) -> bool:
        """Cancel a pipeline execution."""
        if run_id not in self.active_pipelines:
            return False

        pipeline_run = self.active_pipelines[run_id]
        pipeline_run.status = PipelineStatus.CANCELLED

        # Cancel all active tasks
        for task in pipeline_run.tasks:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.FAILED
                task.error_message = "Cancelled by user"

        self.logger.info("Pipeline cancelled", run_id=run_id)
        return True

    async def _validate_pipeline_config(self, config: PipelineConfig) -> None:
        """Validate pipeline configuration."""
        # Check for circular dependencies
        task_deps = {task.task_id: task.depends_on for task in config.tasks}

        def has_cycle(task_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for dep in task_deps.get(task_id, []):
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        visited = set()
        for task_id in task_deps:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    raise ValueError(f"Circular dependency detected in pipeline {config.pipeline_id}")

        # Validate task dependencies exist
        task_ids = {task.task_id for task in config.tasks}
        for task in config.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on non-existent task {dep}"
                    )

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self.is_running:
            try:
                # Update metrics
                if self.metrics:
                    await self.metrics.update_orchestrator_metrics(
                        active_pipelines=len(self.active_pipelines),
                        active_tasks=len(self.active_tasks),
                        registered_pipelines=len(self.pipeline_registry)
                    )

                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Heartbeat loop error", error=str(e))
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.is_running:
            try:
                # Clean up completed pipeline runs from memory
                # (they should already be persisted)
                current_time = datetime.now(timezone.utc)

                # Remove old completed runs from active pipelines
                # (this shouldn't happen normally, but just in case)
                to_remove = []
                for run_id, pipeline_run in self.active_pipelines.items():
                    if (pipeline_run.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.CANCELLED] and
                        pipeline_run.end_time and
                        (current_time - pipeline_run.end_time).total_seconds() > 300):  # 5 minutes
                        to_remove.append(run_id)

                for run_id in to_remove:
                    self.active_pipelines.pop(run_id, None)
                    self.logger.debug("Cleaned up completed pipeline run", run_id=run_id)

                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(self.config.cleanup_interval)
