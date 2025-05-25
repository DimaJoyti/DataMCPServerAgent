"""
Pipeline Scheduler for managing scheduled pipeline executions.

This module provides scheduling capabilities for data pipelines,
supporting cron-like scheduling and event-driven triggers.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
import schedule
from croniter import croniter

import structlog
from pydantic import BaseModel

from .pipeline_models import Pipeline, PipelineStatus


class ScheduleEntry(BaseModel):
    """Represents a scheduled pipeline entry."""
    pipeline_id: str
    schedule_expression: str
    timezone: str = "UTC"
    next_run_time: Optional[datetime] = None
    last_run_time: Optional[datetime] = None
    is_active: bool = True
    max_concurrent_runs: int = 1
    current_runs: int = 0


class PipelineScheduler:
    """
    Scheduler for data pipelines.
    
    Supports cron-like scheduling expressions and manages pipeline execution timing.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        check_interval: int = 30  # seconds
    ):
        """
        Initialize the pipeline scheduler.
        
        Args:
            logger: Logger instance
            check_interval: How often to check for scheduled pipelines (seconds)
        """
        self.logger = logger or structlog.get_logger("pipeline_scheduler")
        self.check_interval = check_interval
        
        # Scheduled pipelines
        self.scheduled_pipelines: Dict[str, ScheduleEntry] = {}
        
        # Execution callback
        self.execution_callback: Optional[Callable[[str, str], Any]] = None
        
        # Control
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("Pipeline scheduler initialized")
    
    def set_execution_callback(self, callback: Callable[[str, str], Any]) -> None:
        """
        Set the callback function for pipeline execution.
        
        Args:
            callback: Function to call when a pipeline should be executed
                     Should accept (pipeline_id, triggered_by) parameters
        """
        self.execution_callback = callback
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        self.logger.info("Starting pipeline scheduler")
        
        # Start the scheduling loop
        await self._scheduling_loop()
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping pipeline scheduler")
        self.is_running = False
        self.shutdown_event.set()
    
    async def schedule_pipeline(self, pipeline: Pipeline) -> bool:
        """
        Schedule a pipeline for execution.
        
        Args:
            pipeline: Pipeline to schedule
            
        Returns:
            True if successfully scheduled
        """
        if not pipeline.config.schedule:
            self.logger.warning(
                "Pipeline has no schedule expression",
                pipeline_id=pipeline.pipeline_id
            )
            return False
        
        try:
            # Validate cron expression
            cron = croniter(pipeline.config.schedule)
            next_run = cron.get_next(datetime)
            
            # Create schedule entry
            schedule_entry = ScheduleEntry(
                pipeline_id=pipeline.pipeline_id,
                schedule_expression=pipeline.config.schedule,
                timezone=pipeline.config.timezone,
                next_run_time=next_run,
            )
            
            self.scheduled_pipelines[pipeline.pipeline_id] = schedule_entry
            
            self.logger.info(
                "Pipeline scheduled",
                pipeline_id=pipeline.pipeline_id,
                schedule=pipeline.config.schedule,
                next_run=next_run
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to schedule pipeline",
                pipeline_id=pipeline.pipeline_id,
                schedule=pipeline.config.schedule,
                error=str(e)
            )
            return False
    
    async def unschedule_pipeline(self, pipeline_id: str) -> bool:
        """
        Unschedule a pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            True if successfully unscheduled
        """
        if pipeline_id in self.scheduled_pipelines:
            del self.scheduled_pipelines[pipeline_id]
            self.logger.info("Pipeline unscheduled", pipeline_id=pipeline_id)
            return True
        
        return False
    
    async def get_scheduled_pipelines(self) -> List[ScheduleEntry]:
        """
        Get all scheduled pipelines.
        
        Returns:
            List of scheduled pipeline entries
        """
        return list(self.scheduled_pipelines.values())
    
    async def get_next_scheduled_runs(self, limit: int = 10) -> List[ScheduleEntry]:
        """
        Get the next scheduled pipeline runs.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of next scheduled runs, sorted by next run time
        """
        entries = [
            entry for entry in self.scheduled_pipelines.values()
            if entry.is_active and entry.next_run_time
        ]
        
        # Sort by next run time
        entries.sort(key=lambda x: x.next_run_time or datetime.max)
        
        return entries[:limit]
    
    async def pause_pipeline_schedule(self, pipeline_id: str) -> bool:
        """
        Pause a pipeline schedule.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            True if successfully paused
        """
        if pipeline_id in self.scheduled_pipelines:
            self.scheduled_pipelines[pipeline_id].is_active = False
            self.logger.info("Pipeline schedule paused", pipeline_id=pipeline_id)
            return True
        
        return False
    
    async def resume_pipeline_schedule(self, pipeline_id: str) -> bool:
        """
        Resume a pipeline schedule.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            True if successfully resumed
        """
        if pipeline_id in self.scheduled_pipelines:
            entry = self.scheduled_pipelines[pipeline_id]
            entry.is_active = True
            
            # Recalculate next run time
            try:
                cron = croniter(entry.schedule_expression)
                entry.next_run_time = cron.get_next(datetime)
                
                self.logger.info(
                    "Pipeline schedule resumed",
                    pipeline_id=pipeline_id,
                    next_run=entry.next_run_time
                )
                return True
                
            except Exception as e:
                self.logger.error(
                    "Failed to resume pipeline schedule",
                    pipeline_id=pipeline_id,
                    error=str(e)
                )
                return False
        
        return False
    
    async def trigger_immediate_run(self, pipeline_id: str) -> bool:
        """
        Trigger an immediate run of a scheduled pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            True if successfully triggered
        """
        if pipeline_id not in self.scheduled_pipelines:
            self.logger.warning(
                "Pipeline not scheduled",
                pipeline_id=pipeline_id
            )
            return False
        
        entry = self.scheduled_pipelines[pipeline_id]
        
        # Check concurrent run limit
        if entry.current_runs >= entry.max_concurrent_runs:
            self.logger.warning(
                "Pipeline already at max concurrent runs",
                pipeline_id=pipeline_id,
                current_runs=entry.current_runs,
                max_concurrent=entry.max_concurrent_runs
            )
            return False
        
        # Trigger execution
        if self.execution_callback:
            try:
                entry.current_runs += 1
                await self.execution_callback(pipeline_id, "manual_trigger")
                
                self.logger.info(
                    "Pipeline manually triggered",
                    pipeline_id=pipeline_id
                )
                return True
                
            except Exception as e:
                entry.current_runs -= 1
                self.logger.error(
                    "Failed to trigger pipeline",
                    pipeline_id=pipeline_id,
                    error=str(e)
                )
                return False
        
        return False
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check for pipelines that need to run
                for pipeline_id, entry in self.scheduled_pipelines.items():
                    if not entry.is_active or not entry.next_run_time:
                        continue
                    
                    # Check if it's time to run
                    if current_time >= entry.next_run_time:
                        await self._execute_scheduled_pipeline(entry)
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Scheduling loop error", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _execute_scheduled_pipeline(self, entry: ScheduleEntry) -> None:
        """Execute a scheduled pipeline."""
        # Check concurrent run limit
        if entry.current_runs >= entry.max_concurrent_runs:
            self.logger.warning(
                "Skipping scheduled run due to concurrent limit",
                pipeline_id=entry.pipeline_id,
                current_runs=entry.current_runs,
                max_concurrent=entry.max_concurrent_runs
            )
            
            # Still update next run time
            self._update_next_run_time(entry)
            return
        
        # Trigger execution
        if self.execution_callback:
            try:
                entry.current_runs += 1
                entry.last_run_time = datetime.now(timezone.utc)
                
                await self.execution_callback(entry.pipeline_id, "scheduled")
                
                self.logger.info(
                    "Scheduled pipeline triggered",
                    pipeline_id=entry.pipeline_id,
                    last_run=entry.last_run_time
                )
                
            except Exception as e:
                entry.current_runs -= 1
                self.logger.error(
                    "Failed to execute scheduled pipeline",
                    pipeline_id=entry.pipeline_id,
                    error=str(e)
                )
        
        # Update next run time
        self._update_next_run_time(entry)
    
    def _update_next_run_time(self, entry: ScheduleEntry) -> None:
        """Update the next run time for a schedule entry."""
        try:
            cron = croniter(entry.schedule_expression)
            entry.next_run_time = cron.get_next(datetime)
            
            self.logger.debug(
                "Updated next run time",
                pipeline_id=entry.pipeline_id,
                next_run=entry.next_run_time
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to update next run time",
                pipeline_id=entry.pipeline_id,
                error=str(e)
            )
            entry.is_active = False
    
    async def pipeline_execution_completed(
        self,
        pipeline_id: str,
        status: PipelineStatus
    ) -> None:
        """
        Notify scheduler that a pipeline execution completed.
        
        Args:
            pipeline_id: Pipeline identifier
            status: Final execution status
        """
        if pipeline_id in self.scheduled_pipelines:
            entry = self.scheduled_pipelines[pipeline_id]
            entry.current_runs = max(0, entry.current_runs - 1)
            
            self.logger.debug(
                "Pipeline execution completed",
                pipeline_id=pipeline_id,
                status=status,
                current_runs=entry.current_runs
            )
