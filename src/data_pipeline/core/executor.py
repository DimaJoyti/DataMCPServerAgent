"""
Pipeline Task Executor for executing individual pipeline tasks.

This module provides the execution engine for pipeline tasks,
supporting various task types and execution environments.
"""

import asyncio
import importlib
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable, List
import json
import os

import structlog
from pydantic import BaseModel

from .pipeline_models import PipelineTask, TaskStatus, TaskType


class ExecutorConfig(BaseModel):
    """Configuration for the task executor."""
    max_task_timeout: int = 3600  # 1 hour
    default_retry_delay: int = 60  # 1 minute
    enable_task_isolation: bool = True
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = {}


class TaskExecutionContext(BaseModel):
    """Context for task execution."""
    task: PipelineTask
    working_directory: str
    environment: Dict[str, str]
    timeout: Optional[int] = None
    retry_attempt: int = 0


class PipelineExecutor:
    """
    Executor for pipeline tasks.
    
    Handles execution of different task types including Python functions,
    shell commands, and custom task implementations.
    """
    
    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the pipeline executor.
        
        Args:
            config: Executor configuration
            logger: Logger instance
        """
        self.config = config or ExecutorConfig()
        self.logger = logger or structlog.get_logger("pipeline_executor")
        
        # Task type handlers
        self.task_handlers: Dict[TaskType, Callable] = {
            TaskType.INGESTION: self._execute_ingestion_task,
            TaskType.TRANSFORMATION: self._execute_transformation_task,
            TaskType.VALIDATION: self._execute_validation_task,
            TaskType.PROCESSING: self._execute_processing_task,
            TaskType.EXPORT: self._execute_export_task,
            TaskType.CUSTOM: self._execute_custom_task,
        }
        
        # Custom task registry
        self.custom_task_registry: Dict[str, Callable] = {}
        
        self.logger.info("Pipeline executor initialized")
    
    def register_custom_task(self, task_name: str, handler: Callable) -> None:
        """
        Register a custom task handler.
        
        Args:
            task_name: Name of the custom task
            handler: Function to handle the task execution
        """
        self.custom_task_registry[task_name] = handler
        self.logger.info("Custom task registered", task_name=task_name)
    
    async def execute_task(self, task: PipelineTask) -> None:
        """
        Execute a pipeline task.
        
        Args:
            task: Task to execute
        """
        self.logger.info(
            "Starting task execution",
            task_id=task.task_id,
            task_type=task.config.task_type,
            run_id=task.run_id
        )
        
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now(timezone.utc)
        
        # Create execution context
        context = TaskExecutionContext(
            task=task,
            working_directory=self.config.working_directory or os.getcwd(),
            environment={**os.environ, **self.config.environment_variables},
            timeout=task.config.timeout or self.config.max_task_timeout,
            retry_attempt=task.attempt_count
        )
        
        try:
            # Execute task with retries
            await self._execute_with_retries(context)
            
            task.status = TaskStatus.SUCCESS
            self.logger.info(
                "Task execution completed successfully",
                task_id=task.task_id,
                run_id=task.run_id
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            self.logger.error(
                "Task execution failed",
                task_id=task.task_id,
                run_id=task.run_id,
                error=str(e),
                exc_info=True
            )
            
        finally:
            task.end_time = datetime.now(timezone.utc)
            if task.start_time:
                task.duration = (task.end_time - task.start_time).total_seconds()
    
    async def _execute_with_retries(self, context: TaskExecutionContext) -> None:
        """Execute task with retry logic."""
        task = context.task
        max_retries = task.config.retry_count
        
        for attempt in range(max_retries + 1):
            context.retry_attempt = attempt
            task.attempt_count = attempt + 1
            
            try:
                # Get task handler
                handler = self.task_handlers.get(task.config.task_type)
                if not handler:
                    raise ValueError(f"No handler for task type: {task.config.task_type}")
                
                # Execute task
                result = await handler(context)
                
                # Store result
                if result is not None:
                    task.result = result if isinstance(result, dict) else {"result": result}
                
                # Success - no need to retry
                return
                
            except Exception as e:
                if attempt < max_retries:
                    retry_delay = task.config.retry_delay or self.config.default_retry_delay
                    
                    self.logger.warning(
                        "Task execution failed, retrying",
                        task_id=task.task_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        error=str(e)
                    )
                    
                    task.status = TaskStatus.RETRY
                    await asyncio.sleep(retry_delay)
                else:
                    # Final attempt failed
                    raise e
    
    async def _execute_ingestion_task(self, context: TaskExecutionContext) -> Any:
        """Execute an ingestion task."""
        task = context.task
        
        # Import and execute ingestion function
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            # Use default ingestion logic
            from ..ingestion.batch.batch_ingestion import BatchIngestionEngine
            
            engine = BatchIngestionEngine()
            
            # Extract parameters
            source_config = task.config.parameters.get("source_config", {})
            destination_config = task.config.parameters.get("destination_config", {})
            
            return await engine.ingest_data(source_config, destination_config)
    
    async def _execute_transformation_task(self, context: TaskExecutionContext) -> Any:
        """Execute a transformation task."""
        task = context.task
        
        # Import and execute transformation function
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            # Use default transformation logic
            from ..transformation.etl.etl_engine import ETLEngine
            
            engine = ETLEngine()
            
            # Extract parameters
            transformation_config = task.config.parameters.get("transformation_config", {})
            input_data = task.config.parameters.get("input_data")
            
            return await engine.transform_data(input_data, transformation_config)
    
    async def _execute_validation_task(self, context: TaskExecutionContext) -> Any:
        """Execute a validation task."""
        task = context.task
        
        # Import and execute validation function
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            # Use default validation logic
            from ..transformation.validation.data_validator import DataValidator
            
            validator = DataValidator()
            
            # Extract parameters
            validation_rules = task.config.parameters.get("validation_rules", [])
            input_data = task.config.parameters.get("input_data")
            
            return await validator.validate_data(input_data, validation_rules)
    
    async def _execute_processing_task(self, context: TaskExecutionContext) -> Any:
        """Execute a processing task."""
        task = context.task
        
        # Import and execute processing function
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            # Use default processing logic
            from ..processing.batch.batch_processor import BatchProcessor
            
            processor = BatchProcessor()
            
            # Extract parameters
            processing_config = task.config.parameters.get("processing_config", {})
            input_data = task.config.parameters.get("input_data")
            
            return await processor.process_data(input_data, processing_config)
    
    async def _execute_export_task(self, context: TaskExecutionContext) -> Any:
        """Execute an export task."""
        task = context.task
        
        # Import and execute export function
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            # Use default export logic
            # This would integrate with the storage layer
            export_config = task.config.parameters.get("export_config", {})
            input_data = task.config.parameters.get("input_data")
            
            # Placeholder for export logic
            self.logger.info("Executing export task", config=export_config)
            return {"exported": True, "config": export_config}
    
    async def _execute_custom_task(self, context: TaskExecutionContext) -> Any:
        """Execute a custom task."""
        task = context.task
        
        # Check if custom task is registered
        task_name = task.config.parameters.get("task_name")
        if task_name and task_name in self.custom_task_registry:
            handler = self.custom_task_registry[task_name]
            return await handler(context)
        
        # Fall back to function or command execution
        if task.config.function and task.config.module:
            return await self._execute_python_function(context)
        elif task.config.command:
            return await self._execute_shell_command(context)
        else:
            raise ValueError(f"No handler found for custom task: {task.task_id}")
    
    async def _execute_python_function(self, context: TaskExecutionContext) -> Any:
        """Execute a Python function."""
        task = context.task
        
        try:
            # Import module
            module = importlib.import_module(task.config.module)
            
            # Get function
            func = getattr(module, task.config.function)
            
            # Prepare arguments
            kwargs = task.config.parameters.copy()
            kwargs["context"] = context
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**kwargs))
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Python function execution failed",
                task_id=task.task_id,
                module=task.config.module,
                function=task.config.function,
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise e
    
    async def _execute_shell_command(self, context: TaskExecutionContext) -> Any:
        """Execute a shell command."""
        task = context.task
        
        try:
            # Prepare environment
            env = context.environment.copy()
            
            # Add task parameters as environment variables
            for key, value in task.config.parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    env[f"TASK_{key.upper()}"] = str(value)
                elif isinstance(value, (dict, list)):
                    env[f"TASK_{key.upper()}"] = json.dumps(value)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                task.config.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_directory,
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=context.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Task {task.task_id} timed out after {context.timeout} seconds")
            
            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Command failed"
                raise RuntimeError(f"Command failed with return code {process.returncode}: {error_msg}")
            
            # Return output
            output = stdout.decode() if stdout else ""
            
            # Try to parse as JSON, otherwise return as string
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {"output": output}
                
        except Exception as e:
            self.logger.error(
                "Shell command execution failed",
                task_id=task.task_id,
                command=task.config.command,
                error=str(e)
            )
            raise e
