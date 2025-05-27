"""
Task queue system for distributed document processing.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field

from pydantic import BaseModel


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Task representation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 1.0
    
    # Results and errors
    result: Any = None
    error: Optional[str] = None
    
    # Progress tracking
    progress: float = 0.0
    progress_message: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.name and self.func:
            self.name = f"{self.func.__name__}_{self.id[:8]}"
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        """Get total time from creation to completion."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count,
            'progress': self.progress,
            'progress_message': self.progress_message,
            'error': self.error,
            'execution_time': self.execution_time,
            'total_time': self.total_time,
            'metadata': self.metadata
        }


class TaskQueue:
    """Asynchronous task queue with priority support."""
    
    def __init__(self, maxsize: int = 0):
        """
        Initialize task queue.
        
        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.maxsize = maxsize
        self._queue = asyncio.PriorityQueue(maxsize=maxsize)
        self._tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def put(self, task: Task) -> None:
        """
        Add task to queue.
        
        Args:
            task: Task to add
        """
        # Store task reference
        self._tasks[task.id] = task
        
        # Add to priority queue (negative priority for correct ordering)
        await self._queue.put((-task.priority.value, task.created_at, task))
        
        self.logger.debug(f"Added task {task.id} to queue")
    
    async def get(self) -> Task:
        """
        Get next task from queue.
        
        Returns:
            Task: Next task to process
        """
        _, _, task = await self._queue.get()
        return task
    
    async def get_nowait(self) -> Optional[Task]:
        """
        Get next task without waiting.
        
        Returns:
            Optional[Task]: Next task or None if queue is empty
        """
        try:
            _, _, task = self._queue.get_nowait()
            return task
        except asyncio.QueueEmpty:
            return None
    
    def task_done(self, task: Task) -> None:
        """
        Mark task as done.
        
        Args:
            task: Completed task
        """
        # Move to completed tasks
        if task.id in self._tasks:
            del self._tasks[task.id]
        
        self._completed_tasks[task.id] = task
        self._queue.task_done()
        
        self.logger.debug(f"Task {task.id} marked as done")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Task]: Task if found
        """
        # Check active tasks first
        if task_id in self._tasks:
            return self._tasks[task_id]
        
        # Check completed tasks
        return self._completed_tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get list of pending tasks."""
        return [task for task in self._tasks.values() if task.status == TaskStatus.PENDING]
    
    def get_running_tasks(self) -> List[Task]:
        """Get list of running tasks."""
        return [task for task in self._tasks.values() if task.status == TaskStatus.RUNNING]
    
    def get_completed_tasks(self) -> List[Task]:
        """Get list of completed tasks."""
        return list(self._completed_tasks.values())
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending_tasks = self.get_pending_tasks()
        running_tasks = self.get_running_tasks()
        completed_tasks = self.get_completed_tasks()
        
        # Calculate average execution time
        execution_times = [t.execution_time for t in completed_tasks if t.execution_time]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            'queue_size': self.qsize(),
            'pending_tasks': len(pending_tasks),
            'running_tasks': len(running_tasks),
            'completed_tasks': len(completed_tasks),
            'total_tasks': len(self._tasks) + len(self._completed_tasks),
            'average_execution_time': avg_execution_time,
            'is_empty': self.empty(),
            'is_full': self.full()
        }


class TaskManager:
    """Task manager for coordinating task execution."""
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_maxsize: int = 0,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """
        Initialize task manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
            queue_maxsize: Maximum queue size
            cleanup_interval: Interval for cleaning up old tasks (seconds)
        """
        self.max_workers = max_workers
        self.queue = TaskQueue(maxsize=queue_maxsize)
        self.cleanup_interval = cleanup_interval
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'start_time': None
        }
    
    async def start(self) -> None:
        """Start the task manager."""
        if self.running:
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.workers.append(cleanup_task)
        
        self.logger.info(f"Started task manager with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop the task manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.logger.info("Task manager stopped")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            name: Task name
            priority: Task priority
            max_retries: Maximum retry attempts
            metadata: Task metadata
            **kwargs: Function keyword arguments
            
        Returns:
            str: Task ID
        """
        task = Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            metadata=metadata or {}
        )
        
        await self.queue.put(task)
        
        self.logger.info(f"Submitted task {task.id}: {task.name}")
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task status or None if not found
        """
        task = self.queue.get_task(task_id)
        if task:
            return task.to_dict()
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if cancelled successfully
        """
        task = self.queue.get_task(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self.logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine."""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next task
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                if task.status == TaskStatus.CANCELLED:
                    self.queue.task_done(task)
                    continue
                
                # Execute task
                await self._execute_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
        
        self.logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_task(self, task: Task, worker_name: str) -> None:
        """Execute a single task."""
        async with self.worker_semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            self.logger.info(f"Worker {worker_name} executing task {task.id}")
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.progress = 100.0
                
                self.stats['tasks_processed'] += 1
                if task.execution_time:
                    self.stats['total_execution_time'] += task.execution_time
                
                self.logger.info(f"Task {task.id} completed successfully")
                
            except Exception as e:
                # Task failed
                task.error = str(e)
                task.retry_count += 1
                
                if task.retry_count <= task.max_retries:
                    # Retry task
                    task.status = TaskStatus.RETRYING
                    self.logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                    
                    # Add delay before retry
                    await asyncio.sleep(task.retry_delay * task.retry_count)
                    
                    # Reset task for retry
                    task.started_at = None
                    task.status = TaskStatus.PENDING
                    
                    # Re-queue task
                    await self.queue.put(task)
                    return
                else:
                    # Max retries exceeded
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    
                    self.stats['tasks_failed'] += 1
                    
                    self.logger.error(f"Task {task.id} failed after {task.retry_count} retries: {e}")
            
            finally:
                self.queue.task_done(task)
    
    async def _cleanup_task(self) -> None:
        """Cleanup old completed tasks."""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Remove old completed tasks (older than 24 hours)
                cutoff_time = datetime.now().timestamp() - 86400  # 24 hours
                
                to_remove = []
                for task_id, task in self.queue._completed_tasks.items():
                    if task.completed_at and task.completed_at.timestamp() < cutoff_time:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self.queue._completed_tasks[task_id]
                
                if to_remove:
                    self.logger.info(f"Cleaned up {len(to_remove)} old tasks")
                    
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        queue_stats = self.queue.get_stats()
        
        uptime = 0.0
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        avg_execution_time = 0.0
        if self.stats['tasks_processed'] > 0:
            avg_execution_time = self.stats['total_execution_time'] / self.stats['tasks_processed']
        
        return {
            **queue_stats,
            'workers': len(self.workers),
            'max_workers': self.max_workers,
            'is_running': self.running,
            'uptime': uptime,
            'tasks_processed': self.stats['tasks_processed'],
            'tasks_failed': self.stats['tasks_failed'],
            'average_execution_time': avg_execution_time,
            'success_rate': (self.stats['tasks_processed'] / (self.stats['tasks_processed'] + self.stats['tasks_failed'])) * 100 if (self.stats['tasks_processed'] + self.stats['tasks_failed']) > 0 else 0
        }
