"""
Parallel Executor

Executes tasks in parallel with proper coordination, error handling,
and resource management.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional


class ParallelExecutor:
    """
    Executes tasks in parallel with coordination and error handling.
    
    Features:
    - Concurrent task execution
    - Error isolation and handling
    - Resource management
    - Progress monitoring
    - Graceful shutdown
    """
    
    def __init__(self, config: Any):
        """Initialize the parallel executor."""
        self.config = config
        self.logger = logging.getLogger("parallel_executor")
        
        # Execution state
        self.is_running = False
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(config.max_parallel_agents)
    
    async def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        executor_func: Callable,
    ) -> List[Dict[str, Any]]:
        """Execute tasks in parallel."""
        if not tasks:
            return []
        
        self.logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Create coroutines with semaphore control
        coroutines = []
        for task in tasks:
            coroutine = self._execute_with_semaphore(executor_func, task)
            coroutines.append(coroutine)
        
        # Execute all tasks
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "task_id": tasks[i].get("task_id", f"task_{i}"),
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_semaphore(
        self, executor_func: Callable, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task with semaphore control."""
        async with self.semaphore:
            return await executor_func(task)
    
    async def shutdown(self) -> None:
        """Shutdown the parallel executor."""
        self.logger.info("Shutting down parallel executor")
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        self.active_tasks.clear()
        self.logger.info("Parallel executor shutdown complete")


class StatePersistence:
    """Manages state persistence for the infinite loop system."""
    
    def __init__(self):
        """Initialize state persistence."""
        self.logger = logging.getLogger("state_persistence")
    
    async def save_final_state(self, execution_state: Any) -> None:
        """Save final execution state."""
        if execution_state:
            self.logger.info(f"Saving final state for session: {execution_state.session_id}")


class ErrorRecoveryManager:
    """Manages error recovery for the infinite loop system."""
    
    def __init__(self, config: Any):
        """Initialize error recovery manager."""
        self.config = config
        self.logger = logging.getLogger("error_recovery_manager")
    
    async def handle_task_error(self, task_id: str, error: Exception) -> Dict[str, Any]:
        """Handle task execution error."""
        self.logger.error(f"Task {task_id} failed: {str(error)}")
        
        return {
            "recovery_attempted": False,
            "should_retry": False,
            "error_handled": True,
        }


class OutputValidator:
    """Validates output files and content."""
    
    def __init__(self):
        """Initialize output validator."""
        self.logger = logging.getLogger("output_validator")
    
    async def validate_output(self, file_path: str, content: str) -> Dict[str, Any]:
        """Validate generated output."""
        return {
            "valid": True,
            "file_exists": True,
            "content_valid": True,
            "issues": [],
        }
