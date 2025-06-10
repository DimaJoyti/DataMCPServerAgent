"""
Progress Tracker

Tracks progress of iteration generation tasks and provides real-time
monitoring and reporting capabilities.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class TaskProgress:
    """Progress information for a single task."""
    
    task_id: str
    iteration_number: int
    status: str  # pending, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    current_stage: str = "pending"
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


class ProgressTracker:
    """
    Tracks progress of iteration generation tasks.
    
    Features:
    - Real-time task progress monitoring
    - Completion time estimation
    - Performance metrics calculation
    - Progress reporting and visualization
    - Error tracking and analysis
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.logger = logging.getLogger("progress_tracker")
        
        # Task tracking
        self.tasks: Dict[str, TaskProgress] = {}
        self.completed_tasks: List[TaskProgress] = []
        self.failed_tasks: List[TaskProgress] = []
        
        # Performance metrics
        self.total_tasks_started = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_execution_time = 0.0
        self.average_task_time = 0.0
        
        # Progress stages
        self.progress_stages = [
            "pending",
            "initializing",
            "analyzing_spec",
            "preparing_context",
            "generating_content",
            "validating_content",
            "saving_file",
            "completed",
        ]
    
    def start_task(self, task_id: str, iteration_number: int) -> None:
        """Start tracking a new task."""
        task_progress = TaskProgress(
            task_id=task_id,
            iteration_number=iteration_number,
            status="running",
            start_time=datetime.now(),
            current_stage="initializing",
            progress_percentage=0.0,
        )
        
        self.tasks[task_id] = task_progress
        self.total_tasks_started += 1
        
        # Estimate completion time based on historical data
        if self.average_task_time > 0:
            estimated_duration = timedelta(seconds=self.average_task_time)
            task_progress.estimated_completion = task_progress.start_time + estimated_duration
        
        self.logger.debug(f"Started tracking task {task_id} (iteration {iteration_number})")
    
    def update_task_progress(
        self,
        task_id: str,
        stage: str,
        progress_percentage: Optional[float] = None,
    ) -> None:
        """Update progress for a task."""
        if task_id not in self.tasks:
            self.logger.warning(f"Task {task_id} not found for progress update")
            return
        
        task = self.tasks[task_id]
        task.current_stage = stage
        
        # Calculate progress percentage based on stage
        if progress_percentage is not None:
            task.progress_percentage = progress_percentage
        else:
            task.progress_percentage = self._calculate_stage_progress(stage)
        
        # Update estimated completion
        if task.start_time and self.average_task_time > 0:
            elapsed = (datetime.now() - task.start_time).total_seconds()
            if task.progress_percentage > 0:
                estimated_total = elapsed / (task.progress_percentage / 100.0)
                remaining = estimated_total - elapsed
                task.estimated_completion = datetime.now() + timedelta(seconds=remaining)
        
        self.logger.debug(f"Task {task_id} progress: {stage} ({task.progress_percentage:.1f}%)")
    
    def complete_task(self, task_id: str, success: bool, error_message: Optional[str] = None) -> None:
        """Mark a task as completed."""
        if task_id not in self.tasks:
            self.logger.warning(f"Task {task_id} not found for completion")
            return
        
        task = self.tasks[task_id]
        task.end_time = datetime.now()
        task.progress_percentage = 100.0
        
        if success:
            task.status = "completed"
            task.current_stage = "completed"
            self.completed_tasks.append(task)
            self.total_tasks_completed += 1
        else:
            task.status = "failed"
            task.error_message = error_message
            self.failed_tasks.append(task)
            self.total_tasks_failed += 1
        
        # Update performance metrics
        if task.start_time and task.end_time:
            execution_time = (task.end_time - task.start_time).total_seconds()
            self.total_execution_time += execution_time
            
            completed_count = self.total_tasks_completed + self.total_tasks_failed
            if completed_count > 0:
                self.average_task_time = self.total_execution_time / completed_count
        
        # Remove from active tasks
        del self.tasks[task_id]
        
        status_msg = "completed successfully" if success else f"failed: {error_message}"
        self.logger.info(f"Task {task_id} {status_msg}")
    
    def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.status = "cancelled"
        task.end_time = datetime.now()
        
        del self.tasks[task_id]
        self.logger.info(f"Task {task_id} cancelled")
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        total_tasks = self.total_tasks_started
        active_tasks = len(self.tasks)
        
        if total_tasks == 0:
            return {
                "total_tasks": 0,
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0.0,
                "overall_progress": 0.0,
                "estimated_completion": None,
            }
        
        # Calculate overall progress
        completed_progress = self.total_tasks_completed * 100.0
        active_progress = sum(task.progress_percentage for task in self.tasks.values())
        overall_progress = (completed_progress + active_progress) / total_tasks
        
        # Calculate success rate
        finished_tasks = self.total_tasks_completed + self.total_tasks_failed
        success_rate = self.total_tasks_completed / finished_tasks if finished_tasks > 0 else 0.0
        
        # Estimate overall completion time
        estimated_completion = None
        if active_tasks > 0 and self.average_task_time > 0:
            remaining_time = max(
                (task.estimated_completion - datetime.now()).total_seconds()
                for task in self.tasks.values()
                if task.estimated_completion
            ) if any(task.estimated_completion for task in self.tasks.values()) else 0
            
            if remaining_time > 0:
                estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": active_tasks,
            "completed_tasks": self.total_tasks_completed,
            "failed_tasks": self.total_tasks_failed,
            "success_rate": success_rate,
            "overall_progress": min(overall_progress, 100.0),
            "average_task_time": self.average_task_time,
            "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
        }
    
    def get_active_tasks_status(self) -> List[Dict[str, Any]]:
        """Get status of all active tasks."""
        return [
            {
                "task_id": task.task_id,
                "iteration_number": task.iteration_number,
                "status": task.status,
                "current_stage": task.current_stage,
                "progress_percentage": task.progress_percentage,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "estimated_completion": task.estimated_completion.isoformat() if task.estimated_completion else None,
                "elapsed_time": (datetime.now() - task.start_time).total_seconds() if task.start_time else 0,
            }
            for task in self.tasks.values()
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_tasks_started": self.total_tasks_started,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "success_rate": self.total_tasks_completed / max(1, self.total_tasks_started),
            "average_task_time": self.average_task_time,
            "total_execution_time": self.total_execution_time,
            "tasks_per_minute": (self.total_tasks_completed / (self.total_execution_time / 60.0)) if self.total_execution_time > 0 else 0.0,
        }
    
    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task failures for analysis."""
        recent_failures = sorted(
            self.failed_tasks,
            key=lambda t: t.end_time or datetime.now(),
            reverse=True
        )[:limit]
        
        return [
            {
                "task_id": task.task_id,
                "iteration_number": task.iteration_number,
                "error_message": task.error_message,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None,
                "execution_time": (task.end_time - task.start_time).total_seconds() if task.start_time and task.end_time else 0,
            }
            for task in recent_failures
        ]
    
    def _calculate_stage_progress(self, stage: str) -> float:
        """Calculate progress percentage based on current stage."""
        if stage not in self.progress_stages:
            return 0.0
        
        stage_index = self.progress_stages.index(stage)
        total_stages = len(self.progress_stages) - 1  # Exclude 'pending'
        
        return (stage_index / total_stages) * 100.0
    
    def reset_statistics(self) -> None:
        """Reset all statistics and tracking data."""
        self.tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        
        self.total_tasks_started = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_execution_time = 0.0
        self.average_task_time = 0.0
        
        self.logger.info("Progress tracker statistics reset")
    
    def generate_progress_report(self) -> str:
        """Generate a human-readable progress report."""
        overall = self.get_overall_progress()
        performance = self.get_performance_metrics()
        active_tasks = self.get_active_tasks_status()
        
        report_lines = [
            "=== Infinite Loop Progress Report ===",
            f"Total Tasks: {overall['total_tasks']}",
            f"Active Tasks: {overall['active_tasks']}",
            f"Completed: {overall['completed_tasks']}",
            f"Failed: {overall['failed_tasks']}",
            f"Success Rate: {overall['success_rate']:.1%}",
            f"Overall Progress: {overall['overall_progress']:.1f}%",
            f"Average Task Time: {performance['average_task_time']:.1f}s",
            "",
        ]
        
        if active_tasks:
            report_lines.append("Active Tasks:")
            for task in active_tasks:
                report_lines.append(
                    f"  - Iteration {task['iteration_number']}: {task['current_stage']} "
                    f"({task['progress_percentage']:.1f}%)"
                )
        
        if overall['estimated_completion']:
            report_lines.append(f"Estimated Completion: {overall['estimated_completion']}")
        
        return "\n".join(report_lines)
