"""
Monitoring Scheduler

Automated scheduling and execution of monitoring tasks.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict

import schedule

from .config import MonitoringConfig

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Schedule and run monitoring tasks automatically"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.running = False
        self.scheduler_thread = None
        self.tasks = {}
        self.last_run_times = {}

    def register_task(self, name: str, func: Callable, interval_minutes: int, enabled: bool = True):
        """Register a monitoring task"""
        self.tasks[name] = {
            "function": func,
            "interval_minutes": interval_minutes,
            "enabled": enabled,
            "last_run": None,
            "next_run": None,
            "run_count": 0,
            "error_count": 0
        }

        if enabled:
            # Schedule the task
            schedule.every(interval_minutes).minutes.do(self._run_task, name)
            logger.info(f"Scheduled task '{name}' to run every {interval_minutes} minutes")

    def _run_task(self, task_name: str):
        """Run a specific monitoring task"""
        task = self.tasks.get(task_name)
        if not task or not task["enabled"]:
            return

        logger.info(f"Running monitoring task: {task_name}")
        start_time = time.time()

        try:
            # Run the task function
            result = task["function"]()

            # Update task statistics
            task["last_run"] = datetime.now()
            task["run_count"] += 1
            execution_time = time.time() - start_time

            logger.info(f"Task '{task_name}' completed successfully in {execution_time:.2f}s")

            # Save task result if it returns data
            if result:
                self._save_task_result(task_name, result)

        except Exception as e:
            task["error_count"] += 1
            execution_time = time.time() - start_time
            logger.error(f"Task '{task_name}' failed after {execution_time:.2f}s: {e}")

            # Save error information
            self._save_task_error(task_name, str(e))

    def _save_task_result(self, task_name: str, result: Any):
        """Save task result to data directory"""
        try:
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)

            result_file = data_dir / f"{task_name}_result.json"

            # Convert result to JSON-serializable format
            if hasattr(result, '__dict__'):
                # Handle dataclass or object with attributes
                result_data = {
                    "timestamp": datetime.now().isoformat(),
                    "task": task_name,
                    "data": result.__dict__ if hasattr(result, '__dict__') else str(result)
                }
            else:
                result_data = {
                    "timestamp": datetime.now().isoformat(),
                    "task": task_name,
                    "data": result
                }

            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save result for task '{task_name}': {e}")

    def _save_task_error(self, task_name: str, error_message: str):
        """Save task error information"""
        try:
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)

            error_file = data_dir / f"{task_name}_errors.json"

            error_data = {
                "timestamp": datetime.now().isoformat(),
                "task": task_name,
                "error": error_message
            }

            # Append to existing errors or create new file
            errors = []
            if error_file.exists():
                try:
                    with open(error_file) as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            errors = existing_data
                        elif isinstance(existing_data, dict):
                            errors = [existing_data]
                except:
                    pass

            errors.append(error_data)

            # Keep only last 50 errors
            errors = errors[-50:]

            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save error for task '{task_name}': {e}")

    def setup_default_tasks(self):
        """Setup default monitoring tasks based on configuration"""

        # CI/CD Performance Monitoring
        if self.config.cicd.enabled:
            def cicd_monitor():
                import os

                from ..ci_cd.performance_monitor import monitor_cicd_performance

                github_token = self.config.github.token or os.getenv("GITHUB_TOKEN")
                if github_token:
                    return asyncio.run(monitor_cicd_performance(
                        github_token=github_token,
                        owner=self.config.github.owner,
                        repo=self.config.github.repo,
                        output_path=f"{self.config.data_directory}/cicd_metrics.json"
                    ))
                else:
                    logger.warning("GitHub token not available for CI/CD monitoring")
                    return None

            self.register_task(
                "cicd_monitor",
                cicd_monitor,
                self.config.cicd.check_interval_minutes,
                self.config.cicd.enabled
            )

        # Code Quality Monitoring
        if self.config.code_quality.enabled:
            def quality_monitor():
                from ..code_quality.quality_monitor import monitor_code_quality

                return monitor_code_quality(
                    project_root=self.config.project_root,
                    directories=self.config.code_quality.directories,
                    output_path=f"{self.config.data_directory}/quality_report.json"
                )

            self.register_task(
                "quality_monitor",
                quality_monitor,
                self.config.code_quality.check_interval_minutes,
                self.config.code_quality.enabled
            )

        # Security Monitoring
        if self.config.security.enabled:
            def security_monitor():
                from ..security.security_monitor import monitor_security

                return monitor_security(
                    project_root=self.config.project_root,
                    directories=self.config.code_quality.directories,  # Use same directories
                    output_path=f"{self.config.data_directory}/security_report.json"
                )

            self.register_task(
                "security_monitor",
                security_monitor,
                self.config.security.check_interval_minutes,
                self.config.security.enabled
            )

        # Testing Monitoring
        if self.config.testing.enabled:
            def testing_monitor():
                from ..testing.coverage_monitor import monitor_testing

                return monitor_testing(
                    project_root=self.config.project_root,
                    output_path=f"{self.config.data_directory}/test_health_report.json"
                )

            self.register_task(
                "testing_monitor",
                testing_monitor,
                self.config.testing.check_interval_minutes,
                self.config.testing.enabled
            )

        # Documentation Monitoring
        if self.config.documentation.enabled:
            def documentation_monitor():
                from ..documentation.doc_health_checker import monitor_documentation_health

                return monitor_documentation_health(
                    project_root=self.config.project_root,
                    docs_directories=self.config.documentation.docs_directories,
                    output_path=f"{self.config.data_directory}/documentation_health.json"
                )

            self.register_task(
                "documentation_monitor",
                documentation_monitor,
                self.config.documentation.check_interval_minutes,
                self.config.documentation.enabled
            )

    def start(self):
        """Start the monitoring scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return

        self.running = True
        logger.info("Starting monitoring scheduler...")

        # Setup default tasks
        self.setup_default_tasks()

        # Start scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info(f"Monitoring scheduler started with {len(self.tasks)} tasks")

    def stop(self):
        """Stop the monitoring scheduler"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping monitoring scheduler...")

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        # Clear scheduled jobs
        schedule.clear()

        logger.info("Monitoring scheduler stopped")

    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)  # Wait a bit before retrying

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all monitoring tasks"""
        status = {
            "scheduler_running": self.running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": len([t for t in self.tasks.values() if t["enabled"]]),
            "tasks": {}
        }

        for name, task in self.tasks.items():
            status["tasks"][name] = {
                "enabled": task["enabled"],
                "interval_minutes": task["interval_minutes"],
                "last_run": task["last_run"].isoformat() if task["last_run"] else None,
                "run_count": task["run_count"],
                "error_count": task["error_count"],
                "next_run": self._get_next_run_time(name)
            }

        return status

    def _get_next_run_time(self, task_name: str) -> str:
        """Get next scheduled run time for a task"""
        # This is a simplified version - in practice, you'd need to track this more precisely
        task = self.tasks.get(task_name)
        if not task or not task["last_run"]:
            return "Soon"

        next_run = task["last_run"] + timedelta(minutes=task["interval_minutes"])
        return next_run.isoformat()

    def run_task_now(self, task_name: str) -> bool:
        """Manually run a specific task now"""
        if task_name not in self.tasks:
            logger.error(f"Task '{task_name}' not found")
            return False

        logger.info(f"Manually running task: {task_name}")
        self._run_task(task_name)
        return True

    def enable_task(self, task_name: str) -> bool:
        """Enable a specific task"""
        if task_name not in self.tasks:
            return False

        self.tasks[task_name]["enabled"] = True
        logger.info(f"Enabled task: {task_name}")
        return True

    def disable_task(self, task_name: str) -> bool:
        """Disable a specific task"""
        if task_name not in self.tasks:
            return False

        self.tasks[task_name]["enabled"] = False
        logger.info(f"Disabled task: {task_name}")
        return True


def create_scheduler(config_path: str = "monitoring/config.json") -> MonitoringScheduler:
    """Create and configure a monitoring scheduler"""
    config = MonitoringConfig.from_file(config_path)
    return MonitoringScheduler(config)


if __name__ == "__main__":
    # Example usage
    scheduler = create_scheduler()

    try:
        scheduler.start()

        # Keep running
        while True:
            time.sleep(60)
            status = scheduler.get_task_status()
            logger.info(f"Scheduler status: {status['enabled_tasks']}/{status['total_tasks']} tasks enabled")

    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.stop()
