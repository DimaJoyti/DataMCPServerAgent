"""
Background Metrics Tracker

Continuously tracks all metrics in the background with intelligent scheduling.
"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from .alert_manager import AlertManager
from .config import MonitoringConfig
from .trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric snapshot"""
    timestamp: datetime
    metric_type: str
    value: float
    metadata: Dict[str, Any]
    status: str  # "good", "warning", "critical"


@dataclass
class SystemSnapshot:
    """Complete system snapshot"""
    timestamp: datetime
    metrics: Dict[str, MetricSnapshot]
    overall_health: float
    alerts: List[str]
    recommendations: List[str]


class BackgroundTracker:
    """Tracks all metrics continuously in the background"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.running = False
        self.tracker_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Components
        self.alert_manager = AlertManager(config)
        self.trend_analyzer = TrendAnalyzer(config.data_directory)

        # Data storage
        self.data_dir = Path(config.data_directory)
        self.metrics_history = []
        self.last_snapshots = {}

        # Tracking intervals (in seconds)
        self.intervals = {
            "cicd": config.cicd.check_interval_minutes * 60,
            "quality": config.code_quality.check_interval_minutes * 60,
            "security": config.security.check_interval_minutes * 60,
            "testing": config.testing.check_interval_minutes * 60,
            "documentation": config.documentation.check_interval_minutes * 60,
            "system_health": 300,  # Every 5 minutes
            "trend_analysis": 1800,  # Every 30 minutes
        }

        # Last run times
        self.last_runs = dict.fromkeys(self.intervals.keys(), 0)

        # Performance tracking
        self.execution_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_execution_time": 0.0
        }

    async def start(self):
        """Start background tracking"""
        if self.running:
            logger.warning("Background tracker already running")
            return

        self.running = True
        logger.info("🚀 Starting background metrics tracking...")

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Start alert manager
        await self.alert_manager.start()

        # Start tracking loop
        self.tracker_thread = threading.Thread(target=self._run_tracking_loop, daemon=True)
        self.tracker_thread.start()

        logger.info("✅ Background tracking started successfully")

    def stop(self):
        """Stop background tracking"""
        if not self.running:
            return

        logger.info("🛑 Stopping background tracking...")
        self.running = False

        if self.tracker_thread:
            self.tracker_thread.join(timeout=10)

        self.executor.shutdown(wait=True)
        logger.info("✅ Background tracking stopped")

    def _run_tracking_loop(self):
        """Main tracking loop"""
        logger.info("📊 Background tracking loop started")

        while self.running:
            try:
                current_time = time.time()

                # Check which tasks need to run
                tasks_to_run = []

                for task_name, interval in self.intervals.items():
                    if current_time - self.last_runs[task_name] >= interval:
                        tasks_to_run.append(task_name)
                        self.last_runs[task_name] = current_time

                # Execute tasks
                if tasks_to_run:
                    asyncio.run(self._execute_tasks(tasks_to_run))

                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Tracking loop error: {e}")
                time.sleep(60)  # Wait longer on error

    async def _execute_tasks(self, tasks: List[str]):
        """Execute monitoring tasks"""
        logger.info(f"🔄 Executing tasks: {', '.join(tasks)}")
        start_time = time.time()

        try:
            # Run tasks concurrently
            task_futures = []

            for task_name in tasks:
                if task_name == "cicd" and self.config.cicd.enabled:
                    task_futures.append(self._track_cicd_metrics())
                elif task_name == "quality" and self.config.code_quality.enabled:
                    task_futures.append(self._track_quality_metrics())
                elif task_name == "security" and self.config.security.enabled:
                    task_futures.append(self._track_security_metrics())
                elif task_name == "testing" and self.config.testing.enabled:
                    task_futures.append(self._track_testing_metrics())
                elif task_name == "documentation" and self.config.documentation.enabled:
                    task_futures.append(self._track_documentation_metrics())
                elif task_name == "system_health":
                    task_futures.append(self._track_system_health())
                elif task_name == "trend_analysis":
                    task_futures.append(self._perform_trend_analysis())

            # Wait for all tasks to complete
            if task_futures:
                results = await asyncio.gather(*task_futures, return_exceptions=True)

                # Process results
                successful_tasks = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Task {tasks[i]} failed: {result}")
                    else:
                        successful_tasks += 1

                # Update execution stats
                execution_time = time.time() - start_time
                self.execution_stats["total_runs"] += 1
                self.execution_stats["successful_runs"] += successful_tasks
                self.execution_stats["failed_runs"] += len(tasks) - successful_tasks

                # Update average execution time
                current_avg = self.execution_stats["average_execution_time"]
                total_runs = self.execution_stats["total_runs"]
                self.execution_stats["average_execution_time"] = (
                    (current_avg * (total_runs - 1) + execution_time) / total_runs
                )

                logger.info(f"✅ Tasks completed: {successful_tasks}/{len(tasks)} successful in {execution_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Task execution error: {e}")

    async def _track_cicd_metrics(self):
        """Track CI/CD metrics"""
        try:
            import os
            github_token = self.config.github.token or os.getenv("GITHUB_TOKEN")

            if not github_token:
                logger.warning("⚠️ GitHub token not available for CI/CD tracking")
                return

            from ..ci_cd.performance_monitor import monitor_cicd_performance

            metrics = await monitor_cicd_performance(
                github_token=github_token,
                owner=self.config.github.owner,
                repo=self.config.github.repo,
                output_path=str(self.data_dir / "cicd_metrics.json")
            )

            # Calculate overall CI/CD health
            if metrics:
                success_rates = [m.success_rate for m in metrics.values()]
                avg_success_rate = sum(success_rates) / len(success_rates)

                snapshot = MetricSnapshot(
                    timestamp=datetime.now(),
                    metric_type="cicd_health",
                    value=avg_success_rate,
                    metadata={"workflows": len(metrics), "details": "cicd_metrics.json"},
                    status="good" if avg_success_rate >= 90 else "warning" if avg_success_rate >= 80 else "critical"
                )

                self.last_snapshots["cicd"] = snapshot
                await self._check_alerts("cicd", snapshot)

        except Exception as e:
            logger.error(f"❌ CI/CD tracking error: {e}")

    async def _track_quality_metrics(self):
        """Track code quality metrics"""
        try:
            from ..code_quality.quality_monitor import monitor_code_quality

            report = monitor_code_quality(
                project_root=self.config.project_root,
                directories=self.config.code_quality.directories,
                output_path=str(self.data_dir / "quality_report.json")
            )

            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metric_type="code_quality",
                value=report.overall_score,
                metadata={
                    "total_issues": report.total_issues,
                    "critical_issues": report.critical_issues,
                    "details": "quality_report.json"
                },
                status="good" if report.overall_score >= 80 else "warning" if report.overall_score >= 60 else "critical"
            )

            self.last_snapshots["quality"] = snapshot
            await self._check_alerts("quality", snapshot)

        except Exception as e:
            logger.error(f"❌ Quality tracking error: {e}")

    async def _track_security_metrics(self):
        """Track security metrics"""
        try:
            from ..security.security_monitor import monitor_security

            report = monitor_security(
                project_root=self.config.project_root,
                directories=self.config.code_quality.directories,
                output_path=str(self.data_dir / "security_report.json")
            )

            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metric_type="security_risk",
                value=report.overall_risk_score,
                metadata={
                    "total_issues": report.total_issues,
                    "critical_issues": report.critical_issues,
                    "high_issues": report.high_issues,
                    "details": "security_report.json"
                },
                status="critical" if report.critical_issues > 0 else "warning" if report.high_issues > 5 else "good"
            )

            self.last_snapshots["security"] = snapshot
            await self._check_alerts("security", snapshot)

        except Exception as e:
            logger.error(f"❌ Security tracking error: {e}")

    async def _track_testing_metrics(self):
        """Track testing metrics"""
        try:
            from ..testing.coverage_monitor import monitor_testing

            report = monitor_testing(
                project_root=self.config.project_root,
                output_path=str(self.data_dir / "test_health_report.json")
            )

            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metric_type="test_health",
                value=report.health_score,
                metadata={
                    "coverage": report.coverage_metrics.overall_coverage,
                    "total_tests": report.performance_metrics.total_tests,
                    "failed_tests": report.performance_metrics.failed_tests,
                    "details": "test_health_report.json"
                },
                status="good" if report.health_score >= 80 else "warning" if report.health_score >= 60 else "critical"
            )

            self.last_snapshots["testing"] = snapshot
            await self._check_alerts("testing", snapshot)

        except Exception as e:
            logger.error(f"❌ Testing tracking error: {e}")

    async def _track_documentation_metrics(self):
        """Track documentation metrics"""
        try:
            from ..documentation.doc_health_checker import monitor_documentation_health

            report = monitor_documentation_health(
                project_root=self.config.project_root,
                docs_directories=self.config.documentation.docs_directories,
                output_path=str(self.data_dir / "documentation_health.json")
            )

            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metric_type="documentation_health",
                value=report.overall_score,
                metadata={
                    "total_documents": report.total_documents,
                    "outdated_documents": report.outdated_documents,
                    "broken_links": report.total_broken_links,
                    "details": "documentation_health.json"
                },
                status="good" if report.overall_score >= 80 else "warning" if report.overall_score >= 60 else "critical"
            )

            self.last_snapshots["documentation"] = snapshot
            await self._check_alerts("documentation", snapshot)

        except Exception as e:
            logger.error(f"❌ Documentation tracking error: {e}")

    async def _track_system_health(self):
        """Track overall system health"""
        try:
            # Calculate overall system health from all metrics
            if not self.last_snapshots:
                return

            health_scores = []
            alerts = []
            recommendations = []

            for metric_type, snapshot in self.last_snapshots.items():
                if metric_type == "security_risk":
                    # For security, lower is better, so invert the score
                    health_scores.append(100 - snapshot.value)
                else:
                    health_scores.append(snapshot.value)

                # Collect alerts
                if snapshot.status == "critical":
                    alerts.append(f"🚨 CRITICAL: {metric_type.replace('_', ' ').title()} needs immediate attention")
                elif snapshot.status == "warning":
                    alerts.append(f"⚠️ WARNING: {metric_type.replace('_', ' ').title()} below optimal")

            overall_health = sum(health_scores) / len(health_scores) if health_scores else 0

            # Generate system snapshot
            system_snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                metrics=self.last_snapshots.copy(),
                overall_health=overall_health,
                alerts=alerts,
                recommendations=recommendations
            )

            # Save system snapshot
            await self._save_system_snapshot(system_snapshot)

            # Add to history
            self.metrics_history.append(system_snapshot)

            # Keep only last 1000 snapshots
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            logger.info(f"📊 System health: {overall_health:.1f}/100 ({len(alerts)} alerts)")

        except Exception as e:
            logger.error(f"❌ System health tracking error: {e}")

    async def _perform_trend_analysis(self):
        """Perform trend analysis"""
        try:
            if len(self.metrics_history) < 2:
                return

            trends = await self.trend_analyzer.analyze_trends(self.metrics_history)

            # Save trend analysis
            trend_file = self.data_dir / "trend_analysis.json"
            with open(trend_file, 'w') as f:
                json.dump(trends, f, indent=2, default=str)

            logger.info("📈 Trend analysis completed")

        except Exception as e:
            logger.error(f"❌ Trend analysis error: {e}")

    async def _check_alerts(self, metric_type: str, snapshot: MetricSnapshot):
        """Check if alerts should be triggered"""
        try:
            await self.alert_manager.check_metric_alert(metric_type, snapshot)
        except Exception as e:
            logger.error(f"❌ Alert check error for {metric_type}: {e}")

    async def _save_system_snapshot(self, snapshot: SystemSnapshot):
        """Save system snapshot to file"""
        try:
            snapshot_file = self.data_dir / "system_snapshot.json"

            # Convert to JSON-serializable format
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_health": snapshot.overall_health,
                "alerts": snapshot.alerts,
                "recommendations": snapshot.recommendations,
                "metrics": {}
            }

            for metric_type, metric_snapshot in snapshot.metrics.items():
                snapshot_data["metrics"][metric_type] = {
                    "timestamp": metric_snapshot.timestamp.isoformat(),
                    "value": metric_snapshot.value,
                    "status": metric_snapshot.status,
                    "metadata": metric_snapshot.metadata
                }

            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Failed to save system snapshot: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "metrics": {k: asdict(v) for k, v in self.last_snapshots.items()},
            "execution_stats": self.execution_stats,
            "history_length": len(self.metrics_history)
        }

    def get_metrics_history(self, hours: int = 24) -> List[SystemSnapshot]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self.metrics_history
            if snapshot.timestamp >= cutoff_time
        ]


if __name__ == "__main__":
    # Example usage
    import asyncio

    from .config import MonitoringConfig

    async def main():
        config = MonitoringConfig.from_env()
        tracker = BackgroundTracker(config)

        try:
            await tracker.start()

            # Keep running
            while True:
                await asyncio.sleep(60)
                metrics = tracker.get_current_metrics()
                print(f"Current metrics: {len(metrics['metrics'])} tracked")

        except KeyboardInterrupt:
            print("Stopping tracker...")
        finally:
            tracker.stop()

    asyncio.run(main())
