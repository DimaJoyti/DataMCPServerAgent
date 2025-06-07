"""
Monitor Manager

Central management for all monitoring components.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .config import MonitoringConfig
from .scheduler import MonitoringScheduler

logger = logging.getLogger(__name__)


class MonitorManager:
    """Central manager for all monitoring activities"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.scheduler = MonitoringScheduler(config)
        self.dashboard = None
        self.running = False
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"{self.config.data_directory}/monitoring.log")
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    async def start(self):
        """Start all monitoring components"""
        if self.running:
            logger.warning("Monitor manager is already running")
            return
        
        self.running = True
        logger.info("Starting DataMCPServerAgent monitoring system...")
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Create data directory
        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Start scheduler
        self.scheduler.start()
        
        # Start dashboard if enabled
        if self.config.dashboard.enabled:
            await self.start_dashboard()
        
        # Run initial monitoring sweep
        await self.run_initial_monitoring()
        
        logger.info("Monitoring system started successfully")
    
    async def start_dashboard(self):
        """Start the web dashboard"""
        try:
            from ..dashboard.main_dashboard import MonitoringDashboard
            
            self.dashboard = MonitoringDashboard(
                data_directory=self.config.data_directory,
                host=self.config.dashboard.host,
                port=self.config.dashboard.port
            )
            
            # Start dashboard in background
            import threading
            dashboard_thread = threading.Thread(
                target=self.dashboard.run,
                daemon=True
            )
            dashboard_thread.start()
            
            logger.info(f"Dashboard started at http://{self.config.dashboard.host}:{self.config.dashboard.port}")
            
        except ImportError:
            logger.warning("Dashboard dependencies not available. Install with: pip install fastapi uvicorn jinja2")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    async def run_initial_monitoring(self):
        """Run initial monitoring sweep to populate data"""
        logger.info("Running initial monitoring sweep...")
        
        tasks = []
        
        # Run each monitoring component once
        if self.config.code_quality.enabled:
            tasks.append(self._run_code_quality_check())
        
        if self.config.security.enabled:
            tasks.append(self._run_security_check())
        
        if self.config.testing.enabled:
            tasks.append(self._run_testing_check())
        
        if self.config.documentation.enabled:
            tasks.append(self._run_documentation_check())
        
        # Run CI/CD check if GitHub token is available
        if self.config.cicd.enabled and self.config.github.token:
            tasks.append(self._run_cicd_check())
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Initial monitoring complete: {success_count}/{len(tasks)} checks successful")
        
        # Generate summary report
        await self.generate_summary_report()
    
    async def _run_code_quality_check(self):
        """Run code quality check"""
        try:
            from ..code_quality.quality_monitor import monitor_code_quality
            
            logger.info("Running code quality check...")
            result = monitor_code_quality(
                project_root=self.config.project_root,
                directories=self.config.code_quality.directories,
                output_path=f"{self.config.data_directory}/quality_report.json"
            )
            logger.info(f"Code quality check complete. Score: {result.overall_score}/100")
            return result
            
        except Exception as e:
            logger.error(f"Code quality check failed: {e}")
            raise
    
    async def _run_security_check(self):
        """Run security check"""
        try:
            from ..security.security_monitor import monitor_security
            
            logger.info("Running security check...")
            result = monitor_security(
                project_root=self.config.project_root,
                directories=self.config.code_quality.directories,
                output_path=f"{self.config.data_directory}/security_report.json"
            )
            logger.info(f"Security check complete. Risk score: {result.overall_risk_score}/100")
            return result
            
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            raise
    
    async def _run_testing_check(self):
        """Run testing check"""
        try:
            from ..testing.coverage_monitor import monitor_testing
            
            logger.info("Running testing check...")
            result = monitor_testing(
                project_root=self.config.project_root,
                output_path=f"{self.config.data_directory}/test_health_report.json"
            )
            logger.info(f"Testing check complete. Health score: {result.health_score}/100")
            return result
            
        except Exception as e:
            logger.error(f"Testing check failed: {e}")
            raise
    
    async def _run_documentation_check(self):
        """Run documentation check"""
        try:
            from ..documentation.doc_health_checker import monitor_documentation_health
            
            logger.info("Running documentation check...")
            result = monitor_documentation_health(
                project_root=self.config.project_root,
                docs_directories=self.config.documentation.docs_directories,
                output_path=f"{self.config.data_directory}/documentation_health.json"
            )
            logger.info(f"Documentation check complete. Health score: {result.overall_score:.1f}/100")
            return result
            
        except Exception as e:
            logger.error(f"Documentation check failed: {e}")
            raise
    
    async def _run_cicd_check(self):
        """Run CI/CD check"""
        try:
            from ..ci_cd.performance_monitor import monitor_cicd_performance
            
            logger.info("Running CI/CD check...")
            result = await monitor_cicd_performance(
                github_token=self.config.github.token,
                owner=self.config.github.owner,
                repo=self.config.github.repo,
                output_path=f"{self.config.data_directory}/cicd_metrics.json"
            )
            logger.info("CI/CD check complete")
            return result
            
        except Exception as e:
            logger.error(f"CI/CD check failed: {e}")
            raise
    
    async def generate_summary_report(self):
        """Generate overall monitoring summary report"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "system_health": {},
                "recommendations": [],
                "alerts": []
            }
            
            data_dir = Path(self.config.data_directory)
            
            # Load all monitoring reports
            reports = {}
            report_files = {
                "quality": "quality_report.json",
                "security": "security_report.json", 
                "testing": "test_health_report.json",
                "documentation": "documentation_health.json",
                "cicd": "cicd_metrics.json"
            }
            
            for report_type, filename in report_files.items():
                file_path = data_dir / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        reports[report_type] = json.load(f)
            
            # Calculate system health scores
            if "quality" in reports:
                summary["system_health"]["code_quality"] = reports["quality"].get("overall_score", 0)
            
            if "security" in reports:
                summary["system_health"]["security_risk"] = reports["security"].get("overall_risk_score", 0)
            
            if "testing" in reports:
                summary["system_health"]["test_health"] = reports["testing"].get("health_score", 0)
            
            if "documentation" in reports:
                summary["system_health"]["documentation_health"] = reports["documentation"]["scores"]["overall_score"]
            
            if "cicd" in reports:
                # Calculate average CI/CD success rate
                metrics = reports["cicd"].get("metrics", {})
                if metrics:
                    success_rates = [m.get("success_rate", 0) for m in metrics.values()]
                    summary["system_health"]["cicd_health"] = sum(success_rates) / len(success_rates)
            
            # Collect all recommendations
            for report_type, report_data in reports.items():
                recommendations = report_data.get("recommendations", [])
                for rec in recommendations[:3]:  # Limit to top 3 per report
                    summary["recommendations"].append(f"[{report_type.title()}] {rec}")
            
            # Generate alerts for critical issues
            if summary["system_health"].get("security_risk", 0) > 70:
                summary["alerts"].append("🚨 HIGH SECURITY RISK: Immediate attention required")
            
            if summary["system_health"].get("code_quality", 100) < 50:
                summary["alerts"].append("⚠️ LOW CODE QUALITY: Code quality below acceptable threshold")
            
            if summary["system_health"].get("test_health", 100) < 60:
                summary["alerts"].append("🧪 POOR TEST HEALTH: Test coverage or performance issues")
            
            # Save summary report
            summary_file = data_dir / "monitoring_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Monitoring summary report generated")
            
            # Log key metrics
            health_scores = summary["system_health"]
            if health_scores:
                logger.info("System Health Summary:")
                for metric, score in health_scores.items():
                    logger.info(f"  {metric}: {score:.1f}")
            
            if summary["alerts"]:
                logger.warning("Active Alerts:")
                for alert in summary["alerts"]:
                    logger.warning(f"  {alert}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
    
    def stop(self):
        """Stop all monitoring components"""
        if not self.running:
            return
        
        logger.info("Stopping monitoring system...")
        self.running = False
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Stop dashboard
        if self.dashboard:
            # Dashboard runs in a separate thread, it will stop when the main process exits
            pass
        
        logger.info("Monitoring system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall monitoring system status"""
        return {
            "running": self.running,
            "scheduler": self.scheduler.get_task_status(),
            "dashboard_enabled": self.config.dashboard.enabled,
            "data_directory": self.config.data_directory,
            "last_summary": self._get_last_summary()
        }
    
    def _get_last_summary(self) -> Optional[Dict[str, Any]]:
        """Get the last monitoring summary"""
        try:
            summary_file = Path(self.config.data_directory) / "monitoring_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load last summary: {e}")
        
        return None


async def main():
    """Main entry point for monitoring system"""
    # Load configuration
    config = MonitoringConfig.from_env()
    
    # Create and start monitor manager
    manager = MonitorManager(config)
    
    try:
        await manager.start()
        
        # Keep running
        while manager.running:
            await asyncio.sleep(60)
            
            # Periodic status check
            status = manager.get_status()
            logger.debug(f"System status: {status['scheduler']['enabled_tasks']} tasks running")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
