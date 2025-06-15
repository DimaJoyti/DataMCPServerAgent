"""
Real-Time Monitoring System for DataMCPServerAgent.
This module provides comprehensive real-time monitoring, alerting,
and performance analytics for the entire system.
"""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Optional dependencies with fallbacks
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from app.core.config import get_settings

try:
    from app.core.logging import get_logger
except ImportError:
    from app.core.simple_logging import get_logger

try:
    from app.monitoring.rl_analytics import get_metrics_collector
except ImportError:
    # Create a simple fallback metrics collector
    class SimpleMetricsCollector:
        def record_metric(self, name, value, tags=None):
            pass
        def record_event(self, name, data, level="info"):
            pass

    def get_metrics_collector():
        return SimpleMetricsCollector()

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringStatus(str, Enum):
    """Monitoring system status."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["load_average"] = list(result["load_average"])
        return result


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    active_sessions: int
    cache_hit_rate: float
    database_connections: int
    queue_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    metric_value: float
    threshold: float
    source: str
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


class MetricsCollector:
    """Collects system and application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.app_metrics_history = deque(maxlen=1440)

        # Network baseline
        self.network_baseline = psutil.net_io_counters()
        self.last_network_check = time.time()

        # Application metrics simulation
        self.request_counter = 0
        self.response_times = deque(maxlen=1000)
        self.error_counter = 0

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            System metrics
        """
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Network
            current_network = psutil.net_io_counters()
            current_time = time.time()
            time_delta = current_time - self.last_network_check

            bytes_sent = current_network.bytes_sent - self.network_baseline.bytes_sent
            bytes_recv = current_network.bytes_recv - self.network_baseline.bytes_recv

            # Update baseline
            self.network_baseline = current_network
            self.last_network_check = current_time

            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                load_avg = (0.0, 0.0, 0.0)  # Windows fallback

            # Active connections
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0

            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk_percent,
                network_bytes_sent=bytes_sent,
                network_bytes_recv=bytes_recv,
                active_connections=connections,
                load_average=load_avg,
            )

            self.system_metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                load_average=(0.0, 0.0, 0.0),
            )

    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect current application metrics.
        
        Returns:
            Application metrics
        """
        try:
            current_time = time.time()

            # Simulate application metrics
            self.request_counter += np.random.poisson(10)  # ~10 requests per collection

            # Generate realistic response times
            base_response_time = 200 + np.random.exponential(100)  # ms
            self.response_times.append(base_response_time)

            # Calculate response time percentiles
            if self.response_times:
                response_times_array = np.array(list(self.response_times))
                avg_response_time = np.mean(response_times_array)
                p95_response_time = np.percentile(response_times_array, 95)
                p99_response_time = np.percentile(response_times_array, 99)
            else:
                avg_response_time = p95_response_time = p99_response_time = 0.0

            # Error rate simulation
            if np.random.random() < 0.05:  # 5% chance of error
                self.error_counter += 1

            error_rate = (self.error_counter / max(self.request_counter, 1)) * 100

            # Other metrics simulation
            active_sessions = max(0, int(np.random.normal(100, 20)))
            cache_hit_rate = min(100, max(0, np.random.normal(85, 10)))
            database_connections = max(0, int(np.random.normal(20, 5)))
            queue_size = max(0, int(np.random.exponential(5)))

            metrics = ApplicationMetrics(
                timestamp=current_time,
                request_count=self.request_counter,
                response_time_avg=avg_response_time,
                response_time_p95=p95_response_time,
                response_time_p99=p99_response_time,
                error_rate=error_rate,
                active_sessions=active_sessions,
                cache_hit_rate=cache_hit_rate,
                database_connections=database_connections,
                queue_size=queue_size,
            )

            self.app_metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=time.time(),
                request_count=0,
                response_time_avg=0.0,
                response_time_p95=0.0,
                response_time_p99=0.0,
                error_rate=0.0,
                active_sessions=0,
                cache_hit_rate=0.0,
                database_connections=0,
                queue_size=0,
            )


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = []

    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default alert rules.
        
        Returns:
            Dictionary of alert rules
        """
        return {
            "high_cpu": {
                "metric": "cpu_percent",
                "threshold": 90.0,
                "severity": AlertSeverity.WARNING,
                "title": "High CPU Usage",
                "description": "CPU usage is above 90%",
            },
            "critical_cpu": {
                "metric": "cpu_percent",
                "threshold": 95.0,
                "severity": AlertSeverity.CRITICAL,
                "title": "Critical CPU Usage",
                "description": "CPU usage is above 95%",
            },
            "high_memory": {
                "metric": "memory_percent",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "title": "High Memory Usage",
                "description": "Memory usage is above 85%",
            },
            "high_response_time": {
                "metric": "response_time_p95",
                "threshold": 2000.0,  # 2 seconds
                "severity": AlertSeverity.WARNING,
                "title": "High Response Time",
                "description": "95th percentile response time is above 2 seconds",
            },
            "high_error_rate": {
                "metric": "error_rate",
                "threshold": 5.0,  # 5%
                "severity": AlertSeverity.ERROR,
                "title": "High Error Rate",
                "description": "Error rate is above 5%",
            },
            "disk_space_low": {
                "metric": "disk_usage_percent",
                "threshold": 90.0,
                "severity": AlertSeverity.WARNING,
                "title": "Low Disk Space",
                "description": "Disk usage is above 90%",
            },
        }

    def check_alerts(
        self,
        system_metrics: SystemMetrics,
        app_metrics: ApplicationMetrics
    ) -> List[Alert]:
        """Check for alert conditions.
        
        Args:
            system_metrics: Current system metrics
            app_metrics: Current application metrics
            
        Returns:
            List of new alerts
        """
        new_alerts = []
        current_time = time.time()

        # Combine metrics for checking
        all_metrics = {
            **system_metrics.to_dict(),
            **app_metrics.to_dict(),
        }

        for rule_id, rule in self.alert_rules.items():
            metric_name = rule["metric"]
            threshold = rule["threshold"]
            metric_value = all_metrics.get(metric_name, 0)

            # Check if threshold is exceeded
            if metric_value > threshold:
                alert_id = f"{rule_id}_{int(current_time)}"

                # Check if similar alert already exists and is not resolved
                existing_alert = self._find_existing_alert(rule_id)
                if existing_alert and not existing_alert.resolved:
                    continue  # Don't create duplicate alerts

                alert = Alert(
                    alert_id=alert_id,
                    timestamp=current_time,
                    severity=rule["severity"],
                    title=rule["title"],
                    description=f"{rule['description']} (Current: {metric_value:.2f})",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=threshold,
                    source="monitoring_system",
                )

                self.alerts[alert_id] = alert
                new_alerts.append(alert)

                logger.warning(f"🚨 Alert triggered: {alert.title}")

        return new_alerts

    def _find_existing_alert(self, rule_id: str) -> Optional[Alert]:
        """Find existing alert for a rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            Existing alert or None
        """
        for alert in self.alerts.values():
            if rule_id in alert.alert_id and not alert.resolved:
                return alert
        return None

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if acknowledged successfully
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alert acknowledged: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if resolved successfully
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            logger.info(f"✅ Alert resolved: {alert_id}")
            return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts.
        
        Returns:
            List of active alerts
        """
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary.
        
        Returns:
            Alert summary
        """
        active_alerts = self.get_active_alerts()

        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "recent_alerts": [
                alert.to_dict() for alert in
                sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)[:10]
            ],
        }


class RealTimeMonitor:
    """Main real-time monitoring system."""

    def __init__(self):
        """Initialize real-time monitor."""
        self.settings = get_settings()
        self.metrics_collector_internal = MetricsCollector()
        self.alert_manager = AlertManager()
        self.metrics_collector = get_metrics_collector()

        # WebSocket connections for real-time updates
        self.websocket_clients = set()

        # Monitoring state
        self.status = MonitoringStatus.STOPPED
        self.monitoring_task = None
        self.websocket_server = None

        # Performance data
        self.performance_history = deque(maxlen=1440)  # 24 hours

    async def start_monitoring(self):
        """Start the real-time monitoring system."""
        if self.status == MonitoringStatus.RUNNING:
            logger.warning("Monitoring already running")
            return

        self.status = MonitoringStatus.STARTING
        logger.info("🔍 Starting real-time monitoring system")

        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Start WebSocket server for real-time updates
            await self._start_websocket_server()

            self.status = MonitoringStatus.RUNNING
            logger.info("✅ Real-time monitoring started")

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.status = MonitoringStatus.ERROR

    async def stop_monitoring(self):
        """Stop the real-time monitoring system."""
        if self.status == MonitoringStatus.STOPPED:
            return

        logger.info("🛑 Stopping real-time monitoring system")

        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

        self.status = MonitoringStatus.STOPPED
        logger.info("✅ Real-time monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Collect metrics
                system_metrics = self.metrics_collector_internal.collect_system_metrics()
                app_metrics = self.metrics_collector_internal.collect_application_metrics()

                # Check for alerts
                new_alerts = self.alert_manager.check_alerts(system_metrics, app_metrics)

                # Record performance data
                performance_data = {
                    "timestamp": time.time(),
                    "system": system_metrics.to_dict(),
                    "application": app_metrics.to_dict(),
                    "alerts": len(self.alert_manager.get_active_alerts()),
                }

                self.performance_history.append(performance_data)

                # Send real-time updates to WebSocket clients
                await self._broadcast_updates(performance_data, new_alerts)

                # Record metrics in main collector
                self._record_metrics(system_metrics, app_metrics)

                await asyncio.sleep(10)  # Collect every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available. Install websockets package for real-time updates.")
            return

        try:
            async def handle_websocket(websocket, path):
                """Handle WebSocket connection."""
                self.websocket_clients.add(websocket)
                logger.info(f"📡 WebSocket client connected: {websocket.remote_address}")

                try:
                    # Send initial data
                    initial_data = {
                        "type": "initial",
                        "data": self.get_monitoring_dashboard(),
                    }
                    await websocket.send(json.dumps(initial_data))

                    # Keep connection alive
                    await websocket.wait_closed()

                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.discard(websocket)
                    logger.info("📡 WebSocket client disconnected")

            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                handle_websocket,
                "localhost",
                8765,
            )

            logger.info("📡 WebSocket server started on ws://localhost:8765")

        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")

    async def _broadcast_updates(
        self,
        performance_data: Dict[str, Any],
        new_alerts: List[Alert]
    ):
        """Broadcast updates to WebSocket clients.
        
        Args:
            performance_data: Current performance data
            new_alerts: New alerts
        """
        if not self.websocket_clients:
            return

        update_data = {
            "type": "update",
            "timestamp": time.time(),
            "performance": performance_data,
            "new_alerts": [alert.to_dict() for alert in new_alerts],
            "alert_summary": self.alert_manager.get_alert_summary(),
        }

        # Send to all connected clients
        disconnected_clients = set()

        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(update_data))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients

    def _record_metrics(
        self,
        system_metrics: SystemMetrics,
        app_metrics: ApplicationMetrics
    ):
        """Record metrics in main collector.
        
        Args:
            system_metrics: System metrics
            app_metrics: Application metrics
        """
        # Record system metrics
        self.metrics_collector.record_metric(
            "system_cpu_percent", system_metrics.cpu_percent
        )
        self.metrics_collector.record_metric(
            "system_memory_percent", system_metrics.memory_percent
        )
        self.metrics_collector.record_metric(
            "system_disk_usage_percent", system_metrics.disk_usage_percent
        )

        # Record application metrics
        self.metrics_collector.record_metric(
            "app_response_time_avg", app_metrics.response_time_avg
        )
        self.metrics_collector.record_metric(
            "app_response_time_p95", app_metrics.response_time_p95
        )
        self.metrics_collector.record_metric(
            "app_error_rate", app_metrics.error_rate
        )

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get complete monitoring dashboard data.
        
        Returns:
            Dashboard data
        """
        # Get latest metrics
        latest_system = (
            self.metrics_collector_internal.system_metrics_history[-1]
            if self.metrics_collector_internal.system_metrics_history
            else None
        )

        latest_app = (
            self.metrics_collector_internal.app_metrics_history[-1]
            if self.metrics_collector_internal.app_metrics_history
            else None
        )

        # Calculate trends
        trends = self._calculate_trends()

        return {
            "status": self.status.value,
            "timestamp": time.time(),
            "current_metrics": {
                "system": latest_system.to_dict() if latest_system else {},
                "application": latest_app.to_dict() if latest_app else {},
            },
            "trends": trends,
            "alerts": self.alert_manager.get_alert_summary(),
            "performance_history": [
                data for data in list(self.performance_history)[-60:]  # Last hour
            ],
            "websocket_clients": len(self.websocket_clients),
        }

    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate metric trends.
        
        Returns:
            Dictionary of trends
        """
        trends = {}

        if len(self.performance_history) < 2:
            return trends

        # Get recent data points
        recent_data = list(self.performance_history)[-10:]  # Last 10 data points

        if len(recent_data) < 2:
            return trends

        # Calculate trends for key metrics
        metrics_to_trend = [
            ("system.cpu_percent", "CPU Usage"),
            ("system.memory_percent", "Memory Usage"),
            ("application.response_time_avg", "Response Time"),
            ("application.error_rate", "Error Rate"),
        ]

        for metric_path, display_name in metrics_to_trend:
            values = []

            for data in recent_data:
                # Navigate nested dictionary
                current = data
                for key in metric_path.split('.'):
                    current = current.get(key, 0)
                    if not isinstance(current, dict):
                        break

                if isinstance(current, (int, float)):
                    values.append(current)

            if len(values) >= 2:
                # Simple trend calculation
                first_half = np.mean(values[:len(values)//2])
                second_half = np.mean(values[len(values)//2:])

                if second_half > first_half * 1.1:
                    trends[display_name] = "increasing"
                elif second_half < first_half * 0.9:
                    trends[display_name] = "decreasing"
                else:
                    trends[display_name] = "stable"

        return trends


# Global real-time monitor instance
_real_time_monitor: Optional[RealTimeMonitor] = None


def get_real_time_monitor() -> RealTimeMonitor:
    """Get global real-time monitor."""
    global _real_time_monitor
    if _real_time_monitor is None:
        _real_time_monitor = RealTimeMonitor()
    return _real_time_monitor
