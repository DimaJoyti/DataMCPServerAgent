"""
Alert Manager

Immediate alerting system for critical issues with multiple notification channels.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from .config import MonitoringConfig

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: str  # "critical", "warning", "info"
    metric_type: str
    title: str
    message: str
    value: float
    threshold: float
    metadata: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts = {}
        self.alert_history = []
        self.notification_cooldowns = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Alert thresholds
        self.thresholds = {
            "cicd_health": {"critical": 70, "warning": 85},
            "code_quality": {"critical": 50, "warning": 70},
            "security_risk": {"critical": 80, "warning": 60},  # Higher is worse for security
            "test_health": {"critical": 60, "warning": 75},
            "documentation_health": {"critical": 60, "warning": 75}
        }

        # Cooldown periods (minutes)
        self.cooldown_periods = {
            "critical": 15,  # 15 minutes
            "warning": 60,   # 1 hour
            "info": 240      # 4 hours
        }

    async def start(self):
        """Start alert manager"""
        self.session = aiohttp.ClientSession()
        logger.info("🚨 Alert manager started")

    async def stop(self):
        """Stop alert manager"""
        if self.session:
            await self.session.close()
        logger.info("🛑 Alert manager stopped")

    async def check_metric_alert(self, metric_type: str, snapshot):
        """Check if metric should trigger an alert"""
        try:
            if metric_type not in self.thresholds:
                return

            thresholds = self.thresholds[metric_type]
            value = snapshot.value

            # Determine severity
            severity = None
            threshold = None

            if metric_type == "security_risk":
                # For security risk, higher values are worse
                if value >= thresholds["critical"]:
                    severity = "critical"
                    threshold = thresholds["critical"]
                elif value >= thresholds["warning"]:
                    severity = "warning"
                    threshold = thresholds["warning"]
            else:
                # For other metrics, lower values are worse
                if value <= thresholds["critical"]:
                    severity = "critical"
                    threshold = thresholds["critical"]
                elif value <= thresholds["warning"]:
                    severity = "warning"
                    threshold = thresholds["warning"]

            if severity:
                await self._create_alert(metric_type, severity, value, threshold, snapshot)
            else:
                # Check if we should resolve existing alerts
                await self._resolve_alerts(metric_type)

        except Exception as e:
            logger.error(f"❌ Alert check error: {e}")

    async def _create_alert(self, metric_type: str, severity: str, value: float,
                          threshold: float, snapshot):
        """Create and send alert"""
        try:
            alert_id = f"{metric_type}_{severity}_{int(datetime.now().timestamp())}"

            # Check cooldown
            cooldown_key = f"{metric_type}_{severity}"
            if self._is_in_cooldown(cooldown_key):
                return

            # Create alert
            alert = Alert(
                id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                metric_type=metric_type,
                title=self._generate_alert_title(metric_type, severity, value),
                message=self._generate_alert_message(metric_type, severity, value, threshold, snapshot),
                value=value,
                threshold=threshold,
                metadata=snapshot.metadata
            )

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Set cooldown
            self.notification_cooldowns[cooldown_key] = datetime.now()

            # Send notifications
            await self._send_notifications(alert)

            # Log alert
            logger.warning(f"🚨 {severity.upper()} ALERT: {alert.title}")

            # Save alert to file
            await self._save_alert(alert)

        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")

    async def _resolve_alerts(self, metric_type: str):
        """Resolve alerts for a metric type"""
        try:
            resolved_alerts = []

            for alert_id, alert in self.active_alerts.items():
                if alert.metric_type == metric_type and not alert.resolved:
                    alert.resolved = True
                    resolved_alerts.append(alert)
                    logger.info(f"✅ Resolved alert: {alert.title}")

            # Remove resolved alerts from active alerts
            for alert in resolved_alerts:
                if alert.id in self.active_alerts:
                    del self.active_alerts[alert.id]

            # Send resolution notifications if any alerts were resolved
            if resolved_alerts:
                await self._send_resolution_notifications(metric_type, resolved_alerts)

        except Exception as e:
            logger.error(f"❌ Failed to resolve alerts: {e}")

    def _is_in_cooldown(self, cooldown_key: str) -> bool:
        """Check if notification is in cooldown period"""
        if cooldown_key not in self.notification_cooldowns:
            return False

        last_notification = self.notification_cooldowns[cooldown_key]
        severity = cooldown_key.split('_')[-1]
        cooldown_minutes = self.cooldown_periods.get(severity, 60)

        return datetime.now() - last_notification < timedelta(minutes=cooldown_minutes)

    def _generate_alert_title(self, metric_type: str, severity: str, value: float) -> str:
        """Generate alert title"""
        metric_name = metric_type.replace('_', ' ').title()

        if severity == "critical":
            return f"🚨 CRITICAL: {metric_name} at {value:.1f}"
        elif severity == "warning":
            return f"⚠️ WARNING: {metric_name} at {value:.1f}"
        else:
            return f"ℹ️ INFO: {metric_name} at {value:.1f}"

    def _generate_alert_message(self, metric_type: str, severity: str, value: float,
                              threshold: float, snapshot) -> str:
        """Generate detailed alert message"""
        metric_name = metric_type.replace('_', ' ').title()

        message = "DataMCPServerAgent Alert\n\n"
        message += f"Metric: {metric_name}\n"
        message += f"Current Value: {value:.1f}\n"
        message += f"Threshold: {threshold:.1f}\n"
        message += f"Severity: {severity.upper()}\n"
        message += f"Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Add specific details based on metric type
        if metric_type == "security_risk":
            metadata = snapshot.metadata
            message += "Security Issues:\n"
            message += f"- Total Issues: {metadata.get('total_issues', 0)}\n"
            message += f"- Critical Issues: {metadata.get('critical_issues', 0)}\n"
            message += f"- High Issues: {metadata.get('high_issues', 0)}\n"

        elif metric_type == "code_quality":
            metadata = snapshot.metadata
            message += "Code Quality Issues:\n"
            message += f"- Total Issues: {metadata.get('total_issues', 0)}\n"
            message += f"- Critical Issues: {metadata.get('critical_issues', 0)}\n"

        elif metric_type == "test_health":
            metadata = snapshot.metadata
            message += "Test Health Details:\n"
            message += f"- Coverage: {metadata.get('coverage', 0):.1f}%\n"
            message += f"- Total Tests: {metadata.get('total_tests', 0)}\n"
            message += f"- Failed Tests: {metadata.get('failed_tests', 0)}\n"

        elif metric_type == "cicd_health":
            metadata = snapshot.metadata
            message += "CI/CD Details:\n"
            message += f"- Workflows: {metadata.get('workflows', 0)}\n"

        elif metric_type == "documentation_health":
            metadata = snapshot.metadata
            message += "Documentation Details:\n"
            message += f"- Total Documents: {metadata.get('total_documents', 0)}\n"
            message += f"- Outdated Documents: {metadata.get('outdated_documents', 0)}\n"
            message += f"- Broken Links: {metadata.get('broken_links', 0)}\n"

        message += "\nRecommended Actions:\n"
        message += self._get_recommendations(metric_type, severity, value)

        return message

    def _get_recommendations(self, metric_type: str, severity: str, value: float) -> str:
        """Get recommendations based on alert"""
        recommendations = {
            "security_risk": {
                "critical": "- Immediately review and fix critical security vulnerabilities\n- Run security scan manually\n- Consider blocking deployments until resolved",
                "warning": "- Review high-priority security issues\n- Schedule security fixes\n- Update dependencies"
            },
            "code_quality": {
                "critical": "- Run automated code fixes\n- Review critical code quality issues\n- Consider code review requirements",
                "warning": "- Schedule code quality improvements\n- Run linting tools\n- Update coding standards"
            },
            "test_health": {
                "critical": "- Fix failing tests immediately\n- Increase test coverage\n- Review test performance",
                "warning": "- Add more test cases\n- Optimize slow tests\n- Review test strategy"
            },
            "cicd_health": {
                "critical": "- Check CI/CD pipeline failures\n- Review build logs\n- Fix infrastructure issues",
                "warning": "- Optimize build performance\n- Review pipeline configuration\n- Monitor resource usage"
            },
            "documentation_health": {
                "critical": "- Update outdated documentation\n- Fix broken links\n- Add missing documentation",
                "warning": "- Review documentation quality\n- Update content\n- Improve structure"
            }
        }

        return recommendations.get(metric_type, {}).get(severity, "- Review the issue and take appropriate action")

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through all configured channels"""
        try:
            # Send Slack notification
            if self.config.notifications.slack_enabled:
                await self._send_slack_notification(alert)

            # Send Discord notification
            if self.config.notifications.discord_enabled:
                await self._send_discord_notification(alert)

            # Send email notification
            if self.config.notifications.email_enabled:
                await self._send_email_notification(alert)

        except Exception as e:
            logger.error(f"❌ Failed to send notifications: {e}")

    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            if not self.config.notifications.slack_webhook_url:
                return

            color = {"critical": "danger", "warning": "warning", "info": "good"}[alert.severity]

            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "footer": "DataMCPServerAgent Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            async with self.session.post(
                self.config.notifications.slack_webhook_url,
                json=payload
            ) as response:
                if response.status == 200:
                    logger.info("✅ Slack notification sent")
                else:
                    logger.error(f"❌ Slack notification failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Slack notification error: {e}")

    async def _send_discord_notification(self, alert: Alert):
        """Send Discord notification"""
        try:
            if not self.config.notifications.discord_webhook_url:
                return

            color = {"critical": 0xFF0000, "warning": 0xFFA500, "info": 0x00FF00}[alert.severity]

            payload = {
                "embeds": [{
                    "title": alert.title,
                    "description": alert.message,
                    "color": color,
                    "footer": {"text": "DataMCPServerAgent Monitoring"},
                    "timestamp": alert.timestamp.isoformat()
                }]
            }

            async with self.session.post(
                self.config.notifications.discord_webhook_url,
                json=payload
            ) as response:
                if response.status in [200, 204]:
                    logger.info("✅ Discord notification sent")
                else:
                    logger.error(f"❌ Discord notification failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Discord notification error: {e}")

    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            if not self.config.notifications.email_recipients:
                return

            # This would require SMTP configuration
            # For now, just log that email would be sent
            logger.info(f"📧 Email notification would be sent to {len(self.config.notifications.email_recipients)} recipients")

        except Exception as e:
            logger.error(f"❌ Email notification error: {e}")

    async def _send_resolution_notifications(self, metric_type: str, resolved_alerts: List[Alert]):
        """Send notifications when alerts are resolved"""
        try:
            message = f"✅ RESOLVED: {len(resolved_alerts)} alert(s) for {metric_type.replace('_', ' ').title()}"

            # Send to Slack
            if self.config.notifications.slack_enabled and self.config.notifications.slack_webhook_url:
                payload = {
                    "attachments": [{
                        "color": "good",
                        "title": "Alert Resolution",
                        "text": message,
                        "footer": "DataMCPServerAgent Monitoring"
                    }]
                }

                async with self.session.post(
                    self.config.notifications.slack_webhook_url,
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Slack resolution notification sent")

            logger.info(message)

        except Exception as e:
            logger.error(f"❌ Failed to send resolution notifications: {e}")

    async def _save_alert(self, alert: Alert):
        """Save alert to file"""
        try:
            alerts_file = Path(self.config.data_directory) / "alerts.json"

            # Load existing alerts
            alerts_data = []
            if alerts_file.exists():
                with open(alerts_file) as f:
                    alerts_data = json.load(f)

            # Add new alert
            alert_data = {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "metric_type": alert.metric_type,
                "title": alert.title,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "metadata": alert.metadata,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }

            alerts_data.append(alert_data)

            # Keep only last 1000 alerts
            alerts_data = alerts_data[-1000:]

            # Save to file
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Failed to save alert: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alert acknowledged: {alert_id}")
            return True
        return False

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(24)

        return {
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in active_alerts if a.severity == "warning"]),
            "alerts_last_24h": len(recent_alerts),
            "most_recent_alert": recent_alerts[-1].timestamp.isoformat() if recent_alerts else None
        }
