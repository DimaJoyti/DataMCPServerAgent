"""
Monitoring Configuration Management
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    token: Optional[str] = None
    owner: str = "DimaJoyti"
    repo: str = "DataMCPServerAgent"
    api_base_url: str = "https://api.github.com"


@dataclass
class NotificationConfig:
    """Notification settings"""
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None


@dataclass
class CICDMonitorConfig:
    """CI/CD monitoring configuration"""
    enabled: bool = True
    check_interval_minutes: int = 30
    track_workflows: List[str] = field(default_factory=lambda: [
        "ci.yml", "security.yml", "enhanced-testing.yml", "docs.yml", "deploy.yml"
    ])
    performance_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "max_build_time_minutes": 30,
        "max_queue_time_minutes": 5,
        "min_success_rate_percent": 90
    })


@dataclass
class CodeQualityConfig:
    """Code quality monitoring configuration"""
    enabled: bool = True
    check_interval_minutes: int = 60
    auto_fix_enabled: bool = True
    directories: List[str] = field(default_factory=lambda: [
        "app", "src", "examples", "scripts", "tests"
    ])
    tools: Dict[str, bool] = field(default_factory=lambda: {
        "black": True,
        "isort": True,
        "ruff": True,
        "mypy": True,
        "bandit": True
    })


@dataclass
class SecurityConfig:
    """Security monitoring configuration"""
    enabled: bool = True
    check_interval_minutes: int = 120
    severity_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "critical": 0,  # Alert immediately
        "high": 5,      # Alert if more than 5
        "medium": 20,   # Alert if more than 20
        "low": 50       # Alert if more than 50
    })
    tools: List[str] = field(default_factory=lambda: [
        "bandit", "safety", "semgrep"
    ])


@dataclass
class TestingConfig:
    """Testing metrics configuration"""
    enabled: bool = True
    check_interval_minutes: int = 60
    coverage_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "minimum_coverage": 70.0,
        "target_coverage": 85.0,
        "excellent_coverage": 95.0
    })
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_test_duration_seconds": 300.0,
        "max_average_test_time_seconds": 5.0
    })


@dataclass
class DocumentationConfig:
    """Documentation monitoring configuration"""
    enabled: bool = True
    check_interval_minutes: int = 240  # 4 hours
    docs_directories: List[str] = field(default_factory=lambda: [
        "docs", "README.md"
    ])
    check_links: bool = True
    check_freshness_days: int = 30
    required_sections: List[str] = field(default_factory=lambda: [
        "Installation", "Usage", "API", "Contributing"
    ])


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8080
    refresh_interval_seconds: int = 30
    theme: str = "dark"
    show_historical_data: bool = True
    data_retention_days: int = 90


@dataclass
class MonitoringConfig:
    """Main monitoring configuration"""

    # Core settings
    project_root: str = "."
    data_directory: str = "monitoring/data"
    log_level: str = "INFO"

    # Component configurations
    github: GitHubConfig = field(default_factory=GitHubConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    cicd: CICDMonitorConfig = field(default_factory=CICDMonitorConfig)
    code_quality: CodeQualityConfig = field(default_factory=CodeQualityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_file(cls, config_path: str) -> "MonitoringConfig":
        """Load configuration from JSON file"""
        path = Path(config_path)
        if not path.exists():
            # Create default config file
            config = cls()
            config.save_to_file(config_path)
            return config

        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Load configuration from environment variables"""
        config = cls()

        # GitHub configuration
        if os.getenv("GITHUB_TOKEN"):
            config.github.token = os.getenv("GITHUB_TOKEN")
        if os.getenv("GITHUB_OWNER"):
            config.github.owner = os.getenv("GITHUB_OWNER")
        if os.getenv("GITHUB_REPO"):
            config.github.repo = os.getenv("GITHUB_REPO")

        # Notification configuration
        if os.getenv("SLACK_WEBHOOK_URL"):
            config.notifications.slack_enabled = True
            config.notifications.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if os.getenv("DISCORD_WEBHOOK_URL"):
            config.notifications.discord_enabled = True
            config.notifications.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # Dashboard configuration
        if os.getenv("DASHBOARD_HOST"):
            config.dashboard.host = os.getenv("DASHBOARD_HOST")
        if os.getenv("DASHBOARD_PORT"):
            config.dashboard.port = int(os.getenv("DASHBOARD_PORT"))

        return config

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        config_dict = self._to_dict()

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            else:
                return obj

        return convert_dataclass(self)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check GitHub token if CI/CD monitoring is enabled
        if self.cicd.enabled and not self.github.token:
            issues.append("GitHub token required for CI/CD monitoring")

        # Check notification settings
        if self.notifications.slack_enabled and not self.notifications.slack_webhook_url:
            issues.append("Slack webhook URL required when Slack notifications enabled")

        if self.notifications.discord_enabled and not self.notifications.discord_webhook_url:
            issues.append("Discord webhook URL required when Discord notifications enabled")

        # Check data directory
        data_dir = Path(self.data_directory)
        if not data_dir.exists():
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create data directory: {e}")

        return issues
