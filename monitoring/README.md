# DataMCPServerAgent Monitoring System

Comprehensive monitoring and automation system for tracking CI/CD performance, code quality, security, testing metrics, and documentation health.

## 🚀 Features

### 📊 **CI/CD Performance Monitoring**
- GitHub Actions workflow tracking
- Build time and success rate analysis
- Performance trends and alerts
- Queue time monitoring

### 🔍 **Code Quality Monitoring**
- Automated code quality checks (Black, isort, Ruff, MyPy)
- Quality score calculation and trending
- Issue categorization and reporting
- Automated fixing capabilities

### 🔒 **Security Monitoring**
- Multi-tool security scanning (Bandit, Safety, Semgrep)
- Vulnerability tracking and risk assessment
- Severity-based alerting
- Dependency security monitoring

### 🧪 **Testing Metrics**
- Test coverage analysis
- Performance benchmarking
- Test health scoring
- Failure tracking and analysis

### 📚 **Documentation Health**
- Documentation completeness checking
- Link validation (internal and external)
- Freshness and quality scoring
- Structure analysis

### 🌐 **Web Dashboard**
- Real-time metrics visualization
- Interactive charts and graphs
- WebSocket-based live updates
- Mobile-responsive design

### ⏰ **Automated Scheduling**
- Configurable monitoring intervals
- Background task execution
- Error handling and retry logic
- Manual task triggering

## 📦 Installation

### Prerequisites

```bash
# Core dependencies
pip install aiohttp requests schedule

# Optional dependencies for full functionality
pip install fastapi uvicorn jinja2 websockets
pip install pytest pytest-cov pytest-benchmark
pip install black isort ruff mypy bandit safety semgrep
```

### Quick Setup

```bash
# 1. Clone or navigate to your DataMCPServerAgent project
cd DataMCPServerAgent

# 2. Set up environment variables
export GITHUB_TOKEN="your_github_token_here"

# 3. Start monitoring system
python scripts/start_monitoring.py
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for CI/CD monitoring
export GITHUB_TOKEN="ghp_your_token_here"

# Optional for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Optional for dashboard
export DASHBOARD_HOST="0.0.0.0"
export DASHBOARD_PORT="8080"
```

### Configuration File

The system automatically creates `monitoring/config.json` with default settings:

```json
{
  "project_root": ".",
  "data_directory": "monitoring/data",
  "log_level": "INFO",
  "github": {
    "owner": "DimaJoyti",
    "repo": "DataMCPServerAgent"
  },
  "cicd": {
    "enabled": true,
    "check_interval_minutes": 30
  },
  "code_quality": {
    "enabled": true,
    "check_interval_minutes": 60,
    "auto_fix_enabled": true
  },
  "security": {
    "enabled": true,
    "check_interval_minutes": 120
  },
  "testing": {
    "enabled": true,
    "check_interval_minutes": 60
  },
  "documentation": {
    "enabled": true,
    "check_interval_minutes": 240
  },
  "dashboard": {
    "enabled": true,
    "host": "localhost",
    "port": 8080
  }
}
```

## 🚀 Usage

### Start Full Monitoring System

```bash
# Start with web dashboard and scheduled monitoring
python scripts/start_monitoring.py

# Access dashboard at http://localhost:8080
```

### Quick Health Check

```bash
# Run one-time monitoring check
python scripts/start_monitoring.py --quick
```

### Individual Components

```bash
# Run specific monitoring components
python -m monitoring.ci_cd.performance_monitor
python -m monitoring.code_quality.quality_monitor
python -m monitoring.security.security_monitor
python -m monitoring.testing.coverage_monitor
python -m monitoring.documentation.doc_health_checker
```

### Dashboard Only

```bash
# Start just the web dashboard
python -m monitoring.dashboard.main_dashboard
```

## 📊 Dashboard

The web dashboard provides:

- **Real-time Metrics**: Live updates via WebSocket
- **Interactive Charts**: CI/CD trends, security issues, test coverage
- **Health Scores**: Overall system health at a glance
- **Recommendations**: Actionable improvement suggestions
- **Historical Data**: Trend analysis and performance tracking

### Dashboard Features

- 📈 **CI/CD Performance Charts**: Success rates, build times, queue times
- 🔒 **Security Risk Visualization**: Issue distribution by severity
- 🧪 **Test Metrics**: Coverage, performance, success rates
- 📚 **Documentation Health**: Completeness, quality, freshness scores
- 🔔 **Real-time Alerts**: Critical issues and recommendations

## 📁 Data Structure

```
monitoring/
├── data/                          # Generated monitoring data
│   ├── cicd_metrics.json         # CI/CD performance data
│   ├── quality_report.json       # Code quality analysis
│   ├── security_report.json      # Security scan results
│   ├── test_health_report.json   # Testing metrics
│   ├── documentation_health.json # Documentation analysis
│   ├── monitoring_summary.json   # Overall system summary
│   └── monitoring.log            # System logs
├── core/                         # Core monitoring components
├── ci_cd/                        # CI/CD monitoring
├── code_quality/                 # Code quality monitoring
├── security/                     # Security monitoring
├── testing/                      # Testing monitoring
├── documentation/                # Documentation monitoring
└── dashboard/                    # Web dashboard
```

## 🔧 API Reference

### REST API Endpoints

```bash
# Get current metrics
GET /api/metrics

# Dashboard home
GET /

# WebSocket for real-time updates
WS /ws
```

### Programmatic Usage

```python
from monitoring.core.monitor_manager import MonitorManager
from monitoring.core.config import MonitoringConfig

# Create configuration
config = MonitoringConfig.from_env()

# Start monitoring
manager = MonitorManager(config)
await manager.start()

# Get status
status = manager.get_status()
```

## 📈 Metrics and Scoring

### Health Scores (0-100)

- **Code Quality**: Based on tool results (Black, Ruff, MyPy, etc.)
- **Security Risk**: Based on vulnerability count and severity
- **Test Health**: Coverage, performance, and success rates
- **Documentation Health**: Completeness, quality, and freshness
- **CI/CD Health**: Success rates and performance metrics

### Alert Thresholds

- 🟢 **Good**: Score ≥ 80
- 🟡 **Warning**: Score 60-79
- 🔴 **Critical**: Score < 60

## 🔄 Automation

### Scheduled Tasks

- **CI/CD Monitoring**: Every 30 minutes
- **Code Quality**: Every 60 minutes
- **Security Scanning**: Every 2 hours
- **Testing Analysis**: Every 60 minutes
- **Documentation Check**: Every 4 hours

### Manual Triggers

```python
# Run specific task immediately
scheduler.run_task_now("quality_monitor")

# Enable/disable tasks
scheduler.enable_task("security_monitor")
scheduler.disable_task("cicd_monitor")
```

## 🚨 Alerts and Notifications

### Alert Conditions

- Critical security vulnerabilities detected
- Code quality below threshold
- Test coverage drops significantly
- CI/CD failure rate increases
- Documentation becomes outdated

### Notification Channels

- **Slack**: Via webhook integration
- **Discord**: Via webhook integration
- **Email**: SMTP configuration
- **Dashboard**: Real-time web alerts

## 🛠️ Troubleshooting

### Common Issues

1. **GitHub API Rate Limits**
   ```bash
   # Check rate limit status
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit
   ```

2. **Missing Dependencies**
   ```bash
   # Install all optional dependencies
   pip install -r requirements-ci.txt
   ```

3. **Permission Issues**
   ```bash
   # Ensure data directory is writable
   chmod 755 monitoring/data
   ```

### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python scripts/start_monitoring.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add monitoring components or improvements
4. Test thoroughly
5. Submit a pull request

### Adding New Monitors

```python
# Example: Add custom monitor
from monitoring.core.scheduler import MonitoringScheduler

def custom_monitor():
    # Your monitoring logic here
    return {"status": "success", "data": {...}}

scheduler.register_task(
    "custom_monitor",
    custom_monitor,
    interval_minutes=30,
    enabled=True
)
```

## 📄 License

This monitoring system is part of the DataMCPServerAgent project and follows the same MIT License.

## 🙏 Acknowledgments

- GitHub Actions API for CI/CD data
- Security tools: Bandit, Safety, Semgrep
- Code quality tools: Black, isort, Ruff, MyPy
- Testing tools: pytest, coverage.py
- Web framework: FastAPI
- Visualization: Chart.js

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: 2024
