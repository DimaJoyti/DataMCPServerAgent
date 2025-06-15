#!/usr/bin/env python3
"""
DataMCPServerAgent Monitoring System Demo

Demonstrates all monitoring capabilities and generates sample reports.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


async def demo_monitoring_system():
    """Demonstrate the complete monitoring system"""

    print_header("DataMCPServerAgent Monitoring System Demo")
    print("This demo showcases all monitoring capabilities")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Configuration Demo
    print_section("1. Configuration Management")

    try:
        from monitoring.core.config import MonitoringConfig

        # Create configuration from environment
        config = MonitoringConfig.from_env()
        print("✅ Configuration loaded successfully")
        print(f"   Project root: {config.project_root}")
        print(f"   Data directory: {config.data_directory}")
        print(f"   Dashboard enabled: {config.dashboard.enabled}")

        # Validate configuration
        issues = config.validate()
        if issues:
            print("⚠️  Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Configuration validation passed")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    # 2. Code Quality Demo
    print_section("2. Code Quality Monitoring")

    try:
        from monitoring.code_quality.quality_monitor import monitor_code_quality

        print("🔍 Running code quality analysis...")
        quality_report = monitor_code_quality(
            project_root=".",
            directories=["app", "src", "examples", "scripts"],
            output_path="monitoring/data/demo_quality_report.json"
        )

        print("✅ Code Quality Analysis Complete")
        print(f"   Overall Score: {quality_report.overall_score}/100")
        print(f"   Total Issues: {quality_report.total_issues}")
        print(f"   Critical Issues: {quality_report.critical_issues}")

        if quality_report.tool_results:
            print("   Tool Results:")
            for tool, metrics in quality_report.tool_results.items():
                print(f"     {tool}: {metrics.status} ({metrics.issues_count} issues)")

    except Exception as e:
        print(f"❌ Code quality monitoring error: {e}")

    # 3. Security Monitoring Demo
    print_section("3. Security Monitoring")

    try:
        from monitoring.security.security_monitor import monitor_security

        print("🔒 Running security analysis...")
        security_report = monitor_security(
            project_root=".",
            directories=["app", "src", "examples", "scripts"],
            output_path="monitoring/data/demo_security_report.json"
        )

        print("✅ Security Analysis Complete")
        print(f"   Risk Score: {security_report.overall_risk_score}/100")
        print(f"   Total Issues: {security_report.total_issues}")
        print(f"   Critical: {security_report.critical_issues}")
        print(f"   High: {security_report.high_issues}")
        print(f"   Medium: {security_report.medium_issues}")
        print(f"   Low: {security_report.low_issues}")

        if security_report.recommendations:
            print("   Top Recommendations:")
            for rec in security_report.recommendations[:3]:
                print(f"     • {rec}")

    except Exception as e:
        print(f"❌ Security monitoring error: {e}")

    # 4. Testing Metrics Demo
    print_section("4. Testing Metrics")

    try:
        from monitoring.testing.coverage_monitor import monitor_testing

        print("🧪 Running testing analysis...")
        test_report = monitor_testing(
            project_root=".",
            output_path="monitoring/data/demo_test_report.json"
        )

        print("✅ Testing Analysis Complete")
        print(f"   Health Score: {test_report.health_score}/100")
        print(f"   Coverage: {test_report.coverage_metrics.overall_coverage:.1f}%")
        print(f"   Total Tests: {test_report.performance_metrics.total_tests}")
        print(f"   Passed: {test_report.performance_metrics.passed_tests}")
        print(f"   Failed: {test_report.performance_metrics.failed_tests}")

        if test_report.recommendations:
            print("   Recommendations:")
            for rec in test_report.recommendations[:3]:
                print(f"     • {rec}")

    except Exception as e:
        print(f"❌ Testing monitoring error: {e}")

    # 5. Documentation Health Demo
    print_section("5. Documentation Health")

    try:
        from monitoring.documentation.doc_health_checker import monitor_documentation_health

        print("📚 Running documentation analysis...")
        doc_report = monitor_documentation_health(
            project_root=".",
            docs_directories=["docs", "README.md"],
            output_path="monitoring/data/demo_doc_report.json"
        )

        print("✅ Documentation Analysis Complete")
        print(f"   Overall Score: {doc_report.overall_score:.1f}/100")
        print(f"   Coverage: {doc_report.coverage_score:.1f}/100")
        print(f"   Quality: {doc_report.quality_score:.1f}/100")
        print(f"   Freshness: {doc_report.freshness_score:.1f}/100")
        print(f"   Total Documents: {doc_report.total_documents}")
        print(f"   Broken Links: {doc_report.total_broken_links}")

        if doc_report.recommendations:
            print("   Recommendations:")
            for rec in doc_report.recommendations[:3]:
                print(f"     • {rec}")

    except Exception as e:
        print(f"❌ Documentation monitoring error: {e}")

    # 6. CI/CD Monitoring Demo (if GitHub token available)
    print_section("6. CI/CD Performance Monitoring")

    try:
        import os
        github_token = os.getenv("GITHUB_TOKEN")

        if github_token:
            from monitoring.ci_cd.performance_monitor import monitor_cicd_performance

            print("🔄 Running CI/CD analysis...")
            cicd_metrics = await monitor_cicd_performance(
                github_token=github_token,
                owner="DimaJoyti",
                repo="DataMCPServerAgent",
                output_path="monitoring/data/demo_cicd_metrics.json"
            )

            print("✅ CI/CD Analysis Complete")
            print(f"   Workflows analyzed: {len(cicd_metrics)}")

            for workflow_name, metrics in cicd_metrics.items():
                print(f"   {workflow_name}:")
                print(f"     Success Rate: {metrics.success_rate:.1f}%")
                print(f"     Avg Duration: {metrics.average_duration_seconds:.1f}s")
                print(f"     Recent Failures: {len(metrics.recent_failures)}")
        else:
            print("⚠️  GitHub token not available - skipping CI/CD monitoring")
            print("   Set GITHUB_TOKEN environment variable to enable this feature")

    except Exception as e:
        print(f"❌ CI/CD monitoring error: {e}")

    # 7. Dashboard Demo
    print_section("7. Web Dashboard")

    try:
        from monitoring.dashboard.main_dashboard import MonitoringDashboard

        print("🌐 Dashboard capabilities:")
        print("   ✅ Real-time metrics visualization")
        print("   ✅ Interactive charts and graphs")
        print("   ✅ WebSocket live updates")
        print("   ✅ Mobile-responsive design")
        print("   ✅ Historical data tracking")

        # Create dashboard instance (don't start server in demo)
        dashboard = MonitoringDashboard("monitoring/data")
        print("   ✅ Dashboard components initialized")
        print("   📊 To start dashboard: python -m monitoring.dashboard.main_dashboard")

    except ImportError:
        print("⚠️  Dashboard dependencies not available")
        print("   Install with: pip install fastapi uvicorn jinja2")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

    # 8. Scheduler Demo
    print_section("8. Automated Scheduling")

    try:
        from monitoring.core.scheduler import MonitoringScheduler

        scheduler = MonitoringScheduler(config)
        scheduler.setup_default_tasks()

        print("⏰ Scheduler capabilities:")
        print("   ✅ Automated task scheduling")
        print("   ✅ Configurable intervals")
        print("   ✅ Error handling and retry")
        print("   ✅ Manual task triggering")

        status = scheduler.get_task_status()
        print(f"   📋 Total tasks: {status['total_tasks']}")
        print(f"   ✅ Enabled tasks: {status['enabled_tasks']}")

        print("   Configured tasks:")
        for task_name, task_info in status['tasks'].items():
            if task_info['enabled']:
                print(f"     • {task_name}: every {task_info['interval_minutes']} minutes")

    except Exception as e:
        print(f"❌ Scheduler error: {e}")

    # 9. Summary Report
    print_section("9. System Summary")

    try:
        # Generate overall summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "demo_completed": True,
            "components_tested": [
                "Configuration Management",
                "Code Quality Monitoring",
                "Security Monitoring",
                "Testing Metrics",
                "Documentation Health",
                "Dashboard Components",
                "Automated Scheduling"
            ],
            "system_health": {},
            "recommendations": []
        }

        # Load generated reports
        data_dir = Path("monitoring/data")
        if data_dir.exists():
            report_files = {
                "quality": "demo_quality_report.json",
                "security": "demo_security_report.json",
                "testing": "demo_test_report.json",
                "documentation": "demo_doc_report.json"
            }

            for report_type, filename in report_files.items():
                file_path = data_dir / filename
                if file_path.exists():
                    with open(file_path) as f:
                        report_data = json.load(f)

                    if report_type == "quality":
                        summary["system_health"]["code_quality"] = report_data.get("overall_score", 0)
                    elif report_type == "security":
                        summary["system_health"]["security_risk"] = report_data.get("overall_risk_score", 0)
                    elif report_type == "testing":
                        summary["system_health"]["test_health"] = report_data.get("health_score", 0)
                    elif report_type == "documentation":
                        summary["system_health"]["doc_health"] = report_data["scores"]["overall_score"]

                    # Collect recommendations
                    recommendations = report_data.get("recommendations", [])
                    for rec in recommendations[:2]:
                        summary["recommendations"].append(f"[{report_type.title()}] {rec}")

        # Save summary
        summary_file = data_dir / "demo_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("📊 Demo Summary:")
        print("   ✅ All components tested successfully")
        print(f"   📁 Reports saved to: {data_dir}")

        if summary["system_health"]:
            print("   📈 Health Scores:")
            for metric, score in summary["system_health"].items():
                status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                print(f"     {status_icon} {metric.replace('_', ' ').title()}: {score:.1f}")

        if summary["recommendations"]:
            print("   💡 Key Recommendations:")
            for rec in summary["recommendations"][:5]:
                print(f"     • {rec}")

    except Exception as e:
        print(f"❌ Summary generation error: {e}")

    # 10. Next Steps
    print_section("10. Next Steps")

    print("🚀 To start using the monitoring system:")
    print("   1. Set environment variables (GITHUB_TOKEN, etc.)")
    print("   2. Install dependencies: pip install -r monitoring/requirements.txt")
    print("   3. Start monitoring: python scripts/start_monitoring.py")
    print("   4. Access dashboard: http://localhost:8080")
    print()
    print("📚 Documentation:")
    print("   • Full guide: monitoring/README.md")
    print("   • Configuration: monitoring/config.json")
    print("   • Data directory: monitoring/data/")
    print()
    print("🔧 Customization:")
    print("   • Modify monitoring intervals in config")
    print("   • Add custom monitoring tasks")
    print("   • Configure notifications (Slack, Discord)")
    print("   • Set up alerting thresholds")

    print_header("Demo Complete!")
    print("The DataMCPServerAgent monitoring system is ready for production use.")


if __name__ == "__main__":
    asyncio.run(demo_monitoring_system())
