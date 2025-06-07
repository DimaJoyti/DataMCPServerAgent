#!/usr/bin/env python3
"""
Start DataMCPServerAgent Monitoring System

This script starts the comprehensive monitoring system with all components.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from monitoring.core.config import MonitoringConfig
    from monitoring.core.monitor_manager import MonitorManager
except ImportError as e:
    print(f"Failed to import monitoring modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              DataMCPServerAgent Monitoring System            ║
    ║                                                              ║
    ║  Comprehensive monitoring for CI/CD, Code Quality,          ║
    ║  Security, Testing, and Documentation                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    # Check for optional dependencies
    optional_deps = {
        "fastapi": "Web dashboard",
        "uvicorn": "Web server",
        "jinja2": "Template engine",
        "aiohttp": "HTTP client",
        "requests": "HTTP requests",
        "schedule": "Task scheduling"
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep} ({description})")
    
    if missing_deps:
        print("⚠️  Optional dependencies missing:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install fastapi uvicorn jinja2 aiohttp requests schedule")
        print("Some features may not be available.\n")
    
    return len(missing_deps) == 0


def setup_environment():
    """Setup environment variables and configuration"""
    # Check for required environment variables
    required_env = {
        "GITHUB_TOKEN": "GitHub API access (for CI/CD monitoring)"
    }
    
    missing_env = []
    for env_var, description in required_env.items():
        if not os.getenv(env_var):
            missing_env.append(f"{env_var}: {description}")
    
    if missing_env:
        print("⚠️  Environment variables not set:")
        for env in missing_env:
            print(f"   - {env}")
        print("\nSome monitoring features may not work without these variables.\n")
    
    # Create default configuration if it doesn't exist
    config_path = project_root / "monitoring" / "config.json"
    if not config_path.exists():
        print("📝 Creating default configuration...")
        config = MonitoringConfig.from_env()
        config.save_to_file(str(config_path))
        print(f"   Configuration saved to: {config_path}")
    
    return config_path


async def main():
    """Main function"""
    print_banner()
    
    print("🔍 Checking system requirements...")
    check_dependencies()
    
    print("⚙️  Setting up environment...")
    config_path = setup_environment()
    
    print("📊 Loading configuration...")
    try:
        config = MonitoringConfig.from_file(str(config_path))
        
        # Validate configuration
        issues = config.validate()
        if issues:
            print("⚠️  Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
            print()
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    print("🚀 Starting monitoring system...")
    print(f"   Data directory: {config.data_directory}")
    print(f"   Dashboard: {'Enabled' if config.dashboard.enabled else 'Disabled'}")
    if config.dashboard.enabled:
        print(f"   Dashboard URL: http://{config.dashboard.host}:{config.dashboard.port}")
    print()
    
    # Create and start monitor manager
    manager = MonitorManager(config)
    
    try:
        await manager.start()
        
        print("✅ Monitoring system started successfully!")
        print("\n📋 Active monitoring:")
        
        status = manager.get_status()
        scheduler_status = status["scheduler"]
        
        for task_name, task_info in scheduler_status["tasks"].items():
            if task_info["enabled"]:
                print(f"   ✓ {task_name.replace('_', ' ').title()}")
        
        print(f"\n📊 Total: {scheduler_status['enabled_tasks']} monitoring tasks active")
        
        if config.dashboard.enabled:
            print(f"\n🌐 Dashboard available at: http://{config.dashboard.host}:{config.dashboard.port}")
        
        print("\n🔄 Monitoring will run continuously...")
        print("   Press Ctrl+C to stop")
        
        # Keep running
        while manager.running:
            await asyncio.sleep(60)
            
            # Periodic status update
            status = manager.get_status()
            last_summary = status.get("last_summary")
            
            if last_summary:
                health = last_summary.get("system_health", {})
                alerts = last_summary.get("alerts", [])
                
                if alerts:
                    print(f"\n⚠️  Active alerts ({len(alerts)}):")
                    for alert in alerts[:3]:  # Show first 3 alerts
                        print(f"   {alert}")
                
                # Show key metrics every 10 minutes
                import time
                if int(time.time()) % 600 == 0:  # Every 10 minutes
                    print(f"\n📊 System Health Update:")
                    for metric, score in health.items():
                        status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                        print(f"   {status_icon} {metric.replace('_', ' ').title()}: {score:.1f}")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
    except Exception as e:
        print(f"\n❌ Monitoring system error: {e}")
    finally:
        print("🔄 Stopping monitoring system...")
        manager.stop()
        print("✅ Monitoring system stopped")


def run_quick_check():
    """Run a quick monitoring check without starting the full system"""
    print("🔍 Running quick monitoring check...")
    
    config = MonitoringConfig.from_env()
    manager = MonitorManager(config)
    
    async def quick_check():
        # Run just the initial monitoring sweep
        await manager.run_initial_monitoring()
        
        # Show summary
        summary_file = Path(config.data_directory) / "monitoring_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("\n📊 Quick Check Results:")
            health = summary.get("system_health", {})
            for metric, score in health.items():
                status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                print(f"   {status_icon} {metric.replace('_', ' ').title()}: {score:.1f}")
            
            recommendations = summary.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Top Recommendations:")
                for rec in recommendations[:5]:
                    print(f"   • {rec}")
            
            alerts = summary.get("alerts", [])
            if alerts:
                print(f"\n⚠️  Alerts:")
                for alert in alerts:
                    print(f"   {alert}")
        
        print(f"\n📁 Detailed reports saved to: {config.data_directory}")
    
    asyncio.run(quick_check())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_check()
    else:
        asyncio.run(main())
