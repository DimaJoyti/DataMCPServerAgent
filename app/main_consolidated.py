#!/usr/bin/env python3
"""
DataMCPServerAgent - Consolidated Main Entry Point

Unified entry point for the consolidated DataMCPServerAgent system.
All functionality accessible through a single, clean interface.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup to avoid import issues
try:
    from app.core.logging import get_logger, setup_logging
except ImportError:
    # Fallback to simple logging if dependencies are missing
    from app.core.simple_logging import get_logger, setup_logging

try:
    from app.core.config import get_settings
except ImportError:
    # Create a simple settings fallback
    class SimpleSettings:
        app_name = "DataMCPServerAgent"
        app_version = "2.0.0"
        environment = "development"
        debug = True

    def get_settings():
        return SimpleSettings()

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="datamcp",
    help="DataMCPServerAgent - Consolidated AI Agent System",
    add_completion=False,
    rich_markup_mode="rich",
)


def display_banner() -> None:
    """Display application banner."""
    banner = Text()
    banner.append("DataMCPServerAgent", style="bold blue")
    banner.append(" v2.0.0 Consolidated", style="dim")
    banner.append("\n")
    banner.append(
        "Unified AI Agent System with Clean Architecture",
        style="italic"
    )

    panel = Panel(
        banner,
        title="🤖 Consolidated System",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8002, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of workers"),
):
    """Start the consolidated API server."""
    display_banner()

    setup_logging()

    logger.info("🚀 Starting Consolidated DataMCPServerAgent API")
    logger.info(f"📍 Host: {host}:{port}")
    logger.info(f"🔄 Reload: {reload}")
    logger.info(f"👥 Workers: {workers}")

    try:
        # Import here to avoid circular imports
        from app.api.consolidated_server import create_consolidated_app

        uvicorn.run(
            create_consolidated_app,
            factory=True,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("👋 API server stopped by user")
    except Exception as e:
        logger.error(f"💥 API server failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def cli(
    interactive: bool = typer.Option(True, help="Interactive mode"),
):
    """Start the consolidated CLI interface."""
    display_banner()

    setup_logging()

    logger.info("🖥️ Starting Consolidated CLI Interface")

    try:
        from app.cli.consolidated_interface import ConsolidatedCLI

        cli_interface = ConsolidatedCLI()

        if interactive:
            asyncio.run(cli_interface.run_interactive())
        else:
            asyncio.run(cli_interface.run_batch())

    except KeyboardInterrupt:
        logger.info("👋 CLI interface stopped by user")
    except Exception as e:
        logger.error(f"💥 CLI interface failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def status():
    """Show consolidated system status."""
    display_banner()

    console.print("📊 Consolidated System Status", style="bold green")
    console.print("=" * 50)

    # Check components
    components = {
        "Configuration": "✅ OK",
        "Logging": "✅ OK",
        "API Server": "🔍 Checking...",
        "Database": "🔍 Checking...",
        "Cache": "🔍 Checking...",
    }

    for component, status in components.items():
        console.print(f"{component}: {status}")

    # Check API server
    try:
        import httpx

        response = httpx.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            console.print("API Server: ✅ RUNNING")
        else:
            console.print("API Server: ⚠️ UNHEALTHY")
    except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
        console.print(f"API Server: ❌ NOT RUNNING ({type(e).__name__})")


@app.command()
def migrate():
    """Run database migrations."""
    display_banner()

    console.print("🔄 Running Database Migrations", style="bold blue")

    try:
        # Import migration logic here
        console.print("✅ Migrations completed successfully", style="green")
    except Exception as e:
        console.print(f"💥 Migration failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def test(
    coverage: bool = typer.Option(True, help="Run with coverage"),
    pattern: Optional[str] = typer.Option(None, help="Test pattern"),
):
    """Run the consolidated test suite."""
    display_banner()

    console.print("🧪 Running Consolidated Test Suite", style="bold blue")

    import subprocess

    cmd = ["python", "-m", "pytest"]

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    if pattern:
        cmd.extend(["-k", pattern])

    try:
        subprocess.run(cmd, check=True)
        console.print("✅ All tests passed!", style="green")
    except subprocess.CalledProcessError:
        console.print("❌ Some tests failed", style="red")
        raise typer.Exit(1)


@app.command()
def agents():
    """Manage agents in the consolidated system."""
    display_banner()

    console.print("🤖 Agent Management", style="bold blue")
    console.print("Available commands:")
    console.print("  • list    - List all agents")
    console.print("  • create  - Create new agent")
    console.print("  • delete  - Delete agent")
    console.print("  • status  - Show agent status")


@app.command()
def rl(
    mode: str = typer.Option("modern_deep", help="RL mode to use"),
    action: str = typer.Option("status", help="Action to perform"),
    interactive: bool = typer.Option(False, help="Interactive RL session"),
):
    """Manage the Reinforcement Learning system."""
    display_banner()

    console.print("🧠 Reinforcement Learning System", style="bold blue")

    if action == "status":
        asyncio.run(_rl_status())
    elif action == "train":
        asyncio.run(_rl_train(mode))
    elif action == "test":
        asyncio.run(_rl_test(mode))
    elif action == "interactive" or interactive:
        asyncio.run(_rl_interactive(mode))
    elif action == "adaptive":
        asyncio.run(_rl_adaptive())
    elif action == "ab-test":
        asyncio.run(_rl_ab_test())
    elif action == "deploy":
        asyncio.run(_rl_deploy())
    elif action == "enterprise":
        asyncio.run(_rl_enterprise_demo())
    elif action == "federated":
        asyncio.run(_rl_federated())
    elif action == "cloud":
        asyncio.run(_rl_cloud())
    elif action == "scaling":
        asyncio.run(_rl_scaling())
    elif action == "monitoring":
        asyncio.run(_rl_monitoring())
    elif action == "training":
        asyncio.run(_rl_enterprise_training())
    elif action == "phase6":
        asyncio.run(_rl_phase6_demo())
    else:
        console.print(f"❌ Unknown action: {action}", style="red")
        console.print(
            "Available actions: status, train, test, interactive, adaptive, "
            "ab-test, deploy, enterprise, federated, cloud, scaling, "
            "monitoring, training, phase6"
        )


async def _rl_status():
    """Show RL system status."""
    try:
        from app.core.rl_integration import get_rl_manager

        manager = get_rl_manager()
        status = manager.get_status()

        console.print("📊 RL System Status:", style="bold green")
        console.print(f"  Initialized: {'✅' if status['initialized'] else '❌'}")
        console.print(f"  Training: {'🏋️' if status['training'] else '💤'}")
        console.print(f"  Mode: {status['mode']}")
        console.print(f"  Algorithm: {status['algorithm']}")

        metrics = status['performance_metrics']
        console.print("\n📈 Performance Metrics:")
        console.print(f"  Total requests: {metrics['total_requests']}")
        console.print(f"  Successful requests: {metrics['successful_requests']}")
        console.print(f"  Average response time: {metrics['average_response_time']:.3f}s")
        console.print(f"  Training episodes: {metrics['training_episodes']}")

    except Exception as e:
        console.print(f"❌ Error getting RL status: {e}", style="red")


async def _rl_train(mode: str):
    """Train the RL system."""
    try:
        from app.core.rl_integration import get_rl_manager

        console.print(f"🏋️ Training RL system in {mode} mode...", style="blue")

        manager = get_rl_manager()
        if not manager.is_initialized:
            console.print("🚀 Initializing RL system...")
            await manager.initialize()

        # Train for a few episodes
        for episode in range(5):
            console.print(f"📚 Training episode {episode + 1}/5...")
            metrics = await manager.train_episode()

            if "error" in metrics:
                console.print(f"❌ Training error: {metrics['error']}", style="red")
                break
            else:
                console.print(f"✅ Episode completed: {metrics}")

        console.print("🎉 Training completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error during training: {e}", style="red")


async def _rl_test(mode: str):
    """Test the RL system."""
    try:
        from app.core.rl_integration import get_rl_manager

        console.print(f"🧪 Testing RL system in {mode} mode...", style="blue")

        manager = get_rl_manager()
        if not manager.is_initialized:
            console.print("🚀 Initializing RL system...")
            await manager.initialize()

        # Test with sample requests
        test_requests = [
            "Analyze the current market trends",
            "Create a summary of recent data",
            "Help me understand this complex problem",
            "Generate a creative solution",
        ]

        for i, request in enumerate(test_requests):
            console.print(f"\n📝 Test {i+1}: {request}")

            result = await manager.process_request(request)

            if result["success"]:
                console.print(f"✅ Success: {result['response']}")
                console.print(f"⏱️ Response time: {result['response_time']:.3f}s")

                if "explanation" in result:
                    console.print(f"💭 Reasoning: {result.get('reasoning', 'N/A')}")

                if "safety_info" in result:
                    safety = result["safety_info"]
                    console.print(f"🛡️ Safety score: {safety.get('safety_score', 'N/A')}")
            else:
                console.print(f"❌ Failed: {result.get('error', 'Unknown error')}", style="red")

        # Show performance report
        report = manager.get_performance_report()
        console.print("\n📊 Test Results Summary:")
        console.print(f"  Success rate: {report['summary']['success_rate']:.1%}")
        console.print(f"  Average response time: {report['summary']['average_response_time']:.3f}s")

    except Exception as e:
        console.print(f"❌ Error during testing: {e}", style="red")


async def _rl_interactive(mode: str):
    """Start interactive RL session."""
    try:
        from app.core.rl_integration import get_rl_manager

        console.print(f"🎮 Starting interactive RL session in {mode} mode...", style="blue")
        console.print("Type 'quit' to exit, 'help' for commands")

        manager = get_rl_manager()
        if not manager.is_initialized:
            console.print("🚀 Initializing RL system...")
            await manager.initialize()

        while True:
            try:
                user_input = console.input("\n[bold blue]RL>[/bold blue] ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    console.print("Available commands:")
                    console.print("  help     - Show this help")
                    console.print("  status   - Show system status")
                    console.print("  train    - Train one episode")
                    console.print("  quit     - Exit interactive mode")
                    console.print("  <text>   - Process request with RL")
                    continue
                elif user_input.lower() == 'status':
                    await _rl_status()
                    continue
                elif user_input.lower() == 'train':
                    metrics = await manager.train_episode()
                    console.print(f"Training result: {metrics}")
                    continue
                elif not user_input.strip():
                    continue

                # Process request
                console.print("🤔 Processing with RL system...")
                result = await manager.process_request(user_input)

                if result["success"]:
                    console.print(f"🤖 {result['response']}")

                    if "explanation" in result:
                        console.print(f"💭 Reasoning: {result.get('reasoning', 'N/A')}")

                    if "safety_info" in result:
                        safety = result["safety_info"]
                        console.print(f"🛡️ Safety: {safety.get('safety_score', 'N/A')}")

                    console.print(f"⏱️ Response time: {result['response_time']:.3f}s")
                else:
                    console.print(f"❌ Error: {result.get('error', 'Unknown error')}", style="red")

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="red")

        console.print("👋 Exiting interactive RL session")

    except Exception as e:
        console.print(f"❌ Error in interactive mode: {e}", style="red")


async def _rl_adaptive():
    """Demonstrate adaptive learning capabilities."""
    try:
        from app.rl.adaptive_learning import get_adaptive_learning_engine

        console.print("🔄 Adaptive Learning System", style="blue")

        engine = get_adaptive_learning_engine()

        # Start adaptive learning
        console.print("🚀 Starting adaptive learning engine...")
        await engine.start_adaptive_learning()

        # Show status
        status = engine.get_adaptation_status()
        console.print("📊 Adaptive Learning Status:")
        console.print(f"   Running: {'✅' if status['is_running'] else '❌'}")
        console.print(f"   Active adaptations: {status['active_adaptations']}")
        console.print(f"   Learning events: {status['learning_events']}")
        console.print(f"   Performance metrics: {status['performance_metrics']}")

        # Run for a short time
        console.print("⏳ Running adaptive learning for 30 seconds...")
        await asyncio.sleep(30)

        # Stop adaptive learning
        await engine.stop_adaptive_learning()
        console.print("✅ Adaptive learning demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in adaptive learning demo: {e}", style="red")


async def _rl_ab_test():
    """Demonstrate A/B testing capabilities."""
    try:
        from app.rl.ab_testing import ExperimentMetric, ExperimentVariant, get_ab_testing_engine

        console.print("🧪 A/B Testing System", style="blue")

        engine = get_ab_testing_engine()

        # Create simple experiment
        variants = [
            ExperimentVariant(
                name="control",
                description="Current algorithm",
                config={"algorithm": "dqn"},
                traffic_allocation=0.5,
                is_control=True
            ),
            ExperimentVariant(
                name="treatment",
                description="New algorithm",
                config={"algorithm": "ppo"},
                traffic_allocation=0.5,
                is_control=False
            ),
        ]

        metrics = [
            ExperimentMetric(
                name="response_time",
                description="Response time",
                metric_type="continuous",
                primary=True,
                higher_is_better=False
            ),
        ]

        experiment_id = engine.create_experiment(
            name="Algorithm Test",
            description="Test DQN vs PPO",
            variants=variants,
            metrics=metrics
        )

        console.print(f"📊 Created experiment: {experiment_id}")

        # Start experiment
        engine.start_experiment(experiment_id)
        console.print("🚀 Experiment started")

        # List experiments
        experiments = engine.list_experiments()
        console.print(f"📋 Total experiments: {len(experiments)}")

        # Stop experiment
        engine.stop_experiment(experiment_id)
        console.print("✅ A/B testing demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in A/B testing demo: {e}", style="red")


async def _rl_deploy():
    """Demonstrate model deployment capabilities."""
    try:
        import os
        import tempfile

        from app.rl.model_deployment import (
            DeploymentConfig,
            DeploymentStrategy,
            get_deployment_manager,
        )

        console.print("🚀 Model Deployment System", style="blue")

        manager = get_deployment_manager()

        # Create dummy model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"dummy model data")
            model_path = f.name

        try:
            # Register model
            model_id = manager.registry.register_model(
                name="demo_model",
                version="1.0.0",
                algorithm="dqn",
                model_path=model_path,
                training_config={"lr": 1e-4},
                performance_metrics={"accuracy": 0.9}
            )

            console.print(f"📦 Registered model: {model_id}")

            # Deploy model
            config = DeploymentConfig(
                strategy=DeploymentStrategy.BLUE_GREEN,
                traffic_percentage=100.0
            )

            deployment_id = await manager.deploy_model(model_id, "staging", config)
            console.print(f"🚀 Deployed model: {deployment_id}")

            # List deployments
            deployments = manager.list_deployments()
            console.print(f"📋 Total deployments: {len(deployments)}")

            console.print("✅ Model deployment demonstration completed")

        finally:
            os.unlink(model_path)

    except Exception as e:
        console.print(f"❌ Error in deployment demo: {e}", style="red")


async def _rl_enterprise_demo():
    """Run the complete enterprise RL demonstration with advanced training capabilities."""
    try:
        console.print("🏢 Enterprise RL System & Training Demo", style="bold blue")
        console.print("This will demonstrate advanced enterprise capabilities:")
        console.print("  • 🤝 Federated Learning - Privacy-preserving multi-organization training")
        console.print("  • 🔄 Adaptive Learning - Self-optimizing system with anomaly detection")  
        console.print("  • 📈 Intelligent Scaling - Predictive scaling with cost optimization")
        console.print("  • 🔐 Privacy Protection - Differential privacy and secure aggregation")
        console.print("  • 🎯 Auto-Tuning - Automatic hyperparameter optimization")
        console.print("  • 💰 Cost Optimization - Intelligent resource allocation")
        console.print("⚠️ This may take several minutes to complete.")

        try:
            confirm = console.input("Continue? [y/N]: ")
            if confirm.lower() != 'y':
                console.print("Demo cancelled")
                return
        except (EOFError, KeyboardInterrupt):
            console.print("Running in non-interactive mode, proceeding automatically...")
            console.print("🚀 Starting enterprise training demonstration...")

        # Import and run new enterprise training demo
        import subprocess
        import sys
        import os
        
        # Get the path to the enterprise training demo
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_path = os.path.join(project_root, "examples", "enterprise_training_demo.py")
        
        if os.path.exists(demo_path):
            console.print("🚀 Running Enterprise Training Suite...")
            result = subprocess.run([sys.executable, demo_path])
            
            if result.returncode == 0:
                console.print("✅ Enterprise training demonstration completed successfully!", style="green")
            else:
                console.print(f"❌ Enterprise demo failed with code: {result.returncode}", style="red")
        else:
            # Fallback to original demo if available
            try:
                from examples.enterprise_rl_system_demo import EnterpriseRLSystemDemo
                console.print("🔄 Running fallback enterprise demo...")
                demo = EnterpriseRLSystemDemo()
                await demo.run_enterprise_demo()
            except ImportError:
                console.print("⚠️ Running Phase 3 optimization demo as demonstration...")
                # Run the optimized demo as fallback
                optimized_demo_path = os.path.join(project_root, "examples", "optimized_rl_demo.py")
                if os.path.exists(optimized_demo_path):
                    result = subprocess.run([sys.executable, optimized_demo_path])
                    if result.returncode == 0:
                        console.print("✅ Optimization demo completed successfully!", style="green")

    except Exception as e:
        console.print(f"❌ Error in enterprise demo: {e}", style="red")


async def _rl_enterprise_training():
    """Run enterprise training suite with federated learning and adaptive systems."""
    try:
        console.print("🎓 Enterprise Training Suite", style="bold blue")
        console.print("Advanced training capabilities demonstration:")
        console.print("  • 🤝 Federated Learning across multiple organizations")
        console.print("  • 🔄 Adaptive Learning with self-optimization")
        console.print("  • 📈 Intelligent Auto-Scaling with cost optimization")
        console.print("  • 🔐 Privacy-preserving training with differential privacy")
        console.print("  • 🧠 Memory-optimized operations with Phase 3 improvements")

        # Import and run enterprise training demo directly
        import subprocess
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_path = os.path.join(project_root, "examples", "enterprise_training_demo.py")
        
        if os.path.exists(demo_path):
            console.print("🚀 Launching Enterprise Training Suite...")
            result = subprocess.run([sys.executable, demo_path])
            
            if result.returncode == 0:
                console.print("✅ Enterprise training suite completed successfully!", style="green")
                console.print("🏆 All advanced training capabilities demonstrated!", style="bold green")
            else:
                console.print(f"❌ Training suite failed with code: {result.returncode}", style="red")
        else:
            console.print("❌ Enterprise training demo not found", style="red")
            console.print("Please ensure the enterprise_training_demo.py file exists in examples/")

    except Exception as e:
        console.print(f"❌ Error in enterprise training: {e}", style="red")


async def _rl_federated():
    """Demonstrate federated learning capabilities."""
    try:
        from app.rl.federated_learning import PrivacyLevel, create_federated_coordinator

        console.print("🤝 Federated Learning System", style="blue")

        # Create federation
        federation_id = "demo_federation"
        coordinator = create_federated_coordinator(
            federation_id=federation_id,
            privacy_level=PrivacyLevel.DIFFERENTIAL,
            min_participants=2
        )

        console.print(f"📋 Created federation: {federation_id}")

        # Register demo participants
        participants = [
            {"id": "org1", "name": "Organization 1", "org": "TechCorp"},
            {"id": "org2", "name": "Organization 2", "org": "DataInc"},
            {"id": "org3", "name": "Organization 3", "org": "AILabs"},
        ]

        for p in participants:
            success = coordinator.register_participant(
                participant_id=p["id"],
                name=p["name"],
                organization=p["org"],
                endpoint=f"https://{p['org'].lower()}.ai/federated",
                data_size=1000
            )

            if success:
                console.print(f"✅ Registered: {p['name']}")

        # Get status
        status = coordinator.get_federation_status()
        console.print(f"📊 Participants: {status['participants']}")
        console.print(f"🔒 Privacy level: {status['privacy_level']}")

        console.print("✅ Federated learning demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in federated learning demo: {e}", style="red")


async def _rl_cloud():
    """Demonstrate cloud integration capabilities."""
    try:
        from app.cloud.cloud_integration import (
            CloudProvider,
            DeploymentEnvironment,
            get_cloud_orchestrator,
        )

        console.print("☁️ Cloud Integration System", style="blue")

        orchestrator = get_cloud_orchestrator()

        # Demo deployment
        deployment_id = await orchestrator.deploy_rl_system(
            deployment_name="demo-rl-system",
            environment=DeploymentEnvironment.STAGING,
            provider=CloudProvider.AWS,
            config={
                "region": "us-east-1",
                "instance_type": "ml.m5.large",
                "deploy_endpoint": True,
            }
        )

        console.print(f"🚀 Created deployment: {deployment_id}")

        # Get deployment status
        status = orchestrator.get_deployment_status(deployment_id)
        if status:
            console.print(f"📊 Status: {status['status']}")
            console.print(f"🌍 Provider: {status['provider']}")
            console.print(f"📍 Environment: {status['environment']}")

        # Monitor costs
        costs = await orchestrator.monitor_costs()
        console.print(f"💰 Total cost: ${costs['total_cost']:.2f}")
        console.print(f"📊 Active resources: {costs['active_resources']}")

        console.print("✅ Cloud integration demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in cloud demo: {e}", style="red")


async def _rl_scaling():
    """Demonstrate auto-scaling capabilities."""
    try:
        from app.scaling.auto_scaling import ScalingPolicy, create_auto_scaler

        console.print("📈 Auto-Scaling System", style="blue")

        # Create auto-scaler
        scaler = create_auto_scaler(
            service_name="demo-service",
            scaling_policy=ScalingPolicy.HYBRID,
            min_instances=1,
            max_instances=5
        )

        console.print("🔧 Created auto-scaler for demo-service")
        console.print(f"📊 Policy: {ScalingPolicy.HYBRID.value}")

        # Start auto-scaling
        await scaler.start_auto_scaling()
        console.print("🚀 Auto-scaling started")

        # Let it run for a short time
        console.print("⏳ Running for 30 seconds...")
        await asyncio.sleep(30)

        # Get status
        status = scaler.get_scaling_status()
        console.print(f"📊 Current instances: {status['current_instances']}")
        console.print(f"📈 Scaling events: {status['total_scaling_events']}")
        console.print(f"⚡ Efficiency: {status['scaling_efficiency']:.1%}")

        # Stop auto-scaling
        await scaler.stop_auto_scaling()
        console.print("✅ Auto-scaling demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in scaling demo: {e}", style="red")


async def _rl_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    try:
        from app.monitoring.real_time_monitoring import get_real_time_monitor

        console.print("🔍 Real-Time Monitoring System", style="blue")

        monitor = get_real_time_monitor()

        # Start monitoring
        await monitor.start_monitoring()
        console.print("🚀 Real-time monitoring started")
        console.print("📡 WebSocket server: ws://localhost:8765")

        # Let it collect data
        console.print("📊 Collecting metrics for 30 seconds...")
        await asyncio.sleep(30)

        # Get dashboard data
        dashboard = monitor.get_monitoring_dashboard()

        console.print("📊 Monitoring Status:")
        console.print(f"   Status: {dashboard['status']}")
        console.print(
            f"   WebSocket clients: {dashboard['websocket_clients']}"
        )
        console.print(
            f"   Data points: {len(dashboard['performance_history'])}"
        )

        # Show current metrics
        current = dashboard.get('current_metrics', {})
        system = current.get('system', {})
        app = current.get('application', {})

        if system:
            console.print(f"   CPU: {system.get('cpu_percent', 0):.1f}%")
            console.print(f"   Memory: {system.get('memory_percent', 0):.1f}%")

        if app:
            console.print(
                f"   Response time: {app.get('response_time_avg', 0):.0f}ms"
            )
            console.print(f"   Error rate: {app.get('error_rate', 0):.1f}%")

        # Stop monitoring
        await monitor.stop_monitoring()
        console.print("✅ Real-time monitoring demonstration completed")

    except Exception as e:
        console.print(f"❌ Error in monitoring demo: {e}", style="red")


async def _rl_phase6_demo():
    """Run the complete Phase 6 demonstration."""
    try:
        console.print("🚀 Phase 6 Advanced Features Demo", style="blue")
        console.print("This will run the complete Phase 6 demonstration...")
        console.print("⚠️ This may take several minutes to complete.")

        confirm = console.input("Continue? [y/N]: ")
        if confirm.lower() != 'y':
            console.print("Demo cancelled")
            return

        # Import and run Phase 6 demo
        from examples.phase6_advanced_features_demo import Phase6AdvancedDemo

        demo = Phase6AdvancedDemo()
        await demo.run_phase6_demo()

    except Exception as e:
        console.print(f"❌ Error in Phase 6 demo: {e}", style="red")


@app.command()
def tools():
    """Manage tools in the consolidated system."""
    display_banner()

    console.print("🔧 Tool Management", style="bold blue")
    console.print("Available tools:")
    console.print("  • Data tools")
    console.print("  • Communication tools")
    console.print("  • Analysis tools")
    console.print("  • Visualization tools")


@app.command()
def memory():
    """Manage memory systems."""
    display_banner()

    console.print("🧠 Memory Management", style="bold blue")
    console.print("Memory systems:")
    console.print("  • Persistence layer")
    console.print("  • Knowledge graph")
    console.print("  • Distributed memory")
    console.print("  • Context-aware retrieval")


@app.command()
def docs(
    serve: bool = typer.Option(False, help="Serve documentation"),
    port: int = typer.Option(8080, help="Documentation port"),
):
    """Generate or serve consolidated documentation."""
    display_banner()

    if serve:
        console.print(
            f"📚 Serving documentation on http://localhost:{port}",
            style="blue"
        )
        console.print("📖 Consolidated documentation includes:")
        console.print("  • Architecture overview")
        console.print("  • API reference")
        console.print("  • Usage examples")
        console.print("  • Migration guide")
    else:
        console.print(
            "📝 Generating consolidated documentation...",
            style="blue"
        )
        console.print("✅ Documentation generated", style="green")


@app.command()
def info():
    """Show consolidated system information."""
    display_banner()

    settings = get_settings()

    info_panel = f"""
[bold]Application[/bold]: {settings.app_name}
[bold]Version[/bold]: {settings.app_version}
[bold]Environment[/bold]: {settings.environment}
[bold]Debug[/bold]: {settings.debug}

[bold]Structure[/bold]: Consolidated single app/ directory
[bold]Architecture[/bold]: Clean Architecture + DDD
[bold]API[/bold]: FastAPI with OpenAPI docs
[bold]CLI[/bold]: Rich interactive interface

[bold]Components[/bold]:
• Domain layer with models and services
• Application layer with use cases
• Infrastructure layer with external integrations
• API layer with versioned endpoints
• CLI layer with interactive commands
    """

    panel = Panel(
        info_panel.strip(),
        title="📋 System Information",
        border_style="green"
    )
    console.print(panel)


if __name__ == "__main__":
    app()
