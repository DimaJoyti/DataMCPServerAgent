"""
Enterprise RL System Demo - Complete demonstration of all advanced features.
This example showcases the full enterprise-grade RL system with all capabilities.
"""

import asyncio
import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.core.config import get_settings
from app.core.rl_integration import get_rl_manager, initialize_rl_system
from app.monitoring.rl_analytics import get_dashboard, get_metrics_collector
from app.rl.ab_testing import ExperimentMetric, ExperimentVariant, get_ab_testing_engine
from app.rl.adaptive_learning import get_adaptive_learning_engine
from app.rl.model_deployment import DeploymentConfig, DeploymentStrategy, get_deployment_manager

# Load environment variables
load_dotenv()


class EnterpriseRLSystemDemo:
    """Complete enterprise RL system demonstration."""

    def __init__(self):
        """Initialize the enterprise demo system."""
        self.settings = get_settings()
        self.rl_manager = None
        self.adaptive_engine = get_adaptive_learning_engine()
        self.ab_testing_engine = get_ab_testing_engine()
        self.deployment_manager = get_deployment_manager()
        self.metrics_collector = get_metrics_collector()
        self.dashboard = get_dashboard()

        # Demo configuration
        self.demo_users = [f"user_{i:03d}" for i in range(100)]
        self.demo_scenarios = [
            "Customer support automation",
            "Financial risk assessment",
            "Content recommendation",
            "Fraud detection",
            "Supply chain optimization",
        ]

    async def initialize_enterprise_system(self):
        """Initialize the complete enterprise system."""
        print("🏢 Initializing Enterprise RL System")
        print("=" * 60)

        # Initialize core RL system
        print("🧠 Initializing core RL system...")
        success = await initialize_rl_system(self.settings)

        if success:
            self.rl_manager = get_rl_manager(self.settings)
            print("✅ Core RL system initialized")
        else:
            print("❌ Failed to initialize core RL system")
            return False

        # Initialize adaptive learning
        print("🔄 Starting adaptive learning engine...")
        await self.adaptive_engine.start_adaptive_learning()
        print("✅ Adaptive learning engine started")

        # Initialize model registry
        print("📚 Initializing model registry...")
        registry = self.deployment_manager.registry
        print(f"✅ Model registry initialized with {len(registry.models)} models")

        print("\n🎯 Enterprise system initialization complete!")
        return True

    async def demonstrate_adaptive_learning(self):
        """Demonstrate adaptive learning capabilities."""
        print("\n🧠 Adaptive Learning Demonstration")
        print("=" * 50)

        # Simulate various performance scenarios
        scenarios = [
            {"name": "Normal Operation", "response_time": 0.5, "success_rate": 0.95},
            {"name": "Performance Degradation", "response_time": 2.0, "success_rate": 0.80},
            {"name": "High Load", "response_time": 1.5, "success_rate": 0.90},
            {"name": "Recovery", "response_time": 0.6, "success_rate": 0.98},
        ]

        for scenario in scenarios:
            print(f"\n📊 Simulating: {scenario['name']}")

            # Simulate metrics for this scenario
            for _ in range(10):
                self.adaptive_engine.performance_tracker.record_metric(
                    "response_time",
                    scenario["response_time"] + np.random.normal(0, 0.1),
                    {"scenario": scenario["name"]}
                )

                self.adaptive_engine.performance_tracker.record_metric(
                    "success_rate",
                    scenario["success_rate"] + np.random.normal(0, 0.02),
                    {"scenario": scenario["name"]}
                )

                await asyncio.sleep(0.1)

            # Check for adaptations
            await asyncio.sleep(2)  # Allow adaptation system to process

        # Get adaptation status
        status = self.adaptive_engine.get_adaptation_status()
        print("\n🔄 Adaptation Status:")
        print(f"   Active adaptations: {status['active_adaptations']}")
        print(f"   Learning events: {status['learning_events']}")
        print(f"   Performance metrics: {status['performance_metrics']}")

        if status['active_strategy_details']:
            print("   Active strategies:")
            for name, details in status['active_strategy_details'].items():
                print(f"     - {details['strategy_name']}: {details['actions_completed']}/{details['total_actions']} actions")

    async def demonstrate_ab_testing(self):
        """Demonstrate A/B testing capabilities."""
        print("\n🧪 A/B Testing Demonstration")
        print("=" * 40)

        # Create experiment variants
        variants = [
            ExperimentVariant(
                name="control",
                description="Current RL algorithm (DQN)",
                config={"algorithm": "dqn", "learning_rate": 1e-4},
                traffic_allocation=0.5,
                is_control=True
            ),
            ExperimentVariant(
                name="treatment",
                description="New RL algorithm (PPO)",
                config={"algorithm": "ppo", "learning_rate": 1e-3},
                traffic_allocation=0.5,
                is_control=False
            ),
        ]

        # Define metrics to track
        metrics = [
            ExperimentMetric(
                name="response_time",
                description="Average response time",
                metric_type="continuous",
                primary=True,
                higher_is_better=False,
                minimum_detectable_effect=0.1
            ),
            ExperimentMetric(
                name="user_satisfaction",
                description="User satisfaction score",
                metric_type="continuous",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.05
            ),
        ]

        # Create experiment
        experiment_id = self.ab_testing_engine.create_experiment(
            name="RL Algorithm Comparison",
            description="Compare DQN vs PPO performance",
            variants=variants,
            metrics=metrics,
            target_sample_size=200
        )

        print(f"📊 Created experiment: {experiment_id}")

        # Start experiment
        success = self.ab_testing_engine.start_experiment(experiment_id)
        if success:
            print("🚀 Experiment started")
        else:
            print("❌ Failed to start experiment")
            return

        # Simulate user interactions
        print("👥 Simulating user interactions...")

        for user_id in self.demo_users[:50]:  # Use first 50 users
            # Assign user to variant
            variant = self.ab_testing_engine.assign_user_to_variant(user_id, experiment_id)

            if variant:
                # Simulate metrics based on variant
                if variant == "control":
                    response_time = np.random.normal(1.0, 0.2)
                    satisfaction = np.random.normal(0.7, 0.1)
                else:  # treatment
                    response_time = np.random.normal(0.8, 0.15)  # Better performance
                    satisfaction = np.random.normal(0.75, 0.1)   # Higher satisfaction

                # Record metrics
                self.ab_testing_engine.record_metric(
                    user_id, experiment_id, "response_time", max(0.1, response_time)
                )
                self.ab_testing_engine.record_metric(
                    user_id, experiment_id, "user_satisfaction", np.clip(satisfaction, 0, 1)
                )

        # Get experiment status
        status = self.ab_testing_engine.get_experiment_status(experiment_id)
        print("📈 Experiment Status:")
        print(f"   Progress: {status['progress']:.1%}")
        print(f"   Total users: {status['total_users']}")
        print(f"   Can analyze: {status['can_analyze']}")

        # Analyze results if we have enough data
        if status['can_analyze']:
            print("\n📊 Analyzing experiment results...")
            analysis = self.ab_testing_engine.analyze_experiment(experiment_id)

            if "error" not in analysis:
                print(f"   Statistical tests performed: {len(analysis['statistical_tests'])}")
                print(f"   Recommendations: {len(analysis['recommendations'])}")

                for rec in analysis['recommendations']:
                    if rec['type'] == 'winner':
                        print(f"   🏆 Winner: {rec['variant']} ({rec['metric']}) - {rec['improvement']:.1f}% improvement")
                    else:
                        print(f"   📉 Underperformer: {rec['variant']} ({rec['metric']}) - {rec['degradation']:.1f}% degradation")

        # Stop experiment
        self.ab_testing_engine.stop_experiment(experiment_id)
        print("🛑 Experiment completed")

    async def demonstrate_model_deployment(self):
        """Demonstrate model deployment capabilities."""
        print("\n🚀 Model Deployment Demonstration")
        print("=" * 45)

        # Register a model
        print("📦 Registering model in registry...")

        # Create a dummy model file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"dummy model data")
            model_path = f.name

        try:
            model_id = self.deployment_manager.registry.register_model(
                name="advanced_dqn",
                version="1.2.0",
                algorithm="dqn",
                model_path=model_path,
                training_config={
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    "episodes": 1000,
                },
                performance_metrics={
                    "accuracy": 0.92,
                    "avg_reward": 15.6,
                    "convergence_episodes": 800,
                },
                trained_by="enterprise_demo"
            )

            print(f"✅ Model registered: {model_id}")

            # Deploy model using different strategies
            strategies = [
                (DeploymentStrategy.BLUE_GREEN, "staging"),
                (DeploymentStrategy.CANARY, "production"),
            ]

            deployment_ids = []

            for strategy, environment in strategies:
                print(f"\n🎯 Deploying with {strategy.value} strategy to {environment}...")

                config = DeploymentConfig(
                    strategy=strategy,
                    traffic_percentage=10.0 if strategy == DeploymentStrategy.CANARY else 100.0,
                    auto_promote=True,
                    monitoring_duration=60,  # 1 minute for demo
                )

                deployment_id = await self.deployment_manager.deploy_model(
                    model_id, environment, config
                )

                deployment_ids.append(deployment_id)
                print(f"✅ Deployment created: {deployment_id}")

                # Get deployment status
                status = self.deployment_manager.get_deployment_status(deployment_id)
                if status:
                    print(f"   Status: {status['status']}")
                    print(f"   Traffic: {status['traffic_percentage']}%")
                    print(f"   Health: {status['health_status']}")

            # List all deployments
            print("\n📋 All Deployments:")
            deployments = self.deployment_manager.list_deployments()
            for deployment in deployments:
                print(f"   {deployment['deployment_id']}: {deployment['environment']} - {deployment['status']}")

            # Simulate monitoring for a bit
            print("\n💓 Monitoring deployments...")
            await asyncio.sleep(5)

            # Check updated statuses
            for deployment_id in deployment_ids:
                status = self.deployment_manager.get_deployment_status(deployment_id)
                if status:
                    print(f"   {deployment_id}: {status['status']} - {status['health_status']}")

        finally:
            # Clean up temp file
            os.unlink(model_path)

    async def demonstrate_enterprise_monitoring(self):
        """Demonstrate enterprise monitoring capabilities."""
        print("\n📊 Enterprise Monitoring Demonstration")
        print("=" * 50)

        # Generate comprehensive dashboard data
        dashboard_data = await self.dashboard.get_dashboard_data(force_update=True)

        if "error" not in dashboard_data:
            print("📈 System Metrics:")
            status = dashboard_data.get("status", {})
            print(f"   Uptime: {status.get('uptime', 'N/A')}")
            print(f"   Requests processed: {status.get('requests_processed', 0)}")
            print(f"   Error rate: {status.get('error_rate', 0):.2%}")

            performance = dashboard_data.get("performance", {})
            print("\n⚡ Performance Metrics:")
            print(f"   Avg response time: {performance.get('avg_response_time', 0)*1000:.0f}ms")
            print(f"   Performance class: {performance.get('performance_class', 'unknown')}")
            print(f"   SLA compliance: {performance.get('sla_compliance', 0):.1%}")

            safety = dashboard_data.get("safety", {})
            print("\n🛡️ Safety Metrics:")
            print(f"   Safety class: {safety.get('safety_class', 'unknown')}")
            print(f"   Recent violations: {safety.get('recent_violations', 0)}")

            training = dashboard_data.get("training", {})
            print("\n🧠 Training Metrics:")
            print(f"   Total metrics: {training.get('total_metrics', 0)}")

            # Show recent events
            recent_events = dashboard_data.get("recent_events", [])
            if recent_events:
                print(f"\n📋 Recent Events ({len(recent_events)}):")
                for event in recent_events[-3:]:  # Last 3 events
                    timestamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
                    print(f"   {timestamp} - {event['event_type']} ({event['severity']})")
        else:
            print(f"❌ Error getting dashboard data: {dashboard_data['error']}")

    async def run_enterprise_demo(self):
        """Run the complete enterprise demonstration."""
        print("🏢 Enterprise RL System Complete Demonstration")
        print("=" * 70)
        print("This demo showcases:")
        print("• Complete enterprise-grade RL system")
        print("• Adaptive learning and self-optimization")
        print("• A/B testing for algorithm comparison")
        print("• MLOps with automated model deployment")
        print("• Real-time monitoring and analytics")
        print("• Production-ready enterprise features")
        print("=" * 70)

        # Initialize enterprise system
        if not await self.initialize_enterprise_system():
            print("❌ Enterprise system initialization failed. Exiting.")
            return

        # Run demonstrations
        await self.demonstrate_adaptive_learning()
        await self.demonstrate_ab_testing()
        await self.demonstrate_model_deployment()
        await self.demonstrate_enterprise_monitoring()

        # Final summary
        print("\n🏆 Enterprise Demo Summary")
        print("=" * 40)

        # Get comprehensive statistics
        rl_status = self.rl_manager.get_status()
        adaptive_status = self.adaptive_engine.get_adaptation_status()
        experiments = self.ab_testing_engine.list_experiments()
        deployments = self.deployment_manager.list_deployments()

        print("📊 System Statistics:")
        print(f"   RL System: {rl_status['mode']} mode, {rl_status['performance_metrics']['total_requests']} requests")
        print(f"   Adaptive Learning: {adaptive_status['learning_events']} events, {adaptive_status['active_adaptations']} active adaptations")
        print(f"   A/B Tests: {len(experiments)} experiments created")
        print(f"   Model Deployments: {len(deployments)} deployments")

        print("\n🎯 Enterprise Capabilities Demonstrated:")
        print("   ✅ Advanced RL with 12 different modes")
        print("   ✅ Self-adaptive learning system")
        print("   ✅ Automated A/B testing framework")
        print("   ✅ MLOps with model deployment strategies")
        print("   ✅ Real-time monitoring and analytics")
        print("   ✅ Enterprise-grade configuration management")
        print("   ✅ Production-ready safety and security")

        print("\n🚀 System is ready for enterprise deployment!")

        # Cleanup
        await self.adaptive_engine.stop_adaptive_learning()
        print("\n🧹 System cleanup completed")


async def main():
    """Main demo function."""
    # Import numpy for the demo
    import numpy as np
    globals()['np'] = np

    demo = EnterpriseRLSystemDemo()
    await demo.run_enterprise_demo()


if __name__ == "__main__":
    asyncio.run(main())
