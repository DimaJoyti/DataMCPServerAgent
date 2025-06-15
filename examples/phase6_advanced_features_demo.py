"""
Phase 6 Advanced Features Demo - Federated Learning, Cloud Integration, and Real-Time Monitoring.
This example demonstrates the most advanced enterprise features added in Phase 6.
"""

import asyncio
import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.cloud.cloud_integration import CloudProvider, DeploymentEnvironment, get_cloud_orchestrator
from app.core.config import get_settings
from app.monitoring.real_time_monitoring import get_real_time_monitor
from app.rl.federated_learning import PrivacyLevel, create_federated_coordinator
from app.scaling.auto_scaling import ScalingPolicy, create_auto_scaler

# Load environment variables
load_dotenv()


class Phase6AdvancedDemo:
    """Demonstrates Phase 6 advanced features."""

    def __init__(self):
        """Initialize the advanced demo system."""
        self.settings = get_settings()

        # Initialize components
        self.federated_coordinator = None
        self.cloud_orchestrator = get_cloud_orchestrator()
        self.auto_scaler = None
        self.real_time_monitor = get_real_time_monitor()

        # Demo configuration
        self.demo_organizations = [
            {"name": "TechCorp", "data_size": 10000},
            {"name": "DataInc", "data_size": 8000},
            {"name": "AILabs", "data_size": 12000},
            {"name": "MLSystems", "data_size": 9000},
        ]

    async def demonstrate_federated_learning(self):
        """Demonstrate federated learning capabilities."""
        print("\n🤝 Federated Learning Demonstration")
        print("=" * 50)

        # Create federated learning coordinator
        federation_id = "enterprise_federation_2024"
        self.federated_coordinator = create_federated_coordinator(
            federation_id=federation_id,
            privacy_level=PrivacyLevel.DIFFERENTIAL,
            min_participants=2,
            max_rounds=5
        )

        print(f"📋 Created federation: {federation_id}")
        print(f"🔒 Privacy level: {PrivacyLevel.DIFFERENTIAL.value}")

        # Register participants from different organizations
        print(f"\n👥 Registering {len(self.demo_organizations)} participants...")

        for i, org in enumerate(self.demo_organizations):
            participant_id = f"participant_{i+1}"
            success = self.federated_coordinator.register_participant(
                participant_id=participant_id,
                name=f"{org['name']} AI Division",
                organization=org["name"],
                endpoint=f"https://{org['name'].lower()}.ai/federated",
                data_size=org["data_size"]
            )

            if success:
                print(f"  ✅ {org['name']}: {org['data_size']} samples")
            else:
                print(f"  ❌ Failed to register {org['name']}")

        # Create mock neural network model
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        initial_model = SimpleModel()

        # Start federated learning
        print("\n🚀 Starting federated learning...")
        success = self.federated_coordinator.start_federation(initial_model)

        if success:
            print("✅ Federation started successfully")

            # Run federated learning rounds
            print("\n🔄 Running federated learning rounds...")

            for round_num in range(3):
                print(f"\n📊 Round {round_num + 1}/3:")

                fed_round = await self.federated_coordinator.run_federated_round()

                if fed_round:
                    print("  ✅ Round completed")
                    print(f"  👥 Participants: {len(fed_round.participants)}")
                    print(f"  📈 Participation rate: {fed_round.aggregated_metrics.get('participation_rate', 0):.1%}")
                    print(f"  🔒 Privacy budget used: {fed_round.privacy_metrics.get('epsilon_spent', 0):.3f}")
                else:
                    print("  ❌ Round failed")
                    break

            # Get federation status
            status = self.federated_coordinator.get_federation_status()
            print("\n📊 Federation Summary:")
            print(f"  Status: {status['status']}")
            print(f"  Participants: {status['participants']}")
            print(f"  Rounds completed: {status['rounds_completed']}")
            print(f"  Privacy level: {status['privacy_level']}")

            # Stop federation
            await self.federated_coordinator.stop_federation()
            print("🛑 Federation completed")
        else:
            print("❌ Failed to start federation")

    async def demonstrate_cloud_integration(self):
        """Demonstrate cloud integration capabilities."""
        print("\n☁️ Cloud Integration Demonstration")
        print("=" * 45)

        # Demonstrate multi-cloud deployment
        cloud_providers = [
            (CloudProvider.AWS, "AWS deployment with SageMaker"),
            (CloudProvider.AZURE, "Azure deployment with ML Studio"),
            (CloudProvider.GCP, "GCP deployment with Vertex AI"),
        ]

        deployment_ids = []

        for provider, description in cloud_providers:
            print(f"\n🚀 {description}...")

            config = {
                "region": "us-east-1" if provider == CloudProvider.AWS else "eastus" if provider == CloudProvider.AZURE else "us-central1",
                "training_instance": "ml.m5.large" if provider == CloudProvider.AWS else "Standard_D2s_v3" if provider == CloudProvider.AZURE else "n1-standard-4",
                "deploy_endpoint": True,
                "image": "datamcp/rl-system:latest",
                "training_cost": 1.5,
            }

            deployment_id = await self.cloud_orchestrator.deploy_rl_system(
                deployment_name=f"datamcp-{provider.value}",
                environment=DeploymentEnvironment.STAGING,
                provider=provider,
                config=config
            )

            deployment_ids.append(deployment_id)

            # Get deployment status
            status = self.cloud_orchestrator.get_deployment_status(deployment_id)
            if status:
                print(f"  ✅ Deployment: {deployment_id}")
                print(f"  📍 Region: {status['config'].get('region', 'N/A')}")
                print(f"  🔗 Endpoints: {len(status.get('endpoints', {}))}")
                print(f"  ⏱️ Uptime: {status['uptime']:.1f}s")
            else:
                print("  ❌ Deployment failed")

        # Demonstrate scaling
        if deployment_ids:
            print("\n📈 Demonstrating auto-scaling...")

            scale_config = {
                "target_capacity": 3,
                "scaling_policy": "target_tracking",
                "metric": "cpu_utilization",
                "target_value": 70.0,
            }

            for deployment_id in deployment_ids[:1]:  # Scale first deployment
                success = await self.cloud_orchestrator.scale_deployment(
                    deployment_id, scale_config
                )

                if success:
                    print(f"  ✅ Scaled deployment {deployment_id}")
                else:
                    print(f"  ❌ Failed to scale deployment {deployment_id}")

        # Monitor costs
        print("\n💰 Cloud Cost Monitoring...")
        cost_summary = await self.cloud_orchestrator.monitor_costs()

        print(f"  Total cost: ${cost_summary['total_cost']:.2f}")
        print(f"  Active resources: {cost_summary['active_resources']}")
        print(f"  Active deployments: {cost_summary['active_deployments']}")

        if cost_summary['cost_by_provider']:
            print("  Cost by provider:")
            for provider, cost in cost_summary['cost_by_provider'].items():
                print(f"    {provider}: ${cost:.2f}")

    async def demonstrate_auto_scaling(self):
        """Demonstrate intelligent auto-scaling."""
        print("\n📈 Auto-Scaling Demonstration")
        print("=" * 40)

        # Create auto-scaler for RL service
        service_name = "datamcp-rl-service"
        self.auto_scaler = create_auto_scaler(
            service_name=service_name,
            scaling_policy=ScalingPolicy.HYBRID,
            min_instances=2,
            max_instances=10
        )

        print(f"🔧 Created auto-scaler for {service_name}")
        print(f"📊 Policy: {ScalingPolicy.HYBRID.value}")
        print("📏 Range: 2-10 instances")

        # Start auto-scaling
        await self.auto_scaler.start_auto_scaling()
        print("🚀 Auto-scaling started")

        # Simulate workload patterns
        print("\n🎭 Simulating workload patterns...")

        workload_scenarios = [
            {"name": "Normal Load", "duration": 30, "cpu_target": 50},
            {"name": "High Load", "duration": 45, "cpu_target": 85},
            {"name": "Peak Load", "duration": 30, "cpu_target": 95},
            {"name": "Cool Down", "duration": 60, "cpu_target": 30},
        ]

        for scenario in workload_scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            print(f"   Duration: {scenario['duration']}s")
            print(f"   Target CPU: {scenario['cpu_target']}%")

            # Let auto-scaler run for scenario duration
            start_time = time.time()
            while time.time() - start_time < scenario['duration']:
                await asyncio.sleep(10)

                # Get current status
                status = self.auto_scaler.get_scaling_status()
                print(f"   Instances: {status['current_instances']}, "
                      f"CPU: {status['current_metrics'].get('cpu_utilization', 0):.1f}%")

        # Get final scaling status
        final_status = self.auto_scaler.get_scaling_status()
        print("\n📊 Auto-Scaling Summary:")
        print(f"  Current instances: {final_status['current_instances']}")
        print(f"  Total scaling events: {final_status['total_scaling_events']}")
        print(f"  Scaling efficiency: {final_status['scaling_efficiency']:.1%}")
        print(f"  Active rules: {len([r for r in final_status['scaling_rules'].values() if r['enabled']])}")

        # Show predictions
        predictions = self.auto_scaler.get_predictions(horizon_minutes=30)
        print("\n🔮 Workload Predictions (30 min):")
        for metric, (value, confidence) in predictions.items():
            print(f"  {metric}: {value:.1f} (confidence: {confidence:.1%})")

        # Stop auto-scaling
        await self.auto_scaler.stop_auto_scaling()
        print("🛑 Auto-scaling stopped")

    async def demonstrate_real_time_monitoring(self):
        """Demonstrate real-time monitoring capabilities."""
        print("\n🔍 Real-Time Monitoring Demonstration")
        print("=" * 50)

        # Start real-time monitoring
        await self.real_time_monitor.start_monitoring()
        print("🚀 Real-time monitoring started")
        print("📡 WebSocket server available at ws://localhost:8765")

        # Let monitoring run for a while
        print("\n📊 Collecting metrics for 60 seconds...")

        for i in range(6):  # 6 iterations of 10 seconds each
            await asyncio.sleep(10)

            # Get current dashboard data
            dashboard = self.real_time_monitor.get_monitoring_dashboard()

            current_metrics = dashboard.get('current_metrics', {})
            system = current_metrics.get('system', {})
            app = current_metrics.get('application', {})

            print(f"  [{i+1}/6] CPU: {system.get('cpu_percent', 0):.1f}%, "
                  f"Memory: {system.get('memory_percent', 0):.1f}%, "
                  f"Response: {app.get('response_time_avg', 0):.0f}ms, "
                  f"Errors: {app.get('error_rate', 0):.1f}%")

        # Get final monitoring summary
        final_dashboard = self.real_time_monitor.get_monitoring_dashboard()

        print("\n📊 Monitoring Summary:")
        print(f"  Status: {final_dashboard['status']}")
        print(f"  WebSocket clients: {final_dashboard['websocket_clients']}")
        print(f"  Data points collected: {len(final_dashboard['performance_history'])}")

        # Show alerts summary
        alerts = final_dashboard.get('alerts', {})
        print(f"  Active alerts: {alerts.get('active_alerts', 0)}")

        if alerts.get('severity_breakdown'):
            print("  Alert breakdown:")
            for severity, count in alerts['severity_breakdown'].items():
                print(f"    {severity}: {count}")

        # Show trends
        trends = final_dashboard.get('trends', {})
        if trends:
            print("  Performance trends:")
            for metric, trend in trends.items():
                print(f"    {metric}: {trend}")

        # Stop monitoring
        await self.real_time_monitor.stop_monitoring()
        print("🛑 Real-time monitoring stopped")

    async def run_phase6_demo(self):
        """Run the complete Phase 6 demonstration."""
        print("🚀 Phase 6 Advanced Features Demonstration")
        print("=" * 70)
        print("This demo showcases the most advanced enterprise features:")
        print("• Federated Learning with Privacy Protection")
        print("• Multi-Cloud Integration and Deployment")
        print("• Intelligent Auto-Scaling with Predictions")
        print("• Real-Time Monitoring and Alerting")
        print("=" * 70)

        try:
            # Run all demonstrations
            await self.demonstrate_federated_learning()
            await self.demonstrate_cloud_integration()
            await self.demonstrate_auto_scaling()
            await self.demonstrate_real_time_monitoring()

            # Final summary
            print("\n🏆 Phase 6 Demo Summary")
            print("=" * 40)

            print("🎯 Advanced Features Demonstrated:")
            print("   ✅ Federated Learning with differential privacy")
            print("   ✅ Multi-cloud deployment (AWS, Azure, GCP)")
            print("   ✅ Intelligent auto-scaling with predictions")
            print("   ✅ Real-time monitoring with WebSocket updates")
            print("   ✅ Cost monitoring and optimization")
            print("   ✅ Alert management and notifications")

            print("\n🚀 Enterprise Capabilities:")
            print("   • Privacy-preserving collaborative learning")
            print("   • Cloud-agnostic deployment strategies")
            print("   • Predictive resource management")
            print("   • Real-time system observability")
            print("   • Automated cost optimization")
            print("   • Proactive alerting and monitoring")

            print("\n🎉 Phase 6 demonstration completed successfully!")
            print("🌟 DataMCPServerAgent now includes the most advanced")
            print("    enterprise features available in the industry!")

        except Exception as e:
            print(f"❌ Error in Phase 6 demo: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function."""
    demo = Phase6AdvancedDemo()
    await demo.run_phase6_demo()


if __name__ == "__main__":
    asyncio.run(main())
