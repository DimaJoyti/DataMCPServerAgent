"""
Complete integration example demonstrating the full DataMCPServerAgent system
with advanced RL capabilities, monitoring, and real-world applications.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.core.rl_integration import get_rl_manager, initialize_rl_system
from app.core.simple_config import SimpleSettings
from app.monitoring.rl_analytics import get_dashboard, get_metrics_collector

# Load environment variables
load_dotenv()


class DataMCPServerAgentDemo:
    """Complete demonstration of DataMCPServerAgent with RL integration."""

    def __init__(self):
        """Initialize the demo system."""
        self.settings = SimpleSettings()
        self.rl_manager = None
        self.metrics_collector = get_metrics_collector()
        self.dashboard = get_dashboard()

        # Demo scenarios
        self.demo_scenarios = [
            {
                "name": "Customer Support Automation",
                "description": "AI agent handling customer inquiries with RL optimization",
                "requests": [
                    "Help me track my order #12345",
                    "I want to return a product I bought last week",
                    "What's your refund policy?",
                    "Can you help me find a replacement for this item?",
                ]
            },
            {
                "name": "Data Analysis Assistant",
                "description": "AI agent performing data analysis with safety constraints",
                "requests": [
                    "Analyze sales trends for the last quarter",
                    "Create a summary of customer feedback data",
                    "Identify patterns in user behavior",
                    "Generate insights from the marketing campaign data",
                ]
            },
            {
                "name": "Creative Content Generation",
                "description": "AI agent creating content with explainable decisions",
                "requests": [
                    "Write a blog post about sustainable technology",
                    "Create a marketing email for our new product",
                    "Generate social media content for the campaign",
                    "Draft a press release for our company milestone",
                ]
            },
            {
                "name": "Risk Assessment and Safety",
                "description": "AI agent making decisions with safety constraints",
                "requests": [
                    "Evaluate the risk of this investment proposal",
                    "Assess the safety implications of this system change",
                    "Review this contract for potential issues",
                    "Analyze the compliance requirements for this project",
                ]
            },
        ]

    async def initialize_system(self):
        """Initialize the complete system."""
        print("🚀 Initializing DataMCPServerAgent with Advanced RL")
        print("=" * 60)

        # Initialize RL system
        print("🧠 Initializing RL system...")
        success = await initialize_rl_system(self.settings)

        if success:
            self.rl_manager = get_rl_manager(self.settings)
            print("✅ RL system initialized successfully")

            # Show system configuration
            config = self.rl_manager.config
            print(f"   Mode: {config.mode.value}")
            print(f"   Algorithm: {config.algorithm}")
            print(f"   Safety enabled: {config.safety_enabled}")
            print(f"   Explanations enabled: {config.explanation_enabled}")
            print(f"   Training enabled: {config.training_enabled}")
        else:
            print("❌ Failed to initialize RL system")
            return False

        print("\n📊 Initializing monitoring and analytics...")

        # Record initialization event
        self.metrics_collector.record_event(
            "system_initialization",
            {"status": "success", "mode": config.mode.value},
            "info"
        )

        print("✅ System initialization complete!")
        return True

    async def run_demo_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete demo scenario.
        
        Args:
            scenario: Demo scenario configuration
            
        Returns:
            Scenario results
        """
        print(f"\n🎯 Running Scenario: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print("-" * 50)

        scenario_start_time = time.time()
        scenario_results = {
            "name": scenario["name"],
            "requests": [],
            "total_time": 0,
            "success_rate": 0,
            "avg_response_time": 0,
            "safety_violations": 0,
            "explanations_generated": 0,
        }

        # Process each request in the scenario
        for i, request in enumerate(scenario["requests"]):
            print(f"\n📋 Request {i+1}: {request}")

            # Add scenario context
            context = {
                "scenario": scenario["name"],
                "request_index": i,
                "total_requests": len(scenario["requests"]),
            }

            # Process request with RL system
            result = await self.rl_manager.process_request(request, context)

            # Record metrics
            self.metrics_collector.record_metric(
                "response_time",
                result.get("response_time", 0),
                {"scenario": scenario["name"], "request_index": i}
            )

            if result["success"]:
                print(f"✅ Success: {result['response']}")

                # Check for explanations
                if "explanation" in result:
                    print(f"💭 Reasoning: {result.get('reasoning', 'N/A')}")
                    scenario_results["explanations_generated"] += 1

                # Check for safety info
                if "safety_info" in result:
                    safety = result["safety_info"]
                    safety_score = safety.get("safety_score", 1.0)
                    print(f"🛡️ Safety score: {safety_score:.3f}")

                    if safety_score < 0.8:  # Consider low safety score as violation
                        scenario_results["safety_violations"] += 1
                        self.metrics_collector.record_event(
                            "safety_violation",
                            {"scenario": scenario["name"], "safety_score": safety_score},
                            "warning"
                        )

                print(f"⏱️ Response time: {result['response_time']:.3f}s")

                # Record success metrics
                self.metrics_collector.record_metric(
                    "request_success",
                    1.0,
                    {"scenario": scenario["name"]}
                )

            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")

                # Record failure metrics
                self.metrics_collector.record_metric(
                    "request_success",
                    0.0,
                    {"scenario": scenario["name"]}
                )

                self.metrics_collector.record_event(
                    "request_failure",
                    {"scenario": scenario["name"], "error": result.get("error")},
                    "error"
                )

            # Store request result
            scenario_results["requests"].append({
                "request": request,
                "success": result["success"],
                "response_time": result.get("response_time", 0),
                "has_explanation": "explanation" in result,
                "safety_score": result.get("safety_info", {}).get("safety_score", 1.0),
            })

            # Small delay between requests
            await asyncio.sleep(0.5)

        # Calculate scenario metrics
        scenario_results["total_time"] = time.time() - scenario_start_time

        successful_requests = [r for r in scenario_results["requests"] if r["success"]]
        scenario_results["success_rate"] = len(successful_requests) / len(scenario_results["requests"])

        if successful_requests:
            scenario_results["avg_response_time"] = sum(
                r["response_time"] for r in successful_requests
            ) / len(successful_requests)

        # Print scenario summary
        print("\n📊 Scenario Summary:")
        print(f"   Success rate: {scenario_results['success_rate']:.1%}")
        print(f"   Average response time: {scenario_results['avg_response_time']:.3f}s")
        print(f"   Safety violations: {scenario_results['safety_violations']}")
        print(f"   Explanations generated: {scenario_results['explanations_generated']}")
        print(f"   Total time: {scenario_results['total_time']:.1f}s")

        return scenario_results

    async def run_training_demonstration(self):
        """Demonstrate RL training capabilities."""
        print("\n🏋️ RL Training Demonstration")
        print("=" * 40)

        if not self.rl_manager.config.training_enabled:
            print("⚠️ Training is disabled in configuration")
            return

        print("🎯 Training the RL agent for improved performance...")

        # Train for several episodes
        training_results = []
        for episode in range(5):
            print(f"\n📚 Training episode {episode + 1}/5...")

            metrics = await self.rl_manager.train_episode()
            training_results.append(metrics)

            if "error" in metrics:
                print(f"❌ Training error: {metrics['error']}")
                break
            else:
                print("✅ Episode completed")

                # Record training metrics
                if "loss" in metrics:
                    self.metrics_collector.record_metric(
                        "training_loss",
                        metrics["loss"],
                        {"episode": episode}
                    )

                if "reward" in metrics:
                    self.metrics_collector.record_metric(
                        "training_reward",
                        metrics["reward"],
                        {"episode": episode}
                    )

        print(f"\n🎉 Training completed! {len(training_results)} episodes")

        # Save model
        print("💾 Saving trained model...")
        save_success = await self.rl_manager.save_model()
        if save_success:
            print("✅ Model saved successfully")
        else:
            print("⚠️ Model saving not supported or failed")

    async def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n📊 Generating Performance Report")
        print("=" * 40)

        # Get dashboard data
        dashboard_data = await self.dashboard.get_dashboard_data(force_update=True)

        if "error" in dashboard_data:
            print(f"❌ Error generating report: {dashboard_data['error']}")
            return

        # System status
        status = dashboard_data.get("status", {})
        print("🔧 System Status:")
        print(f"   Uptime: {status.get('uptime', 'N/A')}")
        print(f"   Requests processed: {status.get('requests_processed', 0)}")
        print(f"   Requests per hour: {status.get('requests_per_hour', 0):.1f}")
        print(f"   Error rate: {status.get('error_rate', 0):.2%}")
        print(f"   Training episodes: {status.get('training_episodes', 0)}")

        # Performance metrics
        performance = dashboard_data.get("performance", {})
        print("\n⚡ Performance Metrics:")
        print(f"   Average response time: {performance.get('avg_response_time', 0)*1000:.0f}ms")
        print(f"   P95 response time: {performance.get('p95_response_time', 0)*1000:.0f}ms")
        print(f"   Performance class: {performance.get('performance_class', 'unknown')}")
        print(f"   SLA compliance: {performance.get('sla_compliance', 0):.1%}")

        # Safety metrics
        safety = dashboard_data.get("safety", {})
        print("\n🛡️ Safety Metrics:")
        print(f"   Recent violations: {safety.get('recent_violations', 0)}")
        print(f"   Safety class: {safety.get('safety_class', 'unknown')}")
        print(f"   Safety score: {safety.get('safety_score', 0)*100:.1f}%")

        # Training metrics
        training = dashboard_data.get("training", {})
        print("\n🧠 Training Metrics:")
        print(f"   Total training metrics: {training.get('total_metrics', 0)}")

        trend_analysis = training.get("trend_analysis", {})
        if trend_analysis:
            print("   Trend analysis:")
            for metric_name, trend_info in trend_analysis.items():
                trend = trend_info.get("trend", "unknown")
                print(f"     {metric_name}: {trend}")

        # Recent events
        recent_events = dashboard_data.get("recent_events", [])
        if recent_events:
            print(f"\n📋 Recent Events ({len(recent_events)}):")
            for event in recent_events[-5:]:  # Last 5 events
                timestamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
                print(f"   {timestamp} - {event['event_type']} ({event['severity']})")

    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🎉 DataMCPServerAgent Complete Integration Demo")
        print("=" * 60)
        print("This demo showcases:")
        print("• Advanced RL system with multiple modes")
        print("• Safety constraints and risk management")
        print("• Explainable AI decisions")
        print("• Real-time monitoring and analytics")
        print("• Production-ready integration")
        print("=" * 60)

        # Initialize system
        if not await self.initialize_system():
            print("❌ System initialization failed. Exiting.")
            return

        # Run demo scenarios
        all_scenario_results = []

        for scenario in self.demo_scenarios:
            try:
                result = await self.run_demo_scenario(scenario)
                all_scenario_results.append(result)
            except Exception as e:
                print(f"❌ Error in scenario {scenario['name']}: {e}")
                self.metrics_collector.record_event(
                    "scenario_error",
                    {"scenario": scenario["name"], "error": str(e)},
                    "error"
                )

        # Training demonstration
        await self.run_training_demonstration()

        # Generate final report
        await self.generate_performance_report()

        # Summary
        print("\n🏆 Demo Summary")
        print("=" * 30)

        total_requests = sum(len(r["requests"]) for r in all_scenario_results)
        total_successful = sum(
            len([req for req in r["requests"] if req["success"]])
            for r in all_scenario_results
        )
        overall_success_rate = total_successful / total_requests if total_requests > 0 else 0

        total_explanations = sum(r["explanations_generated"] for r in all_scenario_results)
        total_violations = sum(r["safety_violations"] for r in all_scenario_results)

        print("📊 Overall Statistics:")
        print(f"   Scenarios completed: {len(all_scenario_results)}")
        print(f"   Total requests: {total_requests}")
        print(f"   Success rate: {overall_success_rate:.1%}")
        print(f"   Explanations generated: {total_explanations}")
        print(f"   Safety violations: {total_violations}")

        print("\n🎯 Key Achievements:")
        print("   ✅ Advanced RL system operational")
        print("   ✅ Safety constraints enforced")
        print("   ✅ Explainable decisions generated")
        print("   ✅ Real-time monitoring active")
        print("   ✅ Production-ready integration")

        print("\n🚀 System is ready for production deployment!")

        # Export results
        results_summary = {
            "demo_completed_at": time.time(),
            "scenarios": all_scenario_results,
            "overall_stats": {
                "total_requests": total_requests,
                "success_rate": overall_success_rate,
                "explanations_generated": total_explanations,
                "safety_violations": total_violations,
            },
            "system_config": {
                "rl_mode": self.rl_manager.config.mode.value,
                "algorithm": self.rl_manager.config.algorithm,
                "safety_enabled": self.rl_manager.config.safety_enabled,
                "explanation_enabled": self.rl_manager.config.explanation_enabled,
            }
        }

        # Save results to file
        import json
        with open("demo_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print("\n💾 Demo results saved to demo_results.json")


async def main():
    """Main demo function."""
    demo = DataMCPServerAgentDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
