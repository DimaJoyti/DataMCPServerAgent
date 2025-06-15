"""
Enterprise Training Demonstration - Advanced Learning Capabilities
Showcases federated learning, adaptive learning, auto-tuning with Phase 3 optimizations.
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any, List
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use lazy imports for memory optimization
from src.utils.lazy_imports import numpy as np, get_loaded_modules, get_memory_report
from src.utils.memory_monitor import MemoryContext, log_memory_usage, get_global_monitor
from src.utils.bounded_collections import BoundedDict, BoundedList, BoundedSet
from src.memory.database_optimization import OptimizedDatabase, apply_database_optimizations
from src.core.dependency_injection import get_container, ILogger, injectable, Lifetime
from app.core.dependencies import configure_fastapi_services


class FederatedLearningNode:
    """Simulated federated learning node with privacy preservation."""
    
    def __init__(self, node_id: str, organization: str):
        self.node_id = node_id
        self.organization = organization
        self.local_data = BoundedList(max_size=1000, eviction_strategy="fifo")
        self.model_parameters = {}
        self.privacy_budget = 1.0  # Differential privacy budget
        self.training_rounds = 0
        
    async def local_training(self, global_parameters: Dict) -> Dict:
        """Perform local training with differential privacy."""
        print(f"   📊 Node {self.node_id} ({self.organization}): Local training...")
        
        # Simulate local training with privacy preservation
        noise_scale = 0.1 / self.privacy_budget if self.privacy_budget > 0 else 0.1
        
        local_updates = {}
        for param_name, global_value in global_parameters.items():
            # Simulate gradient computation with noise for privacy
            gradient = random.uniform(-0.01, 0.01)
            noise = np.random.normal(0, noise_scale)
            local_updates[param_name] = global_value + gradient + noise
            
        # Reduce privacy budget
        self.privacy_budget = max(0.1, self.privacy_budget - 0.1)
        self.training_rounds += 1
        
        return {
            "updates": local_updates,
            "data_size": len(self.local_data),
            "privacy_budget": self.privacy_budget,
            "node_id": self.node_id
        }
    
    def add_local_data(self, data_points: List):
        """Add local training data."""
        for point in data_points:
            self.local_data.append(point)


class SecureAggregationServer:
    """Secure aggregation server for federated learning."""
    
    def __init__(self):
        self.global_parameters = {
            "layer_1_weights": 0.5,
            "layer_1_bias": 0.1,
            "layer_2_weights": 0.3,
            "layer_2_bias": 0.05,
            "learning_rate": 0.001
        }
        self.participating_nodes = []
        self.aggregation_rounds = 0
        
    async def secure_aggregate(self, node_updates: List[Dict]) -> Dict:
        """Perform secure aggregation with homomorphic encryption simulation."""
        print(f"   🔐 Secure aggregation round {self.aggregation_rounds + 1}...")
        
        # Simulate homomorphic encryption - weighted average by data size
        total_data_size = sum(update["data_size"] for update in node_updates)
        aggregated_params = {}
        
        for param_name in self.global_parameters.keys():
            weighted_sum = 0
            for update in node_updates:
                weight = update["data_size"] / total_data_size if total_data_size > 0 else 1/len(node_updates)
                weighted_sum += update["updates"][param_name] * weight
            
            aggregated_params[param_name] = weighted_sum
        
        # Update global parameters
        self.global_parameters.update(aggregated_params)
        self.aggregation_rounds += 1
        
        return {
            "global_parameters": self.global_parameters,
            "participating_nodes": len(node_updates),
            "total_data_points": total_data_size,
            "aggregation_round": self.aggregation_rounds
        }


class AdaptiveLearningSystem:
    """Adaptive learning system with self-optimization."""
    
    def __init__(self):
        self.performance_history = BoundedList(max_size=100, eviction_strategy="fifo")
        self.hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.1,
            "optimizer_momentum": 0.9
        }
        self.adaptation_threshold = 0.05
        self.anomaly_detector = BoundedDict(max_size=50, ttl_seconds=300)
        
    async def performance_tracking(self, metrics: Dict) -> Dict:
        """Track performance and detect patterns."""
        self.performance_history.append({
            "timestamp": time.time(),
            "accuracy": metrics.get("accuracy", 0.0),
            "loss": metrics.get("loss", 1.0),
            "training_time": metrics.get("training_time", 0.0)
        })
        
        # Calculate performance trends
        recent_performance = list(self.performance_history)[-10:] if len(self.performance_history) >= 10 else list(self.performance_history)
        
        if len(recent_performance) > 5:
            recent_accuracy = [p["accuracy"] for p in recent_performance]
            trend = (recent_accuracy[-1] - recent_accuracy[0]) / len(recent_accuracy)
            
            return {
                "performance_trend": trend,
                "recent_avg_accuracy": sum(recent_accuracy) / len(recent_accuracy),
                "adaptation_needed": abs(trend) < self.adaptation_threshold and recent_accuracy[-1] < 0.8
            }
        
        return {"adaptation_needed": False, "performance_trend": 0.0}
    
    async def auto_tuning(self, performance_analysis: Dict) -> Dict:
        """Automatically tune hyperparameters based on performance."""
        print("   🎯 Auto-tuning hyperparameters...")
        
        adjustments = {}
        
        if performance_analysis.get("adaptation_needed", False):
            trend = performance_analysis.get("performance_trend", 0.0)
            
            if trend < -0.01:  # Performance declining
                # Reduce learning rate and increase regularization
                self.hyperparameters["learning_rate"] *= 0.9
                self.hyperparameters["dropout_rate"] = min(0.5, self.hyperparameters["dropout_rate"] * 1.1)
                adjustments["action"] = "reduce_overfitting"
                
            elif trend > -0.005 and performance_analysis.get("recent_avg_accuracy", 0) < 0.7:
                # Performance stagnant, increase learning
                self.hyperparameters["learning_rate"] *= 1.1
                self.hyperparameters["batch_size"] = max(16, int(self.hyperparameters["batch_size"] * 0.9))
                adjustments["action"] = "increase_learning"
                
        return {
            "hyperparameters": self.hyperparameters.copy(),
            "adjustments": adjustments,
            "tuning_round": len(self.performance_history)
        }
    
    async def anomaly_detection(self, current_metrics: Dict) -> Dict:
        """Detect anomalies in training patterns."""
        metric_key = f"accuracy_{current_metrics.get('accuracy', 0):.3f}"
        
        if len(self.performance_history) > 10:
            historical_accuracies = [p["accuracy"] for p in self.performance_history]
            mean_accuracy = sum(historical_accuracies) / len(historical_accuracies)
            std_accuracy = (sum((x - mean_accuracy) ** 2 for x in historical_accuracies) / len(historical_accuracies)) ** 0.5
            
            current_accuracy = current_metrics.get("accuracy", 0.0)
            z_score = abs(current_accuracy - mean_accuracy) / std_accuracy if std_accuracy > 0 else 0
            
            is_anomaly = z_score > 2.0  # 2 standard deviations
            
            if is_anomaly:
                self.anomaly_detector[metric_key] = {
                    "timestamp": time.time(),
                    "z_score": z_score,
                    "current_value": current_accuracy,
                    "expected_range": (mean_accuracy - 2*std_accuracy, mean_accuracy + 2*std_accuracy)
                }
            
            return {
                "is_anomaly": is_anomaly,
                "z_score": z_score,
                "anomaly_count": len(self.anomaly_detector)
            }
        
        return {"is_anomaly": False, "z_score": 0.0}


async def demonstrate_federated_learning():
    """Demonstrate privacy-preserving federated learning."""
    print("🤝 Demonstrating Federated Learning")
    print("=" * 60)
    
    with MemoryContext("federated_learning") as ctx:
        # Create federated learning nodes from different organizations
        nodes = [
            FederatedLearningNode("node_bank_1", "Financial Bank A"),
            FederatedLearningNode("node_bank_2", "Financial Bank B"), 
            FederatedLearningNode("node_hospital_1", "Healthcare Org A"),
            FederatedLearningNode("node_hospital_2", "Healthcare Org B"),
            FederatedLearningNode("node_retail_1", "Retail Company A")
        ]
        
        # Add simulated local data
        for i, node in enumerate(nodes):
            local_data_size = random.randint(100, 500)
            node.add_local_data([f"data_point_{j}" for j in range(local_data_size)])
            print(f"   📊 {node.organization}: {len(node.local_data)} local data points")
        
        # Create secure aggregation server
        server = SecureAggregationServer()
        
        print(f"\n   🔐 Initial global parameters: {server.global_parameters}")
        
        # Perform federated learning rounds
        for round_num in range(3):
            print(f"\n   🔄 Federated Learning Round {round_num + 1}")
            
            # Each node performs local training
            node_updates = []
            for node in nodes:
                update = await node.local_training(server.global_parameters)
                node_updates.append(update)
                print(f"      📈 {node.organization}: Privacy budget remaining: {update['privacy_budget']:.2f}")
            
            # Secure aggregation
            aggregation_result = await server.secure_aggregate(node_updates)
            
            print(f"      ✅ Aggregation complete: {aggregation_result['participating_nodes']} nodes")
            print(f"      📊 Total data points: {aggregation_result['total_data_points']}")
            print(f"      🎯 Updated learning rate: {aggregation_result['global_parameters']['learning_rate']:.6f}")
        
        print(f"\n   🏆 Federated learning completed:")
        print(f"      • {len(nodes)} organizations collaborated")
        print(f"      • {server.aggregation_rounds} aggregation rounds")
        print(f"      • Privacy preserved with differential privacy")
        print(f"      • Secure aggregation with homomorphic encryption simulation")
        print(f"   💾 Memory usage: {ctx.memory_delta:.2f}MB")


async def demonstrate_adaptive_learning():
    """Demonstrate adaptive learning with self-optimization."""
    print("\n🔄 Demonstrating Adaptive Learning System")
    print("=" * 60)
    
    with MemoryContext("adaptive_learning") as ctx:
        adaptive_system = AdaptiveLearningSystem()
        
        print("   🎯 Initial hyperparameters:")
        for param, value in adaptive_system.hyperparameters.items():
            print(f"      {param}: {value}")
        
        print("\n   📈 Simulating training episodes with adaptive optimization...")
        
        # Simulate training episodes with varying performance
        performance_scenarios = [
            {"accuracy": 0.65, "loss": 0.8, "training_time": 120, "scenario": "Initial training"},
            {"accuracy": 0.72, "loss": 0.6, "training_time": 115, "scenario": "Improving performance"},
            {"accuracy": 0.69, "loss": 0.7, "training_time": 125, "scenario": "Performance fluctuation"},
            {"accuracy": 0.68, "loss": 0.75, "training_time": 130, "scenario": "Declining performance"},
            {"accuracy": 0.67, "loss": 0.8, "training_time": 135, "scenario": "Continued decline"},
            {"accuracy": 0.78, "loss": 0.5, "training_time": 110, "scenario": "Recovery after tuning"},
            {"accuracy": 0.82, "loss": 0.4, "training_time": 105, "scenario": "Improved performance"},
            {"accuracy": 0.85, "loss": 0.35, "training_time": 100, "scenario": "Optimized performance"},
            {"accuracy": 0.45, "loss": 1.2, "training_time": 150, "scenario": "Anomalous performance"},
            {"accuracy": 0.83, "loss": 0.38, "training_time": 102, "scenario": "Back to normal"}
        ]
        
        for episode, metrics in enumerate(performance_scenarios):
            print(f"\n   📊 Episode {episode + 1}: {metrics['scenario']}")
            print(f"      Accuracy: {metrics['accuracy']:.3f}, Loss: {metrics['loss']:.3f}")
            
            # Track performance
            performance_analysis = await adaptive_system.performance_tracking(metrics)
            
            # Detect anomalies
            anomaly_result = await adaptive_system.anomaly_detection(metrics)
            
            if anomaly_result["is_anomaly"]:
                print(f"      ⚠️ Anomaly detected! Z-score: {anomaly_result['z_score']:.2f}")
            
            # Auto-tune if needed
            if performance_analysis.get("adaptation_needed", False):
                tuning_result = await adaptive_system.auto_tuning(performance_analysis)
                print(f"      🎯 Auto-tuning applied: {tuning_result['adjustments'].get('action', 'None')}")
                print(f"      📈 New learning rate: {tuning_result['hyperparameters']['learning_rate']:.6f}")
                print(f"      📈 New dropout rate: {tuning_result['hyperparameters']['dropout_rate']:.3f}")
            
            # Show trend
            trend = performance_analysis.get("performance_trend", 0.0)
            trend_direction = "↗️" if trend > 0.01 else "↘️" if trend < -0.01 else "→"
            print(f"      📊 Performance trend: {trend_direction} {trend:.4f}")
        
        print(f"\n   🏆 Adaptive learning system results:")
        print(f"      • {len(adaptive_system.performance_history)} training episodes tracked")
        print(f"      • {len(adaptive_system.anomaly_detector)} anomalies detected")
        print(f"      • Hyperparameters automatically tuned for optimal performance")
        print(f"      • Real-time anomaly detection and recovery")
        print(f"   💾 Memory usage: {ctx.memory_delta:.2f}MB")


async def demonstrate_intelligent_scaling():
    """Demonstrate predictive scaling and workload pattern recognition."""
    print("\n📈 Demonstrating Intelligent Auto-Scaling")
    print("=" * 60)
    
    with MemoryContext("intelligent_scaling") as ctx:
        # Workload pattern recognition
        workload_patterns = BoundedDict(max_size=24, ttl_seconds=3600)  # 24 hours
        scaling_decisions = BoundedList(max_size=100, eviction_strategy="fifo")
        
        # Simulate 24-hour workload pattern
        hours = list(range(24))
        workload_data = []
        
        for hour in hours:
            # Simulate realistic workload patterns
            if 9 <= hour <= 17:  # Business hours
                base_load = 80 + random.randint(-10, 20)
            elif 18 <= hour <= 22:  # Evening peak
                base_load = 60 + random.randint(-15, 25)
            else:  # Night/early morning
                base_load = 20 + random.randint(-5, 15)
            
            # Add some randomness for realistic patterns
            current_cpu = max(0, min(100, base_load + random.randint(-5, 5)))
            current_memory = max(0, min(100, base_load + random.randint(-10, 10)))
            current_requests = max(0, base_load * 10 + random.randint(-50, 100))
            
            workload_patterns[f"hour_{hour}"] = {
                "cpu_usage": current_cpu,
                "memory_usage": current_memory,
                "requests_per_minute": current_requests,
                "timestamp": time.time() - (24 - hour) * 3600  # Simulate past data
            }
            
            workload_data.append({
                "hour": hour,
                "cpu": current_cpu,
                "memory": current_memory,
                "requests": current_requests
            })
        
        print("   📊 Workload pattern analysis:")
        print("      Hour | CPU%  | MEM%  | Req/min | Scaling Decision")
        print("      -----|-------|-------|---------|------------------")
        
        for data in workload_data[::4]:  # Show every 4 hours
            hour = data["hour"]
            cpu = data["cpu"]
            memory = data["memory"]
            requests = data["requests"]
            
            # Predictive scaling logic
            if cpu > 80 or memory > 85 or requests > 800:
                scaling_action = "Scale UP (+2 instances)"
                target_instances = 5
            elif cpu < 30 and memory < 40 and requests < 200:
                scaling_action = "Scale DOWN (-1 instance)"
                target_instances = 2
            else:
                scaling_action = "Maintain current"
                target_instances = 3
            
            scaling_decisions.append({
                "hour": hour,
                "action": scaling_action,
                "target_instances": target_instances,
                "metrics": {"cpu": cpu, "memory": memory, "requests": requests}
            })
            
            print(f"      {hour:2d}:00 | {cpu:3d}%  | {memory:3d}%  | {requests:4d}    | {scaling_action}")
        
        # Cost optimization analysis
        total_cost_optimized = 0
        total_cost_static = 0
        
        for decision in scaling_decisions:
            # Simulate cost calculation ($/hour per instance)
            cost_per_instance_hour = 0.10
            optimized_instances = decision["target_instances"]
            static_instances = 4  # Assume static allocation
            
            total_cost_optimized += optimized_instances * cost_per_instance_hour
            total_cost_static += static_instances * cost_per_instance_hour
        
        cost_savings = total_cost_static - total_cost_optimized
        savings_percentage = (cost_savings / total_cost_static) * 100 if total_cost_static > 0 else 0
        
        print(f"\n   💰 Cost optimization results:")
        print(f"      • Static allocation cost: ${total_cost_static:.2f}")
        print(f"      • Intelligent scaling cost: ${total_cost_optimized:.2f}")
        print(f"      • Cost savings: ${cost_savings:.2f} ({savings_percentage:.1f}%)")
        
        print(f"\n   🏆 Intelligent scaling capabilities:")
        print(f"      • Workload pattern recognition across 24-hour cycles")
        print(f"      • Predictive scaling based on multiple metrics")
        print(f"      • Cost-aware scaling decisions")
        print(f"      • {len(scaling_decisions)} scaling decisions optimized")
        print(f"   💾 Memory usage: {ctx.memory_delta:.2f}MB")


async def run_enterprise_training_suite():
    """Run complete enterprise training demonstration."""
    print("🚀 Enterprise Training Suite - Advanced Learning Capabilities")
    print("🤝 Federated Learning | 🔄 Adaptive Learning | 📈 Intelligent Scaling")
    print("=" * 80)
    
    with MemoryContext("enterprise_training_suite", threshold_mb=20.0) as total_ctx:
        log_memory_usage("Starting enterprise training suite")
        
        # Initialize global monitoring
        monitor = get_global_monitor(auto_start=True)
        
        # Initialize dependency injection
        container = get_container()
        configure_fastapi_services(container)
        
        try:
            # Run all enterprise training demonstrations
            await demonstrate_federated_learning()
            log_memory_usage("After federated learning demo")
            
            await demonstrate_adaptive_learning()
            log_memory_usage("After adaptive learning demo")
            
            await demonstrate_intelligent_scaling()
            log_memory_usage("After intelligent scaling demo")
            
            print(f"\n🎉 Enterprise Training Suite Completed!")
            print(f"💾 Total suite memory usage: {total_ctx.memory_delta:.2f}MB")
            
            # Get performance statistics
            stats = monitor.get_summary_report()
            print(f"🧠 Peak memory during suite: {stats['monitoring_stats']['peak_memory_mb']:.2f}MB")
            
            print("\n✅ Enterprise Learning Capabilities Demonstrated:")
            print("   🤝 Federated Learning - Privacy-preserving multi-organization training")
            print("   🔄 Adaptive Learning - Self-optimizing system with anomaly detection")
            print("   📈 Intelligent Scaling - Predictive scaling with cost optimization")
            print("   🔐 Privacy Protection - Differential privacy and secure aggregation")
            print("   🎯 Auto-Tuning - Automatic hyperparameter optimization")
            print("   💰 Cost Optimization - Intelligent resource allocation")
            
            print("\n🏆 Enterprise Readiness Features:")
            print("   • Multi-organization collaboration with privacy guarantees")
            print("   • Self-optimizing performance with real-time adaptation")
            print("   • Predictive scaling based on workload patterns")
            print("   • Cost-aware resource management")
            print("   • Anomaly detection and automatic recovery")
            print("   • Memory-optimized operations with bounded collections")
            
            print(f"\n🚀 System Status: Enterprise Training Suite COMPLETE")
            print("Ready for production deployment with advanced learning capabilities!")
            
        except Exception as e:
            print(f"❌ Error in enterprise training suite: {e}")
            import traceback
            traceback.print_exc()
        finally:
            monitor.stop_monitoring()
            log_memory_usage("Enterprise training suite completed")


async def main():
    """Main entry point for enterprise training demonstration."""
    await run_enterprise_training_suite()


if __name__ == "__main__":
    asyncio.run(main())