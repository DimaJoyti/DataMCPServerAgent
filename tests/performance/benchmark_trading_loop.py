"""
Performance benchmarking for Trading Infinite Loop system.

This module provides comprehensive performance tests and benchmarks
for the trading strategy generation system.
"""

import asyncio
import json
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd

from src.agents.trading_infinite_loop.trading_strategy_orchestrator import (
    TradingStrategyOrchestrator,
    TradingStrategyConfig
)
from src.api.services.trading_strategy_service import TradingStrategyService


class PerformanceBenchmark:
    """
    Performance benchmarking suite for trading infinite loop system.
    
    Measures throughput, latency, memory usage, and scalability
    of the strategy generation system.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: Dict[str, Any] = {}
        self.process = psutil.Process()
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": self.process.memory_percent()
        }
    
    def measure_cpu_usage(self) -> float:
        """Measure current CPU usage."""
        return self.process.cpu_percent(interval=1)
    
    async def benchmark_strategy_generation_throughput(
        self,
        strategy_counts: List[int] = [10, 50, 100, 500]
    ) -> Dict[str, Any]:
        """
        Benchmark strategy generation throughput.
        
        Args:
            strategy_counts: List of strategy counts to test
            
        Returns:
            Throughput benchmark results
        """
        print("🚀 Benchmarking Strategy Generation Throughput...")
        
        results = {
            "strategy_counts": strategy_counts,
            "execution_times": [],
            "throughput_strategies_per_second": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        for count in strategy_counts:
            print(f"  Testing {count} strategies...")
            
            # Measure initial resources
            initial_memory = self.measure_memory_usage()
            
            # Time the generation process
            start_time = time.time()
            
            # Mock strategy generation (replace with actual implementation)
            await self._mock_strategy_generation(count)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Measure final resources
            final_memory = self.measure_memory_usage()
            cpu_usage = self.measure_cpu_usage()
            
            # Calculate metrics
            throughput = count / execution_time
            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
            
            results["execution_times"].append(execution_time)
            results["throughput_strategies_per_second"].append(throughput)
            results["memory_usage"].append(memory_increase)
            results["cpu_usage"].append(cpu_usage)
            
            print(f"    ✅ {count} strategies: {execution_time:.2f}s, {throughput:.2f} strategies/s")
        
        return results
    
    async def benchmark_concurrent_sessions(
        self,
        session_counts: List[int] = [1, 2, 5, 10]
    ) -> Dict[str, Any]:
        """
        Benchmark concurrent strategy generation sessions.
        
        Args:
            session_counts: List of concurrent session counts to test
            
        Returns:
            Concurrency benchmark results
        """
        print("🔄 Benchmarking Concurrent Sessions...")
        
        results = {
            "session_counts": session_counts,
            "total_execution_times": [],
            "average_session_times": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        for session_count in session_counts:
            print(f"  Testing {session_count} concurrent sessions...")
            
            # Measure initial resources
            initial_memory = self.measure_memory_usage()
            
            # Time concurrent execution
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(session_count):
                task = asyncio.create_task(self._mock_strategy_generation(20))
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            # Measure final resources
            final_memory = self.measure_memory_usage()
            cpu_usage = self.measure_cpu_usage()
            
            # Calculate metrics
            average_session_time = total_execution_time / session_count
            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
            
            results["total_execution_times"].append(total_execution_time)
            results["average_session_times"].append(average_session_time)
            results["memory_usage"].append(memory_increase)
            results["cpu_usage"].append(cpu_usage)
            
            print(f"    ✅ {session_count} sessions: {total_execution_time:.2f}s total, {average_session_time:.2f}s avg")
        
        return results
    
    async def benchmark_backtest_performance(
        self,
        data_sizes: List[int] = [30, 90, 180, 365]  # Days of data
    ) -> Dict[str, Any]:
        """
        Benchmark backtesting performance with different data sizes.
        
        Args:
            data_sizes: List of data sizes (days) to test
            
        Returns:
            Backtesting benchmark results
        """
        print("📊 Benchmarking Backtest Performance...")
        
        results = {
            "data_sizes_days": data_sizes,
            "execution_times": [],
            "memory_usage": [],
            "trades_processed": []
        }
        
        for days in data_sizes:
            print(f"  Testing {days} days of data...")
            
            # Measure initial memory
            initial_memory = self.measure_memory_usage()
            
            # Time backtest execution
            start_time = time.time()
            
            # Mock backtesting (replace with actual implementation)
            trades_processed = await self._mock_backtesting(days)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Measure final memory
            final_memory = self.measure_memory_usage()
            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
            
            results["execution_times"].append(execution_time)
            results["memory_usage"].append(memory_increase)
            results["trades_processed"].append(trades_processed)
            
            print(f"    ✅ {days} days: {execution_time:.2f}s, {trades_processed} trades")
        
        return results
    
    async def benchmark_api_latency(
        self,
        request_counts: List[int] = [10, 50, 100, 500]
    ) -> Dict[str, Any]:
        """
        Benchmark API endpoint latency.
        
        Args:
            request_counts: List of request counts to test
            
        Returns:
            API latency benchmark results
        """
        print("🌐 Benchmarking API Latency...")
        
        results = {
            "request_counts": request_counts,
            "average_latencies": [],
            "p95_latencies": [],
            "p99_latencies": [],
            "throughput_requests_per_second": []
        }
        
        for count in request_counts:
            print(f"  Testing {count} API requests...")
            
            latencies = []
            start_time = time.time()
            
            # Simulate API requests
            for i in range(count):
                request_start = time.time()
                
                # Mock API call (replace with actual API testing)
                await self._mock_api_request()
                
                request_end = time.time()
                latencies.append((request_end - request_start) * 1000)  # Convert to ms
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            throughput = count / total_time
            
            results["average_latencies"].append(avg_latency)
            results["p95_latencies"].append(p95_latency)
            results["p99_latencies"].append(p99_latency)
            results["throughput_requests_per_second"].append(throughput)
            
            print(f"    ✅ {count} requests: {avg_latency:.2f}ms avg, {throughput:.2f} req/s")
        
        return results
    
    async def benchmark_memory_scalability(
        self,
        strategy_counts: List[int] = [100, 500, 1000, 5000]
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage scalability.
        
        Args:
            strategy_counts: List of strategy counts to test
            
        Returns:
            Memory scalability benchmark results
        """
        print("💾 Benchmarking Memory Scalability...")
        
        results = {
            "strategy_counts": strategy_counts,
            "memory_usage_mb": [],
            "memory_per_strategy_kb": []
        }
        
        for count in strategy_counts:
            print(f"  Testing memory usage with {count} strategies...")
            
            # Measure initial memory
            initial_memory = self.measure_memory_usage()
            
            # Create mock strategies in memory
            strategies = {}
            for i in range(count):
                strategies[f"strategy_{i}"] = {
                    "strategy_id": f"strategy_{i}",
                    "performance": {
                        "sharpe_ratio": 1.5 + (i % 100) * 0.01,
                        "total_return": 0.1 + (i % 50) * 0.002,
                        "max_drawdown": -0.05 - (i % 20) * 0.001
                    },
                    "backtest_results": {
                        "trades": [{"pnl": 100 + j} for j in range(50)],  # 50 trades per strategy
                        "daily_returns": [0.001 * (j % 10) for j in range(252)]  # 1 year of returns
                    }
                }
            
            # Measure final memory
            final_memory = self.measure_memory_usage()
            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
            memory_per_strategy = (memory_increase * 1024) / count  # KB per strategy
            
            results["memory_usage_mb"].append(memory_increase)
            results["memory_per_strategy_kb"].append(memory_per_strategy)
            
            print(f"    ✅ {count} strategies: {memory_increase:.2f}MB, {memory_per_strategy:.2f}KB/strategy")
            
            # Clean up
            del strategies
        
        return results
    
    async def _mock_strategy_generation(self, count: int) -> None:
        """Mock strategy generation for benchmarking."""
        # Simulate strategy generation time
        base_time = 0.01  # 10ms per strategy
        for i in range(count):
            await asyncio.sleep(base_time + (i % 10) * 0.001)  # Variable time
    
    async def _mock_backtesting(self, days: int) -> int:
        """Mock backtesting for benchmarking."""
        # Simulate backtesting time based on data size
        trades_per_day = 5
        total_trades = days * trades_per_day
        
        # Simulate processing time
        processing_time = days * 0.001  # 1ms per day of data
        await asyncio.sleep(processing_time)
        
        return total_trades
    
    async def _mock_api_request(self) -> None:
        """Mock API request for benchmarking."""
        # Simulate API processing time
        await asyncio.sleep(0.005 + (time.time() % 0.01))  # 5-15ms
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Formatted performance report
        """
        report = []
        report.append("=" * 80)
        report.append("🚀 TRADING INFINITE LOOP PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Strategy Generation Throughput
        if "throughput" in results:
            throughput_data = results["throughput"]
            report.append("📈 STRATEGY GENERATION THROUGHPUT")
            report.append("-" * 40)
            for i, count in enumerate(throughput_data["strategy_counts"]):
                time_taken = throughput_data["execution_times"][i]
                throughput = throughput_data["throughput_strategies_per_second"][i]
                memory = throughput_data["memory_usage"][i]
                report.append(f"  {count:4d} strategies: {time_taken:6.2f}s | {throughput:6.2f} strategies/s | {memory:6.2f}MB")
            report.append("")
        
        # Concurrent Sessions
        if "concurrency" in results:
            concurrency_data = results["concurrency"]
            report.append("🔄 CONCURRENT SESSIONS PERFORMANCE")
            report.append("-" * 40)
            for i, count in enumerate(concurrency_data["session_counts"]):
                total_time = concurrency_data["total_execution_times"][i]
                avg_time = concurrency_data["average_session_times"][i]
                memory = concurrency_data["memory_usage"][i]
                report.append(f"  {count:2d} sessions: {total_time:6.2f}s total | {avg_time:6.2f}s avg | {memory:6.2f}MB")
            report.append("")
        
        # API Latency
        if "api_latency" in results:
            api_data = results["api_latency"]
            report.append("🌐 API LATENCY PERFORMANCE")
            report.append("-" * 40)
            for i, count in enumerate(api_data["request_counts"]):
                avg_lat = api_data["average_latencies"][i]
                p95_lat = api_data["p95_latencies"][i]
                throughput = api_data["throughput_requests_per_second"][i]
                report.append(f"  {count:3d} requests: {avg_lat:6.2f}ms avg | {p95_lat:6.2f}ms p95 | {throughput:6.2f} req/s")
            report.append("")
        
        # Memory Scalability
        if "memory" in results:
            memory_data = results["memory"]
            report.append("💾 MEMORY SCALABILITY")
            report.append("-" * 40)
            for i, count in enumerate(memory_data["strategy_counts"]):
                memory_mb = memory_data["memory_usage_mb"][i]
                memory_per_strategy = memory_data["memory_per_strategy_kb"][i]
                report.append(f"  {count:4d} strategies: {memory_mb:6.2f}MB total | {memory_per_strategy:6.2f}KB/strategy")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            filename: Optional filename
            
        Returns:
            Filename where results were saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_loop_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filename


async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark suite."""
    print("🚀 Starting Comprehensive Trading Infinite Loop Benchmark")
    print("=" * 80)
    
    benchmark = PerformanceBenchmark()
    all_results = {}
    
    # Run all benchmarks
    all_results["throughput"] = await benchmark.benchmark_strategy_generation_throughput()
    all_results["concurrency"] = await benchmark.benchmark_concurrent_sessions()
    all_results["backtest"] = await benchmark.benchmark_backtest_performance()
    all_results["api_latency"] = await benchmark.benchmark_api_latency()
    all_results["memory"] = await benchmark.benchmark_memory_scalability()
    
    # Generate and save report
    report = benchmark.generate_performance_report(all_results)
    print("\n" + report)
    
    # Save results
    results_file = benchmark.save_results(all_results)
    print(f"\n📁 Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
