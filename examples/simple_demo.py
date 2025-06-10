#!/usr/bin/env python3
"""
Simple demonstration of the Infinite Agentic Loop system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.infinite_loop import (
    InfiniteAgenticLoopOrchestrator,
    InfiniteLoopConfig,
    SpecificationParser,
    DirectoryAnalyzer,
)


async def demo_specification_parsing():
    """Demonstrate specification parsing."""
    print("🔍 Demonstrating Specification Parsing")
    print("=" * 50)
    
    parser = SpecificationParser()
    
    # Parse the demo YAML specification
    spec_file = Path(__file__).parent / "demo_spec.yaml"
    spec_analysis = await parser.parse_specification(spec_file)
    
    print(f"📄 Parsed specification: {spec_file.name}")
    print(f"   Content Type: {spec_analysis.get('content_type', 'unknown')}")
    print(f"   Format: {spec_analysis.get('format', 'unknown')}")
    print(f"   Evolution Pattern: {spec_analysis.get('evolution_pattern', 'unknown')}")
    print(f"   Requirements: {len(spec_analysis.get('requirements', []))}")
    print(f"   Constraints: {len(spec_analysis.get('constraints', []))}")
    print(f"   Innovation Areas: {len(spec_analysis.get('innovation_areas', []))}")
    print(f"   Naming Pattern: {spec_analysis.get('naming_pattern', 'default')}")
    
    print("\n📋 Requirements:")
    for i, req in enumerate(spec_analysis.get('requirements', []), 1):
        print(f"   {i}. {req}")
    
    print("\n🚫 Constraints:")
    for i, constraint in enumerate(spec_analysis.get('constraints', []), 1):
        print(f"   {i}. {constraint}")
    
    print("\n💡 Innovation Areas:")
    for i, area in enumerate(spec_analysis.get('innovation_areas', []), 1):
        print(f"   {i}. {area}")
    
    return spec_analysis


async def demo_directory_analysis():
    """Demonstrate directory analysis."""
    print("\n\n📁 Demonstrating Directory Analysis")
    print("=" * 50)
    
    analyzer = DirectoryAnalyzer()
    
    # Analyze the test output directory
    output_dir = Path(__file__).parent / "test_output"
    directory_state = await analyzer.analyze_directory(output_dir)
    
    print(f"📂 Analyzed directory: {output_dir}")
    print(f"   Exists: {directory_state.get('exists', False)}")
    print(f"   Is Empty: {directory_state.get('is_empty', True)}")
    print(f"   Total Files: {len(directory_state.get('existing_files', []))}")
    print(f"   Iteration Files: {len(directory_state.get('iteration_files', []))}")
    print(f"   Highest Iteration: {directory_state.get('highest_iteration', 0)}")
    print(f"   Naming Patterns: {len(directory_state.get('naming_patterns', []))}")
    
    if directory_state.get('opportunities'):
        print("\n🎯 Opportunities:")
        for i, opportunity in enumerate(directory_state.get('opportunities', []), 1):
            print(f"   {i}. {opportunity}")
    
    return directory_state


async def demo_system_configuration():
    """Demonstrate system configuration."""
    print("\n\n⚙️ Demonstrating System Configuration")
    print("=" * 50)
    
    # Create different configurations
    configs = {
        "Basic": InfiniteLoopConfig(),
        "High Performance": InfiniteLoopConfig(
            max_parallel_agents=10,
            wave_size_max=8,
            context_threshold=0.9,
        ),
        "Quality Focused": InfiniteLoopConfig(
            max_parallel_agents=3,
            quality_threshold=0.9,
            uniqueness_threshold=0.9,
            validation_enabled=True,
        ),
        "Debug Mode": InfiniteLoopConfig(
            max_parallel_agents=1,
            log_level="DEBUG",
            detailed_logging=True,
            max_retries=1,
        ),
    }
    
    for name, config in configs.items():
        print(f"\n📊 {name} Configuration:")
        print(f"   Max Parallel Agents: {config.max_parallel_agents}")
        print(f"   Wave Size: {config.wave_size_min}-{config.wave_size_max}")
        print(f"   Quality Threshold: {config.quality_threshold}")
        print(f"   Uniqueness Threshold: {config.uniqueness_threshold}")
        print(f"   Context Threshold: {config.context_threshold}")
        print(f"   Log Level: {config.log_level}")


async def demo_innovation_dimensions():
    """Demonstrate innovation dimensions."""
    print("\n\n🎨 Demonstrating Innovation Dimensions")
    print("=" * 50)
    
    from src.agents.infinite_loop.task_assignment_engine import TaskAssignmentEngine
    
    engine = TaskAssignmentEngine()
    
    # Show complexity factors for different dimensions
    print("💡 Innovation Dimensions & Complexity Factors:")
    
    dimensions = engine.complexity_factors["innovation_dimension"]
    
    # Group by complexity level
    basic_dims = {k: v for k, v in dimensions.items() if v <= 1.3}
    advanced_dims = {k: v for k, v in dimensions.items() if 1.3 < v <= 1.6}
    expert_dims = {k: v for k, v in dimensions.items() if v > 1.6}
    
    print("\n🟢 Basic Dimensions (Complexity ≤ 1.3):")
    for dim, complexity in sorted(basic_dims.items(), key=lambda x: x[1]):
        print(f"   • {dim.replace('_', ' ').title()}: {complexity}x")
    
    print("\n🟡 Advanced Dimensions (1.3 < Complexity ≤ 1.6):")
    for dim, complexity in sorted(advanced_dims.items(), key=lambda x: x[1]):
        print(f"   • {dim.replace('_', ' ').title()}: {complexity}x")
    
    print("\n🔴 Expert Dimensions (Complexity > 1.6):")
    for dim, complexity in sorted(expert_dims.items(), key=lambda x: x[1]):
        print(f"   • {dim.replace('_', ' ').title()}: {complexity}x")


async def demo_wave_strategy():
    """Demonstrate wave strategy planning."""
    print("\n\n🌊 Demonstrating Wave Strategy")
    print("=" * 50)
    
    from src.agents.infinite_loop.orchestrator import InfiniteAgenticLoopOrchestrator
    
    # Create a mock orchestrator to access the strategy method
    config = InfiniteLoopConfig()
    
    # Simulate different count scenarios
    scenarios = [
        ("3 iterations", 3),
        ("8 iterations", 8),
        ("25 iterations", 25),
        ("infinite", "infinite"),
    ]
    
    print("📋 Wave Strategies for Different Scenarios:")
    
    for scenario_name, count in scenarios:
        print(f"\n🎯 {scenario_name}:")
        
        # This would normally be called by the orchestrator
        if count == "infinite":
            strategy = {
                "type": "infinite_waves",
                "wave_size": config.wave_size_min,
                "max_waves": None,
                "context_monitoring": True,
            }
        elif isinstance(count, int):
            if count <= 5:
                strategy = {
                    "type": "single_wave",
                    "wave_size": count,
                    "max_waves": 1,
                    "context_monitoring": False,
                }
            elif count <= 20:
                strategy = {
                    "type": "batched_waves",
                    "wave_size": 5,
                    "max_waves": (count + 4) // 5,
                    "context_monitoring": True,
                }
            else:
                strategy = {
                    "type": "large_batched_waves",
                    "wave_size": config.wave_size_max,
                    "max_waves": (count + config.wave_size_max - 1) // config.wave_size_max,
                    "context_monitoring": True,
                }
        
        print(f"   Strategy Type: {strategy['type']}")
        print(f"   Wave Size: {strategy['wave_size']}")
        print(f"   Max Waves: {strategy['max_waves'] or 'Unlimited'}")
        print(f"   Context Monitoring: {strategy['context_monitoring']}")


async def main():
    """Run the complete demonstration."""
    print("🚀 Infinite Agentic Loop System Demonstration")
    print("=" * 60)
    print("This demo shows the key components and capabilities")
    print("of the Infinite Agentic Loop system.\n")
    
    try:
        # Run all demonstrations
        await demo_specification_parsing()
        await demo_directory_analysis()
        await demo_system_configuration()
        await demo_innovation_dimensions()
        await demo_wave_strategy()
        
        print("\n\n🎉 Demonstration Complete!")
        print("=" * 60)
        print("The Infinite Agentic Loop system is ready for use!")
        print("\nNext steps:")
        print("1. Set up your Anthropic API key in environment variables")
        print("2. Create your own specification file")
        print("3. Run: python scripts/run_infinite_loop.py your_spec.md ./output 5")
        print("4. For infinite mode: python scripts/run_infinite_loop.py your_spec.md ./output infinite")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
