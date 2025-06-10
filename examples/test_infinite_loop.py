"""
Test script for the Infinite Agentic Loop system.

This script demonstrates how to use the infinite loop system to generate
iterations based on a specification file.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.infinite_loop_main import execute_infinite_loop_command
from src.agents.infinite_loop import InfiniteLoopConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_infinite_loop")


async def test_basic_infinite_loop():
    """Test basic infinite loop functionality."""
    print("=== Testing Basic Infinite Loop ===")
    
    # Setup paths
    spec_file = Path(__file__).parent / "infinite_loop_spec.md"
    output_dir = Path(__file__).parent / "test_output"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration for testing
    config = InfiniteLoopConfig(
        max_parallel_agents=2,  # Reduced for testing
        wave_size_min=1,
        wave_size_max=2,
        context_threshold=0.7,
        quality_threshold=0.6,
        uniqueness_threshold=0.7,
        log_level="INFO",
        detailed_logging=True,
    )
    
    try:
        # Test with a small number of iterations
        print(f"Spec file: {spec_file}")
        print(f"Output directory: {output_dir}")
        print("Generating 3 iterations...")
        
        results = await execute_infinite_loop_command(
            spec_file=spec_file,
            output_dir=output_dir,
            count=3,
            config=config,
        )
        
        # Display results
        print("\n=== Results ===")
        if results.get("success", False):
            print("✅ Test completed successfully!")
            
            # Print statistics
            stats = results.get("statistics", {})
            print(f"Total iterations: {stats.get('total_iterations', 0)}")
            print(f"Execution time: {stats.get('execution_time_seconds', 0):.1f}s")
            print(f"Success rate: {stats.get('success_rate', 0):.1%}")
            
            # Print execution state
            execution_state = results.get("execution_state")
            if execution_state:
                print(f"Completed: {len(execution_state.completed_iterations)}")
                print(f"Failed: {len(execution_state.failed_iterations)}")
                
                if execution_state.completed_iterations:
                    print("Completed iterations:", execution_state.completed_iterations)
                
                if execution_state.failed_iterations:
                    print("Failed iterations:", execution_state.failed_iterations)
        else:
            print("❌ Test failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        logger.exception("Test failed")
        return {"success": False, "error": str(e)}


async def test_specification_parsing():
    """Test specification parsing functionality."""
    print("\n=== Testing Specification Parsing ===")
    
    from src.agents.infinite_loop import SpecificationParser
    
    spec_file = Path(__file__).parent / "infinite_loop_spec.md"
    
    try:
        parser = SpecificationParser()
        spec_analysis = await parser.parse_specification(spec_file)
        
        print("✅ Specification parsed successfully!")
        print(f"Content type: {spec_analysis.get('content_type', 'unknown')}")
        print(f"Format: {spec_analysis.get('format', 'unknown')}")
        print(f"Evolution pattern: {spec_analysis.get('evolution_pattern', 'unknown')}")
        print(f"Requirements: {len(spec_analysis.get('requirements', []))}")
        print(f"Constraints: {len(spec_analysis.get('constraints', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Specification parsing failed: {str(e)}")
        logger.exception("Specification parsing failed")
        return False


async def test_directory_analysis():
    """Test directory analysis functionality."""
    print("\n=== Testing Directory Analysis ===")
    
    from src.agents.infinite_loop import DirectoryAnalyzer
    
    output_dir = Path(__file__).parent / "test_output"
    
    try:
        analyzer = DirectoryAnalyzer()
        directory_state = await analyzer.analyze_directory(output_dir)
        
        print("✅ Directory analyzed successfully!")
        print(f"Directory exists: {directory_state.get('exists', False)}")
        print(f"Is empty: {directory_state.get('is_empty', True)}")
        print(f"Existing files: {len(directory_state.get('existing_files', []))}")
        print(f"Iteration files: {len(directory_state.get('iteration_files', []))}")
        print(f"Highest iteration: {directory_state.get('highest_iteration', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory analysis failed: {str(e)}")
        logger.exception("Directory analysis failed")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Infinite Agentic Loop Tests")
    print("=" * 50)
    
    # Test individual components
    spec_test = await test_specification_parsing()
    dir_test = await test_directory_analysis()
    
    # Test full system if components work
    if spec_test and dir_test:
        loop_test = await test_basic_infinite_loop()
        
        if loop_test.get("success", False):
            print("\n🎉 All tests passed!")
            return True
        else:
            print("\n❌ Full system test failed")
            return False
    else:
        print("\n❌ Component tests failed, skipping full system test")
        return False


async def main():
    """Main test function."""
    try:
        success = await run_all_tests()
        
        if success:
            print("\n✅ Test suite completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Test suite failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        logger.exception("Test suite crashed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
