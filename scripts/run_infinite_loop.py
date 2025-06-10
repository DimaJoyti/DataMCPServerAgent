#!/usr/bin/env python3
"""
Infinite Agentic Loop Runner

Convenient script to run the infinite agentic loop system with various options.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.infinite_loop_main import execute_infinite_loop_command, interactive_infinite_loop
from src.agents.infinite_loop import InfiniteLoopConfig


def setup_logging(level: str, detailed: bool = False) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if detailed:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('infinite_loop.log', mode='a')
        ]
    )


def create_config_from_args(args) -> InfiniteLoopConfig:
    """Create configuration from command line arguments."""
    return InfiniteLoopConfig(
        max_parallel_agents=args.max_agents,
        wave_size_min=args.wave_min,
        wave_size_max=args.wave_max,
        context_threshold=args.context_threshold,
        quality_threshold=args.quality_threshold,
        uniqueness_threshold=args.uniqueness_threshold,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        log_level=args.log_level,
        detailed_logging=args.detailed_logging,
        validation_enabled=not args.skip_validation,
        error_recovery_enabled=not args.skip_recovery,
        batch_processing=not args.no_batch,
        async_execution=not args.no_async,
        memory_optimization=not args.no_memory_opt,
    )


async def run_infinite_loop(args) -> None:
    """Run the infinite agentic loop with given arguments."""
    # Setup logging
    setup_logging(args.log_level, args.detailed_logging)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Validate inputs
    spec_file = Path(args.spec_file)
    if not spec_file.exists():
        print(f"❌ Specification file not found: {spec_file}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Parse count
    if args.count.lower() == "infinite":
        count = "infinite"
    else:
        try:
            count = int(args.count)
            if count < 1:
                print("❌ Count must be a positive integer or 'infinite'")
                sys.exit(1)
        except ValueError:
            print(f"❌ Invalid count: {args.count}")
            sys.exit(1)
    
    # Execute the infinite loop
    print("🚀 Starting Infinite Agentic Loop")
    print(f"Spec file: {spec_file}")
    print(f"Output directory: {output_dir}")
    print(f"Count: {count}")
    print(f"Max agents: {config.max_parallel_agents}")
    print("=" * 50)
    
    try:
        results = await execute_infinite_loop_command(
            spec_file=spec_file,
            output_dir=output_dir,
            count=count,
            config=config,
        )
        
        # Display results
        if results.get("success", False):
            print("\n✅ Execution completed successfully!")
            
            # Print statistics
            stats = results.get("statistics", {})
            print(f"📊 Statistics:")
            print(f"  Total iterations: {stats.get('total_iterations', 0)}")
            print(f"  Execution time: {stats.get('execution_time_seconds', 0):.1f}s")
            print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"  Average iteration time: {stats.get('average_iteration_time', 0):.1f}s")
            print(f"  Waves completed: {stats.get('waves_completed', 0)}")
            
            # Print execution state
            execution_state = results.get("execution_state")
            if execution_state:
                print(f"📈 Execution State:")
                print(f"  Completed iterations: {len(execution_state.completed_iterations)}")
                if execution_state.failed_iterations:
                    print(f"  Failed iterations: {len(execution_state.failed_iterations)}")
                print(f"  Quality score: {execution_state.quality_score:.2f}")
                
                if args.verbose and execution_state.completed_iterations:
                    print(f"  Completed: {', '.join(execution_state.completed_iterations)}")
        
        else:
            print("\n❌ Execution failed!")
            error = results.get("error", "Unknown error")
            print(f"Error: {error}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Execution failed: {str(e)}")
        logging.exception("Execution failed")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Infinite Agentic Loop System - Generate infinite iterations based on specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s spec.md ./output 5                    # Generate 5 iterations
  %(prog)s spec.yaml ./iterations infinite       # Generate infinite iterations
  %(prog)s --interactive                         # Interactive mode
  %(prog)s spec.md ./output 10 --max-agents 3   # Use 3 parallel agents
  %(prog)s spec.md ./output infinite --quality-threshold 0.8  # Higher quality threshold
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "spec_file", 
        nargs="?",
        help="Path to specification file (markdown, yaml, json, or text)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?", 
        help="Directory where iterations will be saved"
    )
    parser.add_argument(
        "count",
        nargs="?",
        help="Number of iterations (positive integer or 'infinite')"
    )
    
    # Mode selection
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Agent configuration
    parser.add_argument(
        "--max-agents",
        type=int,
        default=5,
        help="Maximum number of parallel agents (default: 5)"
    )
    parser.add_argument(
        "--wave-min",
        type=int,
        default=3,
        help="Minimum wave size for infinite mode (default: 3)"
    )
    parser.add_argument(
        "--wave-max",
        type=int,
        default=5,
        help="Maximum wave size for infinite mode (default: 5)"
    )
    
    # Quality thresholds
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Quality threshold (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--uniqueness-threshold",
        type=float,
        default=0.8,
        help="Uniqueness threshold (0.0-1.0, default: 0.8)"
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=0.8,
        help="Context usage threshold (0.0-1.0, default: 0.8)"
    )
    
    # Error handling
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per task (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds (default: 1.0)"
    )
    
    # Feature toggles
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip content validation"
    )
    parser.add_argument(
        "--skip-recovery",
        action="store_true",
        help="Skip error recovery"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch processing"
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async execution"
    )
    parser.add_argument(
        "--no-memory-opt",
        action="store_true",
        help="Disable memory optimization"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--detailed-logging",
        action="store_true",
        help="Enable detailed logging with file/line info"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.interactive:
        # Interactive mode
        asyncio.run(interactive_infinite_loop())
    else:
        # Command line mode
        if not all([args.spec_file, args.output_dir, args.count]):
            parser.error("spec_file, output_dir, and count are required unless using --interactive")
        
        asyncio.run(run_infinite_loop(args))


if __name__ == "__main__":
    main()
