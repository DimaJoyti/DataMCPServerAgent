"""
Infinite Agentic Loop Main Interface

Main entry point for the infinite agentic loop system that provides
command-line interface and integration with the existing agent system.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.infinite_loop import (
    InfiniteAgenticLoopOrchestrator,
    InfiniteLoopConfig,
)
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("infinite_loop_main")


async def setup_infinite_loop_system() -> tuple[ChatAnthropic, List[BaseTool]]:
    """Set up the infinite loop system with model and tools."""
    # Initialize the language model
    model = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0.1,
        max_tokens=4000,
    )
    
    # Initialize tools
    tools = []
    
    try:
        # Add Bright Data tools if available
        bright_data_toolkit = BrightDataToolkit()
        bright_data_tools = await bright_data_toolkit.get_tools()
        tools.extend(bright_data_tools)
        logger.info(f"Loaded {len(bright_data_tools)} Bright Data tools")
    except Exception as e:
        logger.warning(f"Could not load Bright Data tools: {e}")
    
    logger.info(f"Infinite loop system initialized with {len(tools)} tools")
    return model, tools


async def execute_infinite_loop_command(
    spec_file: Union[str, Path],
    output_dir: Union[str, Path],
    count: Union[int, str],
    config: Optional[InfiniteLoopConfig] = None,
) -> Dict[str, Any]:
    """
    Execute the infinite agentic loop command.
    
    Args:
        spec_file: Path to the specification file
        output_dir: Directory for output iterations
        count: Number of iterations (integer or "infinite")
        config: Optional configuration override
        
    Returns:
        Execution results
    """
    try:
        logger.info("Starting infinite agentic loop execution")
        logger.info(f"Spec file: {spec_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Count: {count}")
        
        # Setup system
        model, tools = await setup_infinite_loop_system()
        
        # Create configuration
        if config is None:
            config = InfiniteLoopConfig()
        
        # Create orchestrator
        orchestrator = InfiniteAgenticLoopOrchestrator(
            model=model,
            tools=tools,
            config=config,
        )
        
        # Execute the infinite loop
        results = await orchestrator.execute_infinite_loop(
            spec_file=spec_file,
            output_dir=output_dir,
            count=count,
        )
        
        # Shutdown orchestrator
        await orchestrator.shutdown()
        
        return results
        
    except Exception as e:
        error_message = format_error_for_user(e)
        logger.error(f"Infinite loop execution failed: {error_message}")
        
        return {
            "success": False,
            "error": error_message,
            "session_id": "unknown",
        }


async def parse_arguments_and_execute(arguments: str) -> Dict[str, Any]:
    """
    Parse arguments and execute the infinite loop command.
    
    Args:
        arguments: Command arguments string
        
    Returns:
        Execution results
    """
    try:
        # Parse arguments
        args = arguments.strip().split()
        
        if len(args) < 3:
            return {
                "success": False,
                "error": "Insufficient arguments. Required: spec_file output_dir count",
                "usage": "infinite_loop <spec_file> <output_dir> <count>",
            }
        
        spec_file = args[0]
        output_dir = args[1]
        count_str = args[2]
        
        # Parse count
        if count_str.lower() == "infinite":
            count = "infinite"
        else:
            try:
                count = int(count_str)
                if count < 1:
                    return {
                        "success": False,
                        "error": "Count must be a positive integer or 'infinite'",
                    }
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid count: {count_str}. Must be a positive integer or 'infinite'",
                }
        
        # Validate spec file
        if not Path(spec_file).exists():
            return {
                "success": False,
                "error": f"Specification file not found: {spec_file}",
            }
        
        # Execute the command
        return await execute_infinite_loop_command(
            spec_file=spec_file,
            output_dir=output_dir,
            count=count,
        )
        
    except Exception as e:
        error_message = format_error_for_user(e)
        return {
            "success": False,
            "error": f"Argument parsing failed: {error_message}",
        }


async def interactive_infinite_loop() -> None:
    """Run interactive infinite loop interface."""
    print("=== Infinite Agentic Loop System ===")
    print("Generate infinite iterations based on specifications")
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("infinite_loop> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() in ["help", "h"]:
                print_help()
                continue
            
            # Execute command
            print(f"Executing: {user_input}")
            results = await parse_arguments_and_execute(user_input)
            
            # Display results
            print_results(results)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {format_error_for_user(e)}")


def print_help() -> None:
    """Print help information."""
    help_text = """
Infinite Agentic Loop Commands:

Usage: <spec_file> <output_dir> <count>

Arguments:
  spec_file   - Path to specification file (markdown, yaml, json, or text)
  output_dir  - Directory where iterations will be saved
  count       - Number of iterations (positive integer or 'infinite')

Examples:
  spec.md ./iterations 5        - Generate 5 iterations
  spec.yaml ./output infinite   - Generate infinite iterations
  
Special Commands:
  help, h     - Show this help
  quit, q     - Exit the system

Specification File Format:
The specification file defines what to generate. It can be:
- Markdown with requirements and constraints
- YAML with structured configuration
- JSON with detailed specifications
- Plain text with instructions

The system will analyze existing iterations and generate unique content
based on the specification and assigned innovation dimensions.
"""
    print(help_text)


def print_results(results: Dict[str, Any]) -> None:
    """Print execution results."""
    if results.get("success", False):
        print("✅ Execution completed successfully!")
        
        session_id = results.get("session_id", "unknown")
        print(f"Session ID: {session_id}")
        
        # Print statistics
        stats = results.get("statistics", {})
        if stats:
            print(f"Total iterations: {stats.get('total_iterations', 0)}")
            print(f"Execution time: {stats.get('execution_time_seconds', 0):.1f}s")
            print(f"Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"Waves completed: {stats.get('waves_completed', 0)}")
        
        # Print execution state
        execution_state = results.get("execution_state")
        if execution_state:
            print(f"Completed iterations: {len(execution_state.completed_iterations)}")
            if execution_state.failed_iterations:
                print(f"Failed iterations: {len(execution_state.failed_iterations)}")
    
    else:
        print("❌ Execution failed!")
        error = results.get("error", "Unknown error")
        print(f"Error: {error}")
        
        if "usage" in results:
            print(f"Usage: {results['usage']}")


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command line mode
        arguments = " ".join(sys.argv[1:])
        results = await parse_arguments_and_execute(arguments)
        print_results(results)
    else:
        # Interactive mode
        await interactive_infinite_loop()


if __name__ == "__main__":
    asyncio.run(main())
