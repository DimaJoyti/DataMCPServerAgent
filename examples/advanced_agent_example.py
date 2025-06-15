"""
Example of using the advanced agent with specialized sub-agents.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_main import chat_with_advanced_agent


async def run_example():
    """Run the advanced agent example."""
    print("Running advanced agent example with specialized sub-agents...")
    print("This agent architecture includes:")
    print("- Specialized sub-agents for different tasks")
    print("- Tool selection based on task requirements")
    print("- In-memory conversation history")
    print("- Coordinator agent for managing sub-agents")

    await chat_with_advanced_agent()

if __name__ == "__main__":
    asyncio.run(run_example())
