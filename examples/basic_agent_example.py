"""
Example of using the basic agent.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import chat_with_agent


async def run_example():
    """Run the basic agent example."""
    print("Running basic agent example...")
    await chat_with_agent()


if __name__ == "__main__":
    asyncio.run(run_example())