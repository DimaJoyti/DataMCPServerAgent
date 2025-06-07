"""
Example of using the advanced enhanced agent with context-aware memory and adaptive learning.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent

async def run_example():
    """Run the advanced enhanced agent example."""
    print("Running advanced enhanced agent example with context-aware memory and adaptive learning...")
    print("This agent architecture includes:")
    print("- Context-aware memory with semantic search")
    print("- Adaptive learning based on user preferences")
    print("- Enhanced tool selection with performance tracking")
    print("- Learning capabilities with feedback collection")
    print("- User preference modeling")

    print("\nSpecial commands:")
    print("- 'context': View the current context")
    print("- 'preferences': View the current user preferences")
    print("- 'learn': Trigger learning from feedback")
    print("- 'metrics': View performance metrics")
    print("- 'feedback <your feedback>': Provide feedback on the last response")

    await chat_with_advanced_enhanced_agent()

if __name__ == "__main__":
    asyncio.run(run_example())
