"""
Example of using the enhanced agent with memory persistence and learning capabilities.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.enhanced_main import chat_with_enhanced_agent


async def run_example():
    """Run the enhanced agent example."""
    print("Running enhanced agent example with memory persistence and learning capabilities...")
    print("This agent architecture includes:")
    print("- Memory persistence using SQLite")
    print("- Enhanced tool selection with performance tracking")
    print("- Learning capabilities with feedback collection")
    print("- Self-evaluation and improvement")
    
    print("\nSpecial commands:")
    print("- 'memory': View the current memory state")
    print("- 'learn': Trigger learning from feedback")
    print("- 'insights': View learning insights")
    print("- 'feedback <your feedback>': Provide feedback on the last response")
    
    await chat_with_enhanced_agent()


if __name__ == "__main__":
    asyncio.run(run_example())