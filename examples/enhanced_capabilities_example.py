"""
Example script demonstrating the enhanced capabilities of DataMCPServerAgent.
This example shows memory persistence, enhanced tool selection, and learning capabilities.
"""

import asyncio
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bright_data_tools import BrightDataToolkit
from dotenv import load_dotenv
from enhanced_agent_architecture import create_enhanced_agent_architecture
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Set up the MCP server parameters
server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)

# Example tasks to demonstrate the enhanced capabilities
EXAMPLE_TASKS = [
    "Find the current price of Bitcoin and Ethereum",
    "Compare the features of the latest iPhone and Samsung Galaxy models",
    "Analyze the social media presence of Apple on Twitter",
    "Find the top news stories about artificial intelligence from the past week"
]

# Example feedback to demonstrate learning capabilities
EXAMPLE_FEEDBACK = [
    "Great response, very thorough and well-organized!",
    "The information was accurate but could have been more concise.",
    "I would have liked more specific details about pricing.",
    "Excellent comparison, the table format made it easy to understand."
]

async def load_all_tools(session: ClientSession) -> list[BaseTool]:
    """Load both standard MCP tools and custom Bright Data tools."""
    # Load standard MCP tools
    standard_tools = await load_mcp_tools(session)

    # Load custom Bright Data tools
    bright_data_toolkit = BrightDataToolkit(session)
    custom_tools = await bright_data_toolkit.create_custom_tools()

    # Combine tools, with custom tools taking precedence if there are name conflicts
    tool_dict = {tool.name: tool for tool in standard_tools}

    # Add custom tools, potentially overriding standard tools with the same name
    for tool in custom_tools:
        tool_dict[tool.name] = tool

    return list(tool_dict.values())

async def run_enhanced_capabilities_example():
    """Run the enhanced capabilities example."""
    print("Starting enhanced capabilities example...")

    # Use a temporary database for the example
    db_path = "example_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing enhanced agent architecture...")

            # Initialize the language model
            model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

            # Load all tools
            tools = await load_all_tools(session)

            # Create the enhanced agent architecture
            coordinator = await create_enhanced_agent_architecture(model, tools, db_path)

            print("\nEnhanced agent architecture initialized.")

            # Demonstrate memory persistence
            print("\n" + "=" * 80)
            print("DEMONSTRATING MEMORY PERSISTENCE")
            print("=" * 80)

            # Process a sequence of related tasks
            print("\nProcessing a sequence of related tasks...")

            task1 = "What is the current price of Bitcoin?"
            print(f"\nTask 1: {task1}")
            response1 = await coordinator.process_request(task1)
            print(f"Response: {response1}")

            task2 = "How does that compare to last month?"
            print(f"\nTask 2: {task2}")
            response2 = await coordinator.process_request(task2)
            print(f"Response: {response2}")

            task3 = "And what about Ethereum?"
            print(f"\nTask 3: {task3}")
            response3 = await coordinator.process_request(task3)
            print(f"Response: {response3}")

            # Show memory state
            print("\nMemory state after sequence of tasks:")
            memory_summary = coordinator.memory_db.get_memory_summary()
            print(memory_summary)

            # Demonstrate enhanced tool selection
            print("\n" + "=" * 80)
            print("DEMONSTRATING ENHANCED TOOL SELECTION")
            print("=" * 80)

            # Process tasks that require different tools
            for i, task in enumerate(EXAMPLE_TASKS, 1):
                print(f"\nTask {i}: {task}")

                # Get tool selection
                history = coordinator.conversation_history[-5:] if len(coordinator.conversation_history) > 5 else coordinator.conversation_history
                tool_selection = await coordinator.tool_selector.select_tools(task, history)

                print(f"Selected tools: {', '.join(tool_selection['selected_tools'])}")
                print(f"Reasoning: {tool_selection['reasoning']}")
                print(f"Execution order: {', '.join(tool_selection['execution_order'])}")
                print(f"Fallback tools: {', '.join(tool_selection['fallback_tools'])}")

                # Process the task
                response = await coordinator.process_request(task)
                print(f"Response: {response[:200]}...")  # Show first 200 chars

                # Provide feedback
                if i <= len(EXAMPLE_FEEDBACK):
                    feedback = EXAMPLE_FEEDBACK[i-1]
                    print(f"Providing feedback: {feedback}")
                    await coordinator.collect_user_feedback(task, response, feedback)

            # Demonstrate learning capabilities
            print("\n" + "=" * 80)
            print("DEMONSTRATING LEARNING CAPABILITIES")
            print("=" * 80)

            # Trigger learning from feedback
            print("\nLearning from collected feedback...")
            insights = await coordinator.learn_from_feedback()

            # Show learning insights
            print("\nLearning insights:")
            insights_summary = await coordinator.get_learning_insights()
            print(insights_summary)

            # Process a final task to demonstrate improved performance
            final_task = "What are the main differences between iPhone 15 Pro and Samsung Galaxy S23 Ultra?"
            print(f"\nFinal task: {final_task}")
            final_response = await coordinator.process_request(final_task)
            print(f"Response: {final_response}")

            print("\nEnhanced capabilities example completed.")

if __name__ == "__main__":
    asyncio.run(run_enhanced_capabilities_example())
