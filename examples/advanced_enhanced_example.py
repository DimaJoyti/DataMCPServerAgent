"""
Example script demonstrating the advanced enhanced capabilities of DataMCPServerAgent.
This example shows context-aware memory, adaptive learning, and user preference modeling.
"""

import asyncio
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_enhanced_main import create_advanced_enhanced_agent
from bright_data_tools import BrightDataToolkit
from dotenv import load_dotenv
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

# Example tasks to demonstrate the advanced enhanced capabilities
EXAMPLE_TASKS = [
    # Initial tasks to establish context and preferences
    "What is the current price of Bitcoin?",
    "I prefer information with more technical details.",
    "Can you show me a price comparison between Bitcoin and Ethereum?",
    "I like when you present data in table format.",

    # Tasks to demonstrate context awareness
    "How has it changed since yesterday?",
    "What factors are influencing these changes?",

    # Tasks to demonstrate learning and adaptation
    "Find information about the latest iPhone model.",
    "I'm more interested in the technical specifications than the price.",
    "Compare it with the latest Samsung Galaxy.",
    "That was too detailed. I prefer more concise comparisons.",

    # Tasks to demonstrate memory persistence
    "What was the Bitcoin price you told me earlier?",
    "Summarize all the information you've provided so far."
]

# Example feedback to demonstrate learning capabilities
EXAMPLE_FEEDBACK = [
    "Great response, very thorough!",
    "Too much information, please be more concise.",
    "I like the table format, that's perfect.",
    "Could you include more technical details next time?",
    "This is exactly what I was looking for, thank you!",
    "The comparison was helpful but too wordy."
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

async def run_advanced_enhanced_example():
    """Run the advanced enhanced capabilities example."""
    print("Starting advanced enhanced capabilities example...")

    # Use a temporary database for the example
    db_path = "advanced_example_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing advanced enhanced agent...")

            # Initialize the language model
            model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

            # Load all tools
            tools = await load_all_tools(session)

            # Create the advanced enhanced agent
            agent = await create_advanced_enhanced_agent(model, tools, db_path)

            print("\nAdvanced enhanced agent initialized.")

            # Demonstrate context-aware memory
            print("\n" + "=" * 80)
            print("DEMONSTRATING CONTEXT-AWARE MEMORY")
            print("=" * 80)

            # Process initial tasks to establish context
            print("\nEstablishing initial context...")

            for i, task in enumerate(EXAMPLE_TASKS[:4], 1):
                print(f"\nTask {i}: {task}")
                response = await agent.process_request(task)
                print(f"Response: {response}")

                # Provide feedback for some tasks
                if i <= len(EXAMPLE_FEEDBACK):
                    feedback = EXAMPLE_FEEDBACK[i-1]
                    print(f"Providing feedback: {feedback}")
                    await agent.collect_feedback(feedback)

            # Show current context
            print("\nCurrent context after initial tasks:")
            context = agent.get_context()
            print(context)

            # Demonstrate context-aware responses
            print("\nDemonstrating context-aware responses...")

            for i, task in enumerate(EXAMPLE_TASKS[4:6], 5):
                print(f"\nTask {i}: {task}")
                response = await agent.process_request(task)
                print(f"Response: {response}")

            # Demonstrate adaptive learning
            print("\n" + "=" * 80)
            print("DEMONSTRATING ADAPTIVE LEARNING")
            print("=" * 80)

            # Process tasks to demonstrate learning
            print("\nProcessing tasks to demonstrate learning...")

            for i, task in enumerate(EXAMPLE_TASKS[6:10], 7):
                print(f"\nTask {i}: {task}")
                response = await agent.process_request(task)
                print(f"Response: {response}")

                # Provide feedback for some tasks
                if i - 6 < len(EXAMPLE_FEEDBACK) - 4:
                    feedback = EXAMPLE_FEEDBACK[i - 6 + 4]
                    print(f"Providing feedback: {feedback}")
                    await agent.collect_feedback(feedback)

            # Trigger learning
            print("\nTriggering learning from feedback and performance data...")
            strategies = await agent.learn()

            print("\nLearning Strategies:")
            print(f"Focus Area: {strategies.get('learning_focus', 'None')}")

            if "improvement_strategies" in strategies:
                print("\nImprovement Strategies:")
                for i, strategy in enumerate(strategies["improvement_strategies"], 1):
                    print(f"{i}. {strategy.get('strategy', 'Unknown')} (Priority: {strategy.get('priority', 'medium')})")

            # Show current user preferences
            print("\nCurrent user preferences after learning:")
            preferences = agent.get_preferences()
            print(preferences)

            # Demonstrate memory persistence
            print("\n" + "=" * 80)
            print("DEMONSTRATING MEMORY PERSISTENCE")
            print("=" * 80)

            # Process tasks to demonstrate memory persistence
            print("\nProcessing tasks to demonstrate memory persistence...")

            for i, task in enumerate(EXAMPLE_TASKS[10:], 11):
                print(f"\nTask {i}: {task}")
                response = await agent.process_request(task)
                print(f"Response: {response}")

            # Show performance metrics
            print("\n" + "=" * 80)
            print("PERFORMANCE METRICS")
            print("=" * 80)

            metrics = agent.get_metrics()
            print("\nPerformance Metrics:")
            print(f"Requests Processed: {metrics['requests_processed']}")
            print(f"Successful Responses: {metrics['successful_responses']}")
            print(f"Errors: {metrics['errors']}")
            print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"Feedback Received: {metrics['feedback_received']}")

            if metrics['feedback_received'] > 0:
                positive_rate = (metrics['positive_feedback'] / metrics['feedback_received']) * 100
                negative_rate = (metrics['negative_feedback'] / metrics['feedback_received']) * 100
                print(f"Positive Feedback Rate: {positive_rate:.2f}%")
                print(f"Negative Feedback Rate: {negative_rate:.2f}%")

            print("\nAdvanced enhanced capabilities example completed.")

if __name__ == "__main__":
    asyncio.run(run_advanced_enhanced_example())
