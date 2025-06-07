"""
Enhanced main entry point for DataMCPServerAgent with memory persistence,
enhanced tool selection, and learning capabilities.
"""

import asyncio
import os
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.tools.bright_data_tools import BrightDataToolkit
from src.agents.enhanced_agent_architecture import create_enhanced_agent_architecture
from src.utils.error_handlers import format_error_for_user
from src.memory.memory_persistence import MemoryDatabase

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)

async def load_all_tools(session: ClientSession) -> List[BaseTool]:
    """Load both standard MCP tools and custom Bright Data tools.

    Args:
        session: An initialized MCP ClientSession

    Returns:
        A combined list of standard and custom tools
    """
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

async def chat_with_enhanced_agent():
    """Run the enhanced agent with memory persistence, tool selection, and learning."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing DataMCPServerAgent with enhanced capabilities...")

            # Load all tools
            tools = await load_all_tools(session)

            # Create the enhanced agent architecture
            coordinator = await create_enhanced_agent_architecture(model, tools)

            print("DataMCPServerAgent initialized with enhanced capabilities:")
            print("- Memory persistence")
            print("- Enhanced tool selection")
            print("- Learning capabilities")

            print("\nType 'exit' or 'quit' to end the chat.")
            print("Type 'memory' to see the current memory state.")
            print("Type 'learn' to trigger learning from feedback.")
            print("Type 'insights' to see learning insights.")
            print("Type 'feedback <your feedback>' to provide feedback on the last response.")

            last_request = ""
            last_response = ""

            while True:
                user_input = input("\nYou: ")

                # Check for special commands
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                elif user_input.strip().lower() == "memory":
                    # Get memory summary from the database
                    memory_summary = coordinator.memory_db.get_memory_summary()
                    print("\nCurrent Memory State:")
                    print(memory_summary)
                    continue

                elif user_input.strip().lower() == "learn":
                    print("\nLearning from feedback...")
                    insights = await coordinator.learn_from_feedback()
                    print("Learning complete. Use 'insights' to see the results.")
                    continue

                elif user_input.strip().lower() == "insights":
                    print("\nLearning Insights:")
                    insights_summary = await coordinator.get_learning_insights()
                    print(insights_summary)
                    continue

                elif user_input.strip().lower().startswith("feedback "):
                    if last_request and last_response:
                        feedback = user_input[9:].strip()
                        await coordinator.collect_user_feedback(last_request, last_response, feedback)
                        print("Thank you for your feedback! It will help me improve.")
                    else:
                        print("No previous interaction to provide feedback on.")
                    continue

                # Process regular user input
                print("Processing your request...")

                try:
                    # Save the request for potential feedback
                    last_request = user_input

                    # Process the request through the coordinator
                    response = await coordinator.process_request(user_input)

                    # Save the response for potential feedback
                    last_response = response

                    print(f"Agent: {response}")

                    # Perform self-evaluation in the background
                    asyncio.create_task(coordinator.feedback_collector.perform_self_evaluation(
                        user_input, response, "coordinator"
                    ))

                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"Agent: An error occurred: {error_message}")

if __name__ == "__main__":
    asyncio.run(chat_with_enhanced_agent())
