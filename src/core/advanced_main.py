"""
Advanced main entry point for DataMCPServerAgent with specialized sub-agents.
This version uses a more sophisticated agent architecture with specialized sub-agents,
tool selection, and memory integration.
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

from src.agents.agent_architecture import (
    AgentMemory,
    CoordinatorAgent,
    SpecializedSubAgent,
    ToolSelectionAgent,
    create_specialized_sub_agents
)
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user

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

async def chat_with_advanced_agent():
    """Run the advanced agent with specialized sub-agents and memory."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing DataMCPServerAgent with advanced architecture...")

            # Load all tools
            tools = await load_all_tools(session)

            # Initialize agent memory
            memory = AgentMemory(max_history_length=50)

            # Initialize tool selection agent
            tool_selector = ToolSelectionAgent(model, tools)

            # Create specialized sub-agents
            sub_agents = create_specialized_sub_agents(model, tools)

            # Initialize coordinator agent
            coordinator = CoordinatorAgent(model, sub_agents, tool_selector, memory)

            # Add initial system message to memory
            memory.add_message({
                "role": "system",
                "content": "You are an advanced AI assistant with specialized capabilities for web automation and data collection using Bright Data MCP tools."
            })

            print("DataMCPServerAgent initialized with advanced architecture.")
            print("Available specialized agents:")
            for name, agent in sub_agents.items():
                print(f"- {agent.name} ({name})")

            print("\nType 'exit' or 'quit' to end the chat.")
            print("Type 'memory' to see the current memory state.")
            print("Type 'agents' to see the available specialized agents.")

            while True:
                user_input = input("\nYou: ")

                # Check for special commands
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                elif user_input.strip().lower() == "memory":
                    print("\nCurrent Memory State:")
                    print(memory.get_memory_summary())
                    continue

                elif user_input.strip().lower() == "agents":
                    print("\nAvailable Specialized Agents:")
                    for name, agent in sub_agents.items():
                        print(f"- {agent.name} ({name})")
                        print(f"  Tools: {', '.join(tool.name for tool in agent.tools)}")
                    continue

                # Process regular user input
                print("Processing your request...")

                try:
                    # Process the request through the coordinator
                    response = await coordinator.process_request(user_input)
                    print(f"Agent: {response}")
                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"Agent: An error occurred: {error_message}")

                    # Add error message to memory
                    memory.add_message({
                        "role": "assistant",
                        "content": f"An error occurred: {error_message}"
                    })

if __name__ == "__main__":
    asyncio.run(chat_with_advanced_agent())
