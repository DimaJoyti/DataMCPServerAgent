"""
Main entry point for DataMCPServerAgent.
"""

import asyncio
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.env_config import get_mcp_server_params, get_model_config

# Initialize model with configuration from environment
model_config = get_model_config()
model = ChatAnthropic(model=model_config["model_name"])

# Initialize server parameters with configuration from environment
server_params = StdioServerParameters(
    command="npx",
    env=get_mcp_server_params()["env"],
    args=["@brightdata/mcp"],
)

# Enhanced system prompt with detailed guidance on using Bright Data MCP tools
SYSTEM_PROMPT = """You are an advanced AI assistant with specialized capabilities for web automation and data collection using Bright Data MCP tools.

Your capabilities include:
1. Web searching with enhanced result formatting
2. Web scraping with content extraction and filtering
3. Product data collection and comparison across e-commerce sites
4. Social media content analysis across platforms

When approaching tasks:
- Break down complex requests into a sequence of tool operations
- Choose the most appropriate specialized tool for each subtask
- Provide clear explanations of what you're doing at each step
- Format results in a readable, structured way
- Handle errors gracefully and suggest alternatives when needed

For web scraping tasks:
- Use enhanced_web_scraper for general content extraction
- Extract specific content types (main content, tables, links) when appropriate
- Be mindful of website terms of service and privacy considerations

For product research:
- Use product_comparison for comparing multiple products
- Provide structured comparisons of features, prices, and reviews
- Highlight key differences between products

For social media analysis:
- Use social_media_analyzer to extract insights from posts and profiles
- Analyze engagement metrics when relevant
- Identify trends and patterns in social content

Always think step by step and provide clear, actionable insights from the data you collect.
"""

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

async def chat_with_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load both standard and custom tools
            tools = await load_all_tools(session)

            # Create the agent with the enhanced tools
            agent = create_react_agent(model, tools)

            # Start conversation history with enhanced system prompt
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print(
                "DataMCPServerAgent initialized with enhanced Bright Data capabilities."
            )
            print("Type 'exit' or 'quit' to end the chat.")

            while True:
                user_input = input("\nYou: ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                # Add user message to history
                messages.append({"role": "user", "content": user_input})

                # Provide feedback that the agent is working
                print("Processing your request...")

                try:
                    # Call the agent with the full message history
                    agent_response = await agent.ainvoke({"messages": messages})

                    # Extract agent's reply and add to history
                    ai_message = agent_response["messages"][-1].content
                    messages.append({"role": "assistant", "content": ai_message})
                    print(f"Agent: {ai_message}")
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    print(f"Agent: {error_message}")
                    # Add error message to history
                    messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    asyncio.run(chat_with_agent())
