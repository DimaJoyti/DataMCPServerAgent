"""
SEO Agent entry point for DataMCPServerAgent.
This module provides the main entry point for the SEO agent.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.agent_architecture import AgentMemory
from src.agents.seo.seo_agent import SEOAgent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user

# Load environment variables
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

# Initialize the language model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))


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


async def chat_with_seo_agent(config: Optional[Dict[str, Any]] = None):
    """Run the SEO agent.
    
    Args:
        config: Optional configuration for the agent
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Initializing DataMCPServerAgent with SEO capabilities...")
            
            # Load all tools
            tools = await load_all_tools(session)
            
            # Initialize agent memory
            memory = AgentMemory(max_history_length=50)
            
            # Initialize memory database
            db_path = os.getenv("MEMORY_DB_PATH", "seo_agent_memory.db")
            db = MemoryDatabase(db_path)
            
            # Initialize SEO agent
            seo_agent = SEOAgent(model, memory, db, tools)
            
            # Process user input in a loop
            print("\nSEO Agent initialized. Type 'exit' to quit.")
            print("How can I help you with your SEO needs today?")
            
            while True:
                # Get user input
                user_input = input("\nYou: ")
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Exiting SEO Agent. Goodbye!")
                    break
                
                try:
                    # Process the user input
                    response = await seo_agent.process_request(user_input)
                    
                    # Display the response
                    print(f"\nSEO Agent: {response}")
                    
                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"\nError: {error_message}")
                    print("Please try again with a different request.")


if __name__ == "__main__":
    asyncio.run(chat_with_seo_agent())
