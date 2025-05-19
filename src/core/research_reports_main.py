"""
Research reports entry point for DataMCPServerAgent.
This version implements a specialized agent for generating comprehensive research reports.
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

from src.agents.research_reports.research_reports_agent import ResearchReportsAgent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user
from src.utils.env_config import get_mcp_server_params

# Load environment variables
load_dotenv()


async def load_all_tools(session: ClientSession = None) -> List[BaseTool]:
    """Load both standard MCP tools and custom tools for research reports.

    Args:
        session: An initialized MCP ClientSession, or None if MCP is not available

    Returns:
        A combined list of standard and custom tools
    """
    tools = []

    # Load MCP tools if session is available
    if session:
        try:
            # Load standard MCP tools
            standard_tools = await load_mcp_tools(session)
            tools.extend(standard_tools)

            # Load custom Bright Data tools
            bright_data_toolkit = BrightDataToolkit(session)
            custom_tools = await bright_data_toolkit.create_custom_tools()
            tools.extend(custom_tools)
        except Exception as e:
            print(f"Warning: Could not load MCP tools: {e}")

    # Load research tools
    research_tools = _load_research_tools()
    tools.extend(research_tools)

    # Deduplicate tools by name, with later tools taking precedence
    tool_dict = {}
    for tool in tools:
        tool_dict[tool.name] = tool

    return list(tool_dict.values())


def _load_research_tools() -> List[BaseTool]:
    """Load research-specific tools.

    Returns:
        List of research tools
    """
    # Import research tools
    try:
        from src.tools.research_assistant_tools import search_tool, wiki_tool, save_tool
        from src.tools.academic_tools import (
            google_scholar_tool, pubmed_tool, arxiv_tool,
            google_books_tool, open_library_tool
        )
        from src.tools.export_tools import (
            export_to_markdown_tool, export_to_html_tool,
            export_to_pdf_tool, export_to_docx_tool
        )

        return [
            search_tool, wiki_tool, save_tool,
            google_scholar_tool, pubmed_tool, arxiv_tool,
            google_books_tool, open_library_tool,
            export_to_markdown_tool, export_to_html_tool,
            export_to_pdf_tool, export_to_docx_tool
        ]
    except ImportError as e:
        print(f"Warning: Could not import all research tools: {e}")
        # Return empty list if tools are not available
        return []


async def create_research_reports_agent(
    model: ChatAnthropic,
    tools: List[BaseTool],
    db: MemoryDatabase,
    config: Dict[str, Any] = None
) -> ResearchReportsAgent:
    """Create a research reports agent.

    Args:
        model: Language model to use
        tools: List of available tools
        db: Memory database for persistence
        config: Configuration for the agent

    Returns:
        Research reports agent
    """
    # Get report templates
    templates = config.get("templates") if config else None

    # Create the research reports agent
    agent = ResearchReportsAgent(model, tools, db, templates)

    return agent


async def chat_loop(agent: ResearchReportsAgent, session: Optional[ClientSession] = None):
    """Run the chat loop for the research reports agent.

    Args:
        agent: Research reports agent
        session: MCP client session, or None if MCP is not available
    """
    print("Welcome to the Research Reports Agent!")
    print("Type 'research [topic]' to generate a research report.")
    print("Type 'exit' to quit.")
    print()

    if session is None:
        print("Note: Running with local tools only. Some web search capabilities may be limited.")

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Process the user input
        try:
            response = await agent.process_request(user_input)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            error_message = format_error_for_user(e)
            print(f"\nError: {error_message}\n")


async def chat_with_research_reports_agent(config: Dict[str, Any] = None):
    """Chat with the research reports agent.

    Args:
        config: Configuration for the agent
    """
    # Initialize model
    model_name = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620")
    model = ChatAnthropic(model=model_name)

    # Initialize memory database
    db_path = os.getenv("MEMORY_DB_PATH", "research_reports_memory.db")
    db = MemoryDatabase(db_path)

    try:
        # Try to set up the MCP server
        print("Attempting to start MCP client...")
        server_params = StdioServerParameters(
            command="npx",
            env=get_mcp_server_params()["env"],
            args=["@brightdata/mcp"],
        )

        # Start the MCP client
        async with stdio_client(server_params) as session:
            # Load tools
            print("Loading tools with MCP...")
            tools = await load_all_tools(session)

            # Create the research reports agent
            print("Creating research reports agent...")
            agent = await create_research_reports_agent(model, tools, db, config)

            # Start the chat loop
            await chat_loop(agent, session)
    except Exception as e:
        print(f"Warning: Could not start MCP client: {e}")
        print("Falling back to local tools only...")

        # Load tools without MCP
        tools = await load_all_tools()

        # Create the research reports agent
        print("Creating research reports agent with local tools only...")
        agent = await create_research_reports_agent(model, tools, db, config)

        # Start the chat loop without MCP session
        await chat_loop(agent, None)


if __name__ == "__main__":
    asyncio.run(chat_with_research_reports_agent())
