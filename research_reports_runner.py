"""
Runner script for the Research Reports Agent.
This script runs only the research reports agent without loading other agents.
"""

import asyncio
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from src.agents.research_reports.research_reports_agent import ResearchReportsAgent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.research_assistant_tools import search_tool, wiki_tool, save_tool
from src.tools.academic_tools import (
    google_scholar_tool, pubmed_tool, arxiv_tool,
    google_books_tool, open_library_tool
)
from src.tools.export_tools import (
    export_to_markdown_tool, export_to_html_tool,
    export_to_pdf_tool, export_to_docx_tool
)

# Load environment variables
load_dotenv()


async def create_research_reports_agent(
    model: ChatAnthropic,
    tools: list[BaseTool],
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


async def chat_loop(agent: ResearchReportsAgent):
    """Run the chat loop for the research reports agent.
    
    Args:
        agent: Research reports agent
    """
    print("Welcome to the Research Reports Agent!")
    print("Type 'research [topic]' to generate a research report.")
    print("Type 'exit' to quit.")
    print()
    
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
            print(f"\nError: {str(e)}\n")


async def run_research_reports_agent(config: Dict[str, Any] = None):
    """Run the research reports agent.
    
    Args:
        config: Configuration for the agent
    """
    # Initialize model
    model_name = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620")
    model = ChatAnthropic(model=model_name)
    
    # Initialize memory database
    db_path = os.getenv("MEMORY_DB_PATH", "research_reports_memory.db")
    db = MemoryDatabase(db_path)
    
    # Load tools
    tools = [
        search_tool, wiki_tool, save_tool,
        google_scholar_tool, pubmed_tool, arxiv_tool,
        google_books_tool, open_library_tool,
        export_to_markdown_tool, export_to_html_tool,
        export_to_pdf_tool, export_to_docx_tool
    ]
    
    # Create the research reports agent
    print("Creating research reports agent...")
    agent = await create_research_reports_agent(model, tools, db, config)
    
    # Start the chat loop
    await chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(run_research_reports_agent())
