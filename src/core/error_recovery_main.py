"""
Error recovery entry point for DataMCPServerAgent.
This version implements sophisticated retry strategies, automatic fallback mechanisms,
and a self-healing system that learns from errors.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.enhanced_agent_architecture import create_enhanced_agent_architecture
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker
from src.utils.env_config import get_mcp_server_params, get_model_config
from src.utils.error_handlers import format_error_for_user
from src.utils.error_recovery import ErrorRecoverySystem

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ErrorRecoveryCoordinatorAgent:
    """Coordinator agent with enhanced error recovery capabilities."""

    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        db: MemoryDatabase,
        error_recovery: ErrorRecoverySystem,
    ):
        """Initialize the error recovery coordinator agent.

        Args:
            model: Language model to use
            tools: List of available tools
            db: Memory database for persistence
            error_recovery: Error recovery system
        """
        self.model = model
        self.tools = tools
        self.db = db
        self.error_recovery = error_recovery

        # Create tool performance tracker
        self.tool_performance_tracker = ToolPerformanceTracker(db)

        # Create enhanced tool selector
        self.tool_selector = EnhancedToolSelector(model, tools, db, self.tool_performance_tracker)

        # Create specialized sub-agents
        self.sub_agents = create_specialized_sub_agents(model, tools)

        # Create enhanced agent architecture
        self.enhanced_agent = create_enhanced_agent_architecture(
            model, tools, db, self.tool_selector, self.tool_performance_tracker
        )

        # Conversation history
        self.conversation_history = []

    async def process_request(self, request: str) -> str:
        """Process a user request with enhanced error recovery.

        Args:
            request: User request

        Returns:
            Agent response
        """
        try:
            # Add request to conversation history
            self.conversation_history.append({"role": "user", "content": request})

            # Get tool selection
            tool_selection = await self.tool_selector.select_tools(
                request,
                (
                    self.conversation_history[-5:]
                    if len(self.conversation_history) > 5
                    else self.conversation_history
                ),
            )

            # Log tool selection
            logger.info(f"Selected tools: {', '.join(tool_selection['selected_tools'])}")
            logger.info(f"Fallback tools: {', '.join(tool_selection['fallback_tools'])}")

            # Process the request with error recovery
            if tool_selection["selected_tools"]:
                primary_tool = tool_selection["selected_tools"][0]

                # Prepare context for error recovery
                context = {
                    "request": request,
                    "selected_tools": tool_selection["selected_tools"],
                    "fallback_tools": tool_selection["fallback_tools"],
                    "reasoning": tool_selection["reasoning"],
                }

                # Try with fallbacks
                try:
                    # Extract arguments for the tool
                    tool_args = self._extract_tool_args(request, primary_tool)

                    # Try with fallbacks
                    result, tool_used, success = await self.error_recovery.try_with_fallbacks(
                        primary_tool,
                        tool_args,
                        context,
                        max_fallbacks=len(tool_selection["fallback_tools"]),
                    )

                    # Log the result
                    if success:
                        logger.info(f"Successfully used tool: {tool_used}")
                    else:
                        logger.warning(f"All tools failed for request: {request}")

                    # Format the response
                    response = self._format_response(request, result, tool_used)
                except Exception as e:
                    # All tools failed, use the enhanced agent as a fallback
                    logger.warning(f"Error recovery failed: {str(e)}")
                    response = await self.enhanced_agent.process_request(request)
            else:
                # No tools selected, use the enhanced agent
                response = await self.enhanced_agent.process_request(request)

            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            return response
        except Exception as e:
            error_message = format_error_for_user(e)

            # Add error to conversation history
            self.conversation_history.append(
                {"role": "assistant", "content": f"Error: {error_message}"}
            )

            return f"An error occurred: {error_message}"

    def _extract_tool_args(self, request: str, tool_name: str) -> Dict[str, Any]:
        """Extract arguments for a tool from the request.

        Args:
            request: User request
            tool_name: Name of the tool

        Returns:
            Tool arguments
        """
        # Simple argument extraction based on tool name
        if "scrape" in tool_name.lower() or "web_data" in tool_name.lower():
            # Extract URL from request
            import re

            url_match = re.search(r'https?://[^\s"\']+', request)
            if url_match:
                return {"url": url_match.group(0)}
            else:
                return {"url": "https://www.example.com"}  # Default URL
        elif "search" in tool_name.lower():
            # Use the request as the search query
            return {"query": request}
        else:
            # Default arguments
            return {"input": request}

    def _format_response(self, request: str, result: Any, tool_used: str) -> str:
        """Format the response based on the tool result.

        Args:
            request: Original user request
            result: Tool result
            tool_used: Name of the tool used

        Returns:
            Formatted response
        """
        # Format the response based on the tool result
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Convert dictionary to string
            import json

            return json.dumps(result, indent=2)
        else:
            # Default formatting
            return f"Result from {tool_used}: {str(result)}"


async def setup_error_recovery_agent():
    """Set up the error recovery agent.

    Returns:
        Tuple of (coordinator, model, session)
    """
    # Initialize model
    model_config = get_model_config()
    model = ChatAnthropic(model=model_config["model_name"])

    # Initialize memory database
    db_path = os.getenv("MEMORY_DB_PATH", "error_recovery_agent.db")
    db = MemoryDatabase(db_path)

    # Set up MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        env=get_mcp_server_params()["env"],
        args=["@brightdata/mcp"],
    )

    # Initialize MCP client session
    async with stdio_client(server_params) as session:
        # Load MCP tools
        mcp_tools = await load_mcp_tools(session)

        # Create Bright Data toolkit
        bright_data_toolkit = BrightDataToolkit(session)

        # Create custom tools
        custom_tools = await bright_data_toolkit.create_custom_tools()

        # Combine all tools
        all_tools = mcp_tools + custom_tools

        # Create error recovery system
        error_recovery = ErrorRecoverySystem(model, db, all_tools)

        # Create error recovery coordinator agent
        coordinator = ErrorRecoveryCoordinatorAgent(model, all_tools, db, error_recovery)

        return coordinator, model, session


async def chat_with_error_recovery_agent():
    """Chat with the error recovery agent."""
    print("Starting DataMCPServerAgent with Enhanced Error Recovery...")
    print("Type 'exit' to quit, 'learn' to trigger learning from errors.")

    try:
        # Set up the error recovery agent
        coordinator, model, session = await setup_error_recovery_agent()

        # Chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            # Check for learn command
            if user_input.lower() == "learn":
                print("\nLearning from errors...")
                learning_result = await coordinator.error_recovery.learn_from_errors()
                print(f"Learning result: {learning_result}")
                continue

            # Process regular user input
            print("Processing your request...")

            try:
                # Process the request
                response = await coordinator.process_request(user_input)
                print(f"\nAgent: {response}")
            except Exception as e:
                error_message = format_error_for_user(e)
                print(f"\nAgent: An error occurred: {error_message}")
    except Exception as e:
        print(f"Error setting up agent: {str(e)}")


if __name__ == "__main__":
    asyncio.run(chat_with_error_recovery_agent())
