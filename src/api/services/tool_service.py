"""
Tool service for the API.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set

from ..config import config
from ..models.response_models import ToolResponse

class ToolService:
    """Service for tool operations."""

    async def list_tools(
        self,
        agent_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available tools.

        Args:
            agent_mode (Optional[str]): Agent mode to filter tools

        Returns:
            List[Dict[str, Any]]: List of tools
        """
        # Define tools for each agent mode
        tools_by_mode = {
            "basic": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
            ],
            "advanced": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
            ],
            "enhanced": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
            ],
            "advanced_enhanced": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "sentiment_analysis", "description": "Analyze sentiment"},
                {"name": "translation", "description": "Translate text"},
            ],
            "multi_agent": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "sentiment_analysis", "description": "Analyze sentiment"},
                {"name": "translation", "description": "Translate text"},
                {"name": "collaborative_search", "description": "Collaborative search with multiple agents"},
                {"name": "knowledge_sharing", "description": "Share knowledge between agents"},
            ],
            "reinforcement_learning": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "reinforcement_learning", "description": "Reinforcement learning tools"},
            ],
            "distributed_memory": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "distributed_memory", "description": "Distributed memory tools"},
            ],
            "knowledge_graph": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "knowledge_graph", "description": "Knowledge graph tools"},
            ],
            "error_recovery": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "error_recovery", "description": "Error recovery tools"},
            ],
            "research_reports": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "research", "description": "Research tools"},
                {"name": "report_generation", "description": "Report generation tools"},
            ],
            "seo": [
                {"name": "web_search", "description": "Search the web for information"},
                {"name": "web_browse", "description": "Browse a web page"},
                {"name": "calculator", "description": "Perform calculations"},
                {"name": "product_search", "description": "Search for products"},
                {"name": "image_search", "description": "Search for images"},
                {"name": "data_analysis", "description": "Analyze data"},
                {"name": "code_generation", "description": "Generate code"},
                {"name": "seo_analysis", "description": "SEO analysis tools"},
                {"name": "seo_optimization", "description": "SEO optimization tools"},
            ],
        }

        # Get tools for the agent mode
        if agent_mode:
            return tools_by_mode.get(agent_mode, [])

        # If no agent mode is specified, return all tools
        all_tools = []
        for mode_tools in tools_by_mode.values():
            for tool in mode_tools:
                if tool not in all_tools:
                    all_tools.append(tool)

        return all_tools

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_mode: Optional[str] = None,
    ) -> ToolResponse:
        """
        Execute a tool.

        Args:
            tool_name (str): Name of the tool to use
            tool_input (Dict[str, Any]): Input for the tool
            session_id (Optional[str]): Session ID for the tool operation
            agent_mode (Optional[str]): Agent mode to use for the tool operation

        Returns:
            ToolResponse: Tool response
        """
        # Use default agent mode if not provided
        agent_mode = agent_mode or config.default_agent_mode

        # Start timing
        start_time = time.time()

        try:
            # Get the tool function
            tool_function = self._get_tool_function(tool_name)

            if tool_function:
                # Execute the tool
                result = await self._execute_tool_function(
                    tool_function=tool_function,
                    tool_input=tool_input,
                    session_id=session_id,
                    agent_mode=agent_mode,
                )

                # Calculate execution time
                execution_time = time.time() - start_time

                # Log tool usage
                await self.log_tool_usage(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=result,
                )

                return ToolResponse(
                    tool_name=tool_name,
                    tool_output=result,
                    execution_time=execution_time,
                    status="success",
                    metadata={
                        "session_id": session_id,
                        "agent_mode": agent_mode,
                        "tool_input": tool_input,
                    },
                )
            else:
                # Tool not found
                execution_time = time.time() - start_time

                return ToolResponse(
                    tool_name=tool_name,
                    tool_output=f"Tool '{tool_name}' not found",
                    execution_time=execution_time,
                    status="error",
                    metadata={
                        "session_id": session_id,
                        "agent_mode": agent_mode,
                        "tool_input": tool_input,
                        "error": "Tool not found",
                    },
                )
        except Exception as e:
            # Handle errors
            execution_time = time.time() - start_time

            return ToolResponse(
                tool_name=tool_name,
                tool_output=f"Error executing tool '{tool_name}': {str(e)}",
                execution_time=execution_time,
                status="error",
                metadata={
                    "session_id": session_id,
                    "agent_mode": agent_mode,
                    "tool_input": tool_input,
                    "error": str(e),
                },
            )

    def _get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """
        Get the tool function for a tool name.

        Args:
            tool_name (str): Name of the tool

        Returns:
            Optional[Callable]: Tool function or None if not found
        """
        # Try to import tools from the project
        try:
            # Import tools from different modules
            from src.tools.web_tools import web_search, web_browse
            from src.tools.calculator import calculate
            from src.tools.data_analysis import analyze_data
            from src.tools.code_generation import generate_code

            # Map tool names to functions
            tool_functions = {
                "web_search": web_search,
                "web_browse": web_browse,
                "calculator": calculate,
                "data_analysis": analyze_data,
                "code_generation": generate_code,
            }

            return tool_functions.get(tool_name)
        except ImportError:
            # If tools are not available, return None
            return None

    async def _execute_tool_function(
        self,
        tool_function: Callable,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_mode: Optional[str] = None,
    ) -> Any:
        """
        Execute a tool function.

        Args:
            tool_function (Callable): Tool function to execute
            tool_input (Dict[str, Any]): Input for the tool
            session_id (Optional[str]): Session ID for the tool operation
            agent_mode (Optional[str]): Agent mode to use for the tool operation

        Returns:
            Any: Result of the tool execution
        """
        # Check if the tool function is async
        if asyncio.iscoroutinefunction(tool_function):
            # Execute async function
            result = await tool_function(**tool_input)
        else:
            # Execute sync function
            result = tool_function(**tool_input)

        return result

    async def log_tool_usage(
        self,
        session_id: Optional[str],
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
    ) -> None:
        """
        Log tool usage.

        Args:
            session_id (Optional[str]): Session ID
            tool_name (str): Name of the tool
            tool_input (Dict[str, Any]): Input for the tool
            tool_output (Any): Output from the tool
        """
        if not session_id:
            return

        try:
            # Get session service
            from .session_service import SessionService
            session_service = SessionService()

            # Log tool usage
            await session_service.save_tool_usage(
                session_id=session_id,
                tool_name=tool_name,
                args=tool_input,
                result=tool_output,
            )
        except Exception as e:
            # If there's an error, just print it
            print(f"Error logging tool usage: {str(e)}")
