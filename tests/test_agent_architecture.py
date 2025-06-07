"""
Tests for agent architecture module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agent_architecture import AgentMemory, ToolSelectionAgent, SpecializedSubAgent

class TestAgentMemory(unittest.TestCase):
    """Tests for AgentMemory class."""

    def setUp(self):
        """Set up test environment."""
        self.memory = AgentMemory(max_history_length=5)

    def test_add_message(self):
        """Test adding messages to memory."""
        # Add messages
        self.memory.add_message({"role": "user", "content": "Hello"})
        self.memory.add_message({"role": "assistant", "content": "Hi there!"})

        # Check that messages were added
        self.assertEqual(len(self.memory.conversation_history), 2)
        self.assertEqual(self.memory.conversation_history[0]["role"], "user")
        self.assertEqual(self.memory.conversation_history[0]["content"], "Hello")
        self.assertEqual(self.memory.conversation_history[1]["role"], "assistant")
        self.assertEqual(self.memory.conversation_history[1]["content"], "Hi there!")

    def test_max_history_length(self):
        """Test that history is trimmed to max_history_length."""
        # Add more messages than max_history_length
        for i in range(10):
            self.memory.add_message({"role": "user", "content": f"Message {i}"})

        # Check that only max_history_length messages are kept
        self.assertEqual(len(self.memory.conversation_history), 5)
        self.assertEqual(self.memory.conversation_history[0]["content"], "Message 5")
        self.assertEqual(self.memory.conversation_history[4]["content"], "Message 9")

    def test_add_tool_usage(self):
        """Test adding tool usage to memory."""
        # Add tool usage
        self.memory.add_tool_usage(
            "search_tool",
            {"query": "example search"},
            "Example search results"
        )

        # Check that tool usage was added
        self.assertIn("search_tool", self.memory.tool_usage_history)
        self.assertEqual(len(self.memory.tool_usage_history["search_tool"]), 1)
        self.assertEqual(self.memory.tool_usage_history["search_tool"][0]["args"]["query"], "example search")
        self.assertEqual(self.memory.tool_usage_history["search_tool"][0]["result"], "Example search results")

    def test_add_entity(self):
        """Test adding entities to memory."""
        # Add entity
        self.memory.add_entity(
            "product",
            "product123",
            {"name": "Example Product", "price": 99.99}
        )

        # Check that entity was added
        self.assertIn("product", self.memory.entity_memory)
        self.assertIn("product123", self.memory.entity_memory["product"])
        self.assertEqual(self.memory.entity_memory["product"]["product123"]["name"], "Example Product")
        self.assertEqual(self.memory.entity_memory["product"]["product123"]["price"], 99.99)

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        # Add messages
        for i in range(10):
            self.memory.add_message({"role": "user", "content": f"Message {i}"})

        # Get recent messages
        recent_messages = self.memory.get_recent_messages(3)

        # Check that only the specified number of messages are returned
        self.assertEqual(len(recent_messages), 3)
        self.assertEqual(recent_messages[0]["content"], "Message 7")
        self.assertEqual(recent_messages[2]["content"], "Message 9")

    def test_get_tool_usage(self):
        """Test getting tool usage history."""
        # Add tool usage
        self.memory.add_tool_usage(
            "search_tool",
            {"query": "example search 1"},
            "Example search results 1"
        )
        self.memory.add_tool_usage(
            "search_tool",
            {"query": "example search 2"},
            "Example search results 2"
        )
        self.memory.add_tool_usage(
            "other_tool",
            {"param": "example param"},
            "Example result"
        )

        # Get tool usage for specific tool
        search_tool_usage = self.memory.get_tool_usage("search_tool")

        # Check that only usage for the specified tool is returned
        self.assertIn("search_tool", search_tool_usage)
        self.assertEqual(len(search_tool_usage["search_tool"]), 2)
        self.assertEqual(search_tool_usage["search_tool"][0]["args"]["query"], "example search 1")
        self.assertEqual(search_tool_usage["search_tool"][1]["args"]["query"], "example search 2")

        # Get all tool usage
        all_tool_usage = self.memory.get_tool_usage()

        # Check that usage for all tools is returned
        self.assertIn("search_tool", all_tool_usage)
        self.assertIn("other_tool", all_tool_usage)
        self.assertEqual(len(all_tool_usage["search_tool"]), 2)
        self.assertEqual(len(all_tool_usage["other_tool"]), 1)

    def test_get_entity(self):
        """Test getting entities from memory."""
        # Add entities
        self.memory.add_entity(
            "product",
            "product123",
            {"name": "Example Product", "price": 99.99}
        )
        self.memory.add_entity(
            "product",
            "product456",
            {"name": "Another Product", "price": 49.99}
        )
        self.memory.add_entity(
            "user",
            "user789",
            {"name": "Example User", "email": "user@example.com"}
        )

        # Get specific entity
        product = self.memory.get_entity("product", "product123")

        # Check that the correct entity is returned
        self.assertEqual(product["name"], "Example Product")
        self.assertEqual(product["price"], 99.99)

        # Get entities by type
        products = self.memory.get_entities_by_type("product")

        # Check that all entities of the specified type are returned
        self.assertEqual(len(products), 2)
        self.assertIn("product123", products)
        self.assertIn("product456", products)
        self.assertEqual(products["product123"]["name"], "Example Product")
        self.assertEqual(products["product456"]["name"], "Another Product")

class TestToolSelectionAgent(unittest.TestCase):
    """Tests for ToolSelectionAgent class."""

    @patch('langchain_anthropic.ChatAnthropic')
    def setUp(self, mock_model):
        """Set up test environment."""
        self.mock_model = mock_model
        self.mock_model.ainvoke = AsyncMock()

        # Create mock tools
        self.mock_tools = [
            MagicMock(name="search_tool", description="Search the web"),
            MagicMock(name="scrape_tool", description="Scrape a website")
        ]

        self.agent = ToolSelectionAgent(self.mock_model, self.mock_tools)

    async def test_select_tools(self):
        """Test selecting tools for a request."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "selected_tools": ["search_tool"],
    "reasoning": "The request is asking for information that can be found through a web search.",
    "execution_order": ["search_tool"]
}
```"""
        self.mock_model.ainvoke.return_value = mock_response

        # Select tools
        result = await self.agent.select_tools("What is the capital of France?")

        # Check that the model was called with the correct input
        self.mock_model.ainvoke.assert_called_once()

        # Check that the result matches the expected output
        self.assertEqual(result["selected_tools"], ["search_tool"])
        self.assertEqual(result["reasoning"], "The request is asking for information that can be found through a web search.")
        self.assertEqual(result["execution_order"], ["search_tool"])

    async def test_select_tools_with_invalid_response(self):
        """Test selecting tools with an invalid model response."""
        # Mock model response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"
        self.mock_model.ainvoke.return_value = mock_response

        # Select tools
        result = await self.agent.select_tools("What is the capital of France?")

        # Check that a default selection is returned
        self.assertEqual(result["selected_tools"], ["search_tool"])
        self.assertIn("Error parsing tool selection", result["reasoning"])
        self.assertEqual(result["execution_order"], ["search_tool"])

class TestSpecializedSubAgent(unittest.TestCase):
    """Tests for SpecializedSubAgent class."""

    @patch('langchain_anthropic.ChatAnthropic')
    @patch('langgraph.prebuilt.create_react_agent')
    def setUp(self, mock_create_agent, mock_model):
        """Set up test environment."""
        self.mock_model = mock_model
        self.mock_create_agent = mock_create_agent
        self.mock_agent = MagicMock()
        self.mock_create_agent.return_value = self.mock_agent

        # Create mock tools
        self.mock_tools = [
            MagicMock(name="search_tool", description="Search the web"),
            MagicMock(name="scrape_tool", description="Scrape a website")
        ]

        self.agent = SpecializedSubAgent(
            "Search Agent",
            self.mock_model,
            self.mock_tools,
            "You are a specialized search agent."
        )

    @patch('src.utils.error_handlers.format_error_for_user')
    async def test_execute_success(self, mock_format_error):
        """Test executing a task successfully."""
        # Mock agent response
        mock_response = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        }
        self.mock_agent.ainvoke.return_value = mock_response

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.get_recent_messages.return_value = []

        # Execute task
        result = await self.agent.execute("What is the capital of France?", mock_memory)

        # Check that the agent was called with the correct input
        self.mock_agent.ainvoke.assert_called_once()

        # Check that the result matches the expected output
        self.assertTrue(result["success"])
        self.assertEqual(result["response"], "The capital of France is Paris.")
        self.assertEqual(result["agent"], "Search Agent")

        # Check that the response was added to memory
        mock_memory.add_message.assert_called_once_with(
            {"role": "assistant", "content": "The capital of France is Paris."}
        )

    @patch('src.utils.error_handlers.format_error_for_user')
    async def test_execute_error(self, mock_format_error):
        """Test executing a task with an error."""
        # Mock agent error
        self.mock_agent.ainvoke.side_effect = Exception("Test error")
        mock_format_error.return_value = "Formatted error message"

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.get_recent_messages.return_value = []

        # Execute task
        result = await self.agent.execute("What is the capital of France?", mock_memory)

        # Check that the agent was called with the correct input
        self.mock_agent.ainvoke.assert_called_once()

        # Check that the result matches the expected output
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Formatted error message")
        self.assertEqual(result["agent"], "Search Agent")

        # Check that the error was added to memory
        mock_memory.add_message.assert_called_once_with(
            {"role": "assistant", "content": "Error in Search Agent: Formatted error message"}
        )

if __name__ == '__main__':
    unittest.main()
