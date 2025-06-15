"""
Tests for memory persistence module.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_persistence import FileBackedMemoryDatabase, MemoryDatabase


class TestMemoryDatabase(unittest.TestCase):
    """Tests for MemoryDatabase class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db = MemoryDatabase(self.db_path)

    def tearDown(self):
        """Clean up test environment."""
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_save_load_conversation_history(self):
        """Test saving and loading conversation history."""
        # Test data
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]

        # Save conversation history
        self.db.save_conversation_history(messages)

        # Load conversation history
        loaded_messages = self.db.load_conversation_history()

        # Check that loaded messages match saved messages
        self.assertEqual(len(loaded_messages), len(messages))
        for i, message in enumerate(messages):
            self.assertEqual(loaded_messages[i]["role"], message["role"])
            self.assertEqual(loaded_messages[i]["content"], message["content"])

    def test_save_load_entity(self):
        """Test saving and loading entities."""
        # Test data
        entity_type = "product"
        entity_id = "product123"
        entity_data = {
            "name": "Example Product",
            "price": 99.99,
            "description": "This is an example product"
        }

        # Save entity
        self.db.save_entity(entity_type, entity_id, entity_data)

        # Load entity
        loaded_entity = self.db.load_entity(entity_type, entity_id)

        # Check that loaded entity matches saved entity
        self.assertEqual(loaded_entity["name"], entity_data["name"])
        self.assertEqual(loaded_entity["price"], entity_data["price"])
        self.assertEqual(loaded_entity["description"], entity_data["description"])

    def test_save_load_tool_usage(self):
        """Test saving and loading tool usage."""
        # Test data
        tool_name = "search_tool"
        args = {"query": "example search"}
        result = "Example search results"

        # Save tool usage
        self.db.save_tool_usage(tool_name, args, result)

        # Load tool usage
        loaded_usage = self.db.load_tool_usage(tool_name)

        # Check that loaded usage matches saved usage
        self.assertIn(tool_name, loaded_usage)
        self.assertEqual(len(loaded_usage[tool_name]), 1)
        self.assertEqual(loaded_usage[tool_name][0]["args"], args)
        self.assertEqual(json.loads(loaded_usage[tool_name][0]["result"]), result)

    def test_save_load_learning_feedback(self):
        """Test saving and loading learning feedback."""
        # Test data
        agent_name = "test_agent"
        feedback_type = "user_feedback"
        feedback_data = {
            "request": "Hello, how are you?",
            "response": "I'm doing well, thank you for asking!",
            "feedback": "Great response!"
        }

        # Save learning feedback
        self.db.save_learning_feedback(agent_name, feedback_type, feedback_data)

        # Load learning feedback
        loaded_feedback = self.db.get_learning_feedback(agent_name, feedback_type)

        # Check that loaded feedback matches saved feedback
        self.assertEqual(len(loaded_feedback), 1)
        self.assertEqual(loaded_feedback[0]["agent_name"], agent_name)
        self.assertEqual(loaded_feedback[0]["feedback_type"], feedback_type)
        self.assertEqual(loaded_feedback[0]["feedback_data"], feedback_data)

class TestFileBackedMemoryDatabase(unittest.TestCase):
    """Tests for FileBackedMemoryDatabase class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.db = FileBackedMemoryDatabase(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('aiofiles.open')
    async def test_save_load_conversation_history(self, mock_open):
        """Test saving and loading conversation history."""
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__aenter__.return_value = mock_file
        mock_file.read.return_value = json.dumps([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ])

        # Test data
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]

        # Save conversation history
        await self.db.save_conversation_history(messages)

        # Load conversation history
        loaded_messages = await self.db.load_conversation_history()

        # Check that loaded messages match saved messages
        self.assertEqual(len(loaded_messages), len(messages))
        for i, message in enumerate(messages):
            self.assertEqual(loaded_messages[i]["role"], message["role"])
            self.assertEqual(loaded_messages[i]["content"], message["content"])

if __name__ == '__main__':
    unittest.main()
