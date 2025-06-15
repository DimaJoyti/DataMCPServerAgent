"""
Real-world example of using distributed memory with Redis.
This example demonstrates a practical use case for distributed memory in a multi-session environment.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.memory.distributed_memory_manager import DistributedMemoryManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize distributed memory manager
memory_type = "redis"  # We'll use Redis for this example
memory_config = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "prefix": "datamcp_example:"
}

memory_manager = DistributedMemoryManager(
    memory_type=memory_type,
    config=memory_config,
    namespace="example"
)

class UserSession:
    """Simulates a user session with the agent."""

    def __init__(self, user_id: str, memory_manager: DistributedMemoryManager):
        """Initialize the user session.

        Args:
            user_id: User ID
            memory_manager: Distributed memory manager
        """
        self.user_id = user_id
        self.memory_manager = memory_manager
        self.conversation_id = f"conversation_{user_id}_{int(time.time())}"
        self.history = []

    async def initialize(self) -> None:
        """Initialize the session by loading user preferences and history."""
        # Load user preferences
        user_preferences = await self.memory_manager.load_entity("user_preferences", self.user_id)

        if not user_preferences:
            # Create default preferences if not found
            user_preferences = {
                "theme": "light",
                "language": "en",
                "notifications": True,
                "search_history": [],
                "favorite_tools": [],
                "last_login": time.time()
            }
            await self.memory_manager.save_entity("user_preferences", self.user_id, user_preferences)
            logger.info(f"Created default preferences for user {self.user_id}")
        else:
            # Update last login time
            user_preferences["last_login"] = time.time()
            await self.memory_manager.save_entity("user_preferences", self.user_id, user_preferences)
            logger.info(f"Loaded preferences for user {self.user_id}")

        self.preferences = user_preferences

        # Load previous conversations
        previous_conversations = await self.load_previous_conversations()
        if previous_conversations:
            logger.info(f"Found {len(previous_conversations)} previous conversations for user {self.user_id}")

            # Use the most recent conversation history if available
            if previous_conversations:
                most_recent = max(previous_conversations, key=lambda x: x["timestamp"])
                self.history = most_recent["messages"]
                logger.info(f"Loaded most recent conversation with {len(self.history)} messages")

    async def load_previous_conversations(self) -> List[Dict[str, Any]]:
        """Load previous conversations for the user.

        Returns:
            List of previous conversations
        """
        # In a real implementation, you would query Redis for all conversations
        # belonging to this user. For this example, we'll use a simple approach.
        conversations = []

        # Check if there's a conversation index for this user
        conversation_index = await self.memory_manager.load_entity(
            "conversation_index", self.user_id
        )

        if conversation_index:
            # Load each conversation
            for conv_id in conversation_index["conversations"]:
                conversation = await self.memory_manager.load_entity(
                    "conversation", conv_id
                )
                if conversation:
                    conversations.append(conversation)

        return conversations

    async def save_conversation(self) -> None:
        """Save the current conversation."""
        # Create conversation data
        conversation_data = {
            "messages": self.history,
            "timestamp": time.time(),
            "metadata": {
                "user_id": self.user_id
            }
        }

        # Save conversation
        await self.memory_manager.save_entity(
            "conversation", self.conversation_id, conversation_data
        )

        # Update conversation index
        conversation_index = await self.memory_manager.load_entity(
            "conversation_index", self.user_id
        )

        if not conversation_index:
            conversation_index = {
                "conversations": []
            }

        if self.conversation_id not in conversation_index["conversations"]:
            conversation_index["conversations"].append(self.conversation_id)

        await self.memory_manager.save_entity(
            "conversation_index", self.user_id, conversation_index
        )

        logger.info(f"Saved conversation {self.conversation_id} for user {self.user_id}")

    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        self.history.append({"role": role, "content": content})
        await self.save_conversation()

    async def update_preferences(self, updates: Dict[str, Any]) -> None:
        """Update user preferences.

        Args:
            updates: Preference updates
        """
        for key, value in updates.items():
            self.preferences[key] = value

        await self.memory_manager.save_entity(
            "user_preferences", self.user_id, self.preferences
        )
        logger.info(f"Updated preferences for user {self.user_id}")

    async def add_search_history(self, query: str) -> None:
        """Add a search query to the user's search history.

        Args:
            query: Search query
        """
        if "search_history" not in self.preferences:
            self.preferences["search_history"] = []

        # Add to the beginning of the list (most recent first)
        self.preferences["search_history"].insert(0, query)

        # Keep only the 10 most recent searches
        self.preferences["search_history"] = self.preferences["search_history"][:10]

        await self.memory_manager.save_entity(
            "user_preferences", self.user_id, self.preferences
        )
        logger.info(f"Added '{query}' to search history for user {self.user_id}")

async def simulate_multi_session_scenario():
    """Simulate a multi-session scenario with distributed memory."""
    logger.info("=== Simulating Multi-Session Scenario with Distributed Memory ===")

    # Create users
    user1 = UserSession("user1", memory_manager)
    user2 = UserSession("user2", memory_manager)

    # Initialize sessions
    await user1.initialize()
    await user2.initialize()

    # Simulate user 1 interaction
    logger.info("=== User 1 Session 1 ===")
    await user1.add_message("user", "I'm interested in learning about distributed memory systems.")
    await user1.add_message("assistant", "Distributed memory systems are memory architectures where memory is physically distributed across multiple storage nodes but logically shared. Would you like to know more about specific implementations like Redis or MongoDB?")
    await user1.add_message("user", "Tell me about Redis.")
    await user1.add_message("assistant", "Redis is an in-memory data structure store that can be used as a database, cache, and message broker. It's particularly well-suited for distributed memory because of its high performance, support for complex data structures, and built-in persistence options.")

    # Update user 1 preferences
    await user1.update_preferences({
        "theme": "dark",
        "favorite_tools": ["search", "analyze"]
    })

    # Add search history for user 1
    await user1.add_search_history("Redis distributed memory")
    await user1.add_search_history("Redis vs MongoDB")

    # Simulate user 2 interaction
    logger.info("=== User 2 Session 1 ===")
    await user2.add_message("user", "How can I implement a distributed cache?")
    await user2.add_message("assistant", "Implementing a distributed cache involves several components: a storage backend like Redis or Memcached, a client library, and a caching strategy. Would you like me to explain each component in detail?")
    await user2.add_message("user", "Yes, please explain Redis as a storage backend.")
    await user2.add_message("assistant", "Redis is an excellent choice for a distributed cache backend. It offers in-memory storage with optional persistence, high performance, and various data structures. For caching, you'd typically use Redis's key-value storage with expiration times (TTL) to automatically invalidate stale data.")

    # Update user 2 preferences
    await user2.update_preferences({
        "theme": "light",
        "favorite_tools": ["code", "debug"]
    })

    # Add search history for user 2
    await user2.add_search_history("distributed cache implementation")
    await user2.add_search_history("Redis cache example")

    # Simulate application restart
    logger.info("=== Simulating Application Restart ===")

    # Create new sessions (simulating reconnection after restart)
    user1_new = UserSession("user1", memory_manager)
    user2_new = UserSession("user2", memory_manager)

    # Initialize sessions (should load previous data)
    await user1_new.initialize()
    await user2_new.initialize()

    # Verify user 1 data was persisted
    logger.info("=== User 1 Session 2 ===")
    logger.info(f"User 1 preferences: {user1_new.preferences}")
    logger.info(f"User 1 conversation history: {len(user1_new.history)} messages")
    logger.info(f"User 1 search history: {user1_new.preferences.get('search_history', [])}")

    # Continue user 1 conversation
    await user1_new.add_message("user", "What about Redis Cluster for scaling?")
    await user1_new.add_message("assistant", "Redis Cluster is Redis's built-in solution for horizontal scaling. It automatically partitions data across multiple Redis nodes, supports automatic failover, and allows you to scale your Redis deployment linearly by adding more nodes.")

    # Verify user 2 data was persisted
    logger.info("=== User 2 Session 2 ===")
    logger.info(f"User 2 preferences: {user2_new.preferences}")
    logger.info(f"User 2 conversation history: {len(user2_new.history)} messages")
    logger.info(f"User 2 search history: {user2_new.preferences.get('search_history', [])}")

    # Continue user 2 conversation
    await user2_new.add_message("user", "How do I handle cache invalidation?")
    await user2_new.add_message("assistant", "Cache invalidation is one of the hardest problems in distributed systems. Common strategies include: time-based expiration (TTL), explicit invalidation when data changes, version-based invalidation, and event-based invalidation. Redis supports all of these approaches through its TTL feature, pub/sub mechanism, and atomic operations.")

    # Get memory summary
    summary = await memory_manager.get_memory_summary()
    logger.info(f"Memory Summary:\n{summary}")

if __name__ == "__main__":
    asyncio.run(simulate_multi_session_scenario())
