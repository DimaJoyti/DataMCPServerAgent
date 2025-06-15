"""
Distributed memory module for DataMCPServerAgent.
This module provides distributed memory capabilities using Redis or MongoDB as the backend.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    import redis
    from redis.asyncio import Redis
except ImportError:
    print("Warning: redis package not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "redis"])
    from redis.asyncio import Redis

try:
    import pymongo
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import MongoClient
    from pymongo.collection import Collection as MongoCollection
    from pymongo.database import Database as MongoDatabase
except ImportError:
    print("Warning: pymongo and/or motor packages not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "pymongo", "motor"])


class DistributedMemoryBackend(ABC):
    """Abstract base class for distributed memory backends."""

    @abstractmethod
    async def save_entity(
        self, entity_type: str, entity_id: str, entity_data: Dict[str, Any]
    ) -> None:
        """Save an entity to the distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            entity_data: Entity data
        """
        pass

    @abstractmethod
    async def load_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Load an entity from the distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            Entity data, or None if not found
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from the distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def save_conversation_history(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to the distributed memory.

        Args:
            messages: List of messages to save
        """
        pass

    @abstractmethod
    async def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from the distributed memory.

        Returns:
            List of messages
        """
        pass

    @abstractmethod
    async def save_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Save tool usage to the distributed memory.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
        """
        pass

    @abstractmethod
    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """Check if the memory backend is accessible.

        Returns:
            True if the backend is accessible, False otherwise
        """
        pass

    @abstractmethod
    async def load_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Load all entities of a given type from the distributed memory.

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary of entity data by entity ID
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from the distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def save_conversation_history(self, messages: List[Dict[str, Any]]) -> None:
        """Save conversation history to the distributed memory.

        Args:
            messages: List of message dictionaries
        """
        pass

    @abstractmethod
    async def load_conversation_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from the distributed memory.

        Returns:
            List of message dictionaries
        """
        pass

    @abstractmethod
    async def save_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Save tool usage to the distributed memory.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
        """
        pass

    @abstractmethod
    async def load_tool_usage(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load tool usage from the distributed memory.

        Args:
            tool_name: Name of tool to get history for, or None for all tools

        Returns:
            Dictionary of tool usage by tool name
        """
        pass

    @abstractmethod
    async def save_learning_feedback(
        self, agent_name: str, feedback_type: str, feedback_data: Dict[str, Any]
    ) -> None:
        """Save learning feedback to the distributed memory.

        Args:
            agent_name: Name of the agent
            feedback_type: Type of feedback
            feedback_data: Feedback data
        """
        pass

    @abstractmethod
    async def get_learning_feedback(
        self, agent_name: Optional[str] = None, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learning feedback from the distributed memory.

        Args:
            agent_name: Name of the agent, or None for all agents
            feedback_type: Type of feedback, or None for all types

        Returns:
            List of feedback entries
        """
        pass

    @abstractmethod
    async def get_entity_types(self) -> List[str]:
        """Get all entity types in the distributed memory.

        Returns:
            List of entity types
        """
        pass

    @abstractmethod
    async def get_tool_names(self) -> List[str]:
        """Get all tool names in the distributed memory.

        Returns:
            List of tool names
        """
        pass

    @abstractmethod
    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        pass


class RedisMemoryBackend(DistributedMemoryBackend):
    """Redis-based distributed memory backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "datamcp:",
    ):
        """Initialize the Redis memory backend.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for Redis keys
        """
        self.redis = Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        self.prefix = prefix

    async def save_entity(
        self, entity_type: str, entity_id: str, entity_data: Dict[str, Any]
    ) -> None:
        """Save an entity to Redis.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            entity_data: Entity data
        """
        # Create the entity key
        entity_key = f"{self.prefix}entity:{entity_type}:{entity_id}"

        # Add timestamp to entity data
        entity_data["_timestamp"] = time.time()

        # Save entity data as JSON
        await self.redis.set(entity_key, json.dumps(entity_data))

        # Add entity to the set of entities of this type
        await self.redis.sadd(f"{self.prefix}entity_types:{entity_type}", entity_id)

        # Add entity type to the set of all entity types
        await self.redis.sadd(f"{self.prefix}entity_types", entity_type)

    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from Redis.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        # Create the entity key
        entity_key = f"{self.prefix}entity:{entity_type}:{entity_id}"

        # Delete the entity
        result = await self.redis.delete(entity_key)

        # Remove entity from the set of entities of this type
        await self.redis.srem(f"{self.prefix}entity_types:{entity_type}", entity_id)

        # Check if there are any entities of this type left
        entity_count = await self.redis.scard(f"{self.prefix}entity_types:{entity_type}")

        # If no entities of this type left, remove the entity type
        if entity_count == 0:
            await self.redis.srem(f"{self.prefix}entity_types", entity_type)

        return result > 0

    async def load_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Load an entity from Redis.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            Entity data, or None if not found
        """
        # Create the entity key
        entity_key = f"{self.prefix}entity:{entity_type}:{entity_id}"

        # Load entity data
        entity_data = await self.redis.get(entity_key)

        if entity_data:
            return json.loads(entity_data)

        return None

    async def load_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Load all entities of a given type from Redis.

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary of entity data by entity ID
        """
        # Get all entity IDs of this type
        entity_ids = await self.redis.smembers(f"{self.prefix}entity_types:{entity_type}")

        result = {}

        # Load each entity
        for entity_id in entity_ids:
            entity_data = await self.load_entity(entity_type, entity_id)
            if entity_data:
                result[entity_id] = entity_data

        return result

    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from Redis.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        # Create the entity key
        entity_key = f"{self.prefix}entity:{entity_type}:{entity_id}"

        # Delete entity data
        deleted = await self.redis.delete(entity_key)

        # Remove entity from the set of entities of this type
        await self.redis.srem(f"{self.prefix}entity_types:{entity_type}", entity_id)

        return deleted > 0

    async def save_conversation_history(self, messages: List[Dict[str, Any]]) -> None:
        """Save conversation history to Redis.

        Args:
            messages: List of message dictionaries
        """
        # Convert messages to JSON
        messages_json = json.dumps(messages)

        # Save conversation history
        await self.redis.set(f"{self.prefix}conversation:history", messages_json)

    async def load_conversation_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from Redis.

        Returns:
            List of message dictionaries
        """
        # Load conversation history
        history_json = await self.redis.get(f"{self.prefix}conversation:history")

        if history_json:
            return json.loads(history_json)

        return []

    async def save_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Save tool usage to Redis.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
        """
        # Create usage data
        usage_data = {"args": args, "result": result, "timestamp": time.time()}

        # Convert usage data to JSON
        usage_json = json.dumps(usage_data)

        # Create a unique ID for this usage
        usage_id = f"{int(time.time() * 1000)}_{hash(str(args))}"

        # Save tool usage
        await self.redis.hset(f"{self.prefix}tool_usage:{tool_name}", usage_id, usage_json)

        # Add tool name to the set of all tool names
        await self.redis.sadd(f"{self.prefix}tool_names", tool_name)

    async def load_tool_usage(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load tool usage from Redis.

        Args:
            tool_name: Name of tool to get history for, or None for all tools

        Returns:
            Dictionary of tool usage by tool name
        """
        result = {}

        if tool_name:
            # Load usage for a specific tool
            usage_data = await self.redis.hgetall(f"{self.prefix}tool_usage:{tool_name}")

            result[tool_name] = []

            for _, usage_json in usage_data.items():
                usage = json.loads(usage_json)
                result[tool_name].append(usage)
        else:
            # Load usage for all tools
            tool_names = await self.redis.smembers(f"{self.prefix}tool_names")

            for tool in tool_names:
                tool_result = await self.load_tool_usage(tool)
                result.update(tool_result)

        return result

    async def save_learning_feedback(
        self, agent_name: str, feedback_type: str, feedback_data: Dict[str, Any]
    ) -> None:
        """Save learning feedback to Redis.

        Args:
            agent_name: Name of the agent
            feedback_type: Type of feedback
            feedback_data: Feedback data
        """
        # Create feedback entry
        feedback_entry = {
            "agent_name": agent_name,
            "feedback_type": feedback_type,
            "feedback_data": feedback_data,
            "timestamp": time.time(),
        }

        # Convert feedback entry to JSON
        feedback_json = json.dumps(feedback_entry)

        # Create a unique ID for this feedback
        feedback_id = f"{int(time.time() * 1000)}_{hash(str(feedback_data))}"

        # Save feedback
        await self.redis.hset(f"{self.prefix}learning_feedback", feedback_id, feedback_json)

        # Add feedback type to the set of all feedback types
        await self.redis.sadd(f"{self.prefix}feedback_types", feedback_type)

        # Add agent name to the set of all agent names
        await self.redis.sadd(f"{self.prefix}agent_names", agent_name)

    async def get_learning_feedback(
        self, agent_name: Optional[str] = None, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learning feedback from Redis.

        Args:
            agent_name: Name of the agent, or None for all agents
            feedback_type: Type of feedback, or None for all types

        Returns:
            List of feedback entries
        """
        # Load all feedback
        feedback_data = await self.redis.hgetall(f"{self.prefix}learning_feedback")

        result = []

        for _, feedback_json in feedback_data.items():
            feedback = json.loads(feedback_json)

            # Filter by agent name if specified
            if agent_name and feedback["agent_name"] != agent_name:
                continue

            # Filter by feedback type if specified
            if feedback_type and feedback["feedback_type"] != feedback_type:
                continue

            result.append(feedback)

        # Sort by timestamp (newest first)
        result.sort(key=lambda x: x["timestamp"], reverse=True)

        return result

    async def get_entity_types(self) -> List[str]:
        """Get all entity types in Redis.

        Returns:
            List of entity types
        """
        entity_types = await self.redis.smembers(f"{self.prefix}entity_types")
        return list(entity_types)

    async def get_tool_names(self) -> List[str]:
        """Get all tool names in Redis.

        Returns:
            List of tool names
        """
        tool_names = await self.redis.smembers(f"{self.prefix}tool_names")
        return list(tool_names)

    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        # Get entity types
        entity_types = await self.get_entity_types()

        # Get tool names
        tool_names = await self.get_tool_names()

        # Get feedback types
        feedback_types = await self.redis.smembers(f"{self.prefix}feedback_types")

        # Get agent names
        agent_names = await self.redis.smembers(f"{self.prefix}agent_names")

        # Get conversation history
        history = await self.load_conversation_history()

        # Format the summary
        summary = "## Memory Summary (Redis)\n\n"

        # Conversation summary
        summary += "### Conversation History\n"
        summary += f"- {len(history)} messages in history\n\n"

        # Entity memory summary
        summary += "### Entities in Memory\n"
        for entity_type in entity_types:
            entity_ids = await self.redis.smembers(f"{self.prefix}entity_types:{entity_type}")
            summary += f"- {entity_type}: {len(entity_ids)} entities\n"
        summary += "\n"

        # Tool usage summary
        summary += "### Tool Usage\n"
        for tool_name in tool_names:
            usage_data = await self.redis.hgetall(f"{self.prefix}tool_usage:{tool_name}")
            summary += f"- {tool_name}: {len(usage_data)} uses\n"
        summary += "\n"

        # Feedback summary
        summary += "### Learning Feedback\n"
        for feedback_type in feedback_types:
            feedback = await self.get_learning_feedback(feedback_type=feedback_type)
            summary += f"- {feedback_type}: {len(feedback)} entries\n"
        summary += "\n"

        # Agent summary
        summary += "### Agents\n"
        for agent_name in agent_names:
            feedback = await self.get_learning_feedback(agent_name=agent_name)
            summary += f"- {agent_name}: {len(feedback)} feedback entries\n"

        return summary

    async def ping(self) -> bool:
        """Check if Redis is accessible.

        Returns:
            True if Redis is accessible, False otherwise
        """
        try:
            return await self.redis.ping()
        except Exception:
            return False


class MongoDBMemoryBackend(DistributedMemoryBackend):
    """MongoDB-based distributed memory backend."""

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "datamcp_memory",
    ):
        """Initialize the MongoDB memory backend.

        Args:
            connection_string: MongoDB connection string
            database_name: MongoDB database name
        """
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]

        # Collections
        self.entities = self.db["entities"]
        self.conversation = self.db["conversation"]
        self.tool_usage = self.db["tool_usage"]
        self.learning_feedback = self.db["learning_feedback"]
        self.metadata = self.db["metadata"]

    async def save_entity(
        self, entity_type: str, entity_id: str, entity_data: Dict[str, Any]
    ) -> None:
        """Save an entity to MongoDB.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            entity_data: Entity data
        """
        # Create the document
        document = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "data": entity_data,
            "timestamp": time.time(),
        }

        # Save the document
        await self.entities.update_one(
            {"entity_type": entity_type, "entity_id": entity_id},
            {"$set": document},
            upsert=True,
        )

        # Update entity types metadata
        await self.metadata.update_one(
            {"metadata_type": "entity_types"},
            {"$addToSet": {"types": entity_type}},
            upsert=True,
        )

    async def load_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Load an entity from MongoDB.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            Entity data, or None if not found
        """
        # Find the document
        document = await self.entities.find_one(
            {"entity_type": entity_type, "entity_id": entity_id}
        )

        if document:
            return document["data"]

        return None

    async def load_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Load all entities of a given type from MongoDB.

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary of entity data by entity ID
        """
        # Find all documents of this type
        cursor = self.entities.find({"entity_type": entity_type})

        result = {}

        async for document in cursor:
            result[document["entity_id"]] = document["data"]

        return result

    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from MongoDB.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        # Delete the document
        result = await self.entities.delete_one(
            {"entity_type": entity_type, "entity_id": entity_id}
        )

        return result.deleted_count > 0

    async def save_conversation_history(self, messages: List[Dict[str, Any]]) -> None:
        """Save conversation history to MongoDB.

        Args:
            messages: List of message dictionaries
        """
        # Create the document
        document = {"messages": messages, "timestamp": time.time()}

        # Save the document
        await self.conversation.update_one(
            {"document_type": "history"}, {"$set": document}, upsert=True
        )

    async def load_conversation_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from MongoDB.

        Returns:
            List of message dictionaries
        """
        # Find the document
        document = await self.conversation.find_one({"document_type": "history"})

        if document:
            return document["messages"]

        return []

    async def save_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Save tool usage to MongoDB.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
        """
        # Create the document
        document = {
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "timestamp": time.time(),
        }

        # Save the document
        await self.tool_usage.insert_one(document)

        # Update tool names metadata
        await self.metadata.update_one(
            {"metadata_type": "tool_names"},
            {"$addToSet": {"names": tool_name}},
            upsert=True,
        )

    async def load_tool_usage(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load tool usage from MongoDB.

        Args:
            tool_name: Name of tool to get history for, or None for all tools

        Returns:
            Dictionary of tool usage by tool name
        """
        result = {}

        if tool_name:
            # Find all documents for this tool
            cursor = self.tool_usage.find({"tool_name": tool_name})

            result[tool_name] = []

            async for document in cursor:
                result[tool_name].append(
                    {
                        "args": document["args"],
                        "result": document["result"],
                        "timestamp": document["timestamp"],
                    }
                )
        else:
            # Get all tool names
            metadata = await self.metadata.find_one({"metadata_type": "tool_names"})

            if metadata and "names" in metadata:
                for tool in metadata["names"]:
                    tool_result = await self.load_tool_usage(tool)
                    result.update(tool_result)

        return result

    async def save_learning_feedback(
        self, agent_name: str, feedback_type: str, feedback_data: Dict[str, Any]
    ) -> None:
        """Save learning feedback to MongoDB.

        Args:
            agent_name: Name of the agent
            feedback_type: Type of feedback
            feedback_data: Feedback data
        """
        # Create the document
        document = {
            "agent_name": agent_name,
            "feedback_type": feedback_type,
            "feedback_data": feedback_data,
            "timestamp": time.time(),
        }

        # Save the document
        await self.learning_feedback.insert_one(document)

        # Update feedback types metadata
        await self.metadata.update_one(
            {"metadata_type": "feedback_types"},
            {"$addToSet": {"types": feedback_type}},
            upsert=True,
        )

        # Update agent names metadata
        await self.metadata.update_one(
            {"metadata_type": "agent_names"},
            {"$addToSet": {"names": agent_name}},
            upsert=True,
        )

    async def get_learning_feedback(
        self, agent_name: Optional[str] = None, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learning feedback from MongoDB.

        Args:
            agent_name: Name of the agent, or None for all agents
            feedback_type: Type of feedback, or None for all types

        Returns:
            List of feedback entries
        """
        # Build the query
        query = {}

        if agent_name:
            query["agent_name"] = agent_name

        if feedback_type:
            query["feedback_type"] = feedback_type

        # Find all matching documents
        cursor = self.learning_feedback.find(query).sort("timestamp", pymongo.DESCENDING)

        result = []

        async for document in cursor:
            result.append(
                {
                    "agent_name": document["agent_name"],
                    "feedback_type": document["feedback_type"],
                    "feedback_data": document["feedback_data"],
                    "timestamp": document["timestamp"],
                }
            )

        return result

    async def get_entity_types(self) -> List[str]:
        """Get all entity types in MongoDB.

        Returns:
            List of entity types
        """
        # Get entity types from metadata
        metadata = await self.metadata.find_one({"metadata_type": "entity_types"})

        if metadata and "types" in metadata:
            return metadata["types"]

        return []

    async def get_tool_names(self) -> List[str]:
        """Get all tool names in MongoDB.

        Returns:
            List of tool names
        """
        # Get tool names from metadata
        metadata = await self.metadata.find_one({"metadata_type": "tool_names"})

        if metadata and "names" in metadata:
            return metadata["names"]

        return []

    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        # Get entity types
        entity_types = await self.get_entity_types()

        # Get tool names
        tool_names = await self.get_tool_names()

        # Get feedback types
        feedback_metadata = await self.metadata.find_one({"metadata_type": "feedback_types"})
        feedback_types = feedback_metadata.get("types", []) if feedback_metadata else []

        # Get agent names
        agent_metadata = await self.metadata.find_one({"metadata_type": "agent_names"})
        agent_names = agent_metadata.get("names", []) if agent_metadata else []

        # Get conversation history
        history = await self.load_conversation_history()

        # Count entities by type
        entity_counts = {}
        for entity_type in entity_types:
            count = await self.entities.count_documents({"entity_type": entity_type})
            entity_counts[entity_type] = count

        # Count tool usage by tool name
        tool_usage_counts = {}
        for tool_name in tool_names:
            count = await self.tool_usage.count_documents({"tool_name": tool_name})
            tool_usage_counts[tool_name] = count

        # Count feedback by type
        feedback_counts = {}
        for feedback_type in feedback_types:
            count = await self.learning_feedback.count_documents({"feedback_type": feedback_type})
            feedback_counts[feedback_type] = count

        # Count feedback by agent
        agent_feedback_counts = {}
        for agent_name in agent_names:
            count = await self.learning_feedback.count_documents({"agent_name": agent_name})
            agent_feedback_counts[agent_name] = count

        # Format the summary
        summary = "## Memory Summary (MongoDB)\n\n"

        # Conversation summary
        summary += "### Conversation History\n"
        summary += f"- {len(history)} messages in history\n\n"

        # Entity memory summary
        summary += "### Entities in Memory\n"
        for entity_type, count in entity_counts.items():
            summary += f"- {entity_type}: {count} entities\n"
        summary += "\n"

        # Tool usage summary
        summary += "### Tool Usage\n"
        for tool_name, count in tool_usage_counts.items():
            summary += f"- {tool_name}: {count} uses\n"
        summary += "\n"

        # Feedback summary
        summary += "### Learning Feedback\n"
        for feedback_type, count in feedback_counts.items():
            summary += f"- {feedback_type}: {count} entries\n"
        summary += "\n"

        # Agent summary
        summary += "### Agents\n"
        for agent_name, count in agent_feedback_counts.items():
            summary += f"- {agent_name}: {count} feedback entries\n"

        return summary

    async def ping(self) -> bool:
        """Check if MongoDB is accessible.

        Returns:
            True if MongoDB is accessible, False otherwise
        """
        try:
            await self.client.admin.command("ping")
            return True
        except Exception:
            return False


class DistributedMemoryFactory:
    """Factory for creating distributed memory backends."""

    @staticmethod
    def create_memory_backend(backend_type: str, **kwargs) -> DistributedMemoryBackend:
        """Create a distributed memory backend.

        Args:
            backend_type: Type of backend to create ("redis" or "mongodb")
            **kwargs: Additional arguments to pass to the backend constructor

        Returns:
            Distributed memory backend

        Raises:
            ValueError: If the backend type is not supported
        """
        if backend_type.lower() == "redis":
            return RedisMemoryBackend(**kwargs)
        elif backend_type.lower() == "mongodb":
            return MongoDBMemoryBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    @staticmethod
    async def create_memory_backend_async(backend_type: str, **kwargs) -> DistributedMemoryBackend:
        """Create a distributed memory backend asynchronously.

        This method is the same as create_memory_backend, but it's async to match
        the interface of other async methods.

        Args:
            backend_type: Type of backend to create ("redis" or "mongodb")
            **kwargs: Additional arguments to pass to the backend constructor

        Returns:
            Distributed memory backend

        Raises:
            ValueError: If the backend type is not supported
        """
        return DistributedMemoryFactory.create_memory_backend(backend_type, **kwargs)
