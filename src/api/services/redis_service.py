"""
Redis service for the API.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from redis.asyncio import Redis

from ..config import config

class RedisService:
    """Service for Redis operations."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        prefix: str = None,
    ):
        """
        Initialize the Redis service.

        Args:
            host (str): Redis host
            port (int): Redis port
            db (int): Redis database
            password (str): Redis password
            prefix (str): Key prefix
        """
        self.host = host or config.redis_host
        self.port = port or config.redis_port
        self.db = db or config.redis_db
        self.password = password or config.redis_password
        self.prefix = prefix or config.redis_prefix
        self.redis = None

    async def connect(self) -> None:
        """
        Connect to Redis.
        """
        if self.redis is None:
            self.redis = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )

    async def disconnect(self) -> None:
        """
        Disconnect from Redis.
        """
        if self.redis is not None:
            await self.redis.close()
            self.redis = None

    async def ping(self) -> bool:
        """
        Ping Redis to check if it's available.

        Returns:
            bool: True if Redis is available, False otherwise
        """
        try:
            await self.connect()
            return await self.redis.ping()
        except Exception:
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value from Redis.

        Args:
            key (str): Key

        Returns:
            Optional[str]: Value or None if not found
        """
        await self.connect()
        return await self.redis.get(f"{self.prefix}{key}")

    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None,
    ) -> None:
        """
        Set a value in Redis.

        Args:
            key (str): Key
            value (str): Value
            expire (Optional[int]): Expiration time in seconds
        """
        await self.connect()
        await self.redis.set(f"{self.prefix}{key}", value, ex=expire)

    async def delete(self, key: str) -> int:
        """
        Delete a key from Redis.

        Args:
            key (str): Key

        Returns:
            int: Number of keys deleted
        """
        await self.connect()
        return await self.redis.delete(f"{self.prefix}{key}")

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key (str): Key

        Returns:
            bool: True if the key exists, False otherwise
        """
        await self.connect()
        return await self.redis.exists(f"{self.prefix}{key}") > 0

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set an expiration time for a key.

        Args:
            key (str): Key
            seconds (int): Expiration time in seconds

        Returns:
            bool: True if the expiration was set, False otherwise
        """
        await self.connect()
        return await self.redis.expire(f"{self.prefix}{key}", seconds)

    async def ttl(self, key: str) -> int:
        """
        Get the time to live for a key.

        Args:
            key (str): Key

        Returns:
            int: Time to live in seconds, -1 if the key has no expiration, -2 if the key does not exist
        """
        await self.connect()
        return await self.redis.ttl(f"{self.prefix}{key}")

    async def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern.

        Args:
            pattern (str): Pattern

        Returns:
            List[str]: List of keys
        """
        await self.connect()
        keys = await self.redis.keys(f"{self.prefix}{pattern}")
        return [key.replace(self.prefix, "", 1) for key in keys]

    async def hget(self, key: str, field: str) -> Optional[str]:
        """
        Get a field from a hash.

        Args:
            key (str): Key
            field (str): Field

        Returns:
            Optional[str]: Value or None if not found
        """
        await self.connect()
        return await self.redis.hget(f"{self.prefix}{key}", field)

    async def hset(self, key: str, field: str, value: str) -> int:
        """
        Set a field in a hash.

        Args:
            key (str): Key
            field (str): Field
            value (str): Value

        Returns:
            int: 1 if field is a new field in the hash and value was set, 0 if field already exists in the hash and the value was updated
        """
        await self.connect()
        return await self.redis.hset(f"{self.prefix}{key}", field, value)

    async def hdel(self, key: str, field: str) -> int:
        """
        Delete a field from a hash.

        Args:
            key (str): Key
            field (str): Field

        Returns:
            int: Number of fields removed from the hash
        """
        await self.connect()
        return await self.redis.hdel(f"{self.prefix}{key}", field)

    async def hgetall(self, key: str) -> Dict[str, str]:
        """
        Get all fields and values from a hash.

        Args:
            key (str): Key

        Returns:
            Dict[str, str]: Dictionary of fields and values
        """
        await self.connect()
        return await self.redis.hgetall(f"{self.prefix}{key}")

    async def hexists(self, key: str, field: str) -> bool:
        """
        Check if a field exists in a hash.

        Args:
            key (str): Key
            field (str): Field

        Returns:
            bool: True if the field exists, False otherwise
        """
        await self.connect()
        return await self.redis.hexists(f"{self.prefix}{key}", field)

    async def hkeys(self, key: str) -> List[str]:
        """
        Get all fields from a hash.

        Args:
            key (str): Key

        Returns:
            List[str]: List of fields
        """
        await self.connect()
        return await self.redis.hkeys(f"{self.prefix}{key}")

    async def hvals(self, key: str) -> List[str]:
        """
        Get all values from a hash.

        Args:
            key (str): Key

        Returns:
            List[str]: List of values
        """
        await self.connect()
        return await self.redis.hvals(f"{self.prefix}{key}")

    async def sadd(self, key: str, *values: str) -> int:
        """
        Add values to a set.

        Args:
            key (str): Key
            *values (str): Values

        Returns:
            int: Number of values added to the set
        """
        await self.connect()
        return await self.redis.sadd(f"{self.prefix}{key}", *values)

    async def srem(self, key: str, *values: str) -> int:
        """
        Remove values from a set.

        Args:
            key (str): Key
            *values (str): Values

        Returns:
            int: Number of values removed from the set
        """
        await self.connect()
        return await self.redis.srem(f"{self.prefix}{key}", *values)

    async def smembers(self, key: str) -> Set[str]:
        """
        Get all members of a set.

        Args:
            key (str): Key

        Returns:
            Set[str]: Set of members
        """
        await self.connect()
        return await self.redis.smembers(f"{self.prefix}{key}")

    async def sismember(self, key: str, value: str) -> bool:
        """
        Check if a value is a member of a set.

        Args:
            key (str): Key
            value (str): Value

        Returns:
            bool: True if the value is a member of the set, False otherwise
        """
        await self.connect()
        return await self.redis.sismember(f"{self.prefix}{key}", value)

    async def scard(self, key: str) -> int:
        """
        Get the number of members in a set.

        Args:
            key (str): Key

        Returns:
            int: Number of members in the set
        """
        await self.connect()
        return await self.redis.scard(f"{self.prefix}{key}")

    async def lpush(self, key: str, *values: str) -> int:
        """
        Prepend values to a list.

        Args:
            key (str): Key
            *values (str): Values

        Returns:
            int: Length of the list after the push operation
        """
        await self.connect()
        return await self.redis.lpush(f"{self.prefix}{key}", *values)

    async def rpush(self, key: str, *values: str) -> int:
        """
        Append values to a list.

        Args:
            key (str): Key
            *values (str): Values

        Returns:
            int: Length of the list after the push operation
        """
        await self.connect()
        return await self.redis.rpush(f"{self.prefix}{key}", *values)

    async def lpop(self, key: str) -> Optional[str]:
        """
        Remove and get the first element in a list.

        Args:
            key (str): Key

        Returns:
            Optional[str]: First element or None if the list is empty
        """
        await self.connect()
        return await self.redis.lpop(f"{self.prefix}{key}")

    async def rpop(self, key: str) -> Optional[str]:
        """
        Remove and get the last element in a list.

        Args:
            key (str): Key

        Returns:
            Optional[str]: Last element or None if the list is empty
        """
        await self.connect()
        return await self.redis.rpop(f"{self.prefix}{key}")

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """
        Get a range of elements from a list.

        Args:
            key (str): Key
            start (int): Start index
            end (int): End index

        Returns:
            List[str]: List of elements
        """
        await self.connect()
        return await self.redis.lrange(f"{self.prefix}{key}", start, end)

    async def llen(self, key: str) -> int:
        """
        Get the length of a list.

        Args:
            key (str): Key

        Returns:
            int: Length of the list
        """
        await self.connect()
        return await self.redis.llen(f"{self.prefix}{key}")

    async def incr(self, key: str) -> int:
        """
        Increment the integer value of a key by one.

        Args:
            key (str): Key

        Returns:
            int: Value after the increment
        """
        await self.connect()
        return await self.redis.incr(f"{self.prefix}{key}")

    async def decr(self, key: str) -> int:
        """
        Decrement the integer value of a key by one.

        Args:
            key (str): Key

        Returns:
            int: Value after the decrement
        """
        await self.connect()
        return await self.redis.decr(f"{self.prefix}{key}")

    async def incrby(self, key: str, amount: int) -> int:
        """
        Increment the integer value of a key by the given amount.

        Args:
            key (str): Key
            amount (int): Amount to increment by

        Returns:
            int: Value after the increment
        """
        await self.connect()
        return await self.redis.incrby(f"{self.prefix}{key}", amount)

    async def decrby(self, key: str, amount: int) -> int:
        """
        Decrement the integer value of a key by the given amount.

        Args:
            key (str): Key
            amount (int): Amount to decrement by

        Returns:
            int: Value after the decrement
        """
        await self.connect()
        return await self.redis.decrby(f"{self.prefix}{key}", amount)

    async def save_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        expire: int = 3600,
    ) -> None:
        """
        Save session data to Redis.

        Args:
            session_id (str): Session ID
            data (Dict[str, Any]): Session data
            expire (int): Expiration time in seconds
        """
        await self.set(f"session:{session_id}", json.dumps(data), expire)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data from Redis.

        Args:
            session_id (str): Session ID

        Returns:
            Optional[Dict[str, Any]]: Session data or None if not found
        """
        data = await self.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return None

    async def delete_session(self, session_id: str) -> int:
        """
        Delete session data from Redis.

        Args:
            session_id (str): Session ID

        Returns:
            int: Number of keys deleted
        """
        return await self.delete(f"session:{session_id}")

    async def save_entity(
        self,
        entity_type: str,
        entity_id: str,
        entity_data: Dict[str, Any],
    ) -> None:
        """
        Save an entity to Redis.

        Args:
            entity_type (str): Entity type
            entity_id (str): Entity ID
            entity_data (Dict[str, Any]): Entity data
        """
        # Add timestamp to entity data
        entity_data["_timestamp"] = time.time()

        # Save entity data
        await self.set(f"entity:{entity_type}:{entity_id}", json.dumps(entity_data))

        # Add entity to the set of entities of this type
        await self.sadd(f"entity_types:{entity_type}", entity_id)

        # Add entity type to the set of all entity types
        await self.sadd("entity_types", entity_type)

    async def get_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entity from Redis.

        Args:
            entity_type (str): Entity type
            entity_id (str): Entity ID

        Returns:
            Optional[Dict[str, Any]]: Entity data or None if not found
        """
        data = await self.get(f"entity:{entity_type}:{entity_id}")
        if data:
            return json.loads(data)
        return None

    async def delete_entity(self, entity_type: str, entity_id: str) -> int:
        """
        Delete an entity from Redis.

        Args:
            entity_type (str): Entity type
            entity_id (str): Entity ID

        Returns:
            int: Number of keys deleted
        """
        # Delete entity data
        deleted = await self.delete(f"entity:{entity_type}:{entity_id}")

        # Remove entity from the set of entities of this type
        await self.srem(f"entity_types:{entity_type}", entity_id)

        # Check if there are any entities of this type left
        entity_count = await self.scard(f"entity_types:{entity_type}")

        # If no entities of this type left, remove the entity type
        if entity_count == 0:
            await self.srem("entity_types", entity_type)

        return deleted

    async def get_entities_by_type(
        self,
        entity_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all entities of a given type from Redis.

        Args:
            entity_type (str): Entity type

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of entity data by entity ID
        """
        # Get all entity IDs of this type
        entity_ids = await self.smembers(f"entity_types:{entity_type}")

        result = {}

        # Load each entity
        for entity_id in entity_ids:
            entity_data = await self.get_entity(entity_type, entity_id)
            if entity_data:
                result[entity_id] = entity_data

        return result

    async def get_entity_types(self) -> Set[str]:
        """
        Get all entity types from Redis.

        Returns:
            Set[str]: Set of entity types
        """
        return await self.smembers("entity_types")

    async def save_conversation_history(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """
        Save conversation history to Redis.

        Args:
            session_id (str): Session ID
            messages (List[Dict[str, Any]]): List of message dictionaries
        """
        await self.set(f"conversation:{session_id}", json.dumps(messages))

    async def get_conversation_history(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history from Redis.

        Args:
            session_id (str): Session ID

        Returns:
            List[Dict[str, Any]]: List of message dictionaries
        """
        data = await self.get(f"conversation:{session_id}")
        if data:
            return json.loads(data)
        return []

    async def save_tool_usage(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
    ) -> None:
        """
        Save tool usage to Redis.

        Args:
            session_id (str): Session ID
            tool_name (str): Name of the tool
            args (Dict[str, Any]): Tool arguments
            result (Any): Tool result
        """
        # Create usage data
        usage_data = {
            "args": args,
            "result": result,
            "timestamp": time.time(),
        }

        # Convert usage data to JSON
        usage_json = json.dumps(usage_data)

        # Create a unique ID for this usage
        usage_id = f"{int(time.time() * 1000)}_{hash(str(args))}"

        # Save tool usage
        await self.hset(f"tool_usage:{session_id}:{tool_name}", usage_id, usage_json)

        # Add tool name to the set of all tool names
        await self.sadd(f"tool_names:{session_id}", tool_name)

        # Add session ID to the set of all sessions
        await self.sadd("sessions", session_id)

    async def get_tool_usage(
        self,
        session_id: str,
        tool_name: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get tool usage from Redis.

        Args:
            session_id (str): Session ID
            tool_name (Optional[str]): Name of tool to get history for, or None for all tools

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of tool usage by tool name
        """
        result = {}

        if tool_name:
            # Load usage for a specific tool
            usage_data = await self.hgetall(f"tool_usage:{session_id}:{tool_name}")

            result[tool_name] = []

            for _, usage_json in usage_data.items():
                usage = json.loads(usage_json)
                result[tool_name].append(usage)
        else:
            # Load usage for all tools
            tool_names = await self.smembers(f"tool_names:{session_id}")

            for tool in tool_names:
                tool_result = await self.get_tool_usage(session_id, tool)
                result.update(tool_result)

        return result

    async def get_sessions(self) -> Set[str]:
        """
        Get all session IDs from Redis.

        Returns:
            Set[str]: Set of session IDs
        """
        return await self.smembers("sessions")

    async def get_tool_names(self, session_id: str) -> Set[str]:
        """
        Get all tool names for a session from Redis.

        Args:
            session_id (str): Session ID

        Returns:
            Set[str]: Set of tool names
        """
        return await self.smembers(f"tool_names:{session_id}")
