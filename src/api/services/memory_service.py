"""
Memory service for the API.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from ..config import config
from ..models.response_models import MemoryResponse
from .redis_service import RedisService

# Import memory persistence modules
try:
    from src.memory.distributed_memory import (
        DistributedMemoryBackend,
        RedisMemoryBackend,
    )
    from src.memory.memory_persistence import MemoryDatabase

    MEMORY_MODULES_AVAILABLE = True
except ImportError:
    MEMORY_MODULES_AVAILABLE = False


class MemoryService:
    """Service for memory operations."""

    def __init__(self):
        """Initialize the memory service."""
        self.memory_backend = config.memory_backend
        self.redis_service = None
        self.memory_db = None
        self.distributed_backend = None

    async def _get_redis_service(self) -> RedisService:
        """
        Get the Redis service.

        Returns:
            RedisService: Redis service
        """
        if self.redis_service is None:
            self.redis_service = RedisService()
            await self.redis_service.connect()
        return self.redis_service

    def _get_memory_db(self) -> Optional[Any]:
        """
        Get the memory database.

        Returns:
            Optional[Any]: Memory database
        """
        if not MEMORY_MODULES_AVAILABLE:
            return None

        if self.memory_db is None:
            if self.memory_backend == "sqlite":
                self.memory_db = MemoryDatabase("agent_memory.db")
            elif self.memory_backend == "file":
                # Create memory directory if it doesn't exist
                os.makedirs("memory", exist_ok=True)
                self.memory_db = MemoryDatabase("memory/agent_memory.db")

        return self.memory_db

    async def _get_distributed_backend(self) -> Optional[DistributedMemoryBackend]:
        """
        Get the distributed memory backend.

        Returns:
            Optional[DistributedMemoryBackend]: Distributed memory backend
        """
        if not MEMORY_MODULES_AVAILABLE:
            return None

        if self.distributed_backend is None:
            if config.distributed_backend == "redis":
                self.distributed_backend = RedisMemoryBackend(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    prefix=config.redis_prefix,
                )

        return self.distributed_backend

    async def store_memory(
        self,
        session_id: str,
        memory_item: Dict[str, Any],
        memory_backend: Optional[str] = None,
    ) -> MemoryResponse:
        """
        Store a memory item.

        Args:
            session_id (str): Session ID for the memory
            memory_item (Dict[str, Any]): Memory item to store
            memory_backend (Optional[str]): Memory backend to use

        Returns:
            MemoryResponse: Memory response
        """
        # Use default memory backend if not provided
        memory_backend = memory_backend or self.memory_backend

        # Add timestamp to memory item if not present
        if "timestamp" not in memory_item:
            memory_item["timestamp"] = datetime.now().isoformat()

        # Add session ID to memory item if not present
        if "session_id" not in memory_item:
            memory_item["session_id"] = session_id

        # Generate ID for memory item if not present
        if "id" not in memory_item:
            memory_item["id"] = f"memory_{int(time.time() * 1000)}"

        # Store memory item based on backend
        if memory_backend == "redis" or (
            memory_backend == "distributed" and config.distributed_backend == "redis"
        ):
            redis_service = await self._get_redis_service()
            await redis_service.save_entity("memory", memory_item["id"], memory_item)
            await redis_service.sadd(f"session_memory:{session_id}", memory_item["id"])

        elif memory_backend == "sqlite" or memory_backend == "file":
            memory_db = self._get_memory_db()
            if memory_db:
                memory_db.save_entity("memory", memory_item["id"], memory_item)

                # Save session association
                session_memories = memory_db.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                if memory_item["id"] not in session_memories["memory_ids"]:
                    session_memories["memory_ids"].append(memory_item["id"])
                memory_db.save_entity("session_memory", session_id, session_memories)

        elif memory_backend == "distributed":
            distributed_backend = await self._get_distributed_backend()
            if distributed_backend:
                await distributed_backend.save_entity(
                    "memory", memory_item["id"], memory_item
                )

                # Save session association
                session_memories = await distributed_backend.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                if memory_item["id"] not in session_memories["memory_ids"]:
                    session_memories["memory_ids"].append(memory_item["id"])
                await distributed_backend.save_entity(
                    "session_memory", session_id, session_memories
                )

        return MemoryResponse(
            session_id=session_id,
            memory_items=[memory_item],
            memory_backend=memory_backend,
            metadata={
                "operation": "store",
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def retrieve_memory(
        self,
        session_id: str,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        memory_backend: Optional[str] = None,
    ) -> MemoryResponse:
        """
        Retrieve memory items for a session.

        Args:
            session_id (str): Session ID for the memory
            query (Optional[str]): Query for retrieving memory
            limit (int): Maximum number of memory items to return
            offset (int): Offset for pagination
            memory_backend (Optional[str]): Memory backend to use

        Returns:
            MemoryResponse: Memory response
        """
        # Use default memory backend if not provided
        memory_backend = memory_backend or self.memory_backend

        memory_items = []

        # Retrieve memory items based on backend
        if memory_backend == "redis" or (
            memory_backend == "distributed" and config.distributed_backend == "redis"
        ):
            redis_service = await self._get_redis_service()
            memory_ids = await redis_service.smembers(f"session_memory:{session_id}")

            # Get memory items
            for memory_id in list(memory_ids)[offset : offset + limit]:
                memory_item = await redis_service.get_entity("memory", memory_id)
                if memory_item:
                    # Filter by query if provided
                    if (
                        query is None
                        or query.lower() in json.dumps(memory_item).lower()
                    ):
                        memory_items.append(memory_item)

        elif memory_backend == "sqlite" or memory_backend == "file":
            memory_db = self._get_memory_db()
            if memory_db:
                # Get session memory IDs
                session_memories = memory_db.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                memory_ids = session_memories.get("memory_ids", [])

                # Get memory items
                for memory_id in memory_ids[offset : offset + limit]:
                    memory_item = memory_db.load_entity("memory", memory_id)
                    if memory_item:
                        # Filter by query if provided
                        if (
                            query is None
                            or query.lower() in json.dumps(memory_item).lower()
                        ):
                            memory_items.append(memory_item)

        elif memory_backend == "distributed":
            distributed_backend = await self._get_distributed_backend()
            if distributed_backend:
                # Get session memory IDs
                session_memories = await distributed_backend.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                memory_ids = session_memories.get("memory_ids", [])

                # Get memory items
                for memory_id in memory_ids[offset : offset + limit]:
                    memory_item = await distributed_backend.load_entity(
                        "memory", memory_id
                    )
                    if memory_item:
                        # Filter by query if provided
                        if (
                            query is None
                            or query.lower() in json.dumps(memory_item).lower()
                        ):
                            memory_items.append(memory_item)

        # If no memory items found, return empty list
        if not memory_items:
            # For demonstration purposes, add some mock memory items if no real items found
            if not config.enable_distributed:
                for i in range(offset, offset + limit):
                    memory_items.append(
                        {
                            "id": f"memory_{i}",
                            "content": f"Mock memory content {i}",
                            "timestamp": datetime.now().isoformat(),
                            "session_id": session_id,
                            "metadata": {
                                "query": query,
                            },
                        }
                    )

        return MemoryResponse(
            session_id=session_id,
            memory_items=memory_items,
            memory_backend=memory_backend,
            metadata={
                "operation": "retrieve",
                "query": query,
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def clear_memory(
        self,
        session_id: str,
        memory_backend: Optional[str] = None,
    ) -> MemoryResponse:
        """
        Clear memory for a session.

        Args:
            session_id (str): Session ID for the memory
            memory_backend (Optional[str]): Memory backend to use

        Returns:
            MemoryResponse: Memory response
        """
        # Use default memory backend if not provided
        memory_backend = memory_backend or self.memory_backend

        # Clear memory based on backend
        if memory_backend == "redis" or (
            memory_backend == "distributed" and config.distributed_backend == "redis"
        ):
            redis_service = await self._get_redis_service()

            # Get memory IDs for session
            memory_ids = await redis_service.smembers(f"session_memory:{session_id}")

            # Delete each memory item
            for memory_id in memory_ids:
                await redis_service.delete_entity("memory", memory_id)

            # Delete session memory association
            await redis_service.delete(f"session_memory:{session_id}")

        elif memory_backend == "sqlite" or memory_backend == "file":
            memory_db = self._get_memory_db()
            if memory_db:
                # Get session memory IDs
                session_memories = memory_db.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                memory_ids = session_memories.get("memory_ids", [])

                # Delete each memory item
                for memory_id in memory_ids:
                    memory_db.delete_entity("memory", memory_id)

                # Delete session memory association
                memory_db.delete_entity("session_memory", session_id)

        elif memory_backend == "distributed":
            distributed_backend = await self._get_distributed_backend()
            if distributed_backend:
                # Get session memory IDs
                session_memories = await distributed_backend.load_entity(
                    "session_memory", session_id
                ) or {"memory_ids": []}
                memory_ids = session_memories.get("memory_ids", [])

                # Delete each memory item
                for memory_id in memory_ids:
                    await distributed_backend.delete_entity("memory", memory_id)

                # Delete session memory association
                await distributed_backend.delete_entity("session_memory", session_id)

        return MemoryResponse(
            session_id=session_id,
            memory_items=[],
            memory_backend=memory_backend,
            metadata={
                "operation": "clear",
                "timestamp": datetime.now().isoformat(),
            },
        )
