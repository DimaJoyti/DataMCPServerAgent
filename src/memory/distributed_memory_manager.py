"""
Distributed memory manager for DataMCPServerAgent.
This module provides a unified interface for distributed memory operations.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.memory.distributed_memory import (
    DistributedMemoryFactory,
)
from src.utils.error_handlers import format_error_for_user

# Configure logging
logger = logging.getLogger(__name__)


class DistributedMemoryManager:
    """Manager for distributed memory operations."""

    def __init__(
        self,
        memory_type: str = "redis",
        config: Optional[Dict[str, Any]] = None,
        namespace: str = "datamcp"
    ):
        """Initialize the distributed memory manager.

        Args:
            memory_type: Type of memory backend to use ("redis" or "mongodb")
            config: Configuration for the memory backend
            namespace: Namespace for memory keys to avoid collisions
        """
        self.memory_type = memory_type.lower()
        self.config = config or {}
        self.namespace = namespace

        # Initialize the memory backend
        self._initialize_backend()

        # Cache for frequently accessed data
        self.cache = {}

        # Metrics for monitoring
        self.metrics = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }

    def _initialize_backend(self) -> None:
        """Initialize the memory backend based on configuration."""
        try:
            # Load configuration from environment if not provided
            if not self.config:
                if self.memory_type == "redis":
                    self.config = {
                        "host": os.getenv("REDIS_HOST", "localhost"),
                        "port": int(os.getenv("REDIS_PORT", "6379")),
                        "db": int(os.getenv("REDIS_DB", "0")),
                        "password": os.getenv("REDIS_PASSWORD", None),
                        "prefix": f"{self.namespace}:"
                    }
                elif self.memory_type == "mongodb":
                    self.config = {
                        "connection_string": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                        "database_name": os.getenv("MONGODB_DB", "agent_memory")
                    }

            # Create the memory backend
            self.backend = DistributedMemoryFactory.create_memory_backend(
                self.memory_type,
                **self.config
            )

            logger.info(f"Initialized {self.memory_type} memory backend with namespace {self.namespace}")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to initialize memory backend: {error_message}")
            raise RuntimeError(f"Failed to initialize memory backend: {error_message}")

    async def save_entity(
        self,
        entity_type: str,
        entity_id: str,
        entity_data: Dict[str, Any],
        cache: bool = True
    ) -> None:
        """Save an entity to distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            entity_data: Entity data
            cache: Whether to cache the entity locally
        """
        try:
            # Save to backend
            await self.backend.save_entity(entity_type, entity_id, entity_data)

            # Update cache if enabled
            if cache:
                cache_key = f"{entity_type}:{entity_id}"
                self.cache[cache_key] = {
                    "data": entity_data,
                    "timestamp": time.time()
                }

            # Update metrics
            self.metrics["writes"] += 1

            logger.debug(f"Saved entity {entity_type}:{entity_id} to {self.memory_type} backend")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to save entity {entity_type}:{entity_id}: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def load_entity(
        self,
        entity_type: str,
        entity_id: str,
        use_cache: bool = True,
        cache_ttl: int = 300  # 5 minutes
    ) -> Optional[Dict[str, Any]]:
        """Load an entity from distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            use_cache: Whether to use cached data if available
            cache_ttl: Time-to-live for cached data in seconds

        Returns:
            Entity data or None if not found
        """
        try:
            # Check cache if enabled
            if use_cache:
                cache_key = f"{entity_type}:{entity_id}"
                if cache_key in self.cache:
                    cached_data = self.cache[cache_key]
                    if time.time() - cached_data["timestamp"] < cache_ttl:
                        self.metrics["cache_hits"] += 1
                        logger.debug(f"Cache hit for entity {entity_type}:{entity_id}")
                        return cached_data["data"]
                    else:
                        # Cache expired
                        del self.cache[cache_key]

                self.metrics["cache_misses"] += 1

            # Load from backend
            entity_data = await self.backend.load_entity(entity_type, entity_id)

            # Update cache if data found and caching is enabled
            if entity_data and use_cache:
                cache_key = f"{entity_type}:{entity_id}"
                self.cache[cache_key] = {
                    "data": entity_data,
                    "timestamp": time.time()
                }

            # Update metrics
            self.metrics["reads"] += 1

            logger.debug(f"Loaded entity {entity_type}:{entity_id} from {self.memory_type} backend")
            return entity_data
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to load entity {entity_type}:{entity_id}: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity from distributed memory.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            True if the entity was deleted, False otherwise
        """
        try:
            # Delete from backend
            result = await self.backend.delete_entity(entity_type, entity_id)

            # Remove from cache
            cache_key = f"{entity_type}:{entity_id}"
            if cache_key in self.cache:
                del self.cache[cache_key]

            # Update metrics
            self.metrics["writes"] += 1

            logger.debug(f"Deleted entity {entity_type}:{entity_id} from {self.memory_type} backend")
            return result
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to delete entity {entity_type}:{entity_id}: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def save_conversation_history(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to distributed memory.

        Args:
            messages: List of messages to save
        """
        try:
            await self.backend.save_conversation_history(messages)
            self.metrics["writes"] += 1
            logger.debug(f"Saved conversation history to {self.memory_type} backend")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to save conversation history: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from distributed memory.

        Returns:
            List of messages
        """
        try:
            history = await self.backend.load_conversation_history()
            self.metrics["reads"] += 1
            logger.debug(f"Loaded conversation history from {self.memory_type} backend")
            return history
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to load conversation history: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def save_tool_usage(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any
    ) -> None:
        """Save tool usage to distributed memory.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
        """
        try:
            await self.backend.save_tool_usage(tool_name, args, result)
            self.metrics["writes"] += 1
            logger.debug(f"Saved tool usage for {tool_name} to {self.memory_type} backend")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to save tool usage for {tool_name}: {error_message}")
            self.metrics["errors"] += 1
            raise

    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        try:
            summary = await self.backend.get_memory_summary()

            # Add cache and metrics information
            cache_summary = "\n### Cache Statistics\n"
            cache_summary += f"- Cache size: {len(self.cache)} items\n"
            cache_summary += f"- Cache hits: {self.metrics['cache_hits']}\n"
            cache_summary += f"- Cache misses: {self.metrics['cache_misses']}\n"

            metrics_summary = "\n### Memory Metrics\n"
            metrics_summary += f"- Reads: {self.metrics['reads']}\n"
            metrics_summary += f"- Writes: {self.metrics['writes']}\n"
            metrics_summary += f"- Errors: {self.metrics['errors']}\n"

            return summary + cache_summary + metrics_summary
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get memory summary: {error_message}")
            self.metrics["errors"] += 1
            raise

    def clear_cache(self) -> None:
        """Clear the local cache."""
        self.cache.clear()
        logger.debug("Cleared local cache")

    def get_metrics(self) -> Dict[str, int]:
        """Get memory operation metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    async def ping(self) -> bool:
        """Check if the memory backend is accessible.

        Returns:
            True if the backend is accessible, False otherwise
        """
        try:
            if hasattr(self.backend, "ping"):
                return await self.backend.ping()
            return True
        except Exception:
            return False

    async def initialize(self) -> None:
        """Initialize the memory manager and backend.

        This method can be used for async initialization tasks.
        """
        try:
            # Test connection to backend
            is_connected = await self.ping()
            if not is_connected:
                logger.warning(f"Could not connect to {self.memory_type} backend")
            else:
                logger.info(f"Successfully connected to {self.memory_type} backend")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to initialize memory manager: {error_message}")
            # Don't raise here to allow graceful degradation

    async def cleanup(self) -> None:
        """Cleanup resources and close connections.

        This method should be called when shutting down the memory manager.
        """
        try:
            # Clear cache
            self.clear_cache()

            # Close backend connections if supported
            if hasattr(self.backend, 'close'):
                await self.backend.close()
            elif hasattr(self.backend, 'cleanup'):
                await self.backend.cleanup()

            logger.info(f"Cleaned up {self.memory_type} memory manager")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Error during cleanup: {error_message}")

    async def search_memories(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories based on query.

        Args:
            query: Search query
            agent_id: Optional agent ID to filter by
            limit: Maximum number of results
            filters: Optional additional filters

        Returns:
            List of matching memories
        """
        try:
            # This is a simplified implementation
            # In a real implementation, you would use vector search or full-text search

            # For now, return empty list
            # TODO: Implement proper memory search
            logger.debug(f"Searching memories with query: {query}")
            return []
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to search memories: {error_message}")
            self.metrics["errors"] += 1
            return []

    async def store_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> None:
        """Store memory data with a specific ID.

        Args:
            memory_id: Unique identifier for the memory
            memory_data: Memory data to store
        """
        try:
            # Store as entity with type "memory"
            await self.save_entity("memory", memory_id, memory_data)
            logger.debug(f"Stored memory with ID: {memory_id}")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to store memory {memory_id}: {error_message}")
            raise
