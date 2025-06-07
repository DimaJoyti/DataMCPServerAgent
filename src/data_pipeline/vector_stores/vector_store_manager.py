"""
Vector store manager and factory for creating and managing vector stores.
"""

import logging
from typing import Dict, Optional, Type

from .backends.base_store import BaseVectorStore
from .backends.memory_store import MemoryVectorStore
from .schemas.base_schema import VectorStoreConfig, VectorStoreType

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """Factory for creating vector store instances."""

    def __init__(self):
        """Initialize factory."""
        self._stores: Dict[VectorStoreType, Type[BaseVectorStore]] = {}
        self._register_default_stores()

    def _register_default_stores(self) -> None:
        """Register default vector stores."""
        # Always available
        self.register_store(VectorStoreType.MEMORY, MemoryVectorStore)

        # ChromaDB
        try:
            from .backends.chroma_store import ChromaVectorStore
            self.register_store(VectorStoreType.CHROMA, ChromaVectorStore)
        except ImportError:
            logger.warning("ChromaDB not available - missing dependencies")

        # FAISS
        try:
            from .backends.faiss_store import FAISSVectorStore
            self.register_store(VectorStoreType.FAISS, FAISSVectorStore)
        except ImportError:
            logger.warning("FAISS not available - missing dependencies")

        # Pinecone
        try:
            from .backends.pinecone_store import PineconeVectorStore
            self.register_store(VectorStoreType.PINECONE, PineconeVectorStore)
        except ImportError:
            logger.warning("Pinecone not available - missing dependencies")

        # Weaviate
        try:
            from .backends.weaviate_store import WeaviateVectorStore
            self.register_store(VectorStoreType.WEAVIATE, WeaviateVectorStore)
        except ImportError:
            logger.warning("Weaviate not available - missing dependencies")

        # Qdrant
        try:
            from .backends.qdrant_store import QdrantVectorStore
            self.register_store(VectorStoreType.QDRANT, QdrantVectorStore)
        except ImportError:
            logger.warning("Qdrant not available - missing dependencies")

    def register_store(self, store_type: VectorStoreType, store_class: Type[BaseVectorStore]) -> None:
        """
        Register a vector store class.

        Args:
            store_type: Type of vector store
            store_class: Vector store class
        """
        self._stores[store_type] = store_class
        logger.debug(f"Registered {store_class.__name__} for {store_type}")

    def create_store(self, config: VectorStoreConfig) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            config: Vector store configuration

        Returns:
            BaseVectorStore: Vector store instance

        Raises:
            ValueError: If store type not supported
        """
        store_class = self._stores.get(config.store_type)
        if store_class is None:
            available_types = list(self._stores.keys())
            raise ValueError(
                f"Vector store type {config.store_type} not supported. "
                f"Available types: {available_types}"
            )

        try:
            return store_class(config)
        except Exception as e:
            logger.error(f"Failed to create {config.store_type} store: {e}")
            raise

    def get_available_stores(self) -> list[VectorStoreType]:
        """
        Get list of available vector store types.

        Returns:
            List[VectorStoreType]: Available store types
        """
        return list(self._stores.keys())

    def is_store_available(self, store_type: VectorStoreType) -> bool:
        """
        Check if vector store type is available.

        Args:
            store_type: Store type to check

        Returns:
            bool: True if available
        """
        return store_type in self._stores

class VectorStoreManager:
    """Manager for vector store instances."""

    def __init__(self, factory: Optional[VectorStoreFactory] = None):
        """
        Initialize manager.

        Args:
            factory: Vector store factory (creates default if None)
        """
        self.factory = factory or VectorStoreFactory()
        self.stores: Dict[str, BaseVectorStore] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    async def create_store(
        self,
        name: str,
        config: VectorStoreConfig,
        initialize: bool = True
    ) -> BaseVectorStore:
        """
        Create and optionally initialize a vector store.

        Args:
            name: Store name/identifier
            config: Store configuration
            initialize: Whether to initialize the store

        Returns:
            BaseVectorStore: Created store
        """
        if name in self.stores:
            raise ValueError(f"Store '{name}' already exists")

        # Create store
        store = self.factory.create_store(config)

        # Initialize if requested
        if initialize:
            await store.initialize()

        # Register store
        self.stores[name] = store
        self.logger.info(f"Created vector store '{name}' of type {config.store_type}")

        return store

    async def get_store(self, name: str) -> Optional[BaseVectorStore]:
        """
        Get a vector store by name.

        Args:
            name: Store name

        Returns:
            Optional[BaseVectorStore]: Store if found
        """
        return self.stores.get(name)

    async def remove_store(self, name: str, close: bool = True) -> bool:
        """
        Remove a vector store.

        Args:
            name: Store name
            close: Whether to close the store

        Returns:
            bool: True if removed
        """
        if name not in self.stores:
            return False

        store = self.stores[name]

        if close:
            try:
                await store.close()
            except Exception as e:
                self.logger.error(f"Error closing store '{name}': {e}")

        del self.stores[name]
        self.logger.info(f"Removed vector store '{name}'")

        return True

    async def close_all_stores(self) -> None:
        """Close all managed stores."""
        for name, store in list(self.stores.items()):
            try:
                await store.close()
                self.logger.debug(f"Closed store '{name}'")
            except Exception as e:
                self.logger.error(f"Error closing store '{name}': {e}")

        self.stores.clear()
        self.logger.info("Closed all vector stores")

    def list_stores(self) -> Dict[str, str]:
        """
        List all managed stores.

        Returns:
            Dict[str, str]: Mapping of store names to types
        """
        return {
            name: store.__class__.__name__
            for name, store in self.stores.items()
        }

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all stores.

        Returns:
            Dict[str, bool]: Health status for each store
        """
        results = {}

        for name, store in self.stores.items():
            try:
                results[name] = await store.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for store '{name}': {e}")
                results[name] = False

        return results

    async def get_stats_all(self) -> Dict[str, dict]:
        """
        Get statistics for all stores.

        Returns:
            Dict[str, dict]: Statistics for each store
        """
        results = {}

        for name, store in self.stores.items():
            try:
                stats = await store.get_stats()
                results[name] = stats.dict()
            except Exception as e:
                self.logger.error(f"Failed to get stats for store '{name}': {e}")
                results[name] = {"error": str(e)}

        return results

    def __len__(self) -> int:
        """Return number of managed stores."""
        return len(self.stores)

    def __contains__(self, name: str) -> bool:
        """Check if store exists."""
        return name in self.stores

    def __iter__(self):
        """Iterate over store names."""
        return iter(self.stores.keys())

# Global instances
vector_store_factory = VectorStoreFactory()
vector_store_manager = VectorStoreManager(vector_store_factory)
