"""
Chunker factory for creating appropriate chunkers based on strategy.
"""

import logging
from typing import Dict, Optional, Type

from .base_chunker import BaseChunker, ChunkingConfig
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)

class ChunkerFactory:
    """Factory for creating text chunkers."""

    def __init__(self):
        """Initialize chunker factory."""
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._register_default_chunkers()

    def _register_default_chunkers(self) -> None:
        """Register default chunkers."""
        self.register_chunker("text", TextChunker)

        # Register semantic chunker if available
        try:
            from .semantic_chunker import SemanticChunker
            self.register_chunker("semantic", SemanticChunker)
        except ImportError:
            logger.warning("Semantic chunker not available - missing dependencies")

        # Register adaptive chunker if available
        try:
            from .adaptive_chunker import AdaptiveChunker
            self.register_chunker("adaptive", AdaptiveChunker)
        except ImportError:
            logger.warning("Adaptive chunker not available - missing dependencies")

    def register_chunker(self, strategy: str, chunker_class: Type[BaseChunker]) -> None:
        """
        Register a chunker class for a strategy.

        Args:
            strategy: Chunking strategy name
            chunker_class: Chunker class to register
        """
        self._chunkers[strategy.lower()] = chunker_class
        logger.debug(f"Registered {chunker_class.__name__} for strategy '{strategy}'")

    def get_chunker(
        self,
        strategy: str = "text",
        config: Optional[ChunkingConfig] = None
    ) -> BaseChunker:
        """
        Get chunker for specified strategy.

        Args:
            strategy: Chunking strategy
            config: Chunking configuration

        Returns:
            BaseChunker: Chunker instance

        Raises:
            ValueError: If strategy not supported
        """
        strategy = strategy.lower()

        chunker_class = self._chunkers.get(strategy)
        if chunker_class is None:
            # Try fallback to text chunker
            if strategy != "text":
                logger.warning(f"Strategy '{strategy}' not available, falling back to 'text'")
                chunker_class = self._chunkers.get("text")

            if chunker_class is None:
                raise ValueError(f"No chunker available for strategy: {strategy}")

        try:
            return chunker_class(config)
        except Exception as e:
            logger.error(f"Failed to create chunker {chunker_class.__name__}: {e}")
            raise

    def get_available_strategies(self) -> list[str]:
        """
        Get list of available chunking strategies.

        Returns:
            List[str]: Available strategies
        """
        return list(self._chunkers.keys())

    def is_strategy_available(self, strategy: str) -> bool:
        """
        Check if chunking strategy is available.

        Args:
            strategy: Strategy name

        Returns:
            bool: True if strategy is available
        """
        return strategy.lower() in self._chunkers

# Global chunker factory instance
chunker_factory = ChunkerFactory()
