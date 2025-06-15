"""
Base chunker interface for text chunking.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ..metadata.models import ChunkMetadata, DocumentMetadata


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    # Basic chunking parameters
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters")

    # Chunking strategy
    strategy: str = Field(
        default="text", description="Chunking strategy (text, semantic, adaptive)"
    )

    # Text processing options
    preserve_sentences: bool = Field(
        default=True, description="Try to preserve sentence boundaries"
    )
    preserve_paragraphs: bool = Field(
        default=True, description="Try to preserve paragraph boundaries"
    )
    preserve_sections: bool = Field(default=True, description="Try to preserve section boundaries")

    # Size constraints
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size in characters")

    # Language and encoding
    language: Optional[str] = Field(default=None, description="Document language for processing")

    # Semantic chunking options (if applicable)
    similarity_threshold: float = Field(
        default=0.7, description="Similarity threshold for semantic chunking"
    )
    use_embeddings: bool = Field(default=False, description="Use embeddings for semantic chunking")

    # Custom options
    custom_separators: List[str] = Field(
        default_factory=list, description="Custom chunk separators"
    )
    custom_options: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific options"
    )


class TextChunk(BaseModel):
    """Represents a text chunk."""

    # Identification
    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique chunk identifier"
    )
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Chunk index in document")

    # Content
    text: str = Field(..., description="Chunk text content")

    # Position information
    start_char: int = Field(..., description="Start character position in document")
    end_char: int = Field(..., description="End character position in document")

    # Metadata
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")

    # Optional context
    previous_chunk_id: Optional[str] = Field(default=None, description="Previous chunk ID")
    next_chunk_id: Optional[str] = Field(default=None, description="Next chunk ID")

    def __len__(self) -> int:
        """Return chunk length."""
        return len(self.text)

    def __str__(self) -> str:
        """Return chunk text."""
        return self.text

    def get_context_window(self, window_size: int = 100) -> str:
        """
        Get text with context window.

        Args:
            window_size: Size of context window on each side

        Returns:
            str: Text with context
        """
        # This would need access to the full document text
        # For now, just return the chunk text
        return self.text


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chunk_text(
        self, text: str, document_metadata: DocumentMetadata, **kwargs
    ) -> List[TextChunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text to chunk
            document_metadata: Document metadata
            **kwargs: Additional arguments

        Returns:
            List[TextChunk]: List of text chunks
        """
        pass

    def chunk_document(self, text: str, document_id: str, **metadata_kwargs) -> List[TextChunk]:
        """
        Chunk a document with minimal metadata.

        Args:
            text: Document text
            document_id: Document identifier
            **metadata_kwargs: Additional metadata fields

        Returns:
            List[TextChunk]: List of text chunks
        """
        # Create minimal document metadata
        from ..metadata.extractor import MetadataExtractor

        extractor = MetadataExtractor()

        document_metadata = extractor.extract_from_content(text, document_id, **metadata_kwargs)

        return self.chunk_text(text, document_metadata)

    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        document_metadata: DocumentMetadata,
        **additional_metadata,
    ) -> TextChunk:
        """
        Create a text chunk with metadata.

        Args:
            text: Chunk text
            chunk_index: Chunk index
            start_char: Start character position
            end_char: End character position
            document_metadata: Document metadata
            **additional_metadata: Additional metadata fields

        Returns:
            TextChunk: Created text chunk
        """
        chunk_id = str(uuid4())

        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_metadata.document_id,
            chunk_index=chunk_index,
            text=text,
            character_count=len(text),
            word_count=len(text.split()),
            sentence_count=len([s for s in text.split(".") if s.strip()]),
            start_char=start_char,
            end_char=end_char,
            chunking_strategy=self.__class__.__name__,
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap,
            **additional_metadata,
        )

        return TextChunk(
            chunk_id=chunk_id,
            document_id=document_metadata.document_id,
            chunk_index=chunk_index,
            text=text,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata,
        )

    def _link_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Link chunks together with previous/next references.

        Args:
            chunks: List of chunks to link

        Returns:
            List[TextChunk]: Linked chunks
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id

        return chunks

    def _validate_chunk_size(self, text: str) -> bool:
        """
        Validate if chunk size is within acceptable limits.

        Args:
            text: Chunk text

        Returns:
            bool: True if chunk size is valid
        """
        chunk_length = len(text)
        return self.config.min_chunk_size <= chunk_length <= self.config.max_chunk_size

    def _find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries in text.

        Args:
            text: Text to analyze

        Returns:
            List[int]: List of sentence boundary positions
        """
        import re

        # Simple sentence boundary detection
        sentence_endings = r"[.!?]+\s+"
        boundaries = [0]

        for match in re.finditer(sentence_endings, text):
            boundaries.append(match.end())

        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries

    def _find_paragraph_boundaries(self, text: str) -> List[int]:
        """
        Find paragraph boundaries in text.

        Args:
            text: Text to analyze

        Returns:
            List[int]: List of paragraph boundary positions
        """
        boundaries = [0]

        # Find double newlines (paragraph separators)
        import re

        for match in re.finditer(r"\n\s*\n", text):
            boundaries.append(match.end())

        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries

    def _find_section_boundaries(self, text: str) -> List[int]:
        """
        Find section boundaries in text (headers, etc.).

        Args:
            text: Text to analyze

        Returns:
            List[int]: List of section boundary positions
        """
        boundaries = [0]

        # Find markdown-style headers
        import re

        header_pattern = r"^#{1,6}\s+.+$"

        for match in re.finditer(header_pattern, text, re.MULTILINE):
            boundaries.append(match.start())

        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return sorted(set(boundaries))

    def _get_optimal_split_point(
        self, text: str, target_position: int, search_window: int = 100
    ) -> int:
        """
        Find optimal split point near target position.

        Args:
            text: Text to split
            target_position: Target split position
            search_window: Search window around target position

        Returns:
            int: Optimal split position
        """
        start = max(0, target_position - search_window)
        end = min(len(text), target_position + search_window)
        search_text = text[start:end]

        # Look for sentence boundaries first
        if self.config.preserve_sentences:
            import re

            sentence_endings = list(re.finditer(r"[.!?]+\s+", search_text))
            if sentence_endings:
                # Find closest to target
                target_in_window = target_position - start
                closest = min(sentence_endings, key=lambda m: abs(m.end() - target_in_window))
                return start + closest.end()

        # Look for paragraph boundaries
        if self.config.preserve_paragraphs:
            import re

            para_breaks = list(re.finditer(r"\n\s*\n", search_text))
            if para_breaks:
                target_in_window = target_position - start
                closest = min(para_breaks, key=lambda m: abs(m.end() - target_in_window))
                return start + closest.end()

        # Look for word boundaries
        import re

        word_boundaries = list(re.finditer(r"\s+", search_text))
        if word_boundaries:
            target_in_window = target_position - start
            closest = min(word_boundaries, key=lambda m: abs(m.start() - target_in_window))
            return start + closest.start()

        # Fallback to target position
        return target_position
