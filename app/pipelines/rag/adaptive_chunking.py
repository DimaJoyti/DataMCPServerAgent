"""
Adaptive Chunking for RAG Architecture.

This module provides intelligent document chunking strategies:
- Context-aware chunking
- Semantic boundary detection
- Dynamic chunk sizing
- Overlap optimization
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.core.logging import get_logger, LoggerMixin

class ChunkingStrategy(str, Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"
    SENTENCE_BASED = "sentence_based"

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    chunk_id: str
    start_position: int
    end_position: int
    chunk_size: int
    overlap_size: int
    strategy_used: ChunkingStrategy
    confidence_score: float = 0.0

class ChunkedDocument(BaseModel):
    """Document divided into chunks."""

    document_id: str = Field(..., description="Original document ID")
    chunks: List[str] = Field(..., description="Document chunks")
    metadata: List[ChunkMetadata] = Field(..., description="Chunk metadata")
    total_chunks: int = Field(..., description="Total number of chunks")

    class Config:
        arbitrary_types_allowed = True

class AdaptiveChunker(LoggerMixin):
    """Adaptive document chunker."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive chunker."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.default_chunk_size = self.config.get("default_chunk_size", 1000)
        self.overlap_size = self.config.get("overlap_size", 200)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 2000)

        self.logger.info("AdaptiveChunker initialized")

    async def chunk_document(self, text: str, document_id: str,
                           strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE) -> ChunkedDocument:
        """Chunk a document using the specified strategy."""

        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks, metadata = await self._fixed_size_chunking(text, document_id)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks, metadata = await self._semantic_chunking(text, document_id)
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            chunks, metadata = await self._sentence_based_chunking(text, document_id)
        else:  # ADAPTIVE
            chunks, metadata = await self._adaptive_chunking(text, document_id)

        return ChunkedDocument(
            document_id=document_id,
            chunks=chunks,
            metadata=metadata,
            total_chunks=len(chunks)
        )

    async def _fixed_size_chunking(self, text: str, document_id: str) -> tuple[List[str], List[ChunkMetadata]]:
        """Fixed size chunking."""
        chunks = []
        metadata = []

        for i in range(0, len(text), self.default_chunk_size - self.overlap_size):
            chunk_start = i
            chunk_end = min(i + self.default_chunk_size, len(text))
            chunk_text = text[chunk_start:chunk_end]

            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunks.append(chunk_text)
            metadata.append(ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                start_position=chunk_start,
                end_position=chunk_end,
                chunk_size=len(chunk_text),
                overlap_size=self.overlap_size if i > 0 else 0,
                strategy_used=ChunkingStrategy.FIXED_SIZE,
                confidence_score=1.0
            ))

        return chunks, metadata

    async def _semantic_chunking(self, text: str, document_id: str) -> tuple[List[str], List[ChunkMetadata]]:
        """Semantic boundary-based chunking."""
        # Placeholder implementation
        # In production, use NLP models to detect semantic boundaries

        # Simple sentence-based approach for now
        sentences = text.split('. ')
        chunks = []
        metadata = []

        current_chunk = ""
        chunk_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.default_chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                metadata.append(ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    start_position=chunk_start,
                    end_position=chunk_start + len(current_chunk),
                    chunk_size=len(current_chunk),
                    overlap_size=0,
                    strategy_used=ChunkingStrategy.SEMANTIC,
                    confidence_score=0.8
                ))

                # Start new chunk
                chunk_start += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append(ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                start_position=chunk_start,
                end_position=chunk_start + len(current_chunk),
                chunk_size=len(current_chunk),
                overlap_size=0,
                strategy_used=ChunkingStrategy.SEMANTIC,
                confidence_score=0.8
            ))

        return chunks, metadata

    async def _sentence_based_chunking(self, text: str, document_id: str) -> tuple[List[str], List[ChunkMetadata]]:
        """Sentence-based chunking."""
        sentences = text.split('. ')
        chunks = []
        metadata = []

        current_chunk = ""
        chunk_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.default_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                metadata.append(ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    start_position=chunk_start,
                    end_position=chunk_start + len(current_chunk),
                    chunk_size=len(current_chunk),
                    overlap_size=0,
                    strategy_used=ChunkingStrategy.SENTENCE_BASED,
                    confidence_score=0.9
                ))

                chunk_start += len(current_chunk)
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append(ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                start_position=chunk_start,
                end_position=chunk_start + len(current_chunk),
                chunk_size=len(current_chunk),
                overlap_size=0,
                strategy_used=ChunkingStrategy.SENTENCE_BASED,
                confidence_score=0.9
            ))

        return chunks, metadata

    async def _adaptive_chunking(self, text: str, document_id: str) -> tuple[List[str], List[ChunkMetadata]]:
        """Adaptive chunking that combines multiple strategies."""
        # For now, use semantic chunking as the adaptive approach
        # In production, this would analyze the text and choose the best strategy

        return await self._semantic_chunking(text, document_id)
