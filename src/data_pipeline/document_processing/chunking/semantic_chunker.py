"""
Semantic chunker implementation that uses embeddings for intelligent text splitting.
"""

from typing import List

from ..metadata.models import DocumentMetadata
from .base_chunker import BaseChunker, TextChunk


class SemanticChunker(BaseChunker):
    """Semantic chunker that uses embeddings to create semantically coherent chunks."""

    def __init__(self, *args, **kwargs):
        """Initialize semantic chunker."""
        super().__init__(*args, **kwargs)

        # Check if embeddings are enabled
        if not self.config.use_embeddings:
            self.logger.warning(
                "Semantic chunking works best with embeddings enabled. "
                "Set use_embeddings=True in config for better results."
            )

    def chunk_text(
        self, text: str, document_metadata: DocumentMetadata, **kwargs
    ) -> List[TextChunk]:
        """
        Chunk text using semantic similarity.

        Args:
            text: Text to chunk
            document_metadata: Document metadata
            **kwargs: Additional arguments

        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            return []

        # For now, fall back to sentence-based chunking with semantic awareness
        # This is a simplified implementation - a full semantic chunker would:
        # 1. Split text into sentences
        # 2. Generate embeddings for each sentence
        # 3. Use similarity metrics to group related sentences
        # 4. Create chunks based on semantic boundaries

        self.logger.info("Using simplified semantic chunking (sentence-based)")

        # Split into sentences first
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Group sentences into chunks based on similarity threshold
        chunks = self._group_sentences_semantically(sentences, document_metadata)

        # Link chunks together
        chunks = self._link_chunks(chunks)

        self.logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List[str]: List of sentences
        """
        import re

        # Simple sentence splitting - could be improved with NLP libraries
        sentence_endings = r"[.!?]+\s+"
        sentences = re.split(sentence_endings, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _group_sentences_semantically(
        self, sentences: List[str], document_metadata: DocumentMetadata
    ) -> List[TextChunk]:
        """
        Group sentences into semantically coherent chunks.

        Args:
            sentences: List of sentences
            document_metadata: Document metadata

        Returns:
            List[TextChunk]: List of text chunks
        """
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if (
                current_chunk_length + sentence_length > self.config.chunk_size
                and current_chunk_sentences
            ):

                # Create chunk from current sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, chunk_index, document_metadata
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap if configured
                if self.config.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk_sentences, self.config.chunk_overlap
                    )
                    current_chunk_sentences = overlap_sentences
                    current_chunk_length = sum(len(s) for s in overlap_sentences)
                else:
                    current_chunk_sentences = []
                    current_chunk_length = 0

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length

        # Create final chunk if there are remaining sentences
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, chunk_index, document_metadata
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_sentences(
        self, sentences: List[str], chunk_index: int, document_metadata: DocumentMetadata
    ) -> TextChunk:
        """
        Create a text chunk from sentences.

        Args:
            sentences: List of sentences
            chunk_index: Chunk index
            document_metadata: Document metadata

        Returns:
            TextChunk: Created text chunk
        """
        # Join sentences with appropriate spacing
        chunk_text = ". ".join(sentences)
        if not chunk_text.endswith("."):
            chunk_text += "."

        # For now, we don't have exact character positions
        # In a full implementation, we would track these
        start_char = chunk_index * self.config.chunk_size
        end_char = start_char + len(chunk_text)

        return self._create_chunk(
            text=chunk_text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            document_metadata=document_metadata,
            chunking_method="semantic",
            sentence_count=len(sentences),
        )

    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """
        Get sentences for overlap based on character count.

        Args:
            sentences: List of sentences
            overlap_size: Target overlap size in characters

        Returns:
            List[str]: Sentences for overlap
        """
        overlap_sentences = []
        current_length = 0

        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                current_length += len(sentence)
            else:
                break

        return overlap_sentences

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        This is a placeholder implementation. A full semantic chunker would:
        1. Generate embeddings for both texts
        2. Calculate cosine similarity
        3. Return similarity score

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0-1)
        """
        # Placeholder: simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
