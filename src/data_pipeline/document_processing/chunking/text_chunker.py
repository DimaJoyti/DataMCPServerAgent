"""
Basic text chunker implementation.
"""

from typing import List

from ..metadata.models import DocumentMetadata
from .base_chunker import BaseChunker, TextChunk


class TextChunker(BaseChunker):
    """Basic text chunker that splits text based on character count and boundaries."""

    def chunk_text(
        self, text: str, document_metadata: DocumentMetadata, **kwargs
    ) -> List[TextChunk]:
        """
        Chunk text into smaller pieces based on character count.

        Args:
            text: Text to chunk
            document_metadata: Document metadata
            **kwargs: Additional arguments

        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        current_position = 0
        chunk_index = 0

        while current_position < len(text):
            # Calculate chunk end position
            chunk_end = min(current_position + self.config.chunk_size, len(text))

            # Find optimal split point if not at end of text
            if chunk_end < len(text):
                chunk_end = self._get_optimal_split_point(text, chunk_end)

            # Extract chunk text
            chunk_text = text[current_position:chunk_end].strip()

            # Skip empty chunks
            if not chunk_text:
                current_position = chunk_end
                continue

            # Validate chunk size
            if not self._validate_chunk_size(chunk_text):
                # If chunk is too small, try to extend it
                if len(chunk_text) < self.config.min_chunk_size and chunk_end < len(text):
                    extended_end = min(chunk_end + self.config.min_chunk_size, len(text))
                    extended_text = text[current_position:extended_end].strip()
                    if self._validate_chunk_size(extended_text):
                        chunk_text = extended_text
                        chunk_end = extended_end

                # If chunk is too large, split it
                elif len(chunk_text) > self.config.max_chunk_size:
                    chunk_text = chunk_text[: self.config.max_chunk_size]
                    chunk_end = current_position + len(chunk_text)

            # Create chunk
            chunk = self._create_chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=current_position,
                end_char=chunk_end,
                document_metadata=document_metadata,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next position with overlap
            if chunk_end >= len(text):
                break

            # Calculate next position with overlap
            overlap_start = max(0, chunk_end - self.config.chunk_overlap)
            current_position = overlap_start

            # Ensure we make progress
            if current_position >= chunk_end:
                current_position = chunk_end

        # Link chunks together
        chunks = self._link_chunks(chunks)

        self.logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")

        return chunks

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
        # Use custom separators if provided
        if self.config.custom_separators:
            return self._find_custom_separator_split(text, target_position, search_window)

        # Use base implementation
        return super()._get_optimal_split_point(text, target_position, search_window)

    def _find_custom_separator_split(
        self, text: str, target_position: int, search_window: int
    ) -> int:
        """
        Find split point using custom separators.

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

        best_position = target_position
        best_distance = float("inf")

        for separator in self.config.custom_separators:
            # Find all occurrences of separator in search window
            separator_positions = []
            start_pos = 0

            while True:
                pos = search_text.find(separator, start_pos)
                if pos == -1:
                    break
                separator_positions.append(start + pos + len(separator))
                start_pos = pos + 1

            # Find closest separator to target
            for sep_pos in separator_positions:
                distance = abs(sep_pos - target_position)
                if distance < best_distance:
                    best_distance = distance
                    best_position = sep_pos

        return best_position

    def chunk_by_sentences(
        self, text: str, document_metadata: DocumentMetadata, sentences_per_chunk: int = 5
    ) -> List[TextChunk]:
        """
        Chunk text by sentences.

        Args:
            text: Text to chunk
            document_metadata: Document metadata
            sentences_per_chunk: Number of sentences per chunk

        Returns:
            List[TextChunk]: List of text chunks
        """
        # Find sentence boundaries
        sentence_boundaries = self._find_sentence_boundaries(text)

        chunks = []
        chunk_index = 0

        for i in range(0, len(sentence_boundaries) - 1, sentences_per_chunk):
            start_boundary = sentence_boundaries[i]
            end_boundary = sentence_boundaries[
                min(i + sentences_per_chunk, len(sentence_boundaries) - 1)
            ]

            chunk_text = text[start_boundary:end_boundary].strip()

            if chunk_text and self._validate_chunk_size(chunk_text):
                chunk = self._create_chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_boundary,
                    end_char=end_boundary,
                    document_metadata=document_metadata,
                    chunking_method="sentence_based",
                )

                chunks.append(chunk)
                chunk_index += 1

        return self._link_chunks(chunks)

    def chunk_by_paragraphs(
        self, text: str, document_metadata: DocumentMetadata, paragraphs_per_chunk: int = 2
    ) -> List[TextChunk]:
        """
        Chunk text by paragraphs.

        Args:
            text: Text to chunk
            document_metadata: Document metadata
            paragraphs_per_chunk: Number of paragraphs per chunk

        Returns:
            List[TextChunk]: List of text chunks
        """
        # Find paragraph boundaries
        paragraph_boundaries = self._find_paragraph_boundaries(text)

        chunks = []
        chunk_index = 0

        for i in range(0, len(paragraph_boundaries) - 1, paragraphs_per_chunk):
            start_boundary = paragraph_boundaries[i]
            end_boundary = paragraph_boundaries[
                min(i + paragraphs_per_chunk, len(paragraph_boundaries) - 1)
            ]

            chunk_text = text[start_boundary:end_boundary].strip()

            if chunk_text and self._validate_chunk_size(chunk_text):
                chunk = self._create_chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_boundary,
                    end_char=end_boundary,
                    document_metadata=document_metadata,
                    chunking_method="paragraph_based",
                )

                chunks.append(chunk)
                chunk_index += 1

        return self._link_chunks(chunks)

    def chunk_by_sections(self, text: str, document_metadata: DocumentMetadata) -> List[TextChunk]:
        """
        Chunk text by sections (headers).

        Args:
            text: Text to chunk
            document_metadata: Document metadata

        Returns:
            List[TextChunk]: List of text chunks
        """
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(text)

        chunks = []
        chunk_index = 0

        for i in range(len(section_boundaries) - 1):
            start_boundary = section_boundaries[i]
            end_boundary = section_boundaries[i + 1]

            chunk_text = text[start_boundary:end_boundary].strip()

            if chunk_text:
                # If section is too large, split it further
                if len(chunk_text) > self.config.max_chunk_size:
                    # Split large section into smaller chunks
                    section_chunks = self.chunk_text(chunk_text, document_metadata)
                    for section_chunk in section_chunks:
                        # Adjust positions relative to full text
                        section_chunk.start_char += start_boundary
                        section_chunk.end_char += start_boundary
                        section_chunk.chunk_index = chunk_index
                        section_chunk.metadata.chunk_index = chunk_index
                        section_chunk.metadata.chunking_method = "section_based"
                        chunks.append(section_chunk)
                        chunk_index += 1
                else:
                    chunk = self._create_chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start_boundary,
                        end_char=end_boundary,
                        document_metadata=document_metadata,
                        chunking_method="section_based",
                    )

                    chunks.append(chunk)
                    chunk_index += 1

        return self._link_chunks(chunks)
