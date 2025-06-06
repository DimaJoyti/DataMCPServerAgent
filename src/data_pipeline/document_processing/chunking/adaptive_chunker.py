"""
Adaptive chunker implementation that adjusts chunk size based on content characteristics.
"""

import logging
import re
from typing import List, Tuple

from .base_chunker import BaseChunker, TextChunk
from ..metadata.models import DocumentMetadata


class AdaptiveChunker(BaseChunker):
    """Adaptive chunker that adjusts chunk size based on content characteristics."""
    
    def chunk_text(
        self,
        text: str,
        document_metadata: DocumentMetadata,
        **kwargs
    ) -> List[TextChunk]:
        """
        Chunk text using adaptive sizing based on content characteristics.
        
        Args:
            text: Text to chunk
            document_metadata: Document metadata
            **kwargs: Additional arguments
            
        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            return []
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_characteristics(text)
        
        # Adapt chunking strategy based on analysis
        adapted_config = self._adapt_chunking_config(text_analysis)
        
        self.logger.info(
            f"Adaptive chunking: detected {text_analysis['content_type']}, "
            f"using chunk size {adapted_config['chunk_size']}"
        )
        
        # Perform chunking with adapted configuration
        chunks = self._chunk_with_adapted_config(text, document_metadata, adapted_config)
        
        # Link chunks together
        chunks = self._link_chunks(chunks)
        
        self.logger.info(f"Created {len(chunks)} adaptive chunks")
        
        return chunks
    
    def _analyze_text_characteristics(self, text: str) -> dict:
        """
        Analyze text characteristics to determine optimal chunking strategy.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Text analysis results
        """
        analysis = {
            'total_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'avg_sentence_length': 0,
            'avg_paragraph_length': 0,
            'content_type': 'general',
            'complexity_score': 0,
            'structure_score': 0
        }
        
        # Calculate averages
        if analysis['sentence_count'] > 0:
            analysis['avg_sentence_length'] = analysis['word_count'] / analysis['sentence_count']
        
        if analysis['paragraph_count'] > 0:
            analysis['avg_paragraph_length'] = analysis['word_count'] / analysis['paragraph_count']
        
        # Detect content type
        analysis['content_type'] = self._detect_content_type(text)
        
        # Calculate complexity score
        analysis['complexity_score'] = self._calculate_complexity_score(text)
        
        # Calculate structure score
        analysis['structure_score'] = self._calculate_structure_score(text)
        
        return analysis
    
    def _detect_content_type(self, text: str) -> str:
        """
        Detect the type of content to optimize chunking strategy.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Detected content type
        """
        # Check for code content
        if self._is_code_content(text):
            return 'code'
        
        # Check for academic/technical content
        if self._is_academic_content(text):
            return 'academic'
        
        # Check for narrative content
        if self._is_narrative_content(text):
            return 'narrative'
        
        # Check for structured content (lists, tables)
        if self._is_structured_content(text):
            return 'structured'
        
        # Check for conversational content
        if self._is_conversational_content(text):
            return 'conversational'
        
        return 'general'
    
    def _is_code_content(self, text: str) -> bool:
        """Check if text contains significant code content."""
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*[{:]',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'#include\s*<',  # C/C++ includes
            r'```\w*\n',  # Code blocks
            r'^\s*//.*$',  # Single line comments
            r'^\s*/\*.*\*/$',  # Multi-line comments
        ]
        
        code_matches = sum(len(re.findall(pattern, text, re.MULTILINE)) for pattern in code_indicators)
        return code_matches > len(text.split('\n')) * 0.1  # 10% of lines have code indicators
    
    def _is_academic_content(self, text: str) -> bool:
        """Check if text is academic/technical content."""
        academic_indicators = [
            r'\b(abstract|introduction|methodology|results|conclusion|references)\b',
            r'\b(figure|table|equation|theorem|lemma|proof)\s+\d+',
            r'\b(et al\.|ibid\.|op\. cit\.)',
            r'\[\d+\]',  # Citation numbers
            r'\b\d+\.\d+\b',  # Section numbers
        ]
        
        academic_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in academic_indicators)
        return academic_matches > 5
    
    def _is_narrative_content(self, text: str) -> bool:
        """Check if text is narrative content."""
        narrative_indicators = [
            r'\b(once upon a time|in the beginning|meanwhile|suddenly|finally)\b',
            r'\b(he|she|they)\s+(said|thought|felt|walked|ran)',
            r'"[^"]*"',  # Dialogue
        ]
        
        narrative_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in narrative_indicators)
        return narrative_matches > len(text.split()) * 0.02  # 2% of words are narrative indicators
    
    def _is_structured_content(self, text: str) -> bool:
        """Check if text has structured content."""
        structured_indicators = [
            r'^\s*[-*+]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
            r'\|.*\|',  # Table rows
            r'^#{1,6}\s+',  # Headers
        ]
        
        structured_matches = sum(len(re.findall(pattern, text, re.MULTILINE)) for pattern in structured_indicators)
        return structured_matches > len(text.split('\n')) * 0.2  # 20% of lines are structured
    
    def _is_conversational_content(self, text: str) -> bool:
        """Check if text is conversational."""
        conversational_indicators = [
            r'\b(you|your|we|our|us)\b',
            r'\b(what|how|why|when|where)\b.*\?',
            r'\b(thanks|please|sorry|excuse me)\b',
            r'!{1,3}',  # Exclamation marks
        ]
        
        conversational_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conversational_indicators)
        return conversational_matches > len(text.split()) * 0.05  # 5% of words are conversational
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence length variation
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if sentence_lengths:
            import statistics
            sentence_length_std = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        else:
            sentence_length_std = 0
        
        # Vocabulary diversity (unique words / total words)
        unique_words = set(word.lower() for word in words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Combine metrics (normalized to 0-1)
        complexity = (
            min(avg_word_length / 10, 1.0) * 0.3 +
            min(sentence_length_std / 20, 1.0) * 0.3 +
            vocabulary_diversity * 0.4
        )
        
        return complexity
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate text structure score (0-1)."""
        lines = text.split('\n')
        if not lines:
            return 0.0
        
        # Count structured elements
        headers = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*+\d]+[\.\)]\s+', text, re.MULTILINE))
        paragraphs = len(text.split('\n\n'))
        
        # Calculate structure density
        total_lines = len(lines)
        structure_density = (headers + lists) / total_lines if total_lines > 0 else 0
        
        # Paragraph consistency
        paragraph_lengths = [len(p.split()) for p in text.split('\n\n') if p.strip()]
        if paragraph_lengths:
            import statistics
            paragraph_consistency = 1.0 - (statistics.stdev(paragraph_lengths) / max(paragraph_lengths)) if len(paragraph_lengths) > 1 else 1.0
        else:
            paragraph_consistency = 0.0
        
        # Combine metrics
        structure_score = (structure_density * 0.6 + paragraph_consistency * 0.4)
        
        return min(structure_score, 1.0)
    
    def _adapt_chunking_config(self, analysis: dict) -> dict:
        """
        Adapt chunking configuration based on text analysis.
        
        Args:
            analysis: Text analysis results
            
        Returns:
            dict: Adapted configuration
        """
        base_chunk_size = self.config.chunk_size
        base_overlap = self.config.chunk_overlap
        
        # Adjust based on content type
        if analysis['content_type'] == 'code':
            # Smaller chunks for code to preserve function boundaries
            chunk_size = int(base_chunk_size * 0.7)
            overlap = int(base_overlap * 0.5)
            preserve_boundaries = True
        elif analysis['content_type'] == 'academic':
            # Larger chunks for academic content to preserve context
            chunk_size = int(base_chunk_size * 1.3)
            overlap = int(base_overlap * 1.2)
            preserve_boundaries = True
        elif analysis['content_type'] == 'narrative':
            # Medium chunks for narrative, preserve paragraph boundaries
            chunk_size = base_chunk_size
            overlap = base_overlap
            preserve_boundaries = True
        elif analysis['content_type'] == 'structured':
            # Smaller chunks for structured content
            chunk_size = int(base_chunk_size * 0.8)
            overlap = int(base_overlap * 0.8)
            preserve_boundaries = True
        elif analysis['content_type'] == 'conversational':
            # Smaller chunks for conversational content
            chunk_size = int(base_chunk_size * 0.6)
            overlap = int(base_overlap * 1.5)
            preserve_boundaries = False
        else:
            chunk_size = base_chunk_size
            overlap = base_overlap
            preserve_boundaries = True
        
        # Adjust based on complexity
        complexity_factor = 1.0 + (analysis['complexity_score'] - 0.5) * 0.4
        chunk_size = int(chunk_size * complexity_factor)
        
        # Adjust based on structure
        if analysis['structure_score'] > 0.7:
            # Highly structured content - respect boundaries more
            overlap = int(overlap * 0.8)
        
        # Ensure limits
        chunk_size = max(self.config.min_chunk_size, min(chunk_size, self.config.max_chunk_size))
        overlap = max(0, min(overlap, chunk_size // 2))
        
        return {
            'chunk_size': chunk_size,
            'overlap': overlap,
            'preserve_boundaries': preserve_boundaries,
            'content_type': analysis['content_type']
        }
    
    def _chunk_with_adapted_config(
        self,
        text: str,
        document_metadata: DocumentMetadata,
        adapted_config: dict
    ) -> List[TextChunk]:
        """
        Perform chunking with adapted configuration.
        
        Args:
            text: Text to chunk
            document_metadata: Document metadata
            adapted_config: Adapted configuration
            
        Returns:
            List[TextChunk]: List of text chunks
        """
        chunks = []
        current_position = 0
        chunk_index = 0
        
        chunk_size = adapted_config['chunk_size']
        overlap = adapted_config['overlap']
        preserve_boundaries = adapted_config['preserve_boundaries']
        
        while current_position < len(text):
            # Calculate chunk end position
            chunk_end = min(current_position + chunk_size, len(text))
            
            # Find optimal split point if preserving boundaries
            if preserve_boundaries and chunk_end < len(text):
                chunk_end = self._get_optimal_split_point(text, chunk_end)
            
            # Extract chunk text
            chunk_text = text[current_position:chunk_end].strip()
            
            # Skip empty chunks
            if not chunk_text:
                current_position = chunk_end
                continue
            
            # Create chunk
            chunk = self._create_chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=current_position,
                end_char=chunk_end,
                document_metadata=document_metadata,
                chunking_method="adaptive",
                content_type=adapted_config['content_type'],
                adapted_chunk_size=chunk_size
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move to next position with overlap
            if chunk_end >= len(text):
                break
            
            # Calculate next position with overlap
            overlap_start = max(current_position, chunk_end - overlap)
            current_position = overlap_start
            
            # Ensure we make progress
            if current_position >= chunk_end:
                current_position = chunk_end
        
        return chunks
