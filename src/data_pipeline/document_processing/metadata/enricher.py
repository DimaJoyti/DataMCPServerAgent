"""
Metadata enrichment utilities for documents.
"""

import logging
import re
from typing import Any, Dict, List, Optional

try:
    from textstat import flesch_kincaid_grade, automated_readability_index
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

from .models import DocumentMetadata


logger = logging.getLogger(__name__)


class MetadataEnricher:
    """Enrich document metadata with additional analysis."""
    
    def __init__(self):
        """Initialize metadata enricher."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def enrich_metadata(self, metadata: DocumentMetadata, content: str) -> DocumentMetadata:
        """
        Enrich metadata with additional analysis.
        
        Args:
            metadata: Base metadata to enrich
            content: Document content for analysis
            
        Returns:
            DocumentMetadata: Enriched metadata
        """
        try:
            # Extract title if not present
            if not metadata.title:
                metadata.title = self._extract_title(content)
            
            # Extract keywords
            if not metadata.keywords:
                metadata.keywords = self._extract_keywords(content)
            
            # Calculate complexity score
            if not metadata.complexity_score:
                metadata.complexity_score = self._calculate_complexity(content)
            
            # Extract additional readability metrics
            self._enrich_readability(content, metadata)
            
            # Analyze content structure
            self._analyze_structure(content, metadata)
            
            # Extract entities and topics
            self._extract_entities(content, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to enrich metadata: {e}")
        
        return metadata
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract document title from content."""
        lines = content.split('\n')
        
        # Look for markdown-style headers
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
        
        # Look for HTML title tags
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        
        # Look for first non-empty line as potential title
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                return line
        
        return None
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate text complexity score."""
        if not content.strip():
            return 0.0
        
        # Simple complexity metrics
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable complexity (approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Combine metrics into complexity score (0-100)
        complexity = (
            (avg_word_length - 4) * 10 +
            (avg_sentence_length - 15) * 2 +
            (avg_syllables_per_word - 1.5) * 20
        )
        
        return max(0, min(100, complexity))
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _enrich_readability(self, content: str, metadata: DocumentMetadata) -> None:
        """Add additional readability metrics."""
        if not HAS_TEXTSTAT:
            return
        
        try:
            # Add grade level
            grade_level = flesch_kincaid_grade(content)
            metadata.add_custom_field('grade_level', grade_level)
            
            # Add automated readability index
            ari = automated_readability_index(content)
            metadata.add_custom_field('automated_readability_index', ari)
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate additional readability metrics: {e}")
    
    def _analyze_structure(self, content: str, metadata: DocumentMetadata) -> None:
        """Analyze document structure."""
        # Count different types of elements
        
        # Headers (markdown style)
        h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
        h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
        h3_count = len(re.findall(r'^### ', content, re.MULTILINE))
        
        metadata.add_custom_field('h1_count', h1_count)
        metadata.add_custom_field('h2_count', h2_count)
        metadata.add_custom_field('h3_count', h3_count)
        metadata.add_custom_field('total_headers', h1_count + h2_count + h3_count)
        
        # Lists
        bullet_lists = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE))
        
        metadata.add_custom_field('bullet_lists', bullet_lists)
        metadata.add_custom_field('numbered_lists', numbered_lists)
        
        # Links (markdown and HTML)
        markdown_links = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content))
        html_links = len(re.findall(r'<a[^>]+href', content, re.IGNORECASE))
        
        metadata.add_custom_field('markdown_links', markdown_links)
        metadata.add_custom_field('html_links', html_links)
        metadata.add_custom_field('total_links', markdown_links + html_links)
        
        # Code blocks
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        inline_code = len(re.findall(r'`[^`]+`', content))
        
        metadata.add_custom_field('code_blocks', code_blocks)
        metadata.add_custom_field('inline_code', inline_code)
    
    def _extract_entities(self, content: str, metadata: DocumentMetadata) -> None:
        """Extract entities and topics from content."""
        # Simple entity extraction using regex patterns
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        if emails:
            metadata.add_custom_field('email_addresses', list(set(emails)))
        
        # URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        if urls:
            metadata.add_custom_field('urls', list(set(urls)))
        
        # Phone numbers (simple pattern)
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content)
        if phones:
            metadata.add_custom_field('phone_numbers', list(set(phones)))
        
        # Dates (simple patterns)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content)
        if dates:
            metadata.add_custom_field('dates', list(set(dates)))
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        if len(numbers) > 5:  # Only store if there are significant numbers
            metadata.add_custom_field('number_count', len(numbers))
    
    def add_custom_analysis(
        self,
        metadata: DocumentMetadata,
        analysis_name: str,
        analysis_result: Any
    ) -> None:
        """Add custom analysis result to metadata."""
        metadata.add_custom_field(f"analysis_{analysis_name}", analysis_result)
