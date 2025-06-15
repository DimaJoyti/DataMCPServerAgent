"""
Document Processing Module for DataMCPServerAgent.

This module provides comprehensive document processing capabilities including:
- Document parsing from multiple formats (PDF, DOCX, HTML, Markdown, TXT)
- Text chunking with various strategies
- Metadata extraction and enrichment
- Document preprocessing and cleaning
- Integration with vectorization pipeline
"""

from .chunking import AdaptiveChunker, BaseChunker, ChunkerFactory, SemanticChunker, TextChunker
from .document_processor import DocumentProcessingConfig, DocumentProcessor
from .metadata import DocumentMetadata, MetadataEnricher, MetadataExtractor
from .parsers import (
    BaseParser,
    DOCXParser,
    HTMLParser,
    MarkdownParser,
    ParserFactory,
    PDFParser,
    TextParser,
)

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    # Main processor
    "DocumentProcessor",
    "DocumentProcessingConfig",
    # Parsers
    "BaseParser",
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "MarkdownParser",
    "TextParser",
    "ParserFactory",
    # Chunking
    "BaseChunker",
    "TextChunker",
    "SemanticChunker",
    "AdaptiveChunker",
    "ChunkerFactory",
    # Metadata
    "DocumentMetadata",
    "MetadataExtractor",
    "MetadataEnricher",
]
