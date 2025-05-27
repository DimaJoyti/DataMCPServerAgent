"""
Tests for document processing pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.document_processing import (
    DocumentProcessor,
    DocumentProcessingConfig,
    ParsingConfig,
    ChunkingConfig
)
from src.data_pipeline.document_processing.parsers import TextParser, ParserFactory
from src.data_pipeline.document_processing.chunking import TextChunker, ChunkerFactory
from src.data_pipeline.document_processing.metadata import MetadataExtractor, DocumentType


class TestDocumentProcessor:
    """Test document processor functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor()
        assert processor is not None
        assert processor.config is not None
    
    def test_process_text_content(self):
        """Test processing text content directly."""
        processor = DocumentProcessor()
        
        content = "This is a test document. It has multiple sentences. Each sentence provides information."
        result = processor.process_content(
            content=content,
            document_id="test_doc",
            document_type="text"
        )
        
        assert result is not None
        assert result.document_id == "test_doc"
        assert result.get_text() == content
        assert len(result.chunks) > 0
        assert result.processing_status in ["completed", "completed_with_errors"]
    
    def test_process_with_custom_config(self):
        """Test processing with custom configuration."""
        chunking_config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            strategy="text"
        )
        
        config = DocumentProcessingConfig(
            chunking_config=chunking_config,
            enable_chunking=True
        )
        
        processor = DocumentProcessor(config)
        
        content = "A" * 500  # Long text to ensure chunking
        result = processor.process_content(
            content=content,
            document_id="test_long",
            document_type="text"
        )
        
        assert len(result.chunks) > 1  # Should create multiple chunks
        assert all(len(chunk.text) <= 120 for chunk in result.chunks)  # Respect chunk size + overlap


class TestTextParser:
    """Test text parser functionality."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = TextParser()
        assert parser is not None
        assert "txt" in parser.supported_extensions
        assert DocumentType.TEXT in parser.supported_types
    
    def test_parse_simple_text(self):
        """Test parsing simple text."""
        parser = TextParser()
        
        content = "Hello, world! This is a test."
        result = parser.parse_content(
            content=content,
            document_id="test",
            document_type=DocumentType.TEXT
        )
        
        assert result.text == content
        assert result.metadata.document_id == "test"
        assert result.metadata.character_count == len(content)
        assert result.metadata.word_count == 6
    
    def test_parse_file(self):
        """Test parsing text file."""
        parser = TextParser()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content.\nSecond line.")
            temp_path = Path(f.name)
        
        try:
            result = parser.parse_file(temp_path)
            assert "Test file content" in result.text
            assert result.metadata.filename == temp_path.name
        finally:
            temp_path.unlink()


class TestTextChunker:
    """Test text chunker functionality."""
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = TextChunker()
        assert chunker is not None
        assert chunker.config.chunk_size == 1000  # Default
    
    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        from src.data_pipeline.document_processing.metadata.extractor import MetadataExtractor
        
        chunker = TextChunker()
        extractor = MetadataExtractor()
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        metadata = extractor.extract_from_content(text, "test_doc", DocumentType.TEXT)
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) >= 1
        assert all(chunk.text.strip() for chunk in chunks)
        assert all(chunk.metadata.document_id == "test_doc" for chunk in chunks)
    
    def test_chunk_long_text(self):
        """Test chunking long text."""
        from src.data_pipeline.document_processing.metadata.extractor import MetadataExtractor
        from src.data_pipeline.document_processing.chunking.base_chunker import ChunkingConfig
        
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        extractor = MetadataExtractor()
        
        # Create long text
        text = "This is a test sentence. " * 50  # ~1250 characters
        metadata = extractor.extract_from_content(text, "test_long", DocumentType.TEXT)
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk.text) <= 120 for chunk in chunks)  # Respect size + some tolerance


class TestMetadataExtractor:
    """Test metadata extraction functionality."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = MetadataExtractor()
        assert extractor is not None
    
    def test_extract_from_content(self):
        """Test extracting metadata from content."""
        extractor = MetadataExtractor()
        
        content = "This is a test document with multiple sentences. It has various characteristics."
        metadata = extractor.extract_from_content(
            content=content,
            document_id="test_meta",
            document_type=DocumentType.TEXT
        )
        
        assert metadata.document_id == "test_meta"
        assert metadata.document_type == DocumentType.TEXT
        assert metadata.character_count == len(content)
        assert metadata.word_count > 0
        assert metadata.sentence_count > 0
    
    def test_extract_from_file(self):
        """Test extracting metadata from file."""
        extractor = MetadataExtractor()
        
        # Create temporary file
        content = "File content for metadata extraction."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            metadata = extractor.extract_from_file(temp_path)
            assert metadata.filename == temp_path.name
            assert metadata.file_size > 0
            assert metadata.character_count == len(content)
        finally:
            temp_path.unlink()


class TestParserFactory:
    """Test parser factory functionality."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = ParserFactory()
        assert factory is not None
    
    def test_get_parser_for_text(self):
        """Test getting parser for text file."""
        factory = ParserFactory()
        
        parser = factory.get_parser(document_type=DocumentType.TEXT)
        assert isinstance(parser, TextParser)
    
    def test_get_parser_for_file(self):
        """Test getting parser for file."""
        factory = ParserFactory()
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            parser = factory.get_parser_for_file(temp_path)
            assert isinstance(parser, TextParser)
        finally:
            temp_path.unlink()
    
    def test_can_parse(self):
        """Test can_parse method."""
        factory = ParserFactory()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            txt_path = Path(f.name)
        
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            unknown_path = Path(f.name)
        
        try:
            assert factory.can_parse(txt_path) == True
            assert factory.can_parse(unknown_path) == False
        finally:
            txt_path.unlink()
            unknown_path.unlink()


class TestChunkerFactory:
    """Test chunker factory functionality."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = ChunkerFactory()
        assert factory is not None
    
    def test_get_text_chunker(self):
        """Test getting text chunker."""
        factory = ChunkerFactory()
        
        chunker = factory.get_chunker(strategy="text")
        assert isinstance(chunker, TextChunker)
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        factory = ChunkerFactory()
        
        strategies = factory.get_available_strategies()
        assert "text" in strategies
        assert len(strategies) > 0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end document processing."""
        # Create sample document
        content = """
        # Test Document
        
        This is a test document for integration testing.
        
        ## Section 1
        This section contains some information about testing.
        
        ## Section 2  
        This section contains more information for comprehensive testing.
        The content is designed to create multiple chunks.
        """
        
        # Configure processor
        chunking_config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            strategy="text"
        )
        
        config = DocumentProcessingConfig(
            chunking_config=chunking_config,
            enable_chunking=True,
            enable_metadata_enrichment=True
        )
        
        processor = DocumentProcessor(config)
        
        # Process content
        result = processor.process_content(
            content=content,
            document_id="integration_test",
            document_type="markdown",
            title="Test Document"
        )
        
        # Verify results
        assert result.document_id == "integration_test"
        assert len(result.chunks) > 1
        assert result.get_metadata().title == "Test Document"
        assert result.processing_status in ["completed", "completed_with_errors"]
        
        # Verify chunks
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i
            assert chunk.metadata.document_id == "integration_test"
            assert len(chunk.text) > 0


if __name__ == "__main__":
    pytest.main([__file__])
