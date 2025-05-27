"""
Main document processor that orchestrates parsing, chunking, and metadata extraction.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .parsers import BaseParser, ParsedDocument, ParsingConfig, ParserFactory
from .chunking import BaseChunker, ChunkingConfig, TextChunk, ChunkerFactory
from .metadata import DocumentMetadata, MetadataExtractor, MetadataEnricher


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""
    
    # Parsing configuration
    parsing_config: ParsingConfig = Field(default_factory=ParsingConfig)
    
    # Chunking configuration
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    
    # Processing options
    enable_chunking: bool = Field(default=True, description="Enable text chunking")
    enable_metadata_enrichment: bool = Field(default=True, description="Enable metadata enrichment")
    
    # Error handling
    ignore_parsing_errors: bool = Field(default=False, description="Continue on parsing errors")
    ignore_chunking_errors: bool = Field(default=False, description="Continue on chunking errors")
    
    # Performance options
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes")
    processing_timeout: int = Field(default=300, description="Processing timeout in seconds")


class DocumentProcessingResult(BaseModel):
    """Result of document processing."""
    
    # Input information
    source_path: Optional[str] = Field(None, description="Source file path")
    document_id: str = Field(..., description="Document identifier")
    
    # Processing results
    parsed_document: ParsedDocument = Field(..., description="Parsed document")
    chunks: List[TextChunk] = Field(default_factory=list, description="Text chunks")
    
    # Processing metadata
    processing_time: float = Field(..., description="Total processing time in seconds")
    processing_status: str = Field(..., description="Processing status")
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    
    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return len(self.errors) > 0 or self.parsed_document.has_errors()
    
    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return len(self.warnings) > 0 or self.parsed_document.has_warnings()
    
    def get_text(self) -> str:
        """Get the full document text."""
        return self.parsed_document.text
    
    def get_chunk_texts(self) -> List[str]:
        """Get list of chunk texts."""
        return [chunk.text for chunk in self.chunks]
    
    def get_metadata(self) -> DocumentMetadata:
        """Get document metadata."""
        return self.parsed_document.metadata


class DocumentProcessor:
    """Main document processor that orchestrates parsing, chunking, and metadata extraction."""
    
    def __init__(self, config: Optional[DocumentProcessingConfig] = None):
        """
        Initialize document processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or DocumentProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.parser_factory = ParserFactory()
        self.chunker_factory = ChunkerFactory()
        self.metadata_extractor = MetadataExtractor()
        self.metadata_enricher = MetadataEnricher()
    
    def process_file(self, file_path: Union[str, Path]) -> DocumentProcessingResult:
        """
        Process a document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            DocumentProcessingResult: Processing result
        """
        path = Path(file_path)
        document_id = path.stem
        
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            # Check file size
            if path.stat().st_size > self.config.max_file_size:
                raise ValueError(f"File too large: {path.stat().st_size} bytes")
            
            # Parse document
            self.logger.info(f"Parsing document: {file_path}")
            parsed_document = self._parse_document(path)
            
            # Enrich metadata if enabled
            if self.config.enable_metadata_enrichment:
                self.logger.debug("Enriching metadata")
                parsed_document.metadata = self.metadata_enricher.enrich_metadata(
                    parsed_document.metadata,
                    parsed_document.text
                )
            
            # Chunk document if enabled
            chunks = []
            if self.config.enable_chunking and parsed_document.text.strip():
                self.logger.debug("Chunking document")
                chunks = self._chunk_document(parsed_document)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Collect warnings and errors
            warnings.extend(parsed_document.warnings)
            errors.extend(parsed_document.errors)
            
            # Create result
            result = DocumentProcessingResult(
                source_path=str(path),
                document_id=document_id,
                parsed_document=parsed_document,
                chunks=chunks,
                processing_time=processing_time,
                processing_status="completed" if not errors else "completed_with_errors",
                warnings=warnings,
                errors=errors
            )
            
            self.logger.info(
                f"Document processing completed: {len(parsed_document.text)} chars, "
                f"{len(chunks)} chunks, {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Document processing failed: {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            # Create minimal result with error
            try:
                metadata = self.metadata_extractor.extract_from_file(path)
            except Exception:
                from .metadata.models import DocumentType, ProcessingStatus
                metadata = DocumentMetadata(
                    document_id=document_id,
                    source_path=str(path),
                    filename=path.name,
                    document_type=DocumentType.UNKNOWN,
                    processing_status=ProcessingStatus.FAILED
                )
            
            parsed_document = ParsedDocument(
                text="",
                metadata=metadata,
                parsing_time=0.0,
                parser_name="unknown",
                parser_version="unknown",
                errors=[error_msg]
            )
            
            return DocumentProcessingResult(
                source_path=str(path),
                document_id=document_id,
                parsed_document=parsed_document,
                chunks=[],
                processing_time=processing_time,
                processing_status="failed",
                warnings=warnings,
                errors=errors
            )
    
    def process_content(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: Optional[str] = None,
        **metadata_kwargs
    ) -> DocumentProcessingResult:
        """
        Process document content directly.
        
        Args:
            content: Document content
            document_id: Document identifier
            document_type: Document type hint
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            DocumentProcessingResult: Processing result
        """
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            # Parse content
            self.logger.info(f"Parsing content for document: {document_id}")
            parsed_document = self._parse_content(content, document_id, document_type, **metadata_kwargs)
            
            # Enrich metadata if enabled
            if self.config.enable_metadata_enrichment:
                self.logger.debug("Enriching metadata")
                parsed_document.metadata = self.metadata_enricher.enrich_metadata(
                    parsed_document.metadata,
                    parsed_document.text
                )
            
            # Chunk document if enabled
            chunks = []
            if self.config.enable_chunking and parsed_document.text.strip():
                self.logger.debug("Chunking document")
                chunks = self._chunk_document(parsed_document)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Collect warnings and errors
            warnings.extend(parsed_document.warnings)
            errors.extend(parsed_document.errors)
            
            # Create result
            result = DocumentProcessingResult(
                source_path=None,
                document_id=document_id,
                parsed_document=parsed_document,
                chunks=chunks,
                processing_time=processing_time,
                processing_status="completed" if not errors else "completed_with_errors",
                warnings=warnings,
                errors=errors
            )
            
            self.logger.info(
                f"Content processing completed: {len(parsed_document.text)} chars, "
                f"{len(chunks)} chunks, {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Content processing failed: {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            # Create minimal result with error
            from .metadata.models import DocumentType, ProcessingStatus
            metadata = DocumentMetadata(
                document_id=document_id,
                document_type=DocumentType.UNKNOWN,
                processing_status=ProcessingStatus.FAILED,
                **metadata_kwargs
            )
            
            parsed_document = ParsedDocument(
                text="",
                metadata=metadata,
                parsing_time=0.0,
                parser_name="unknown",
                parser_version="unknown",
                errors=[error_msg]
            )
            
            return DocumentProcessingResult(
                source_path=None,
                document_id=document_id,
                parsed_document=parsed_document,
                chunks=[],
                processing_time=processing_time,
                processing_status="failed",
                warnings=warnings,
                errors=errors
            )
    
    def _parse_document(self, file_path: Path) -> ParsedDocument:
        """Parse document using appropriate parser."""
        try:
            parser = self.parser_factory.get_parser_for_file(file_path, self.config.parsing_config)
            return parser.parse_file(file_path)
        except Exception as e:
            if not self.config.ignore_parsing_errors:
                raise
            
            # Create minimal parsed document with error
            metadata = self.metadata_extractor.extract_from_file(file_path)
            return ParsedDocument(
                text="",
                metadata=metadata,
                parsing_time=0.0,
                parser_name="failed",
                parser_version="unknown",
                errors=[f"Parsing failed: {str(e)}"]
            )
    
    def _parse_content(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: Optional[str] = None,
        **metadata_kwargs
    ) -> ParsedDocument:
        """Parse content using appropriate parser."""
        try:
            # Determine document type
            from .metadata.models import DocumentType
            if document_type:
                doc_type = DocumentType(document_type.lower())
            else:
                doc_type = DocumentType.TEXT
            
            parser = self.parser_factory.get_parser(document_type=doc_type, config=self.config.parsing_config)
            return parser.parse_content(content, document_id, doc_type, **metadata_kwargs)
        except Exception as e:
            if not self.config.ignore_parsing_errors:
                raise
            
            # Create minimal parsed document with error
            text_content = str(content) if isinstance(content, str) else content.decode('utf-8', errors='ignore')
            metadata = self.metadata_extractor.extract_from_content(text_content, document_id, **metadata_kwargs)
            return ParsedDocument(
                text=text_content,
                metadata=metadata,
                parsing_time=0.0,
                parser_name="failed",
                parser_version="unknown",
                errors=[f"Parsing failed: {str(e)}"]
            )
    
    def _chunk_document(self, parsed_document: ParsedDocument) -> List[TextChunk]:
        """Chunk parsed document."""
        try:
            chunker = self.chunker_factory.get_chunker(
                strategy=self.config.chunking_config.strategy,
                config=self.config.chunking_config
            )
            return chunker.chunk_text(parsed_document.text, parsed_document.metadata)
        except Exception as e:
            if not self.config.ignore_chunking_errors:
                raise
            
            self.logger.warning(f"Chunking failed: {e}")
            return []
