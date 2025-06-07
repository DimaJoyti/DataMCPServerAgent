"""
Base parser interface for document processing.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..metadata.models import DocumentMetadata, DocumentType

class ParsingConfig(BaseModel):
    """Configuration for document parsing."""

    # Text extraction options
    extract_text: bool = Field(default=True, description="Extract text content")
    extract_metadata: bool = Field(default=True, description="Extract document metadata")
    extract_images: bool = Field(default=False, description="Extract images")
    extract_tables: bool = Field(default=False, description="Extract tables")

    # Text processing options
    preserve_formatting: bool = Field(default=False, description="Preserve text formatting")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace")
    remove_headers_footers: bool = Field(default=False, description="Remove headers and footers")

    # Language and encoding
    encoding: Optional[str] = Field(default=None, description="Text encoding (auto-detect if None)")
    language: Optional[str] = Field(default=None, description="Document language")

    # Error handling
    ignore_errors: bool = Field(default=False, description="Continue parsing on errors")
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes")

    # Custom options
    custom_options: Dict[str, Any] = Field(default_factory=dict, description="Parser-specific options")

class ParsedDocument(BaseModel):
    """Result of document parsing."""

    # Content
    text: str = Field(..., description="Extracted text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

    # Optional extracted elements
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted images")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    links: List[Dict[str, str]] = Field(default_factory=list, description="Extracted links")

    # Parsing information
    parsing_time: float = Field(..., description="Time taken to parse (seconds)")
    parser_name: str = Field(..., description="Name of parser used")
    parser_version: str = Field(..., description="Version of parser used")

    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Parsing warnings")
    errors: List[str] = Field(default_factory=list, description="Parsing errors")

    # Raw data (optional)
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Raw parser output")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def has_errors(self) -> bool:
        """Check if parsing had errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if parsing had warnings."""
        return len(self.warnings) > 0

class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, config: Optional[ParsingConfig] = None):
        """
        Initialize parser.

        Args:
            config: Parsing configuration
        """
        self.config = config or ParsingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._parser_name = self.__class__.__name__
        self._parser_version = "1.0.0"

    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if parser can handle the given file.

        Args:
            file_path: Path to file

        Returns:
            bool: True if parser can handle the file
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        return extension in self.supported_extensions

    def parse_file(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a document file.

        Args:
            file_path: Path to file to parse

        Returns:
            ParsedDocument: Parsed document result

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported
            Exception: If parsing fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_parse(path):
            raise ValueError(f"File type not supported: {path.suffix}")

        # Check file size
        if path.stat().st_size > self.config.max_file_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes")

        start_time = datetime.now()

        try:
            # Parse the document
            result = self._parse_file_impl(path)

            # Calculate parsing time
            end_time = datetime.now()
            result.parsing_time = (end_time - start_time).total_seconds()
            result.parser_name = self._parser_name
            result.parser_version = self._parser_version

            return result

        except Exception as e:
            if not self.config.ignore_errors:
                raise

            # Return empty result with error
            end_time = datetime.now()
            parsing_time = (end_time - start_time).total_seconds()

            from ..metadata.extractor import MetadataExtractor
            extractor = MetadataExtractor()
            metadata = extractor.extract_from_file(path)

            result = ParsedDocument(
                text="",
                metadata=metadata,
                parsing_time=parsing_time,
                parser_name=self._parser_name,
                parser_version=self._parser_version
            )
            result.add_error(f"Parsing failed: {str(e)}")

            return result

    def parse_content(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs
    ) -> ParsedDocument:
        """
        Parse document content directly.

        Args:
            content: Document content
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields

        Returns:
            ParsedDocument: Parsed document result
        """
        start_time = datetime.now()

        try:
            result = self._parse_content_impl(content, document_id, document_type, **metadata_kwargs)

            # Calculate parsing time
            end_time = datetime.now()
            result.parsing_time = (end_time - start_time).total_seconds()
            result.parser_name = self._parser_name
            result.parser_version = self._parser_version

            return result

        except Exception as e:
            if not self.config.ignore_errors:
                raise

            # Return empty result with error
            end_time = datetime.now()
            parsing_time = (end_time - start_time).total_seconds()

            from ..metadata.extractor import MetadataExtractor
            extractor = MetadataExtractor()
            metadata = extractor.extract_from_content(
                str(content) if isinstance(content, str) else content.decode('utf-8', errors='ignore'),
                document_id,
                document_type,
                **metadata_kwargs
            )

            result = ParsedDocument(
                text="",
                metadata=metadata,
                parsing_time=parsing_time,
                parser_name=self._parser_name,
                parser_version=self._parser_version
            )
            result.add_error(f"Parsing failed: {str(e)}")

            return result

    @abstractmethod
    def _parse_file_impl(self, file_path: Path) -> ParsedDocument:
        """
        Implementation-specific file parsing logic.

        Args:
            file_path: Path to file to parse

        Returns:
            ParsedDocument: Parsed document result
        """
        pass

    @abstractmethod
    def _parse_content_impl(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs
    ) -> ParsedDocument:
        """
        Implementation-specific content parsing logic.

        Args:
            content: Document content
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields

        Returns:
            ParsedDocument: Parsed document result
        """
        pass

    def _normalize_text(self, text: str) -> str:
        """Normalize text content based on configuration."""
        if not text:
            return ""

        if self.config.normalize_whitespace:
            # Normalize whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        return text

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from content."""
        import re
        links = []

        # Extract markdown links
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, url in markdown_links:
            links.append({"text": text, "url": url, "type": "markdown"})

        # Extract HTML links
        html_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', content, re.IGNORECASE)
        for url, text in html_links:
            links.append({"text": text, "url": url, "type": "html"})

        # Extract plain URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        plain_urls = re.findall(url_pattern, content)
        for url in plain_urls:
            links.append({"text": url, "url": url, "type": "plain"})

        return links
