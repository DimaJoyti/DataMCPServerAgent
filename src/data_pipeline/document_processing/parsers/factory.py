"""
Parser factory for creating appropriate parsers based on document type.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from ..metadata.models import DocumentType
from .base_parser import BaseParser, ParsingConfig
from .docx_parser import DOCXParser
from .html_parser import HTMLParser
from .markdown_parser import MarkdownParser
from .pdf_parser import PDFParser
from .text_parser import TextParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """Factory for creating document parsers."""

    def __init__(self):
        """Initialize parser factory."""
        self._parsers: Dict[DocumentType, Type[BaseParser]] = {}
        self._extension_map: Dict[str, DocumentType] = {}
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register default parsers."""
        # Register parsers
        self.register_parser(TextParser)

        # Register PDF parser if available
        try:
            self.register_parser(PDFParser)
        except ImportError:
            logger.warning("PDF parser not available - missing dependencies")

        # Register DOCX parser if available
        try:
            self.register_parser(DOCXParser)
        except ImportError:
            logger.warning("DOCX parser not available - missing dependencies")

        # Register HTML parser if available
        try:
            self.register_parser(HTMLParser)
        except ImportError:
            logger.warning("HTML parser not available - missing dependencies")

        # Register Markdown parser
        self.register_parser(MarkdownParser)

        # Register additional parsers (with error handling for missing dependencies)
        try:
            from .excel_parser import ExcelParser
            self.register_parser(ExcelParser)
        except ImportError as e:
            logger.warning(f"Excel parser not available: {e}")

        try:
            from .powerpoint_parser import PowerPointParser
            self.register_parser(PowerPointParser)
        except ImportError as e:
            logger.warning(f"PowerPoint parser not available: {e}")

        try:
            from .csv_parser import CSVParser
            self.register_parser(CSVParser)
        except ImportError as e:
            logger.warning(f"CSV parser not available: {e}")

    def register_parser(self, parser_class: Type[BaseParser]) -> None:
        """
        Register a parser class.

        Args:
            parser_class: Parser class to register
        """
        # Create temporary instance to get supported types and extensions
        try:
            temp_parser = parser_class()

            # Register for each supported document type
            for doc_type in temp_parser.supported_types:
                self._parsers[doc_type] = parser_class
                logger.debug(f"Registered {parser_class.__name__} for {doc_type}")

            # Register extensions
            for ext in temp_parser.supported_extensions:
                # Map extension to the first supported document type
                if temp_parser.supported_types:
                    self._extension_map[ext.lower()] = temp_parser.supported_types[0]

        except Exception as e:
            logger.warning(f"Failed to register parser {parser_class.__name__}: {e}")

    def get_parser(
        self,
        document_type: Optional[DocumentType] = None,
        file_path: Optional[Union[str, Path]] = None,
        config: Optional[ParsingConfig] = None
    ) -> BaseParser:
        """
        Get appropriate parser for document type or file.

        Args:
            document_type: Document type (if known)
            file_path: File path (for type detection)
            config: Parsing configuration

        Returns:
            BaseParser: Appropriate parser instance

        Raises:
            ValueError: If no suitable parser found
        """
        # Determine document type if not provided
        if document_type is None and file_path is not None:
            document_type = self._detect_document_type(file_path)

        if document_type is None:
            raise ValueError("Cannot determine document type")

        # Get parser class
        parser_class = self._parsers.get(document_type)
        if parser_class is None:
            # Try to find a fallback parser
            parser_class = self._find_fallback_parser(document_type)

        if parser_class is None:
            raise ValueError(f"No parser available for document type: {document_type}")

        # Create parser instance
        try:
            return parser_class(config)
        except Exception as e:
            logger.error(f"Failed to create parser {parser_class.__name__}: {e}")
            raise

    def get_parser_for_file(
        self,
        file_path: Union[str, Path],
        config: Optional[ParsingConfig] = None
    ) -> BaseParser:
        """
        Get appropriate parser for a file.

        Args:
            file_path: Path to file
            config: Parsing configuration

        Returns:
            BaseParser: Appropriate parser instance
        """
        return self.get_parser(file_path=file_path, config=config)

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if any parser can handle the file.

        Args:
            file_path: Path to file

        Returns:
            bool: True if file can be parsed
        """
        try:
            self.get_parser_for_file(file_path)
            return True
        except ValueError:
            return False

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.

        Returns:
            List[str]: Supported extensions
        """
        return list(self._extension_map.keys())

    def get_supported_types(self) -> List[DocumentType]:
        """
        Get list of all supported document types.

        Returns:
            List[DocumentType]: Supported document types
        """
        return list(self._parsers.keys())

    def _detect_document_type(self, file_path: Union[str, Path]) -> Optional[DocumentType]:
        """
        Detect document type from file path.

        Args:
            file_path: Path to file

        Returns:
            Optional[DocumentType]: Detected document type
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')

        # Check extension mapping
        doc_type = self._extension_map.get(extension)
        if doc_type:
            return doc_type

        # Try to detect from MIME type if file exists
        if path.exists():
            try:
                import mimetypes
                mime_type, _ = mimetypes.guess_type(str(path))

                if mime_type:
                    return self._mime_to_document_type(mime_type)
            except Exception:
                pass

        return None

    def _mime_to_document_type(self, mime_type: str) -> Optional[DocumentType]:
        """
        Convert MIME type to document type.

        Args:
            mime_type: MIME type string

        Returns:
            Optional[DocumentType]: Corresponding document type
        """
        mime_mapping = {
            'application/pdf': DocumentType.PDF,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
            'text/html': DocumentType.HTML,
            'text/markdown': DocumentType.MARKDOWN,
            'text/plain': DocumentType.TEXT,
            'application/json': DocumentType.JSON,
            'application/xml': DocumentType.XML,
            'text/xml': DocumentType.XML,
            'text/csv': DocumentType.CSV,
        }

        return mime_mapping.get(mime_type)

    def _find_fallback_parser(self, document_type: DocumentType) -> Optional[Type[BaseParser]]:
        """
        Find fallback parser for unsupported document type.

        Args:
            document_type: Document type

        Returns:
            Optional[Type[BaseParser]]: Fallback parser class
        """
        # For text-based formats, try text parser
        text_based_types = {
            DocumentType.TEXT,
            DocumentType.CSV,
            DocumentType.JSON,
            DocumentType.XML,
            DocumentType.MARKDOWN
        }

        if document_type in text_based_types:
            return self._parsers.get(DocumentType.TEXT)

        return None

    def list_parsers(self) -> Dict[DocumentType, str]:
        """
        List all registered parsers.

        Returns:
            Dict[DocumentType, str]: Mapping of document types to parser names
        """
        return {
            doc_type: parser_class.__name__
            for doc_type, parser_class in self._parsers.items()
        }


# Global parser factory instance
parser_factory = ParserFactory()
