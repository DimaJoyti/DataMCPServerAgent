"""
Text file parser implementation.
"""

import logging
from pathlib import Path
from typing import List, Union

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

from .base_parser import BaseParser, ParsedDocument
from ..metadata.models import DocumentMetadata, DocumentType
from ..metadata.extractor import MetadataExtractor

class TextParser(BaseParser):
    """Parser for plain text files."""

    @property
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        return [DocumentType.TEXT]

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['txt', 'text', 'log', 'csv', 'tsv', 'json', 'xml', 'yaml', 'yml']

    def _parse_file_impl(self, file_path: Path) -> ParsedDocument:
        """
        Parse a text file.

        Args:
            file_path: Path to text file

        Returns:
            ParsedDocument: Parsed document result
        """
        # Detect encoding
        encoding = self.config.encoding or self._detect_encoding(file_path)

        try:
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self.logger.warning(f"Encoding detection failed for {file_path}, using UTF-8 with error handling")

        # Extract metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_file(file_path)
        metadata.encoding = encoding

        # Determine document type based on extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension == 'csv':
            metadata.document_type = DocumentType.CSV
        elif extension in ['json']:
            metadata.document_type = DocumentType.JSON
        elif extension in ['xml']:
            metadata.document_type = DocumentType.XML
        else:
            metadata.document_type = DocumentType.TEXT

        # Normalize text if configured
        if self.config.normalize_whitespace:
            content = self._normalize_text(content)

        # Extract links
        links = self._extract_links(content)

        # Create parsed document
        result = ParsedDocument(
            text=content,
            metadata=metadata,
            links=links,
            parsing_time=0.0,  # Will be set by base class
            parser_name=self._parser_name,
            parser_version=self._parser_version
        )

        # Add format-specific analysis
        self._analyze_text_format(content, result)

        return result

    def _parse_content_impl(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs
    ) -> ParsedDocument:
        """
        Parse text content directly.

        Args:
            content: Text content
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields

        Returns:
            ParsedDocument: Parsed document result
        """
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = content.decode('utf-8', errors='ignore')

        # Extract metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_content(
            content,
            document_id,
            document_type,
            **metadata_kwargs
        )

        # Normalize text if configured
        if self.config.normalize_whitespace:
            content = self._normalize_text(content)

        # Extract links
        links = self._extract_links(content)

        # Create parsed document
        result = ParsedDocument(
            text=content,
            metadata=metadata,
            links=links,
            parsing_time=0.0,  # Will be set by base class
            parser_name=self._parser_name,
            parser_version=self._parser_version
        )

        # Add format-specific analysis
        self._analyze_text_format(content, result)

        return result

    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.

        Args:
            file_path: Path to file

        Returns:
            str: Detected encoding
        """
        if not HAS_CHARDET:
            return 'utf-8'

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)

                # Use detected encoding if confidence is high enough
                if confidence > 0.7:
                    return encoding
                else:
                    self.logger.warning(
                        f"Low confidence ({confidence:.2f}) in encoding detection for {file_path}, "
                        f"using UTF-8"
                    )
                    return 'utf-8'
        except Exception as e:
            self.logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return 'utf-8'

    def _analyze_text_format(self, content: str, result: ParsedDocument) -> None:
        """
        Analyze text format and add specific metadata.

        Args:
            content: Text content
            result: Parsed document result to update
        """
        # Detect if content looks like CSV
        if self._looks_like_csv(content):
            result.metadata.add_custom_field('format_detected', 'csv')
            self._analyze_csv_structure(content, result)

        # Detect if content looks like JSON
        elif self._looks_like_json(content):
            result.metadata.add_custom_field('format_detected', 'json')
            self._analyze_json_structure(content, result)

        # Detect if content looks like XML
        elif self._looks_like_xml(content):
            result.metadata.add_custom_field('format_detected', 'xml')
            self._analyze_xml_structure(content, result)

        # Detect if content looks like log file
        elif self._looks_like_log(content):
            result.metadata.add_custom_field('format_detected', 'log')
            self._analyze_log_structure(content, result)

    def _looks_like_csv(self, content: str) -> bool:
        """Check if content looks like CSV."""
        lines = content.split('\n')[:10]  # Check first 10 lines
        if len(lines) < 2:
            return False

        # Check for consistent comma/tab separation
        separators = [',', '\t', ';']
        for sep in separators:
            first_line_count = lines[0].count(sep)
            if first_line_count > 0:
                # Check if other lines have similar separator count
                consistent = sum(1 for line in lines[1:5] if abs(line.count(sep) - first_line_count) <= 1)
                if consistent >= 3:
                    return True
        return False

    def _looks_like_json(self, content: str) -> bool:
        """Check if content looks like JSON."""
        content = content.strip()
        return (content.startswith('{') and content.endswith('}')) or \
               (content.startswith('[') and content.endswith(']'))

    def _looks_like_xml(self, content: str) -> bool:
        """Check if content looks like XML."""
        content = content.strip()
        return content.startswith('<?xml') or \
               (content.startswith('<') and content.endswith('>'))

    def _looks_like_log(self, content: str) -> bool:
        """Check if content looks like a log file."""
        lines = content.split('\n')[:10]

        # Look for timestamp patterns
        import re
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\[\d{4}-\d{2}-\d{2}',  # [YYYY-MM-DD
        ]

        timestamp_lines = 0
        for line in lines:
            for pattern in timestamp_patterns:
                if re.search(pattern, line):
                    timestamp_lines += 1
                    break

        return timestamp_lines >= len(lines) * 0.5  # At least 50% of lines have timestamps

    def _analyze_csv_structure(self, content: str, result: ParsedDocument) -> None:
        """Analyze CSV structure."""
        lines = content.split('\n')
        if not lines:
            return

        # Detect separator
        separators = [',', '\t', ';']
        separator = ','
        max_count = 0

        for sep in separators:
            count = lines[0].count(sep)
            if count > max_count:
                max_count = count
                separator = sep

        result.metadata.add_custom_field('csv_separator', separator)
        result.metadata.add_custom_field('csv_columns', max_count + 1)
        result.metadata.add_custom_field('csv_rows', len([line for line in lines if line.strip()]))

    def _analyze_json_structure(self, content: str, result: ParsedDocument) -> None:
        """Analyze JSON structure."""
        try:
            import json
            data = json.loads(content)

            if isinstance(data, dict):
                result.metadata.add_custom_field('json_type', 'object')
                result.metadata.add_custom_field('json_keys', len(data.keys()))
            elif isinstance(data, list):
                result.metadata.add_custom_field('json_type', 'array')
                result.metadata.add_custom_field('json_items', len(data))

        except json.JSONDecodeError:
            result.add_warning("Content appears to be JSON but is not valid")

    def _analyze_xml_structure(self, content: str, result: ParsedDocument) -> None:
        """Analyze XML structure."""
        import re

        # Count XML elements
        elements = re.findall(r'<([^/\s>]+)', content)
        unique_elements = set(elements)

        result.metadata.add_custom_field('xml_elements', len(elements))
        result.metadata.add_custom_field('xml_unique_elements', len(unique_elements))

    def _analyze_log_structure(self, content: str, result: ParsedDocument) -> None:
        """Analyze log file structure."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        # Analyze log levels
        import re
        log_levels = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE', 'FATAL']
        level_counts = {}

        for line in non_empty_lines:
            for level in log_levels:
                if level in line.upper():
                    level_counts[level] = level_counts.get(level, 0) + 1

        if level_counts:
            result.metadata.add_custom_field('log_levels', level_counts)

        result.metadata.add_custom_field('log_lines', len(non_empty_lines))
