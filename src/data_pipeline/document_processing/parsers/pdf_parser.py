"""
PDF file parser implementation.
"""

import logging
from pathlib import Path
from typing import List, Union, Dict, Any

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from .base_parser import BaseParser, ParsedDocument
from ..metadata.models import DocumentMetadata, DocumentType
from ..metadata.extractor import MetadataExtractor

class PDFParser(BaseParser):
    """Parser for PDF files."""

    def __init__(self, *args, **kwargs):
        """Initialize PDF parser."""
        super().__init__(*args, **kwargs)

        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            raise ImportError(
                "PDF parsing requires either PyPDF2 or pdfplumber. "
                "Install with: pip install PyPDF2 pdfplumber"
            )

    @property
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        return [DocumentType.PDF]

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['pdf']

    def _parse_file_impl(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument: Parsed document result
        """
        # Extract metadata first
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_file(file_path)
        metadata.document_type = DocumentType.PDF

        # Try pdfplumber first (better text extraction)
        if HAS_PDFPLUMBER:
            return self._parse_with_pdfplumber(file_path, metadata)
        elif HAS_PYPDF2:
            return self._parse_with_pypdf2(file_path, metadata)
        else:
            raise ImportError("No PDF parsing library available")

    def _parse_content_impl(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs
    ) -> ParsedDocument:
        """
        Parse PDF content directly.

        Args:
            content: PDF content as bytes
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields

        Returns:
            ParsedDocument: Parsed document result
        """
        if isinstance(content, str):
            raise ValueError("PDF content must be provided as bytes")

        # Create temporary file for parsing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        try:
            # Parse the temporary file
            result = self._parse_file_impl(tmp_path)

            # Update metadata with provided information
            result.metadata.document_id = document_id
            for key, value in metadata_kwargs.items():
                if hasattr(result.metadata, key):
                    setattr(result.metadata, key, value)
                else:
                    result.metadata.add_custom_field(key, value)

            return result
        finally:
            # Clean up temporary file
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _parse_with_pdfplumber(self, file_path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        """Parse PDF using pdfplumber."""
        import pdfplumber

        text_content = []
        tables = []
        images = []
        warnings = []
        errors = []

        try:
            with pdfplumber.open(file_path) as pdf:
                # Update page count
                metadata.page_count = len(pdf.pages)

                # Extract PDF metadata
                if pdf.metadata:
                    self._extract_pdf_metadata(pdf.metadata, metadata)

                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)

                        # Extract tables if configured
                        if self.config.extract_tables:
                            page_tables = page.extract_tables()
                            for table_idx, table in enumerate(page_tables):
                                tables.append({
                                    'page': page_num,
                                    'table_index': table_idx,
                                    'data': table,
                                    'rows': len(table),
                                    'columns': len(table[0]) if table else 0
                                })

                        # Extract images if configured
                        if self.config.extract_images:
                            # pdfplumber doesn't directly extract images,
                            # but we can get image information
                            if hasattr(page, 'images'):
                                for img_idx, img in enumerate(page.images):
                                    images.append({
                                        'page': page_num,
                                        'image_index': img_idx,
                                        'bbox': img.get('bbox'),
                                        'width': img.get('width'),
                                        'height': img.get('height')
                                    })

                    except Exception as e:
                        error_msg = f"Error processing page {page_num}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)

        except Exception as e:
            error_msg = f"Error opening PDF file: {str(e)}"
            errors.append(error_msg)
            if not self.config.ignore_errors:
                raise

        # Combine all text
        full_text = '\n\n'.join(text_content)

        # Normalize text if configured
        if self.config.normalize_whitespace:
            full_text = self._normalize_text(full_text)

        # Extract links
        links = self._extract_links(full_text)

        # Update text statistics in metadata
        if full_text:
            metadata.character_count = len(full_text)
            metadata.word_count = len(full_text.split())

        # Create result
        result = ParsedDocument(
            text=full_text,
            metadata=metadata,
            tables=tables,
            images=images,
            links=links,
            warnings=warnings,
            errors=errors,
            parsing_time=0.0,
            parser_name=self._parser_name,
            parser_version=self._parser_version
        )

        return result

    def _parse_with_pypdf2(self, file_path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        """Parse PDF using PyPDF2."""
        import PyPDF2

        text_content = []
        warnings = []
        errors = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Update page count
                metadata.page_count = len(pdf_reader.pages)

                # Extract PDF metadata
                if pdf_reader.metadata:
                    self._extract_pdf_metadata(pdf_reader.metadata, metadata)

                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    except Exception as e:
                        error_msg = f"Error extracting text from page {page_num}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)

        except Exception as e:
            error_msg = f"Error reading PDF file: {str(e)}"
            errors.append(error_msg)
            if not self.config.ignore_errors:
                raise

        # Combine all text
        full_text = '\n\n'.join(text_content)

        # Normalize text if configured
        if self.config.normalize_whitespace:
            full_text = self._normalize_text(full_text)

        # Extract links
        links = self._extract_links(full_text)

        # Update text statistics in metadata
        if full_text:
            metadata.character_count = len(full_text)
            metadata.word_count = len(full_text.split())

        # Create result
        result = ParsedDocument(
            text=full_text,
            metadata=metadata,
            links=links,
            warnings=warnings,
            errors=errors,
            parsing_time=0.0,
            parser_name=self._parser_name,
            parser_version=self._parser_version
        )

        return result

    def _extract_pdf_metadata(self, pdf_metadata: Dict[str, Any], metadata: DocumentMetadata) -> None:
        """Extract metadata from PDF metadata dictionary."""
        # Map PDF metadata fields to our metadata model
        field_mapping = {
            '/Title': 'title',
            '/Author': 'author',
            '/Subject': 'subject',
            '/Creator': 'creator',
            '/Producer': 'producer',
            '/CreationDate': 'creation_date',
            '/ModDate': 'modification_date',
            '/Keywords': 'keywords'
        }

        for pdf_field, our_field in field_mapping.items():
            if pdf_field in pdf_metadata:
                value = pdf_metadata[pdf_field]

                # Handle special cases
                if our_field == 'keywords' and isinstance(value, str):
                    # Split keywords by common separators
                    keywords = [k.strip() for k in value.replace(',', ';').split(';') if k.strip()]
                    metadata.keywords = keywords
                elif our_field in ['title', 'author', 'subject']:
                    setattr(metadata, our_field, str(value))
                elif our_field in ['creation_date', 'modification_date']:
                    # PDF dates are in a special format, store as custom field for now
                    metadata.add_custom_field(our_field, str(value))
                else:
                    metadata.add_custom_field(our_field, str(value))
