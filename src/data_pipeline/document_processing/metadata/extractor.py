"""
Metadata extraction utilities for documents.
"""

import hashlib
import logging
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

from .models import DocumentMetadata, DocumentType, ProcessingStatus

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extract metadata from documents and files."""

    def __init__(self):
        """Initialize metadata extractor."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_from_file(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file

        Returns:
            DocumentMetadata: Extracted metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Start with basic metadata from file path
        metadata = DocumentMetadata.from_file_path(path)

        try:
            # Extract file properties
            self._extract_file_properties(path, metadata)

            # Extract MIME type
            self._extract_mime_type(path, metadata)

            # Extract file hash
            self._extract_file_hash(path, metadata)

            # If it's a text-based file, extract text properties
            if self._is_text_file(metadata.document_type):
                try:
                    content = self._read_file_content(path)
                    if content:
                        self._extract_text_properties(content, metadata)
                        self._extract_language(content, metadata)
                        self._extract_readability(content, metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to extract text properties: {e}")

            metadata.processing_status = ProcessingStatus.COMPLETED
            metadata.processed_at = datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {e}")
            metadata.processing_status = ProcessingStatus.FAILED
            metadata.error_message = str(e)

        return metadata

    def extract_from_content(
        self,
        content: str,
        document_id: str,
        document_type: DocumentType = DocumentType.TEXT,
        **kwargs
    ) -> DocumentMetadata:
        """
        Extract metadata from text content.

        Args:
            content: Text content
            document_id: Document identifier
            document_type: Type of document
            **kwargs: Additional metadata fields

        Returns:
            DocumentMetadata: Extracted metadata
        """
        metadata = DocumentMetadata(
            document_id=document_id,
            document_type=document_type,
            **kwargs
        )

        try:
            # Extract text properties
            self._extract_text_properties(content, metadata)

            # Extract language
            self._extract_language(content, metadata)

            # Extract readability
            self._extract_readability(content, metadata)

            metadata.processing_status = ProcessingStatus.COMPLETED
            metadata.processed_at = datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from content: {e}")
            metadata.processing_status = ProcessingStatus.FAILED
            metadata.error_message = str(e)

        return metadata

    def _extract_file_properties(self, path: Path, metadata: DocumentMetadata) -> None:
        """Extract basic file properties."""
        stat = path.stat()

        metadata.file_size = stat.st_size
        metadata.created_at = datetime.fromtimestamp(stat.st_ctime)
        metadata.modified_at = datetime.fromtimestamp(stat.st_mtime)

    def _extract_mime_type(self, path: Path, metadata: DocumentMetadata) -> None:
        """Extract MIME type."""
        # Try python-magic first if available
        if HAS_MAGIC:
            try:
                mime = magic.Magic(mime=True)
                metadata.mime_type = mime.from_file(str(path))
                return
            except Exception as e:
                self.logger.debug(f"Failed to detect MIME type with magic: {e}")

        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))
        metadata.mime_type = mime_type

    def _extract_file_hash(self, path: Path, metadata: DocumentMetadata) -> None:
        """Extract file content hash."""
        try:
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            metadata.file_hash = hasher.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute file hash: {e}")

    def _is_text_file(self, document_type: DocumentType) -> bool:
        """Check if document type is text-based."""
        text_types = {
            DocumentType.TEXT,
            DocumentType.MARKDOWN,
            DocumentType.HTML,
            DocumentType.CSV,
            DocumentType.JSON,
            DocumentType.XML
        }
        return document_type in text_types

    def _read_file_content(self, path: Path) -> Optional[str]:
        """Read file content as text."""
        try:
            # Detect encoding if chardet is available
            encoding = 'utf-8'
            if HAS_CHARDET:
                with open(path, 'rb') as f:
                    raw_data = f.read(10000)  # Read first 10KB for detection
                    result = chardet.detect(raw_data)
                    if result['encoding']:
                        encoding = result['encoding']

            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Failed to read file content: {e}")
            return None

    def _extract_text_properties(self, content: str, metadata: DocumentMetadata) -> None:
        """Extract text statistics."""
        metadata.character_count = len(content)
        metadata.word_count = len(content.split())

        # Count sentences (simple approach)
        sentences = re.split(r'[.!?]+', content)
        metadata.sentence_count = len([s for s in sentences if s.strip()])

        # Count paragraphs
        paragraphs = content.split('\n\n')
        metadata.paragraph_count = len([p for p in paragraphs if p.strip()])

    def _extract_language(self, content: str, metadata: DocumentMetadata) -> None:
        """Detect document language."""
        if not HAS_LANGDETECT:
            return

        try:
            # Use first 1000 characters for language detection
            sample = content[:1000]
            if len(sample.strip()) > 10:
                language = langdetect.detect(sample)
                metadata.language = language
        except Exception as e:
            self.logger.debug(f"Failed to detect language: {e}")

    def _extract_readability(self, content: str, metadata: DocumentMetadata) -> None:
        """Calculate readability score."""
        if not HAS_TEXTSTAT:
            return

        try:
            # Use Flesch Reading Ease score
            score = textstat.flesch_reading_ease(content)
            metadata.readability_score = score
        except Exception as e:
            self.logger.debug(f"Failed to calculate readability: {e}")

    def update_processing_time(self, metadata: DocumentMetadata, start_time: datetime) -> None:
        """Update processing time in metadata."""
        if metadata.processed_at:
            processing_time = (metadata.processed_at - start_time).total_seconds()
            metadata.processing_time = processing_time
