"""
Document metadata models and schemas.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    RTF = "rtf"
    PPTX = "pptx"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    XML = "xml"
    # Generic types
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentMetadata(BaseModel):
    """Comprehensive document metadata model."""

    # Basic identification
    document_id: str = Field(..., description="Unique document identifier")
    source_path: Optional[str] = Field(None, description="Original file path")
    filename: Optional[str] = Field(None, description="Original filename")
    document_type: DocumentType = Field(..., description="Document type")

    # File properties
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File content hash (SHA256)")
    mime_type: Optional[str] = Field(None, description="MIME type")
    encoding: Optional[str] = Field(None, description="Text encoding")

    # Content properties
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    language: Optional[str] = Field(None, description="Detected language")

    # Text statistics
    character_count: Optional[int] = Field(None, description="Total character count")
    word_count: Optional[int] = Field(None, description="Total word count")
    sentence_count: Optional[int] = Field(None, description="Total sentence count")
    paragraph_count: Optional[int] = Field(None, description="Total paragraph count")
    page_count: Optional[int] = Field(None, description="Total page count")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Document creation time")
    modified_at: Optional[datetime] = Field(None, description="Document modification time")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")

    # Processing information
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Processing status"
    )
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

    # Content analysis
    readability_score: Optional[float] = Field(None, description="Readability score")
    sentiment_score: Optional[float] = Field(None, description="Sentiment analysis score")
    complexity_score: Optional[float] = Field(None, description="Text complexity score")

    # Custom metadata
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )

    # Tags and categories
    tags: List[str] = Field(default_factory=list, description="Document tags")
    categories: List[str] = Field(default_factory=list, description="Document categories")

    @validator('document_type', pre=True)
    def validate_document_type(cls, v):
        """Validate and normalize document type."""
        if isinstance(v, str):
            v = v.lower()
            # Try to match known types
            for doc_type in DocumentType:
                if v == doc_type.value or v.endswith(f'.{doc_type.value}'):
                    return doc_type
            return DocumentType.UNKNOWN
        return v

    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size is non-negative."""
        if v is not None and v < 0:
            raise ValueError("File size must be non-negative")
        return v

    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Validate processing time is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Processing time must be non-negative")
        return v

    def add_custom_field(self, key: str, value: Any) -> None:
        """Add a custom metadata field."""
        self.custom_fields[key] = value

    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get a custom metadata field."""
        return self.custom_fields.get(key, default)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the document."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_category(self, category: str) -> None:
        """Add a category to the document."""
        if category not in self.categories:
            self.categories.append(category)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return self.dict()

    @classmethod
    def from_file_path(cls, file_path: Union[str, Path]) -> "DocumentMetadata":
        """Create metadata from file path."""
        path = Path(file_path)

        # Determine document type from extension
        extension = path.suffix.lower().lstrip('.')
        doc_type = DocumentType.UNKNOWN
        for dt in DocumentType:
            if extension == dt.value:
                doc_type = dt
                break

        return cls(
            document_id=str(path.stem),
            source_path=str(path),
            filename=path.name,
            document_type=doc_type,
            file_size=path.stat().st_size if path.exists() else None,
            created_at=datetime.fromtimestamp(path.stat().st_ctime) if path.exists() else None,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None,
        )


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Chunk index in document")

    # Content properties
    text: str = Field(..., description="Chunk text content")
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    sentence_count: int = Field(..., description="Sentence count")

    # Position information
    start_char: Optional[int] = Field(None, description="Start character position in document")
    end_char: Optional[int] = Field(None, description="End character position in document")
    page_number: Optional[int] = Field(None, description="Page number (if applicable)")

    # Chunking strategy
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    chunk_size: int = Field(..., description="Target chunk size")
    overlap_size: int = Field(default=0, description="Overlap with adjacent chunks")

    # Semantic information
    semantic_similarity: Optional[float] = Field(None, description="Semantic similarity score")
    topic: Optional[str] = Field(None, description="Detected topic")

    # Custom metadata
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom chunk metadata"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Chunk creation time")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to dictionary."""
        return self.dict()
