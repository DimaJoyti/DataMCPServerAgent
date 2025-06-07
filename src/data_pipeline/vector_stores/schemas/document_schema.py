"""
Document-specific vector store schema.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_schema import BaseVectorSchema, VectorRecord, VectorStoreConfig
from ...document_processing.metadata.models import DocumentMetadata, ChunkMetadata

class DocumentVectorRecord(VectorRecord):
    """Vector record specifically for document chunks."""

    # Document information
    document_id: str
    document_title: Optional[str] = None
    document_type: Optional[str] = None

    # Chunk information
    chunk_id: str
    chunk_index: int
    chunk_size: int

    # Position information
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_number: Optional[int] = None

    # Content analysis
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    language: Optional[str] = None

    # Processing information
    embedding_model: Optional[str] = None
    processing_time: Optional[float] = None

    @classmethod
    def from_chunk_and_embedding(
        cls,
        chunk_metadata: ChunkMetadata,
        document_metadata: DocumentMetadata,
        vector: List[float],
        embedding_model: str,
        processing_time: float = 0.0
    ) -> "DocumentVectorRecord":
        """
        Create document vector record from chunk metadata and embedding.

        Args:
            chunk_metadata: Chunk metadata
            document_metadata: Document metadata
            vector: Embedding vector
            embedding_model: Model used for embedding
            processing_time: Time taken for processing

        Returns:
            DocumentVectorRecord: Created record
        """
        return cls(
            id=chunk_metadata.chunk_id,
            vector=vector,
            text=chunk_metadata.text,
            document_id=document_metadata.document_id,
            document_title=document_metadata.title,
            document_type=document_metadata.document_type.value if document_metadata.document_type else None,
            chunk_id=chunk_metadata.chunk_id,
            chunk_index=chunk_metadata.chunk_index,
            chunk_size=chunk_metadata.character_count,
            start_char=chunk_metadata.start_char,
            end_char=chunk_metadata.end_char,
            page_number=chunk_metadata.page_number,
            word_count=chunk_metadata.word_count,
            sentence_count=chunk_metadata.sentence_count,
            language=document_metadata.language,
            embedding_model=embedding_model,
            processing_time=processing_time,
            source=document_metadata.source_path,
            source_type="document",
            metadata={
                "document_metadata": document_metadata.dict(),
                "chunk_metadata": chunk_metadata.dict()
            }
        )

class DocumentVectorSchema(BaseVectorSchema):
    """Schema for document-based vector storage."""

    def create_record(
        self,
        chunk_metadata: ChunkMetadata,
        document_metadata: DocumentMetadata,
        vector: List[float],
        embedding_model: str,
        processing_time: float = 0.0,
        **kwargs
    ) -> DocumentVectorRecord:
        """
        Create a document vector record.

        Args:
            chunk_metadata: Chunk metadata
            document_metadata: Document metadata
            vector: Embedding vector
            embedding_model: Model used for embedding
            processing_time: Processing time
            **kwargs: Additional fields

        Returns:
            DocumentVectorRecord: Created record
        """
        record = DocumentVectorRecord.from_chunk_and_embedding(
            chunk_metadata=chunk_metadata,
            document_metadata=document_metadata,
            vector=vector,
            embedding_model=embedding_model,
            processing_time=processing_time
        )

        # Add any additional fields
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
            else:
                record.add_metadata(key, value)

        return record

    def validate_record(self, record: VectorRecord) -> bool:
        """
        Validate a document vector record.

        Args:
            record: Record to validate

        Returns:
            bool: True if valid
        """
        # Check base validation
        if not isinstance(record, (VectorRecord, DocumentVectorRecord)):
            return False

        # Check vector dimension
        if len(record.vector) != self.config.embedding_dimension:
            self.logger.error(
                f"Vector dimension mismatch: {len(record.vector)} != {self.config.embedding_dimension}"
            )
            return False

        # Check required fields for document records
        if isinstance(record, DocumentVectorRecord):
            required_attrs = ["document_id", "chunk_id", "chunk_index"]
            for attr in required_attrs:
                if not hasattr(record, attr) or getattr(record, attr) is None:
                    self.logger.error(f"Missing required attribute: {attr}")
                    return False

        # Check text content
        if not record.text or not record.text.strip():
            self.logger.error("Text content cannot be empty")
            return False

        return True

    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields for document schema.

        Returns:
            List[str]: Required field names
        """
        return [
            "id",
            "vector",
            "text",
            "document_id",
            "chunk_id",
            "chunk_index"
        ]

    def get_searchable_fields(self) -> List[str]:
        """
        Get list of searchable metadata fields for documents.

        Returns:
            List[str]: Searchable field names
        """
        return [
            "document_id",
            "document_title",
            "document_type",
            "chunk_index",
            "page_number",
            "language",
            "embedding_model",
            "source",
            "source_type",
            "word_count",
            "sentence_count"
        ]

    def prepare_for_storage(self, record: DocumentVectorRecord) -> Dict[str, Any]:
        """
        Prepare document record for storage.

        Args:
            record: Record to prepare

        Returns:
            Dict[str, Any]: Prepared data
        """
        # Get base storage data
        storage_data = super().prepare_for_storage(record)

        # Add document-specific fields to metadata
        if isinstance(record, DocumentVectorRecord):
            document_fields = {
                "document_id": record.document_id,
                "document_title": record.document_title,
                "document_type": record.document_type,
                "chunk_id": record.chunk_id,
                "chunk_index": record.chunk_index,
                "chunk_size": record.chunk_size,
                "start_char": record.start_char,
                "end_char": record.end_char,
                "page_number": record.page_number,
                "word_count": record.word_count,
                "sentence_count": record.sentence_count,
                "language": record.language,
                "embedding_model": record.embedding_model,
                "processing_time": record.processing_time
            }

            # Add non-null fields to metadata
            for key, value in document_fields.items():
                if value is not None:
                    storage_data["metadata"][key] = value

        return storage_data

    def restore_from_storage(self, storage_data: Dict[str, Any]) -> DocumentVectorRecord:
        """
        Restore document record from storage format.

        Args:
            storage_data: Data from storage

        Returns:
            DocumentVectorRecord: Restored record
        """
        # Get base record
        base_record = super().restore_from_storage(storage_data)

        # Extract document-specific fields from metadata
        metadata = storage_data.get("metadata", {})

        # Create document vector record
        doc_record = DocumentVectorRecord(
            id=base_record.id,
            vector=base_record.vector,
            text=base_record.text,
            metadata=base_record.metadata,
            created_at=base_record.created_at,
            updated_at=base_record.updated_at,
            source=base_record.source,
            source_type=base_record.source_type,
            document_id=metadata.get("document_id", ""),
            document_title=metadata.get("document_title"),
            document_type=metadata.get("document_type"),
            chunk_id=metadata.get("chunk_id", base_record.id),
            chunk_index=metadata.get("chunk_index", 0),
            chunk_size=metadata.get("chunk_size", len(base_record.text)),
            start_char=metadata.get("start_char"),
            end_char=metadata.get("end_char"),
            page_number=metadata.get("page_number"),
            word_count=metadata.get("word_count"),
            sentence_count=metadata.get("sentence_count"),
            language=metadata.get("language"),
            embedding_model=metadata.get("embedding_model"),
            processing_time=metadata.get("processing_time")
        )

        return doc_record

    def create_collection_schema(self) -> Dict[str, Any]:
        """
        Create collection schema definition for vector stores that support it.

        Returns:
            Dict[str, Any]: Collection schema
        """
        return {
            "name": self.config.collection_name,
            "description": "Document chunks with embeddings",
            "vector_config": {
                "dimension": self.config.embedding_dimension,
                "distance": self.config.distance_metric.value
            },
            "fields": [
                {"name": "id", "type": "string", "primary": True},
                {"name": "text", "type": "text", "indexed": True},
                {"name": "document_id", "type": "string", "indexed": True},
                {"name": "document_title", "type": "string", "indexed": True},
                {"name": "document_type", "type": "string", "indexed": True},
                {"name": "chunk_index", "type": "integer", "indexed": True},
                {"name": "page_number", "type": "integer", "indexed": True},
                {"name": "language", "type": "string", "indexed": True},
                {"name": "embedding_model", "type": "string", "indexed": True},
                {"name": "source", "type": "string", "indexed": True},
                {"name": "word_count", "type": "integer", "indexed": False},
                {"name": "sentence_count", "type": "integer", "indexed": False},
                {"name": "created_at", "type": "datetime", "indexed": True}
            ]
        }
