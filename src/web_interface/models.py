"""
Pydantic models for web interface API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""

    # File information
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")

    # Processing options
    extract_metadata: bool = Field(default=True, description="Extract document metadata")
    enable_chunking: bool = Field(default=True, description="Enable text chunking")
    enable_vectorization: bool = Field(default=True, description="Generate embeddings")

    # Chunking configuration
    chunk_size: int = Field(default=1000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    chunking_strategy: str = Field(default="text", description="Chunking strategy")

    # Vectorization configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    embedding_provider: str = Field(default="huggingface", description="Embedding provider")

    # Storage configuration
    store_vectors: bool = Field(default=True, description="Store vectors in vector store")
    vector_store_type: str = Field(default="memory", description="Vector store type")
    collection_name: Optional[str] = Field(None, description="Collection name")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Document tags")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

class ChunkInfo(BaseModel):
    """Information about a text chunk."""

    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_index: int = Field(..., description="Chunk index in document")
    text: str = Field(..., description="Chunk text content")
    character_count: int = Field(..., description="Number of characters")
    word_count: int = Field(..., description="Number of words")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")

    # Embedding information
    has_embedding: bool = Field(default=False, description="Whether chunk has embedding")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    embedding_dimension: Optional[int] = Field(None, description="Embedding dimension")

class DocumentProcessingResponse(BaseModel):
    """Response model for document processing."""

    # Processing information
    task_id: str = Field(..., description="Processing task ID")
    status: ProcessingStatus = Field(..., description="Processing status")

    # Document information
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")

    # Processing results
    text_length: Optional[int] = Field(None, description="Extracted text length")
    chunks: List[ChunkInfo] = Field(default_factory=list, description="Text chunks")

    # Metadata
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

    # Processing statistics
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    vectorization_time: Optional[float] = Field(None, description="Vectorization time in seconds")

    # Status information
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    # Error information
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")

    # Storage information
    stored_in_vector_store: bool = Field(default=False, description="Whether stored in vector store")
    vector_store_collection: Optional[str] = Field(None, description="Vector store collection")

class VectorSearchRequest(BaseModel):
    """Request model for vector search."""

    # Query
    query_text: Optional[str] = Field(None, description="Text query for semantic search")
    query_vector: Optional[List[float]] = Field(None, description="Vector query")

    # Search configuration
    search_type: str = Field(default="hybrid", description="Search type (vector, keyword, hybrid)")
    limit: int = Field(default=10, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")

    # Similarity configuration
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")

    # Filtering
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    document_types: Optional[List[str]] = Field(None, description="Filter by document types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")

    # Hybrid search weights
    vector_weight: float = Field(default=0.7, description="Weight for vector search")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search")

    # Result configuration
    include_text: bool = Field(default=True, description="Include text in results")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    include_vectors: bool = Field(default=False, description="Include vectors in results")

    # Vector store configuration
    collection_name: Optional[str] = Field(None, description="Collection to search")
    vector_store_type: str = Field(default="memory", description="Vector store type")

class SearchResultItem(BaseModel):
    """Individual search result item."""

    # Identification
    id: str = Field(..., description="Result ID")
    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")

    # Content
    text: str = Field(..., description="Text content")

    # Relevance
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Result rank")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

    # Optional data
    vector: Optional[List[float]] = Field(None, description="Embedding vector")

    # Context
    document_title: Optional[str] = Field(None, description="Document title")
    chunk_index: Optional[int] = Field(None, description="Chunk index")

class VectorSearchResponse(BaseModel):
    """Response model for vector search."""

    # Results
    results: List[SearchResultItem] = Field(..., description="Search results")

    # Query information
    query_text: Optional[str] = Field(None, description="Original query text")
    search_type: str = Field(..., description="Search type used")

    # Statistics
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")

    # Pagination
    offset: int = Field(..., description="Result offset")
    limit: int = Field(..., description="Result limit")
    has_more: bool = Field(..., description="Whether more results available")

    # Aggregations
    document_counts: Dict[str, int] = Field(default_factory=dict, description="Results by document")
    type_counts: Dict[str, int] = Field(default_factory=dict, description="Results by type")

class PipelineStatus(BaseModel):
    """Pipeline status information."""

    # Overall status
    is_healthy: bool = Field(..., description="Whether pipeline is healthy")
    status: str = Field(..., description="Overall status")

    # Component status
    document_processor_status: str = Field(..., description="Document processor status")
    vectorizer_status: str = Field(..., description="Vectorizer status")
    vector_store_status: str = Field(..., description="Vector store status")

    # Statistics
    total_documents: int = Field(default=0, description="Total documents processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    total_vectors: int = Field(default=0, description="Total vectors stored")

    # Performance metrics
    avg_processing_time: Optional[float] = Field(None, description="Average processing time")
    avg_vectorization_time: Optional[float] = Field(None, description="Average vectorization time")

    # Resource usage
    memory_usage: Optional[int] = Field(None, description="Memory usage in bytes")
    disk_usage: Optional[int] = Field(None, description="Disk usage in bytes")

    # Cache statistics
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")
    cache_size: Optional[int] = Field(None, description="Cache size")

    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")

class TaskInfo(BaseModel):
    """Information about a processing task."""

    task_id: str = Field(..., description="Task identifier")
    status: ProcessingStatus = Field(..., description="Task status")
    task_type: str = Field(..., description="Type of task")

    # Progress information
    progress: float = Field(default=0.0, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")

    # Timing
    created_at: datetime = Field(..., description="Task creation time")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")

    # Results
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task metadata")

class BatchProcessingRequest(BaseModel):
    """Request for batch processing multiple documents."""

    # Files information
    files: List[Dict[str, Any]] = Field(..., description="List of files to process")

    # Processing configuration
    processing_config: DocumentUploadRequest = Field(..., description="Processing configuration")

    # Batch options
    batch_size: int = Field(default=10, description="Batch size for processing")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")

    # Error handling
    continue_on_error: bool = Field(default=True, description="Continue on individual file errors")

    # Notification
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")

class BatchProcessingResponse(BaseModel):
    """Response for batch processing."""

    batch_id: str = Field(..., description="Batch processing ID")
    status: ProcessingStatus = Field(..., description="Batch status")

    # Progress
    total_files: int = Field(..., description="Total number of files")
    processed_files: int = Field(default=0, description="Number of processed files")
    failed_files: int = Field(default=0, description="Number of failed files")

    # Individual task IDs
    task_ids: List[str] = Field(..., description="Individual task IDs")

    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="Batch creation time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

    # Results summary
    results_summary: Dict[str, Any] = Field(default_factory=dict, description="Results summary")
