"""
Web interface for document processing pipeline.

This module provides REST API endpoints and web interface for:
- Document upload and processing
- Vector search and retrieval
- Pipeline monitoring and management
- Integration with agent-ui
"""

from .api import DocumentProcessingAPI, create_app
from .models import (
    DocumentProcessingResponse,
    DocumentUploadRequest,
    PipelineStatus,
    VectorSearchRequest,
    VectorSearchResponse,
)

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    "create_app",
    "DocumentProcessingAPI",
    "DocumentUploadRequest",
    "DocumentProcessingResponse",
    "VectorSearchRequest",
    "VectorSearchResponse",
    "PipelineStatus",
]
