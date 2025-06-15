"""
Base classes for multimodal processing.

This module defines the core interfaces and data structures for
multimodal content processing in the DataMCPServerAgent system.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.logging import get_logger


class ModalityType(str, Enum):
    """Types of modalities supported."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class MultiModalContent(BaseModel):
    """Container for multimodal content."""

    # Content data
    text: Optional[str] = Field(None, description="Text content")
    image: Optional[bytes] = Field(None, description="Image data")
    audio: Optional[bytes] = Field(None, description="Audio data")
    video: Optional[bytes] = Field(None, description="Video data")

    # Metadata
    content_id: str = Field(..., description="Unique content identifier")
    modalities: List[ModalityType] = Field(..., description="Present modalities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # File information
    source_path: Optional[str] = Field(None, description="Source file path")
    mime_type: Optional[str] = Field(None, description="MIME type")
    file_size: Optional[int] = Field(None, description="File size in bytes")

    class Config:
        arbitrary_types_allowed = True


class ProcessingMetrics(BaseModel):
    """Metrics for processing operations."""

    processing_time: float = Field(..., description="Total processing time in seconds")
    modalities_processed: List[ModalityType] = Field(..., description="Processed modalities")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1)")

    # Resource usage
    memory_used: Optional[float] = Field(None, description="Memory used in MB")
    cpu_time: Optional[float] = Field(None, description="CPU time in seconds")

    # Quality metrics
    accuracy: Optional[float] = Field(None, description="Accuracy score (0-1)")
    relevance: Optional[float] = Field(None, description="Relevance score (0-1)")


class ProcessedResult(BaseModel):
    """Result of multimodal processing."""

    # Input reference
    content_id: str = Field(..., description="Original content ID")
    input_modalities: List[ModalityType] = Field(..., description="Input modalities")

    # Processing results
    extracted_text: Optional[str] = Field(None, description="Extracted or generated text")
    generated_description: Optional[str] = Field(None, description="Generated description")
    extracted_entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )

    # Embeddings
    text_embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    image_embedding: Optional[List[float]] = Field(None, description="Image embedding vector")
    audio_embedding: Optional[List[float]] = Field(None, description="Audio embedding vector")
    combined_embedding: Optional[List[float]] = Field(None, description="Combined embedding vector")

    # Metadata and metrics
    processing_metrics: ProcessingMetrics = Field(..., description="Processing metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Status
    status: str = Field(..., description="Processing status")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class MultiModalProcessor(ABC):
    """Abstract base class for multimodal processors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the processor with configuration."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize processor-specific components."""
        pass

    @abstractmethod
    async def process(self, content: MultiModalContent) -> ProcessedResult:
        """Process multimodal content."""
        pass

    @abstractmethod
    def get_supported_modalities(self) -> List[ModalityType]:
        """Get list of supported modalities."""
        pass

    async def validate_content(self, content: MultiModalContent) -> bool:
        """Validate that content can be processed."""
        supported = set(self.get_supported_modalities())
        required = set(content.modalities)

        if not required.issubset(supported):
            unsupported = required - supported
            self.logger.warning(f"Unsupported modalities: {unsupported}")
            return False

        return True

    async def preprocess(self, content: MultiModalContent) -> MultiModalContent:
        """Preprocess content before main processing."""
        # Default implementation - no preprocessing
        return content

    async def postprocess(self, result: ProcessedResult) -> ProcessedResult:
        """Postprocess results after main processing."""
        # Default implementation - no postprocessing
        return result

    async def process_with_metrics(self, content: MultiModalContent) -> ProcessedResult:
        """Process content with comprehensive metrics tracking."""
        start_time = time.time()

        try:
            # Validate content
            if not await self.validate_content(content):
                raise ValueError(f"Content validation failed for {content.content_id}")

            # Preprocess
            preprocessed_content = await self.preprocess(content)

            # Main processing
            result = await self.process(preprocessed_content)

            # Postprocess
            final_result = await self.postprocess(result)

            # Update metrics
            processing_time = time.time() - start_time
            final_result.processing_metrics.processing_time = processing_time
            final_result.status = "completed"

            self.logger.info(
                f"Successfully processed {content.content_id} " f"in {processing_time:.2f}s"
            )

            return final_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed for {content.content_id}: {str(e)}"
            self.logger.error(error_msg)

            # Create error result
            return ProcessedResult(
                content_id=content.content_id,
                input_modalities=content.modalities,
                processing_metrics=ProcessingMetrics(
                    processing_time=processing_time, modalities_processed=[]
                ),
                status="failed",
                errors=[error_msg],
            )

    async def process_batch(self, contents: List[MultiModalContent]) -> List[ProcessedResult]:
        """Process multiple contents in batch."""
        self.logger.info(f"Processing batch of {len(contents)} items")

        # Process all items concurrently
        tasks = [self.process_with_metrics(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessedResult(
                    content_id=contents[i].content_id,
                    input_modalities=contents[i].modalities,
                    processing_metrics=ProcessingMetrics(
                        processing_time=0.0, modalities_processed=[]
                    ),
                    status="failed",
                    errors=[str(result)],
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(updates)
        self.logger.info(f"Updated configuration: {list(updates.keys())}")


class ProcessorFactory:
    """Factory for creating multimodal processors."""

    _processors: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, processor_class: type) -> None:
        """Register a processor class."""
        cls._processors[name] = processor_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> MultiModalProcessor:
        """Create a processor instance."""
        if name not in cls._processors:
            raise ValueError(f"Unknown processor: {name}")

        processor_class = cls._processors[name]
        return processor_class(config)

    @classmethod
    def list_processors(cls) -> List[str]:
        """List available processors."""
        return list(cls._processors.keys())
