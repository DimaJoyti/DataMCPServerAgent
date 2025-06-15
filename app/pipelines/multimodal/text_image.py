"""
Text + Image Processing Pipeline.

This module provides comprehensive text and image processing capabilities:
- OCR (Optical Character Recognition) for text extraction
- Image analysis and description generation
- Visual question answering
- Combined text-image embeddings
- Cross-modal search and retrieval
"""

import io
from typing import Any, Dict, List, Optional

import numpy as np

# Optional dependencies
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from app.core.logging import get_logger

from .base import (
    ModalityType,
    MultiModalContent,
    MultiModalProcessor,
    ProcessedResult,
    ProcessingMetrics,
    ProcessorFactory,
)


class ImageAnalyzer:
    """Analyzer for image content and properties."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def extract_text_ocr(self, image_data: bytes) -> str:
        """Extract text from image using OCR."""
        if not PIL_AVAILABLE or not PYTESSERACT_AVAILABLE:
            self.logger.warning("PIL or pytesseract not available, returning placeholder text")
            return "[OCR text would be extracted here - install PIL and pytesseract]"

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Perform OCR
            extracted_text = pytesseract.image_to_string(image)

            self.logger.debug(f"Extracted {len(extracted_text)} characters via OCR")
            return extracted_text.strip()

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""

    async def analyze_image_properties(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze basic image properties."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, returning basic properties")
            return {"size_bytes": len(image_data), "format": "unknown", "analysis_available": False}

        try:
            image = Image.open(io.BytesIO(image_data))

            properties = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(image_data),
                "aspect_ratio": image.width / image.height if image.height > 0 else 0,
            }

            # Color analysis
            if image.mode == "RGB":
                # Get dominant colors (simplified)
                colors = image.getcolors(maxcolors=256 * 256 * 256)
                if colors:
                    dominant_color = max(colors, key=lambda x: x[0])[1]
                    properties["dominant_color"] = dominant_color

            return properties

        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {"size_bytes": len(image_data)}

    async def generate_description(self, image_data: bytes, context: Optional[str] = None) -> str:
        """Generate description of image content."""
        # This is a placeholder for AI-powered image description
        # In a real implementation, this would use models like:
        # - OpenAI GPT-4 Vision
        # - Google Cloud Vision API
        # - Azure Computer Vision
        # - Local models like BLIP, LLaVA, etc.

        try:
            properties = await self.analyze_image_properties(image_data)

            # Basic description based on properties
            description_parts = []

            if properties.get("width") and properties.get("height"):
                size = f"{properties['width']}x{properties['height']}"
                description_parts.append(f"Image with dimensions {size}")

            if properties.get("format"):
                description_parts.append(f"in {properties['format']} format")

            # Add context if provided
            if context:
                description_parts.append(f"Related to: {context}")

            description = ", ".join(description_parts) if description_parts else "Image content"

            self.logger.debug(f"Generated description: {description}")
            return description

        except Exception as e:
            self.logger.error(f"Description generation failed: {e}")
            return "Image content (description unavailable)"


class TextImageProcessor(MultiModalProcessor):
    """Processor for combined text and image content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text-image processor."""
        super().__init__(config)
        self.image_analyzer = ImageAnalyzer()

    def _initialize(self) -> None:
        """Initialize processor-specific components."""
        self.logger.info("Initializing TextImageProcessor")

        # Configuration
        self.enable_ocr = self.get_config("enable_ocr", True)
        self.enable_description = self.get_config("enable_description", True)
        self.enable_embeddings = self.get_config("enable_embeddings", True)
        self.max_image_size = self.get_config("max_image_size", 10 * 1024 * 1024)  # 10MB

        self.logger.info(
            f"TextImageProcessor initialized with OCR: {self.enable_ocr}, "
            f"Description: {self.enable_description}, "
            f"Embeddings: {self.enable_embeddings}"
        )

    def get_supported_modalities(self) -> List[ModalityType]:
        """Get supported modalities."""
        return [ModalityType.TEXT, ModalityType.IMAGE]

    async def validate_content(self, content: MultiModalContent) -> bool:
        """Validate content for text-image processing."""
        # Check base validation
        if not await super().validate_content(content):
            return False

        # Check image size if present
        if content.image and len(content.image) > self.max_image_size:
            self.logger.warning(
                f"Image size {len(content.image)} exceeds limit {self.max_image_size}"
            )
            return False

        # Must have at least text or image
        if not content.text and not content.image:
            self.logger.warning("Content must have either text or image")
            return False

        return True

    async def process_image_only(self, image_data: bytes) -> Dict[str, Any]:
        """Process image-only content."""
        results = {}

        # OCR text extraction
        if self.enable_ocr:
            extracted_text = await self.image_analyzer.extract_text_ocr(image_data)
            results["extracted_text"] = extracted_text

        # Image analysis
        properties = await self.image_analyzer.analyze_image_properties(image_data)
        results["image_properties"] = properties

        # Description generation
        if self.enable_description:
            description = await self.image_analyzer.generate_description(image_data)
            results["description"] = description

        return results

    async def process_text_only(self, text: str) -> Dict[str, Any]:
        """Process text-only content."""
        results = {
            "processed_text": text,
            "text_length": len(text),
            "word_count": len(text.split()) if text else 0,
        }

        # Basic text analysis
        if text:
            # Extract entities (placeholder)
            entities = self.extract_entities(text)
            results["entities"] = entities

        return results

    async def process_combined(self, text: str, image_data: bytes) -> Dict[str, Any]:
        """Process combined text and image content."""
        # Process both modalities
        text_results = await self.process_text_only(text)
        image_results = await self.process_image_only(image_data)

        # Combine results
        combined_results = {**text_results, **image_results}

        # Cross-modal analysis
        if self.enable_description and text:
            # Generate image description with text context
            contextual_description = await self.image_analyzer.generate_description(
                image_data, context=text[:200]  # Use first 200 chars as context
            )
            combined_results["contextual_description"] = contextual_description

        return combined_results

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (placeholder implementation)."""
        # This is a simplified implementation
        # In production, use NLP libraries like spaCy, NLTK, or cloud APIs

        entities = []

        # Simple keyword extraction
        words = text.split()
        for word in words:
            if word.isupper() and len(word) > 2:  # Potential acronym
                entities.append({"text": word, "type": "ACRONYM", "confidence": 0.7})

        return entities

    async def generate_embeddings(self, content: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate embeddings for text and image content."""
        embeddings = {}

        # Text embedding (placeholder)
        if "processed_text" in content and content["processed_text"]:
            # In production, use actual embedding models
            text_embedding = np.random.rand(384).tolist()  # Placeholder
            embeddings["text_embedding"] = text_embedding

        # Image embedding (placeholder)
        if "image_properties" in content:
            # In production, use vision models like CLIP
            image_embedding = np.random.rand(512).tolist()  # Placeholder
            embeddings["image_embedding"] = image_embedding

        # Combined embedding
        if "text_embedding" in embeddings and "image_embedding" in embeddings:
            # Simple concatenation (in production, use more sophisticated fusion)
            combined = embeddings["text_embedding"] + embeddings["image_embedding"]
            embeddings["combined_embedding"] = combined

        return embeddings

    async def process(self, content: MultiModalContent) -> ProcessedResult:
        """Process multimodal text-image content."""
        self.logger.info(f"Processing text-image content: {content.content_id}")

        # Determine processing path
        has_text = bool(content.text)
        has_image = bool(content.image)

        if has_text and has_image:
            processing_results = await self.process_combined(content.text, content.image)
            modalities_processed = [ModalityType.TEXT, ModalityType.IMAGE]
        elif has_image:
            processing_results = await self.process_image_only(content.image)
            modalities_processed = [ModalityType.IMAGE]
        elif has_text:
            processing_results = await self.process_text_only(content.text)
            modalities_processed = [ModalityType.TEXT]
        else:
            raise ValueError("No valid content to process")

        # Generate embeddings
        embeddings = {}
        if self.enable_embeddings:
            embeddings = await self.generate_embeddings(processing_results)

        # Create result
        result = ProcessedResult(
            content_id=content.content_id,
            input_modalities=content.modalities,
            extracted_text=processing_results.get("extracted_text"),
            generated_description=processing_results.get("description")
            or processing_results.get("contextual_description"),
            extracted_entities=processing_results.get("entities", []),
            text_embedding=embeddings.get("text_embedding"),
            image_embedding=embeddings.get("image_embedding"),
            combined_embedding=embeddings.get("combined_embedding"),
            processing_metrics=ProcessingMetrics(
                processing_time=0.0,  # Will be set by parent class
                modalities_processed=modalities_processed,
            ),
            metadata={"processor": "TextImageProcessor", "processing_results": processing_results},
            status="processing",
        )

        return result


# Register the processor
ProcessorFactory.register("text_image", TextImageProcessor)
