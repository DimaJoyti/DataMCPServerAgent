"""
Combined Multimodal Processing Pipeline.

This module provides comprehensive processing for all modality combinations:
- Text + Image + Audio processing
- Cross-modal understanding and fusion
- Unified embeddings and representations
- Advanced multimodal reasoning
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from .base import (
    MultiModalProcessor,
    MultiModalContent,
    ProcessedResult,
    ProcessingMetrics,
    ModalityType,
    ProcessorFactory
)
from .text_image import TextImageProcessor
from .text_audio import TextAudioProcessor
from app.core.logging import get_logger

class ModalityFusion:
    """Handles fusion of multiple modalities."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def fuse_embeddings(self, embeddings: Dict[str, List[float]]) -> List[float]:
        """Fuse embeddings from multiple modalities."""
        try:
            available_embeddings = []

            # Collect available embeddings
            for modality in ["text_embedding", "image_embedding", "audio_embedding"]:
                if modality in embeddings and embeddings[modality]:
                    available_embeddings.append(np.array(embeddings[modality]))

            if not available_embeddings:
                return []

            # Simple fusion strategies
            fusion_method = "concatenation"  # Could be: concatenation, average, weighted_average, attention

            if fusion_method == "concatenation":
                # Concatenate all embeddings
                fused = np.concatenate(available_embeddings)
            elif fusion_method == "average":
                # Average embeddings (requires same dimensions)
                min_dim = min(emb.shape[0] for emb in available_embeddings)
                truncated = [emb[:min_dim] for emb in available_embeddings]
                fused = np.mean(truncated, axis=0)
            else:
                # Default to concatenation
                fused = np.concatenate(available_embeddings)

            self.logger.debug(f"Fused {len(available_embeddings)} embeddings into {len(fused)} dimensions")
            return fused.tolist()

        except Exception as e:
            self.logger.error(f"Embedding fusion failed: {e}")
            return []

    async def cross_modal_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal analysis and reasoning."""
        analysis = {}

        try:
            # Text-Image consistency
            if "processed_text" in results and "description" in results:
                text = results["processed_text"]
                image_desc = results["description"]
                consistency = self.calculate_consistency(text, image_desc)
                analysis["text_image_consistency"] = consistency

            # Text-Audio consistency
            if "processed_text" in results and "transcription" in results:
                text = results["processed_text"]
                transcription = results["transcription"]
                consistency = self.calculate_consistency(text, transcription)
                analysis["text_audio_consistency"] = consistency

            # Multimodal sentiment (placeholder)
            sentiment_scores = []
            if "processed_text" in results:
                # Text sentiment (placeholder)
                sentiment_scores.append(0.5)  # Neutral
            if "audio_classification" in results:
                # Audio emotion (placeholder)
                emotion = results["audio_classification"].get("emotion", "neutral")
                emotion_score = {"positive": 0.8, "negative": 0.2, "neutral": 0.5}.get(emotion, 0.5)
                sentiment_scores.append(emotion_score)

            if sentiment_scores:
                analysis["overall_sentiment"] = np.mean(sentiment_scores)

            # Content coherence
            modalities_count = len([k for k in results.keys()
                                  if k in ["processed_text", "description", "transcription"]])
            analysis["modality_richness"] = modalities_count / 3.0  # Normalized

            return analysis

        except Exception as e:
            self.logger.error(f"Cross-modal analysis failed: {e}")
            return {}

    def calculate_consistency(self, text1: str, text2: str) -> float:
        """Calculate consistency between two text sources."""
        if not text1 or not text2:
            return 0.0

        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

class CombinedProcessor(MultiModalProcessor):
    """Processor for all multimodal combinations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the combined processor."""
        super().__init__(config)

        # Initialize specialized processors
        self.text_image_processor = TextImageProcessor(config)
        self.text_audio_processor = TextAudioProcessor(config)
        self.modality_fusion = ModalityFusion()

    def _initialize(self) -> None:
        """Initialize processor-specific components."""
        self.logger.info("Initializing CombinedProcessor")

        # Configuration
        self.enable_cross_modal = self.get_config("enable_cross_modal", True)
        self.enable_fusion = self.get_config("enable_fusion", True)
        self.enable_reasoning = self.get_config("enable_reasoning", True)

        self.logger.info(f"CombinedProcessor initialized with Cross-modal: {self.enable_cross_modal}, "
                        f"Fusion: {self.enable_fusion}, "
                        f"Reasoning: {self.enable_reasoning}")

    def get_supported_modalities(self) -> List[ModalityType]:
        """Get supported modalities."""
        return [ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]

    async def validate_content(self, content: MultiModalContent) -> bool:
        """Validate content for combined processing."""
        # Check base validation
        if not await super().validate_content(content):
            return False

        # Must have at least two modalities for combined processing
        modality_count = sum([
            bool(content.text),
            bool(content.image),
            bool(content.audio)
        ])

        if modality_count < 2:
            self.logger.warning("Combined processor requires at least 2 modalities")
            return False

        return True

    async def process_text_image_audio(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process content with all three modalities."""
        results = {}

        # Process text-image combination
        text_image_content = MultiModalContent(
            content_id=f"{content.content_id}_text_image",
            text=content.text,
            image=content.image,
            modalities=[ModalityType.TEXT, ModalityType.IMAGE],
            metadata=content.metadata
        )

        text_image_result = await self.text_image_processor.process(text_image_content)
        results.update(text_image_result.metadata.get("processing_results", {}))

        # Process text-audio combination
        text_audio_content = MultiModalContent(
            content_id=f"{content.content_id}_text_audio",
            text=content.text,
            audio=content.audio,
            modalities=[ModalityType.TEXT, ModalityType.AUDIO],
            metadata=content.metadata
        )

        text_audio_result = await self.text_audio_processor.process(text_audio_content)
        audio_results = text_audio_result.metadata.get("processing_results", {})

        # Merge audio-specific results
        for key, value in audio_results.items():
            if key.startswith("audio_") or key in ["transcription"]:
                results[key] = value

        # Collect embeddings
        embeddings = {}
        if text_image_result.text_embedding:
            embeddings["text_embedding"] = text_image_result.text_embedding
        if text_image_result.image_embedding:
            embeddings["image_embedding"] = text_image_result.image_embedding
        if text_audio_result.audio_embedding:
            embeddings["audio_embedding"] = text_audio_result.audio_embedding

        results["embeddings"] = embeddings

        return results

    async def process_two_modalities(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process content with exactly two modalities."""
        has_text = bool(content.text)
        has_image = bool(content.image)
        has_audio = bool(content.audio)

        if has_text and has_image and not has_audio:
            # Text + Image
            result = await self.text_image_processor.process(content)
            return result.metadata.get("processing_results", {})

        elif has_text and has_audio and not has_image:
            # Text + Audio
            result = await self.text_audio_processor.process(content)
            return result.metadata.get("processing_results", {})

        elif has_image and has_audio and not has_text:
            # Image + Audio (less common, process separately)
            results = {}

            # Process image
            image_content = MultiModalContent(
                content_id=f"{content.content_id}_image",
                image=content.image,
                modalities=[ModalityType.IMAGE],
                metadata=content.metadata
            )
            image_result = await self.text_image_processor.process_image_only(content.image)
            results.update(image_result)

            # Process audio
            audio_result = await self.text_audio_processor.process_audio_only(content.audio)
            results.update(audio_result)

            return results

        else:
            raise ValueError("Invalid modality combination")

    async def generate_unified_description(self, results: Dict[str, Any]) -> str:
        """Generate a unified description of all content."""
        description_parts = []

        # Text content
        if "processed_text" in results and results["processed_text"]:
            word_count = results.get("word_count", 0)
            description_parts.append(f"Text content with {word_count} words")

        # Image content
        if "description" in results:
            description_parts.append(f"Image: {results['description']}")
        elif "image_properties" in results:
            props = results["image_properties"]
            if props.get("width") and props.get("height"):
                description_parts.append(f"Image ({props['width']}x{props['height']})")

        # Audio content
        if "audio_classification" in results:
            classification = results["audio_classification"]
            content_type = classification.get("content_type", "audio")
            description_parts.append(f"{content_type.title()} content")
        elif "audio_properties" in results:
            props = results["audio_properties"]
            duration = props.get("duration", 0)
            if duration > 0:
                description_parts.append(f"Audio ({duration:.1f}s)")

        # Cross-modal insights
        if "cross_modal_analysis" in results:
            analysis = results["cross_modal_analysis"]
            if "overall_sentiment" in analysis:
                sentiment = analysis["overall_sentiment"]
                sentiment_label = "positive" if sentiment > 0.6 else "negative" if sentiment < 0.4 else "neutral"
                description_parts.append(f"Overall sentiment: {sentiment_label}")

        return "; ".join(description_parts) if description_parts else "Multimodal content"

    async def process(self, content: MultiModalContent) -> ProcessedResult:
        """Process combined multimodal content."""
        self.logger.info(f"Processing combined multimodal content: {content.content_id}")

        # Count modalities
        modality_count = sum([
            bool(content.text),
            bool(content.image),
            bool(content.audio)
        ])

        # Process based on modality count
        if modality_count == 3:
            processing_results = await self.process_text_image_audio(content)
            modalities_processed = [ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]
        elif modality_count == 2:
            processing_results = await self.process_two_modalities(content)
            modalities_processed = [m for m in content.modalities if m in self.get_supported_modalities()]
        else:
            raise ValueError("Combined processor requires at least 2 modalities")

        # Cross-modal analysis
        cross_modal_results = {}
        if self.enable_cross_modal:
            cross_modal_results = await self.modality_fusion.cross_modal_analysis(processing_results)
            processing_results["cross_modal_analysis"] = cross_modal_results

        # Embedding fusion
        combined_embedding = None
        if self.enable_fusion and "embeddings" in processing_results:
            combined_embedding = await self.modality_fusion.fuse_embeddings(
                processing_results["embeddings"]
            )

        # Generate unified description
        unified_description = await self.generate_unified_description(processing_results)

        # Create result
        result = ProcessedResult(
            content_id=content.content_id,
            input_modalities=content.modalities,
            extracted_text=processing_results.get("extracted_text") or processing_results.get("transcription"),
            generated_description=unified_description,
            extracted_entities=processing_results.get("entities", []),
            text_embedding=processing_results.get("embeddings", {}).get("text_embedding"),
            image_embedding=processing_results.get("embeddings", {}).get("image_embedding"),
            audio_embedding=processing_results.get("embeddings", {}).get("audio_embedding"),
            combined_embedding=combined_embedding,
            processing_metrics=ProcessingMetrics(
                processing_time=0.0,  # Will be set by parent class
                modalities_processed=modalities_processed,
                confidence_score=cross_modal_results.get("overall_sentiment")
            ),
            metadata={
                "processor": "CombinedProcessor",
                "processing_results": processing_results,
                "cross_modal_analysis": cross_modal_results
            },
            status="processing"
        )

        return result

# Register the processor
ProcessorFactory.register("combined", CombinedProcessor)
