"""
Pipeline Router for Intelligent Orchestration.

This module provides intelligent routing of content to appropriate pipelines
based on content analysis, performance metrics, and resource availability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from app.core.logging import get_logger


class PipelineType(str, Enum):
    """Available pipeline types."""
    TEXT_ONLY = "text_only"
    TEXT_IMAGE = "text_image"
    TEXT_AUDIO = "text_audio"
    MULTIMODAL = "multimodal"
    STREAMING = "streaming"
    RAG = "rag"


@dataclass
class RoutingDecision:
    """Decision made by the router."""

    pipeline_type: PipelineType
    confidence: float
    reasoning: str
    estimated_processing_time: float
    resource_requirements: Dict[str, Any]


class PipelineRouter:
    """Intelligent pipeline router."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline router."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Routing rules and weights
        self.routing_rules = self.config.get("routing_rules", {})
        self.performance_weights = self.config.get("performance_weights", {
            "speed": 0.3,
            "accuracy": 0.4,
            "resource_efficiency": 0.3
        })

        self.logger.info("PipelineRouter initialized")

    async def route_content(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route content to the most appropriate pipeline."""

        # Analyze content characteristics
        content_analysis = await self._analyze_content(content, metadata)

        # Determine best pipeline
        pipeline_scores = await self._score_pipelines(content_analysis)

        # Select best pipeline
        best_pipeline = max(pipeline_scores.items(), key=lambda x: x[1]["total_score"])
        pipeline_type, score_info = best_pipeline

        return RoutingDecision(
            pipeline_type=PipelineType(pipeline_type),
            confidence=score_info["total_score"],
            reasoning=score_info["reasoning"],
            estimated_processing_time=score_info["estimated_time"],
            resource_requirements=score_info["resources"]
        )

    async def _analyze_content(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze content to determine characteristics."""
        analysis = {
            "has_text": False,
            "has_image": False,
            "has_audio": False,
            "content_size": 0,
            "complexity": "low",
            "modalities": []
        }

        # Check for different modalities
        if hasattr(content, 'text') and content.text:
            analysis["has_text"] = True
            analysis["modalities"].append("text")
            analysis["content_size"] += len(content.text)

        if hasattr(content, 'image') and content.image:
            analysis["has_image"] = True
            analysis["modalities"].append("image")
            analysis["content_size"] += len(content.image)

        if hasattr(content, 'audio') and content.audio:
            analysis["has_audio"] = True
            analysis["modalities"].append("audio")
            analysis["content_size"] += len(content.audio)

        # Determine complexity
        modality_count = len(analysis["modalities"])
        if modality_count > 2:
            analysis["complexity"] = "high"
        elif modality_count > 1:
            analysis["complexity"] = "medium"

        # Add metadata analysis
        if metadata:
            analysis["metadata"] = metadata

            # Check for streaming indicators
            if metadata.get("streaming", False) or metadata.get("real_time", False):
                analysis["requires_streaming"] = True

            # Check for RAG indicators
            if metadata.get("search_query", False) or metadata.get("retrieval", False):
                analysis["requires_rag"] = True

        return analysis

    async def _score_pipelines(self, content_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Score different pipelines for the given content."""

        scores = {}

        # Text-only pipeline
        scores["text_only"] = await self._score_text_pipeline(content_analysis)

        # Text+Image pipeline
        scores["text_image"] = await self._score_text_image_pipeline(content_analysis)

        # Text+Audio pipeline
        scores["text_audio"] = await self._score_text_audio_pipeline(content_analysis)

        # Multimodal pipeline
        scores["multimodal"] = await self._score_multimodal_pipeline(content_analysis)

        # Streaming pipeline
        scores["streaming"] = await self._score_streaming_pipeline(content_analysis)

        # RAG pipeline
        scores["rag"] = await self._score_rag_pipeline(content_analysis)

        return scores

    async def _score_text_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score text-only pipeline."""

        # High score if only text, low if other modalities
        if analysis["has_text"] and not analysis["has_image"] and not analysis["has_audio"]:
            speed_score = 0.9  # Very fast
            accuracy_score = 0.8  # Good for text
            resource_score = 0.9  # Low resource usage
            reasoning = "Content is text-only, perfect for text pipeline"
        else:
            speed_score = 0.1
            accuracy_score = 0.1
            resource_score = 0.9
            reasoning = "Content has non-text modalities, not suitable for text-only pipeline"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 0.1,  # seconds
            "resources": {"memory_mb": 50, "cpu_cores": 1}
        }

    async def _score_text_image_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score text+image pipeline."""

        if analysis["has_text"] and analysis["has_image"] and not analysis["has_audio"]:
            speed_score = 0.7
            accuracy_score = 0.9
            resource_score = 0.7
            reasoning = "Content has text and image, ideal for text-image pipeline"
        elif analysis["has_image"]:
            speed_score = 0.6
            accuracy_score = 0.7
            resource_score = 0.7
            reasoning = "Content has image, can be processed by text-image pipeline"
        else:
            speed_score = 0.2
            accuracy_score = 0.3
            resource_score = 0.7
            reasoning = "No image content, not optimal for text-image pipeline"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 0.5,
            "resources": {"memory_mb": 200, "cpu_cores": 2}
        }

    async def _score_text_audio_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score text+audio pipeline."""

        if analysis["has_text"] and analysis["has_audio"] and not analysis["has_image"]:
            speed_score = 0.6
            accuracy_score = 0.9
            resource_score = 0.6
            reasoning = "Content has text and audio, ideal for text-audio pipeline"
        elif analysis["has_audio"]:
            speed_score = 0.5
            accuracy_score = 0.7
            resource_score = 0.6
            reasoning = "Content has audio, can be processed by text-audio pipeline"
        else:
            speed_score = 0.2
            accuracy_score = 0.3
            resource_score = 0.6
            reasoning = "No audio content, not optimal for text-audio pipeline"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 1.0,
            "resources": {"memory_mb": 300, "cpu_cores": 2}
        }

    async def _score_multimodal_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score multimodal pipeline."""

        modality_count = len(analysis["modalities"])

        if modality_count >= 3:
            speed_score = 0.4
            accuracy_score = 0.95
            resource_score = 0.4
            reasoning = "Content has multiple modalities, perfect for multimodal pipeline"
        elif modality_count == 2:
            speed_score = 0.5
            accuracy_score = 0.8
            resource_score = 0.5
            reasoning = "Content has two modalities, good for multimodal pipeline"
        else:
            speed_score = 0.3
            accuracy_score = 0.5
            resource_score = 0.4
            reasoning = "Single modality content, multimodal pipeline is overkill"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 2.0,
            "resources": {"memory_mb": 500, "cpu_cores": 4}
        }

    async def _score_streaming_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score streaming pipeline."""

        if analysis.get("requires_streaming", False):
            speed_score = 0.9
            accuracy_score = 0.7
            resource_score = 0.6
            reasoning = "Content requires real-time processing, streaming pipeline is ideal"
        elif analysis["content_size"] > 10000:  # Large content
            speed_score = 0.7
            accuracy_score = 0.7
            resource_score = 0.7
            reasoning = "Large content benefits from streaming processing"
        else:
            speed_score = 0.3
            accuracy_score = 0.6
            resource_score = 0.5
            reasoning = "Small content doesn't require streaming"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 0.3,
            "resources": {"memory_mb": 150, "cpu_cores": 3}
        }

    async def _score_rag_pipeline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score RAG pipeline."""

        if analysis.get("requires_rag", False):
            speed_score = 0.6
            accuracy_score = 0.95
            resource_score = 0.5
            reasoning = "Content requires retrieval-augmented generation"
        elif analysis["has_text"]:
            speed_score = 0.5
            accuracy_score = 0.8
            resource_score = 0.6
            reasoning = "Text content can benefit from RAG for enhanced responses"
        else:
            speed_score = 0.3
            accuracy_score = 0.4
            resource_score = 0.5
            reasoning = "Non-text content not ideal for RAG pipeline"

        total_score = (
            speed_score * self.performance_weights["speed"] +
            accuracy_score * self.performance_weights["accuracy"] +
            resource_score * self.performance_weights["resource_efficiency"]
        )

        return {
            "total_score": total_score,
            "speed_score": speed_score,
            "accuracy_score": accuracy_score,
            "resource_score": resource_score,
            "reasoning": reasoning,
            "estimated_time": 0.8,
            "resources": {"memory_mb": 400, "cpu_cores": 2}
        }
