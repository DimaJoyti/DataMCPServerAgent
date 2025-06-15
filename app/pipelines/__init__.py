"""
LLM-driven Pipelines for DataMCPServerAgent.

This module provides comprehensive pipeline capabilities including:
- Multimodal processing (text, image, audio)
- RAG architecture with hybrid search
- Real-time streaming processing
- Intelligent orchestration and routing
- Performance optimization and monitoring
"""

from .multimodal import (
    CombinedProcessor,
    MultiModalProcessor,
    TextAudioProcessor,
    TextImageProcessor,
)
from .orchestration import DynamicOptimizer, PipelineCoordinator, PipelineRouter
from .rag import AdaptiveChunker, HybridSearchEngine, MultiVectorStore, ReRanker
from .streaming import IncrementalProcessor, LiveMonitor, StreamingPipeline

__version__ = "2.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    # Multimodal
    "MultiModalProcessor",
    "TextImageProcessor",
    "TextAudioProcessor",
    "CombinedProcessor",
    # RAG
    "HybridSearchEngine",
    "AdaptiveChunker",
    "MultiVectorStore",
    "ReRanker",
    # Streaming
    "StreamingPipeline",
    "IncrementalProcessor",
    "LiveMonitor",
    # Orchestration
    "PipelineRouter",
    "DynamicOptimizer",
    "PipelineCoordinator",
]
