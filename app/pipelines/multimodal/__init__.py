"""
Multimodal Processing Pipeline.

This module provides comprehensive multimodal processing capabilities:
- Text + Image processing with OCR and visual analysis
- Text + Audio processing with speech recognition and synthesis
- Combined multimodal content processing
- Cross-modal embeddings and search
"""

from .base import (
    ModalityType,
    MultiModalContent,
    MultiModalProcessor,
    ProcessedResult,
    ProcessorFactory,
)
from .combined import CombinedProcessor
from .text_audio import TextAudioProcessor
from .text_image import TextImageProcessor

# Processors will be added later
# from .processors import (
#     ImageProcessor,
#     AudioProcessor,
#     VideoProcessor
# )

__all__ = [
    # Base
    "MultiModalProcessor",
    "MultiModalContent",
    "ProcessedResult",
    "ModalityType",
    "ProcessorFactory",

    # Processors
    "TextImageProcessor",
    "TextAudioProcessor",
    "CombinedProcessor",

    # Specialized (will be added later)
    # "ImageProcessor",
    # "AudioProcessor",
    # "VideoProcessor",
]
