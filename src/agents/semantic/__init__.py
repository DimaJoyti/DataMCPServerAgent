"""
Semantic Agents Module

This module provides semantic agents with advanced understanding capabilities,
inter-agent communication, and scalable architecture for generative AI solutions.

Phase 3 Integration:
- LLM-driven pipelines integration
- Multimodal processing capabilities
- RAG architecture integration
- Streaming pipeline support
"""

from .base_semantic_agent import BaseSemanticAgent, SemanticAgentConfig
from .communication import AgentCommunicationHub, AgentMessage, MessageBus
from .coordinator import SemanticCoordinator

# Phase 3: Integrated agents with LLM pipelines
from .integrated_agents import (
    IntegratedSemanticCoordinator,
    MultimodalSemanticAgent,
    RAGSemanticAgent,
    StreamingSemanticAgent,
)
from .specialized_agents import (
    DataAnalysisAgent,
    DocumentProcessingAgent,
    KnowledgeExtractionAgent,
    ReasoningAgent,
    SearchAgent,
)

__all__ = [
    "BaseSemanticAgent",
    "SemanticAgentConfig",
    "AgentCommunicationHub",
    "MessageBus",
    "AgentMessage",
    "SemanticCoordinator",
    "DataAnalysisAgent",
    "DocumentProcessingAgent",
    "KnowledgeExtractionAgent",
    "ReasoningAgent",
    "SearchAgent",
    # Phase 3 integrated agents
    "MultimodalSemanticAgent",
    "RAGSemanticAgent",
    "StreamingSemanticAgent",
    "IntegratedSemanticCoordinator",
]
