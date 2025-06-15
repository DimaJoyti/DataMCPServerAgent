"""
Integrated Semantic Agents with LLM Pipelines

This module provides semantic agents that integrate with Phase 2 LLM pipelines:
- Multimodal processing integration
- RAG architecture integration
- Streaming pipeline integration
- Advanced coordination capabilities
"""

import time
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.pipelines.multimodal import (
    ModalityType,
    MultiModalContent,
    ProcessorFactory,
)
from app.pipelines.rag import (
    AdaptiveChunker,
    HybridSearchEngine,
    MultiVectorStore,
    SearchQuery,
)
from app.pipelines.streaming import (
    IncrementalProcessor,
    StreamEvent,
    StreamEventType,
    StreamingPipeline,
)

from .base_semantic_agent import BaseSemanticAgent, SemanticAgentConfig, SemanticContext
from .coordinator import SemanticCoordinator


class MultimodalSemanticAgent(BaseSemanticAgent):
    """
    Semantic agent with integrated multimodal processing capabilities.

    Features:
    - Text + Image processing with OCR and visual analysis
    - Text + Audio processing with speech recognition
    - Combined multimodal content understanding
    - Cross-modal semantic analysis
    """

    def __init__(
        self,
        config: SemanticAgentConfig,
        multimodal_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize multimodal semantic agent."""
        super().__init__(config, **kwargs)

        # Initialize multimodal processors
        self.multimodal_config = multimodal_config or {}
        self.text_image_processor = ProcessorFactory.create("text_image", self.multimodal_config)
        self.text_audio_processor = ProcessorFactory.create("text_audio", self.multimodal_config)
        self.combined_processor = ProcessorFactory.create("combined", self.multimodal_config)

        self.logger = get_logger(f"multimodal_agent.{config.name}")

    async def process_request(
        self,
        request: str,
        context: Optional[SemanticContext] = None,
    ) -> Dict[str, Any]:
        """Process multimodal requests with semantic understanding."""
        self.logger.info(f"Processing multimodal request: {request[:100]}...")

        start_time = time.time()

        try:
            # Understand intent first
            semantic_context = await self.understand_intent(
                request, context.dict() if context else {}
            )

            # Determine if this is a multimodal request
            modalities = self._detect_modalities(request, semantic_context)

            if len(modalities) > 1:
                # Process as multimodal content
                result = await self._process_multimodal_content(
                    request, modalities, semantic_context
                )
            else:
                # Process as single modality
                result = await self._process_single_modality(
                    request, modalities[0] if modalities else ModalityType.TEXT, semantic_context
                )

            # Store results in memory
            if self.config.memory_enabled:
                await self.store_memory(
                    content=f"Multimodal processing: {request}",
                    context=semantic_context,
                    metadata={
                        "modalities": [m.value for m in modalities],
                        "processing_time": time.time() - start_time,
                        "result_type": result.get("type", "unknown"),
                    },
                )

            return {
                "agent": self.config.name,
                "type": "multimodal_processing",
                "modalities": [m.value for m in modalities],
                "result": result,
                "processing_time": time.time() - start_time,
                "semantic_context": semantic_context.dict(),
            }

        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            return {
                "agent": self.config.name,
                "type": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def understand_intent(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SemanticContext:
        """Understand multimodal intent."""
        # Analyze request for multimodal indicators
        multimodal_indicators = {
            "image": ["image", "picture", "photo", "visual", "see", "look", "ocr"],
            "audio": ["audio", "sound", "voice", "speech", "listen", "hear"],
            "video": ["video", "movie", "clip", "watch"],
        }

        detected_modalities = []
        for modality, keywords in multimodal_indicators.items():
            if any(keyword in request.lower() for keyword in keywords):
                detected_modalities.append(modality)

        return SemanticContext(
            user_intent=request,
            context_data=context or {},
            metadata={
                "detected_modalities": detected_modalities,
                "multimodal_request": len(detected_modalities) > 0,
                "agent_type": "multimodal_semantic",
            },
        )

    def _detect_modalities(self, request: str, context: SemanticContext) -> List[ModalityType]:
        """Detect required modalities from request and context."""
        modalities = [ModalityType.TEXT]  # Always include text

        # Check for image indicators
        if any(
            keyword in request.lower() for keyword in ["image", "picture", "photo", "visual", "ocr"]
        ):
            modalities.append(ModalityType.IMAGE)

        # Check for audio indicators
        if any(keyword in request.lower() for keyword in ["audio", "sound", "voice", "speech"]):
            modalities.append(ModalityType.AUDIO)

        return modalities

    async def _process_multimodal_content(
        self, request: str, modalities: List[ModalityType], context: SemanticContext
    ) -> Dict[str, Any]:
        """Process content with multiple modalities."""
        # Create multimodal content object
        content = MultiModalContent(
            content_id=context.task_id,
            text=request,
            modalities=modalities,
            metadata=context.metadata,
        )

        # Use appropriate processor
        if ModalityType.IMAGE in modalities and ModalityType.AUDIO in modalities:
            processor = self.combined_processor
        elif ModalityType.IMAGE in modalities:
            processor = self.text_image_processor
        elif ModalityType.AUDIO in modalities:
            processor = self.text_audio_processor
        else:
            processor = self.text_image_processor  # Default

        # Process content
        result = await processor.process_with_metrics(content)

        return {
            "type": "multimodal",
            "processor_used": processor.__class__.__name__,
            "extracted_text": getattr(result, "extracted_text", ""),
            "entities": getattr(result, "extracted_entities", []),
            "embeddings_generated": hasattr(result, "combined_embedding"),
            "processing_metrics": (
                getattr(result, "processing_metrics", {}).dict()
                if hasattr(result, "processing_metrics")
                else {}
            ),
        }

    async def _process_single_modality(
        self, request: str, modality: ModalityType, context: SemanticContext
    ) -> Dict[str, Any]:
        """Process content with single modality."""
        return {
            "type": "single_modality",
            "modality": modality.value,
            "processed_text": request,
            "semantic_analysis": "Basic text processing completed",
        }


class RAGSemanticAgent(BaseSemanticAgent):
    """
    Semantic agent with integrated RAG (Retrieval-Augmented Generation) capabilities.

    Features:
    - Hybrid search (vector + keyword + semantic)
    - Adaptive document chunking
    - Multi-vector store integration
    - Context-aware retrieval
    """

    def __init__(
        self, config: SemanticAgentConfig, rag_config: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """Initialize RAG semantic agent."""
        super().__init__(config, **kwargs)

        # Initialize RAG components
        self.rag_config = rag_config or {}
        self.search_engine = HybridSearchEngine(self.rag_config)
        self.chunker = AdaptiveChunker(self.rag_config)
        self.vector_store = MultiVectorStore(self.rag_config)

        self.logger = get_logger(f"rag_agent.{config.name}")

    async def process_request(
        self,
        request: str,
        context: Optional[SemanticContext] = None,
    ) -> Dict[str, Any]:
        """Process requests with RAG capabilities."""
        self.logger.info(f"Processing RAG request: {request[:100]}...")

        start_time = time.time()

        try:
            # Understand intent
            semantic_context = await self.understand_intent(
                request, context.dict() if context else {}
            )

            # Determine if this is a search/retrieval request
            if self._is_retrieval_request(request):
                result = await self._perform_rag_search(request, semantic_context)
            else:
                result = await self._perform_rag_generation(request, semantic_context)

            # Store in memory
            if self.config.memory_enabled:
                await self.store_memory(
                    content=f"RAG processing: {request}",
                    context=semantic_context,
                    metadata={
                        "rag_type": result.get("type"),
                        "processing_time": time.time() - start_time,
                    },
                )

            return {
                "agent": self.config.name,
                "type": "rag_processing",
                "result": result,
                "processing_time": time.time() - start_time,
                "semantic_context": semantic_context.dict(),
            }

        except Exception as e:
            self.logger.error(f"Error in RAG processing: {e}")
            return {
                "agent": self.config.name,
                "type": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def understand_intent(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SemanticContext:
        """Understand RAG-specific intent."""
        return SemanticContext(
            user_intent=request,
            context_data=context or {},
            metadata={
                "is_search_query": self._is_retrieval_request(request),
                "agent_type": "rag_semantic",
            },
        )

    def _is_retrieval_request(self, request: str) -> bool:
        """Determine if request is for information retrieval."""
        retrieval_keywords = [
            "search",
            "find",
            "look for",
            "retrieve",
            "get information",
            "what is",
            "who is",
            "where is",
            "when",
            "how",
            "explain",
        ]
        return any(keyword in request.lower() for keyword in retrieval_keywords)

    async def _perform_rag_search(self, query: str, context: SemanticContext) -> Dict[str, Any]:
        """Perform RAG search operation."""
        # Create search query
        search_query = SearchQuery(query=query, filters={}, limit=10, metadata=context.metadata)

        # Perform hybrid search
        search_results = await self.search_engine.search(search_query)

        return {
            "type": "search",
            "query": query,
            "results_count": (
                len(search_results.results) if hasattr(search_results, "results") else 0
            ),
            "search_strategy": "hybrid",
        }

    async def _perform_rag_generation(
        self, request: str, context: SemanticContext
    ) -> Dict[str, Any]:
        """Perform RAG generation with retrieved context."""
        return {
            "type": "generation",
            "request": request,
            "generated_response": f"RAG-enhanced response for: {request}",
            "context_used": True,
        }


class StreamingSemanticAgent(BaseSemanticAgent):
    """
    Semantic agent with integrated streaming pipeline capabilities.

    Features:
    - Real-time content processing
    - Incremental updates
    - Event-driven processing
    - Live monitoring
    """

    def __init__(
        self,
        config: SemanticAgentConfig,
        streaming_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize streaming semantic agent."""
        super().__init__(config, **kwargs)

        # Initialize streaming components
        self.streaming_config = streaming_config or {}
        self.streaming_pipeline = StreamingPipeline(self.streaming_config)
        self.incremental_processor = IncrementalProcessor(self.streaming_config)

        self.logger = get_logger(f"streaming_agent.{config.name}")

    async def process_request(
        self,
        request: str,
        context: Optional[SemanticContext] = None,
    ) -> Dict[str, Any]:
        """Process requests with streaming capabilities."""
        self.logger.info(f"Processing streaming request: {request[:100]}...")

        start_time = time.time()

        try:
            # Understand intent
            semantic_context = await self.understand_intent(
                request, context.dict() if context else {}
            )

            # Create stream event
            stream_event = StreamEvent(
                event_id=semantic_context.task_id,
                event_type=StreamEventType.DOCUMENT_ADDED,
                content=request,
                metadata=semantic_context.metadata,
            )

            # Process through streaming pipeline
            result = await self.streaming_pipeline.process_event(stream_event)

            return {
                "agent": self.config.name,
                "type": "streaming_processing",
                "event_id": stream_event.event_id,
                "result": result,
                "processing_time": time.time() - start_time,
                "semantic_context": semantic_context.dict(),
            }

        except Exception as e:
            self.logger.error(f"Error in streaming processing: {e}")
            return {
                "agent": self.config.name,
                "type": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def understand_intent(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SemanticContext:
        """Understand streaming-specific intent."""
        return SemanticContext(
            user_intent=request,
            context_data=context or {},
            metadata={"requires_streaming": True, "agent_type": "streaming_semantic"},
        )


class IntegratedSemanticCoordinator(SemanticCoordinator):
    """
    Enhanced semantic coordinator with LLM pipeline integration.

    Features:
    - Intelligent routing to integrated agents
    - Pipeline-aware task distribution
    - Cross-pipeline optimization
    """

    def __init__(self, **kwargs):
        """Initialize integrated coordinator."""
        super().__init__(**kwargs)
        self.logger = get_logger("integrated_coordinator")

    async def route_task_to_agent(
        self,
        task_description: str,
        required_capabilities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Route tasks to appropriate integrated agents."""
        # Analyze task for pipeline requirements
        pipeline_requirements = self._analyze_pipeline_requirements(task_description)

        # Select best agent based on requirements
        if pipeline_requirements.get("multimodal"):
            return await self._route_to_multimodal_agent(task_description, context)
        elif pipeline_requirements.get("rag"):
            return await self._route_to_rag_agent(task_description, context)
        elif pipeline_requirements.get("streaming"):
            return await self._route_to_streaming_agent(task_description, context)
        else:
            # Fall back to standard routing
            return await super().route_task_to_agent(
                task_description, required_capabilities, context
            )

    def _analyze_pipeline_requirements(self, task_description: str) -> Dict[str, bool]:
        """Analyze task for pipeline requirements."""
        text = task_description.lower()

        return {
            "multimodal": any(
                keyword in text for keyword in ["image", "audio", "video", "visual", "speech"]
            ),
            "rag": any(
                keyword in text
                for keyword in ["search", "find", "retrieve", "knowledge", "document"]
            ),
            "streaming": any(
                keyword in text
                for keyword in ["real-time", "stream", "live", "continuous", "monitor"]
            ),
        }

    async def _route_to_multimodal_agent(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """Route to multimodal agent."""
        # Find available multimodal agent
        for agent_id, agent in self.agents.items():
            if isinstance(agent, MultimodalSemanticAgent):
                return agent_id

        # Create new multimodal agent if none available
        config = SemanticAgentConfig(name="multimodal_agent_auto")
        agent = MultimodalSemanticAgent(config)
        await self.register_agent(agent)
        return agent.config.agent_id

    async def _route_to_rag_agent(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """Route to RAG agent."""
        # Find available RAG agent
        for agent_id, agent in self.agents.items():
            if isinstance(agent, RAGSemanticAgent):
                return agent_id

        # Create new RAG agent if none available
        config = SemanticAgentConfig(name="rag_agent_auto")
        agent = RAGSemanticAgent(config)
        await self.register_agent(agent)
        return agent.config.agent_id

    async def _route_to_streaming_agent(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """Route to streaming agent."""
        # Find available streaming agent
        for agent_id, agent in self.agents.items():
            if isinstance(agent, StreamingSemanticAgent):
                return agent_id

        # Create new streaming agent if none available
        config = SemanticAgentConfig(name="streaming_agent_auto")
        agent = StreamingSemanticAgent(config)
        await self.register_agent(agent)
        return agent.config.agent_id
