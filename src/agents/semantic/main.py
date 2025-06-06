"""
Semantic Agents Main Entry Point

Initializes and runs the semantic agents system with all components:
- Semantic agents
- Coordinator
- Performance monitoring
- Auto-scaling
- API server
"""

import asyncio
import logging
import signal
import sys
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph_manager import KnowledgeGraphManager
from src.tools.bright_data_tools import BrightDataToolkit

from .api import initialize_semantic_agents_api, router
from .base_semantic_agent import SemanticAgentConfig
from .communication import MessageBus
from .coordinator import SemanticCoordinator
from .integrated_agents import (
    IntegratedSemanticCoordinator,
    MultimodalSemanticAgent,
    RAGSemanticAgent,
    StreamingSemanticAgent,
)
from .performance import CacheManager, PerformanceTracker
from .scaling import AutoScaler, LoadBalancer
from .specialized_agents import (
    DataAnalysisAgent,
    DocumentProcessingAgent,
    KnowledgeExtractionAgent,
    ReasoningAgent,
    SearchAgent,
)


class SemanticAgentsSystem:
    """
    Main system class for semantic agents.

    Manages the lifecycle of all components and provides a unified interface
    for running the semantic agents system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the semantic agents system."""
        self.config_path = config_path
        self.logger = logging.getLogger("semantic_agents_system")

        # Core components
        self.message_bus: Optional[MessageBus] = None
        self.coordinator: Optional[SemanticCoordinator] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.cache_manager: Optional[CacheManager] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.auto_scaler: Optional[AutoScaler] = None

        # External dependencies
        self.model: Optional[ChatAnthropic] = None
        self.tools: List[BaseTool] = []
        self.memory_manager: Optional[DistributedMemoryManager] = None
        self.knowledge_graph: Optional[KnowledgeGraphManager] = None

        # State
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing semantic agents system")

        # Load environment variables
        load_dotenv()

        # Initialize external dependencies
        await self._initialize_dependencies()

        # Initialize core components
        await self._initialize_core_components()

        # Initialize agents
        await self._initialize_agents()

        # Initialize API
        await self._initialize_api()

        self.is_running = True
        self.logger.info("Semantic agents system initialized successfully")

    async def _initialize_dependencies(self) -> None:
        """Initialize external dependencies."""
        self.logger.info("Initializing dependencies")

        # Initialize language model
        self.model = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.1,
            max_tokens=4000,
        )

        # Initialize tools
        await self._load_tools()

        # Initialize memory manager
        self.memory_manager = DistributedMemoryManager()
        await self.memory_manager.initialize()

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraphManager()
        await self.knowledge_graph.initialize()

    async def _load_tools(self) -> None:
        """Load MCP tools and other tools."""
        try:
            # Load Bright Data tools
            bright_data_toolkit = BrightDataToolkit()
            self.tools.extend(bright_data_toolkit.get_tools())

            # Load MCP tools (if available)
            try:
                server_params = StdioServerParameters(
                    command="npx",
                    args=["-y", "@bright-data/mcp-server"],
                )

                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        mcp_tools = await load_mcp_tools(session)
                        self.tools.extend(mcp_tools)

            except Exception as e:
                self.logger.warning(f"Could not load MCP tools: {e}")

        except Exception as e:
            self.logger.error(f"Error loading tools: {e}")

    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        self.logger.info("Initializing core components")

        # Initialize message bus
        self.message_bus = MessageBus()

        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()

        # Initialize cache manager
        self.cache_manager = CacheManager()

        # Initialize load balancer
        self.load_balancer = LoadBalancer()

        # Initialize integrated coordinator (Phase 3)
        self.coordinator = IntegratedSemanticCoordinator(message_bus=self.message_bus)
        await self.coordinator.initialize()

        # Initialize auto scaler
        self.auto_scaler = AutoScaler(
            coordinator=self.coordinator,
            performance_tracker=self.performance_tracker,
            load_balancer=self.load_balancer,
        )
        await self.auto_scaler.initialize()

    async def _initialize_agents(self) -> None:
        """Initialize semantic agents."""
        self.logger.info("Initializing semantic agents")

        # Create agent configurations
        agent_configs = [
            {
                "type": DataAnalysisAgent,
                "config": SemanticAgentConfig(
                    name="data_analysis_agent",
                    specialization="data_analysis",
                    capabilities=[
                        "statistical_analysis",
                        "data_visualization",
                        "pattern_recognition",
                    ],
                ),
            },
            {
                "type": DocumentProcessingAgent,
                "config": SemanticAgentConfig(
                    name="document_processing_agent",
                    specialization="document_processing",
                    capabilities=[
                        "document_parsing",
                        "content_summarization",
                        "entity_extraction",
                    ],
                ),
            },
            {
                "type": KnowledgeExtractionAgent,
                "config": SemanticAgentConfig(
                    name="knowledge_extraction_agent",
                    specialization="knowledge_extraction",
                    capabilities=[
                        "concept_extraction",
                        "relationship_identification",
                        "knowledge_graph_construction",
                    ],
                ),
            },
            {
                "type": ReasoningAgent,
                "config": SemanticAgentConfig(
                    name="reasoning_agent",
                    specialization="reasoning",
                    capabilities=[
                        "logical_inference",
                        "causal_reasoning",
                        "problem_decomposition",
                    ],
                ),
            },
            {
                "type": SearchAgent,
                "config": SemanticAgentConfig(
                    name="search_agent",
                    specialization="search",
                    capabilities=[
                        "semantic_search",
                        "query_expansion",
                        "result_ranking",
                    ],
                ),
            },
            # Phase 3: Integrated agents with LLM pipelines
            {
                "type": MultimodalSemanticAgent,
                "config": SemanticAgentConfig(
                    name="multimodal_agent",
                    specialization="multimodal_processing",
                    capabilities=[
                        "text_image_processing",
                        "text_audio_processing",
                        "cross_modal_analysis",
                        "ocr",
                        "speech_recognition",
                    ],
                ),
            },
            {
                "type": RAGSemanticAgent,
                "config": SemanticAgentConfig(
                    name="rag_agent",
                    specialization="rag_processing",
                    capabilities=[
                        "hybrid_search",
                        "document_retrieval",
                        "context_generation",
                        "adaptive_chunking",
                    ],
                ),
            },
            {
                "type": StreamingSemanticAgent,
                "config": SemanticAgentConfig(
                    name="streaming_agent",
                    specialization="streaming_processing",
                    capabilities=[
                        "real_time_processing",
                        "incremental_updates",
                        "event_driven_processing",
                        "live_monitoring",
                    ],
                ),
            },
        ]

        # Create and register agents
        for agent_config in agent_configs:
            agent_class = agent_config["type"]
            config = agent_config["config"]

            # Create agent
            agent = agent_class(
                config=config,
                tools=self.tools,
                memory_manager=self.memory_manager,
                knowledge_graph=self.knowledge_graph,
            )

            # Initialize agent
            await agent.initialize()

            # Register with coordinator
            await self.coordinator.register_agent(agent)

            # Register with load balancer
            self.load_balancer.register_agent(agent.config.agent_id)

            self.logger.info(f"Initialized agent: {config.name}")

    async def _initialize_api(self) -> None:
        """Initialize the API server."""
        self.logger.info("Initializing API")

        await initialize_semantic_agents_api(
            coordinator=self.coordinator,
            performance_tracker=self.performance_tracker,
            auto_scaler=self.auto_scaler,
            load_balancer=self.load_balancer,
            cache_manager=self.cache_manager,
        )

    async def run(self) -> None:
        """Run the semantic agents system."""
        if not self.is_running:
            await self.initialize()

        self.logger.info("Starting semantic agents system")

        # Set up signal handlers
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the semantic agents system."""
        if not self.is_running:
            return

        self.logger.info("Shutting down semantic agents system")

        self.is_running = False

        # Shutdown components in reverse order
        if self.auto_scaler:
            await self.auto_scaler.shutdown()

        if self.coordinator:
            await self.coordinator.shutdown()

        if self.memory_manager:
            await self.memory_manager.cleanup()

        if self.knowledge_graph:
            await self.knowledge_graph.cleanup()

        self.logger.info("Semantic agents system shutdown complete")

    def get_fastapi_app(self) -> FastAPI:
        """Get FastAPI app with semantic agents router."""
        app = FastAPI(
            title="Semantic Agents API",
            description="Advanced semantic agents with inter-agent communication",
            version="1.0.0",
        )

        app.include_router(router)

        return app


# Main entry point
async def main():
    """Main entry point for the semantic agents system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run system
    system = SemanticAgentsSystem()

    try:
        await system.run()
    except Exception as e:
        logging.error(f"Error running semantic agents system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
