"""
Tests for Semantic Agents System

Comprehensive test suite for semantic agents, coordination, communication,
and performance monitoring.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.semantic.base_semantic_agent import (
    BaseSemanticAgent,
    SemanticAgentConfig,
    SemanticContext,
)
from src.agents.semantic.communication import (
    AgentMessage,
    MessageBus,
    MessageType,
    MessagePriority,
    AgentCommunicationHub,
)
from src.agents.semantic.coordinator import SemanticCoordinator
from src.agents.semantic.performance import PerformanceTracker, CacheManager
from src.agents.semantic.scaling import AutoScaler, LoadBalancer
from src.agents.semantic.specialized_agents import (
    DataAnalysisAgent,
    DocumentProcessingAgent,
    KnowledgeExtractionAgent,
    ReasoningAgent,
    SearchAgent,
)

class TestSemanticAgent(BaseSemanticAgent):
    """Test implementation of BaseSemanticAgent."""

    async def process_request(self, request: str, context=None):
        """Test implementation of process_request."""
        return {
            "success": True,
            "result": f"Processed: {request}",
            "agent": self.config.name,
        }

    async def understand_intent(self, request: str, context=None):
        """Test implementation of understand_intent."""
        return SemanticContext(
            user_intent=request,
            context_data=context or {},
        )

@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return SemanticAgentConfig(
        name="test_agent",
        specialization="testing",
        capabilities=["test_capability_1", "test_capability_2"],
    )

@pytest.fixture
async def test_agent(agent_config):
    """Create a test semantic agent."""
    agent = TestSemanticAgent(agent_config)
    await agent.initialize()
    yield agent
    await agent.shutdown()

@pytest.fixture
def message_bus():
    """Create a message bus for testing."""
    return MessageBus()

@pytest.fixture
async def coordinator(message_bus):
    """Create a semantic coordinator for testing."""
    coordinator = SemanticCoordinator(message_bus=message_bus)
    await coordinator.initialize()
    yield coordinator
    await coordinator.shutdown()

@pytest.fixture
def performance_tracker():
    """Create a performance tracker for testing."""
    return PerformanceTracker()

@pytest.fixture
def cache_manager():
    """Create a cache manager for testing."""
    return CacheManager()

class TestBaseSemanticAgent:
    """Test cases for BaseSemanticAgent."""

    def test_agent_config_creation(self, agent_config):
        """Test agent configuration creation."""
        assert agent_config.name == "test_agent"
        assert agent_config.specialization == "testing"
        assert "test_capability_1" in agent_config.capabilities

    @pytest.mark.asyncio
    async def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.is_active
        assert test_agent.config.name == "test_agent"

    @pytest.mark.asyncio
    async def test_agent_process_request(self, test_agent):
        """Test agent request processing."""
        result = await test_agent.process_request("test request")

        assert result["success"]
        assert "Processed: test request" in result["result"]
        assert result["agent"] == "test_agent"

    @pytest.mark.asyncio
    async def test_agent_understand_intent(self, test_agent):
        """Test agent intent understanding."""
        context = await test_agent.understand_intent("test request")

        assert context.user_intent == "test request"
        assert isinstance(context.context_data, dict)

class TestMessageBus:
    """Test cases for MessageBus."""

    @pytest.mark.asyncio
    async def test_message_publishing(self, message_bus):
        """Test message publishing."""
        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type=MessageType.TASK_REQUEST,
            data={"test": "data"},
        )

        # Mock handler
        handler_called = False

        async def mock_handler(msg):
            nonlocal handler_called
            handler_called = True
            assert msg.sender_id == "agent_1"
            assert msg.data["test"] == "data"

        # Subscribe mock handler
        from src.agents.semantic.communication import MessageHandler
        handler = MessageHandler(
            handler_func=mock_handler,
            message_types={MessageType.TASK_REQUEST},
        )

        await message_bus.subscribe("agent_2", handler)
        await message_bus.publish(message)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        assert handler_called

    @pytest.mark.asyncio
    async def test_topic_subscription(self, message_bus):
        """Test topic-based message subscription."""
        await message_bus.subscribe_topic("agent_1", "test_topic")

        assert "test_topic" in message_bus.topic_subscribers
        assert "agent_1" in message_bus.topic_subscribers["test_topic"]

class TestSemanticCoordinator:
    """Test cases for SemanticCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.is_running
        assert len(coordinator.registered_agents) == 0

    @pytest.mark.asyncio
    async def test_agent_registration(self, coordinator, test_agent):
        """Test agent registration with coordinator."""
        await coordinator.register_agent(test_agent)

        assert test_agent.config.agent_id in coordinator.registered_agents
        assert coordinator.agent_workloads[test_agent.config.agent_id] == 0

    @pytest.mark.asyncio
    async def test_task_execution(self, coordinator, test_agent):
        """Test task execution through coordinator."""
        await coordinator.register_agent(test_agent)

        result = await coordinator.execute_task(
            task_description="test task",
            required_capabilities=["test_capability_1"],
        )

        assert result["success"]
        assert result["agent_id"] == test_agent.config.agent_id

class TestPerformanceTracker:
    """Test cases for PerformanceTracker."""

    def test_operation_tracking(self, performance_tracker):
        """Test operation performance tracking."""
        operation_id = performance_tracker.start_operation(
            agent_id="test_agent",
            operation_type="test_operation",
        )

        assert operation_id in performance_tracker.active_operations

        metrics = performance_tracker.end_operation(
            operation_id,
            success=True,
        )

        assert metrics is not None
        assert metrics.success
        assert metrics.duration_ms is not None

    def test_agent_performance_stats(self, performance_tracker):
        """Test agent performance statistics."""
        # Add some test metrics
        operation_id = performance_tracker.start_operation(
            agent_id="test_agent",
            operation_type="test_operation",
        )

        performance_tracker.end_operation(operation_id, success=True)

        stats = performance_tracker.get_agent_performance("test_agent")

        assert stats["agent_id"] == "test_agent"
        assert stats["total_operations"] >= 1
        assert stats["success_rate"] >= 0

class TestCacheManager:
    """Test cases for CacheManager."""

    @pytest.mark.asyncio
    async def test_cache_operations(self, cache_manager):
        """Test basic cache operations."""
        # Test set and get
        await cache_manager.set("test_key", "test_value")
        value = await cache_manager.get("test_key")

        assert value == "test_value"

        # Test delete
        deleted = await cache_manager.delete("test_key")
        assert deleted

        # Test get after delete
        value = await cache_manager.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_ttl(self, cache_manager):
        """Test cache TTL functionality."""
        # Set with short TTL
        await cache_manager.set("ttl_key", "ttl_value", ttl=1)

        # Should be available immediately
        value = await cache_manager.get("ttl_key")
        assert value == "ttl_value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        value = await cache_manager.get("ttl_key")
        assert value is None

    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        stats = cache_manager.get_stats()

        assert "size" in stats
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats

class TestLoadBalancer:
    """Test cases for LoadBalancer."""

    def test_agent_registration(self):
        """Test agent registration with load balancer."""
        lb = LoadBalancer()

        lb.register_agent("agent_1", weight=1.0)
        lb.register_agent("agent_2", weight=2.0)

        assert "agent_1" in lb.agent_weights
        assert "agent_2" in lb.agent_weights
        assert lb.agent_weights["agent_2"] == 2.0

    def test_agent_selection(self):
        """Test agent selection algorithms."""
        lb = LoadBalancer()

        lb.register_agent("agent_1", weight=1.0)
        lb.register_agent("agent_2", weight=2.0)

        # Test round-robin
        agent = lb.select_agent(strategy="round_robin")
        assert agent in ["agent_1", "agent_2"]

        # Test weighted
        agent = lb.select_agent(strategy="weighted")
        assert agent in ["agent_1", "agent_2"]

class TestSpecializedAgents:
    """Test cases for specialized agents."""

    @pytest.mark.asyncio
    async def test_data_analysis_agent(self):
        """Test DataAnalysisAgent."""
        config = SemanticAgentConfig(
            name="data_analysis_test",
            specialization="data_analysis",
        )

        agent = DataAnalysisAgent(config)
        await agent.initialize()

        try:
            result = await agent.process_request("analyze sales data")

            assert "analysis_type" in result
            assert "result" in result
            assert result["agent"] == "data_analysis_test"
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_document_processing_agent(self):
        """Test DocumentProcessingAgent."""
        config = SemanticAgentConfig(
            name="document_processing_test",
            specialization="document_processing",
        )

        agent = DocumentProcessingAgent(config)
        await agent.initialize()

        try:
            result = await agent.process_request("summarize this document")

            assert "processing_type" in result
            assert "result" in result
            assert result["agent"] == "document_processing_test"
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_knowledge_extraction_agent(self):
        """Test KnowledgeExtractionAgent."""
        config = SemanticAgentConfig(
            name="knowledge_extraction_test",
            specialization="knowledge_extraction",
        )

        agent = KnowledgeExtractionAgent(config)
        await agent.initialize()

        try:
            result = await agent.process_request("extract concepts from text")

            assert "concepts" in result
            assert "relationships" in result
            assert result["agent"] == "knowledge_extraction_test"
        finally:
            await agent.shutdown()

class TestIntegration:
    """Integration tests for the semantic agents system."""

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration."""
        # Create components
        message_bus = MessageBus()
        coordinator = SemanticCoordinator(message_bus=message_bus)
        performance_tracker = PerformanceTracker()

        await coordinator.initialize()

        try:
            # Create and register agents
            config = SemanticAgentConfig(
                name="integration_test_agent",
                specialization="testing",
                capabilities=["integration_test"],
            )

            agent = TestSemanticAgent(config)
            await agent.initialize()

            await coordinator.register_agent(agent)

            # Execute task
            result = await coordinator.execute_task(
                task_description="integration test task",
                required_capabilities=["integration_test"],
            )

            assert result["success"]
            assert result["agent_id"] == agent.config.agent_id

            # Check performance tracking
            stats = performance_tracker.get_system_performance()
            assert "timestamp" in stats

        finally:
            await coordinator.shutdown()

if __name__ == "__main__":
    pytest.main([__file__])
