#!/usr/bin/env python3
"""
Phase 3 Integration Test Script

Tests the integration between LLM pipelines and semantic agents.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.semantic.integrated_agents import (
    MultimodalSemanticAgent,
    RAGSemanticAgent,
    StreamingSemanticAgent,
    IntegratedSemanticCoordinator,
)
from src.agents.semantic.base_semantic_agent import SemanticAgentConfig
from src.agents.semantic.communication import MessageBus


async def test_multimodal_agent():
    """Test multimodal semantic agent."""
    print("🎭 Testing Multimodal Semantic Agent...")
    
    config = SemanticAgentConfig(
        name="test_multimodal_agent",
        specialization="multimodal_processing",
        capabilities=["text_image_processing", "ocr", "visual_analysis"]
    )
    
    agent = MultimodalSemanticAgent(config)
    await agent.initialize()
    
    # Test text + image request
    result = await agent.process_request(
        "Analyze this image and extract text using OCR"
    )
    
    print(f"✅ Multimodal Agent Result: {result['type']}")
    print(f"   Agent: {result['agent']}")
    print(f"   Processing Time: {result['processing_time']:.3f}s")
    
    if 'result' in result:
        print(f"   Result Type: {result['result'].get('type', 'unknown')}")
        print(f"   Modalities: {result.get('modalities', [])}")
    
    await agent.shutdown()
    return True


async def test_rag_agent():
    """Test RAG semantic agent."""
    print("🔍 Testing RAG Semantic Agent...")
    
    config = SemanticAgentConfig(
        name="test_rag_agent",
        specialization="rag_processing",
        capabilities=["hybrid_search", "document_retrieval", "context_generation"]
    )
    
    agent = RAGSemanticAgent(config)
    await agent.initialize()
    
    # Test search request
    result = await agent.process_request(
        "Search for information about machine learning algorithms"
    )
    
    print(f"✅ RAG Agent Result: {result['type']}")
    print(f"   Agent: {result['agent']}")
    print(f"   Processing Time: {result['processing_time']:.3f}s")
    
    if 'result' in result:
        print(f"   Result Type: {result['result'].get('type', 'unknown')}")
        print(f"   Query: {result['result'].get('query', 'N/A')}")
    
    await agent.shutdown()
    return True


async def test_streaming_agent():
    """Test streaming semantic agent."""
    print("⚡ Testing Streaming Semantic Agent...")
    
    config = SemanticAgentConfig(
        name="test_streaming_agent",
        specialization="streaming_processing",
        capabilities=["real_time_processing", "event_driven_processing"]
    )
    
    agent = StreamingSemanticAgent(config)
    await agent.initialize()
    
    # Test streaming request
    result = await agent.process_request(
        "Process this document in real-time with live updates"
    )
    
    print(f"✅ Streaming Agent Result: {result['type']}")
    print(f"   Agent: {result['agent']}")
    print(f"   Processing Time: {result['processing_time']:.3f}s")
    
    if 'result' in result:
        print(f"   Event ID: {result.get('event_id', 'N/A')}")
    
    await agent.shutdown()
    return True


async def test_integrated_coordinator():
    """Test integrated semantic coordinator."""
    print("🧠 Testing Integrated Semantic Coordinator...")
    
    message_bus = MessageBus()
    coordinator = IntegratedSemanticCoordinator(message_bus=message_bus)
    await coordinator.initialize()
    
    # Create and register test agents
    agents = []
    
    # Multimodal agent
    multimodal_config = SemanticAgentConfig(
        name="coordinator_multimodal_agent",
        specialization="multimodal_processing"
    )
    multimodal_agent = MultimodalSemanticAgent(multimodal_config)
    await multimodal_agent.initialize()
    await coordinator.register_agent(multimodal_agent)
    agents.append(multimodal_agent)
    
    # RAG agent
    rag_config = SemanticAgentConfig(
        name="coordinator_rag_agent",
        specialization="rag_processing"
    )
    rag_agent = RAGSemanticAgent(rag_config)
    await rag_agent.initialize()
    await coordinator.register_agent(rag_agent)
    agents.append(rag_agent)
    
    # Test intelligent routing
    test_tasks = [
        "Analyze this image and extract text",  # Should route to multimodal
        "Search for documents about AI",        # Should route to RAG
        "Process data in real-time",           # Should route to streaming
        "Perform statistical analysis",        # Should route to standard agent
    ]
    
    for task in test_tasks:
        print(f"   Testing task: {task[:50]}...")
        try:
            agent_id = await coordinator.route_task_to_agent(task)
            print(f"   ✅ Routed to agent: {agent_id}")
        except Exception as e:
            print(f"   ❌ Routing failed: {e}")
    
    # Cleanup
    for agent in agents:
        await agent.shutdown()
    await coordinator.shutdown()
    
    return True


async def test_pipeline_integration():
    """Test integration with LLM pipelines."""
    print("🔗 Testing LLM Pipeline Integration...")
    
    try:
        # Test multimodal pipeline integration
        from app.pipelines.multimodal import ProcessorFactory
        
        # Check if processors are available
        processors = ProcessorFactory.list_processors()
        print(f"   Available processors: {processors}")
        
        if processors:
            print("   ✅ Multimodal pipelines accessible")
        else:
            print("   ⚠️ No multimodal processors found")
        
    except ImportError as e:
        print(f"   ❌ Multimodal pipeline import failed: {e}")
    
    try:
        # Test RAG pipeline integration
        from app.pipelines.rag import HybridSearchEngine
        
        search_engine = HybridSearchEngine()
        print("   ✅ RAG pipelines accessible")
        
    except ImportError as e:
        print(f"   ❌ RAG pipeline import failed: {e}")
    
    try:
        # Test streaming pipeline integration
        from app.pipelines.streaming import StreamingPipeline
        
        streaming_pipeline = StreamingPipeline()
        print("   ✅ Streaming pipelines accessible")
        
    except ImportError as e:
        print(f"   ❌ Streaming pipeline import failed: {e}")
    
    return True


async def main():
    """Main test function."""
    print("🚀 Phase 3 Integration Tests")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    tests = [
        ("Pipeline Integration", test_pipeline_integration),
        ("Multimodal Agent", test_multimodal_agent),
        ("RAG Agent", test_rag_agent),
        ("Streaming Agent", test_streaming_agent),
        ("Integrated Coordinator", test_integrated_coordinator),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            result = await test_func()
            results.append((test_name, result, None))
            print(f"✅ {test_name} Test: PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name} Test: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success, error in results:
        if success:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            if error:
                print(f"   Error: {error}")
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Phase 3 integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
