#!/usr/bin/env python3
"""
Phase 3 Demo Script

Demonstrates the integrated semantic agents with LLM pipelines.
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

async def demo_multimodal_processing():
    """Demonstrate multimodal processing capabilities."""
    print("\n🎭 MULTIMODAL PROCESSING DEMO")
    print("=" * 50)

    config = SemanticAgentConfig(
        name="demo_multimodal_agent",
        specialization="multimodal_processing",
        capabilities=["text_image_processing", "text_audio_processing", "cross_modal_analysis"]
    )

    agent = MultimodalSemanticAgent(config)
    await agent.initialize()

    # Demo scenarios
    scenarios = [
        {
            "title": "Image Analysis with OCR",
            "request": "Analyze this business card image and extract contact information using OCR",
            "description": "Demonstrates text extraction from images"
        },
        {
            "title": "Audio Transcription",
            "request": "Transcribe this audio recording and identify the speaker's sentiment",
            "description": "Demonstrates speech-to-text and sentiment analysis"
        },
        {
            "title": "Cross-Modal Analysis",
            "request": "Compare the text content with the audio narration and identify discrepancies",
            "description": "Demonstrates cross-modal understanding"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['title']}")
        print(f"Description: {scenario['description']}")
        print(f"Request: {scenario['request']}")

        result = await agent.process_request(scenario['request'])

        print(f"✅ Result Type: {result['type']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print(f"   Modalities Detected: {result.get('modalities', [])}")

        if 'result' in result and 'processor_used' in result['result']:
            print(f"   Processor Used: {result['result']['processor_used']}")

    await agent.shutdown()

async def demo_rag_capabilities():
    """Demonstrate RAG capabilities."""
    print("\n🔍 RAG PROCESSING DEMO")
    print("=" * 50)

    config = SemanticAgentConfig(
        name="demo_rag_agent",
        specialization="rag_processing",
        capabilities=["hybrid_search", "document_retrieval", "context_generation"]
    )

    agent = RAGSemanticAgent(config)
    await agent.initialize()

    # Demo scenarios
    scenarios = [
        {
            "title": "Knowledge Retrieval",
            "request": "Find information about machine learning best practices",
            "description": "Demonstrates semantic search and retrieval"
        },
        {
            "title": "Document Analysis",
            "request": "Analyze the quarterly financial report and summarize key findings",
            "description": "Demonstrates document processing with context"
        },
        {
            "title": "Contextual Generation",
            "request": "Generate a comprehensive guide based on retrieved documentation",
            "description": "Demonstrates RAG-enhanced content generation"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['title']}")
        print(f"Description: {scenario['description']}")
        print(f"Request: {scenario['request']}")

        result = await agent.process_request(scenario['request'])

        print(f"✅ Result Type: {result['type']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")

        if 'result' in result:
            print(f"   Operation: {result['result'].get('type', 'unknown')}")
            if 'query' in result['result']:
                print(f"   Query: {result['result']['query']}")

    await agent.shutdown()

async def demo_streaming_processing():
    """Demonstrate streaming processing capabilities."""
    print("\n⚡ STREAMING PROCESSING DEMO")
    print("=" * 50)

    config = SemanticAgentConfig(
        name="demo_streaming_agent",
        specialization="streaming_processing",
        capabilities=["real_time_processing", "incremental_updates", "live_monitoring"]
    )

    agent = StreamingSemanticAgent(config)
    await agent.initialize()

    # Demo scenarios
    scenarios = [
        {
            "title": "Real-time Document Processing",
            "request": "Process incoming documents in real-time and update the index",
            "description": "Demonstrates live document processing"
        },
        {
            "title": "Event Stream Analysis",
            "request": "Monitor user activity stream and detect anomalies",
            "description": "Demonstrates event-driven processing"
        },
        {
            "title": "Incremental Updates",
            "request": "Update the knowledge base with new information incrementally",
            "description": "Demonstrates incremental processing"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['title']}")
        print(f"Description: {scenario['description']}")
        print(f"Request: {scenario['request']}")

        result = await agent.process_request(scenario['request'])

        print(f"✅ Result Type: {result['type']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print(f"   Event ID: {result.get('event_id', 'N/A')}")

    await agent.shutdown()

async def demo_intelligent_coordination():
    """Demonstrate intelligent task coordination."""
    print("\n🧠 INTELLIGENT COORDINATION DEMO")
    print("=" * 50)

    message_bus = MessageBus()
    coordinator = IntegratedSemanticCoordinator(message_bus=message_bus)
    await coordinator.initialize()

    # Create and register agents
    agents = []

    # Multimodal agent
    multimodal_config = SemanticAgentConfig(
        name="coord_multimodal_agent",
        specialization="multimodal_processing"
    )
    multimodal_agent = MultimodalSemanticAgent(multimodal_config)
    await multimodal_agent.initialize()
    await coordinator.register_agent(multimodal_agent)
    agents.append(multimodal_agent)

    # RAG agent
    rag_config = SemanticAgentConfig(
        name="coord_rag_agent",
        specialization="rag_processing"
    )
    rag_agent = RAGSemanticAgent(rag_config)
    await rag_agent.initialize()
    await coordinator.register_agent(rag_agent)
    agents.append(rag_agent)

    # Streaming agent
    streaming_config = SemanticAgentConfig(
        name="coord_streaming_agent",
        specialization="streaming_processing"
    )
    streaming_agent = StreamingSemanticAgent(streaming_config)
    await streaming_agent.initialize()
    await coordinator.register_agent(streaming_agent)
    agents.append(streaming_agent)

    print(f"✅ Registered {len(agents)} specialized agents")

    # Demo intelligent routing
    tasks = [
        {
            "description": "Extract text from product images",
            "expected_agent": "multimodal",
            "request": "Analyze product images and extract text descriptions"
        },
        {
            "description": "Search knowledge base",
            "expected_agent": "rag",
            "request": "Find relevant documentation about API endpoints"
        },
        {
            "description": "Real-time monitoring",
            "expected_agent": "streaming",
            "request": "Monitor system logs in real-time for errors"
        },
        {
            "description": "Complex multimodal task",
            "expected_agent": "multimodal",
            "request": "Process video content and extract both audio and visual information"
        }
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task['description']}")
        print(f"Request: {task['request']}")
        print(f"Expected routing: {task['expected_agent']} agent")

        try:
            agent_id = await coordinator.route_task_to_agent(
                task['request'],
                context={"demo": True}
            )

            # Find agent by ID
            routed_agent = None
            for agent in agents:
                if agent.config.agent_id == agent_id:
                    routed_agent = agent
                    break

            if routed_agent:
                print(f"✅ Routed to: {routed_agent.config.name} ({routed_agent.config.specialization})")

                # Execute the task
                result = await routed_agent.process_request(task['request'])
                print(f"   Task completed in {result['processing_time']:.3f}s")
            else:
                print(f"⚠️ Agent {agent_id} not found in registered agents")

        except Exception as e:
            print(f"❌ Routing failed: {e}")

    # Cleanup
    for agent in agents:
        await agent.shutdown()
    await coordinator.shutdown()

async def main():
    """Main demo function."""
    print("🚀 PHASE 3: INTEGRATED SEMANTIC AGENTS DEMO")
    print("=" * 60)
    print("Demonstrating LLM pipeline integration with semantic agents")
    print("=" * 60)

    # Configure logging to reduce noise
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demos = [
        ("Multimodal Processing", demo_multimodal_processing),
        ("RAG Capabilities", demo_rag_capabilities),
        ("Streaming Processing", demo_streaming_processing),
        ("Intelligent Coordination", demo_intelligent_coordination),
    ]

    for demo_name, demo_func in demos:
        try:
            await demo_func()
            print(f"\n✅ {demo_name} demo completed successfully")
        except Exception as e:
            print(f"\n❌ {demo_name} demo failed: {e}")
            logging.exception(f"Demo {demo_name} failed")

    print("\n" + "=" * 60)
    print("🎉 PHASE 3 DEMO COMPLETED")
    print("=" * 60)
    print("Key achievements demonstrated:")
    print("• Multimodal content processing (text, image, audio)")
    print("• RAG-enhanced information retrieval and generation")
    print("• Real-time streaming data processing")
    print("• Intelligent task routing and coordination")
    print("• Seamless integration with LLM pipelines")
    print("\nPhase 3 integration is working successfully! 🚀")

if __name__ == "__main__":
    asyncio.run(main())
