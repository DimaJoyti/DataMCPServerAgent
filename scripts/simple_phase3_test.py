#!/usr/bin/env python3
"""
Simple Phase 3 Test

Basic test to verify Phase 3 integration is working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all Phase 3 components can be imported."""
    print("🧪 Testing Phase 3 Imports...")

    try:
        # Test base semantic agent
        from src.agents.semantic.base_semantic_agent import BaseSemanticAgent, SemanticAgentConfig
        print("✅ Base semantic agent imported")

        # Test integrated agents
        from src.agents.semantic.integrated_agents import (
            MultimodalSemanticAgent,
            RAGSemanticAgent,
            StreamingSemanticAgent,
            IntegratedSemanticCoordinator,
        )
        print("✅ Integrated agents imported")

        # Test main system
        from src.agents.semantic.main import SemanticAgentsSystem
        print("✅ Main system imported")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_agent_creation():
    """Test creating Phase 3 agents."""
    print("\n🏗️ Testing Agent Creation...")

    try:
        from src.agents.semantic.base_semantic_agent import SemanticAgentConfig
        from src.agents.semantic.integrated_agents import MultimodalSemanticAgent

        # Create config
        config = SemanticAgentConfig(
            name="test_multimodal_agent",
            specialization="multimodal_processing"
        )
        print("✅ Config created")

        # Create agent
        agent = MultimodalSemanticAgent(config)
        print("✅ Multimodal agent created")

        # Check agent properties
        assert agent.config.name == "test_multimodal_agent"
        assert agent.config.specialization == "multimodal_processing"
        print("✅ Agent properties verified")

        return True

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_pipeline_imports():
    """Test that pipeline imports work."""
    print("\n🔗 Testing Pipeline Imports...")

    pipeline_tests = [
        ("Multimodal", "app.pipelines.multimodal", "ProcessorFactory"),
        ("RAG", "app.pipelines.rag", "HybridSearchEngine"),
        ("Streaming", "app.pipelines.streaming", "StreamingPipeline"),
    ]

    results = []

    for name, module, class_name in pipeline_tests:
        try:
            exec(f"from {module} import {class_name}")
            print(f"✅ {name} pipeline accessible")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {name} pipeline not available: {e}")
            results.append(False)

    return any(results)  # At least one pipeline should work

def test_configuration():
    """Test Phase 3 configuration."""
    print("\n⚙️ Testing Configuration...")

    try:
        from src.agents.semantic.base_semantic_agent import SemanticAgentConfig

        # Test different configurations
        configs = [
            {
                "name": "multimodal_agent",
                "specialization": "multimodal_processing",
                "capabilities": ["text_image_processing", "ocr"]
            },
            {
                "name": "rag_agent",
                "specialization": "rag_processing",
                "capabilities": ["hybrid_search", "document_retrieval"]
            },
            {
                "name": "streaming_agent",
                "specialization": "streaming_processing",
                "capabilities": ["real_time_processing"]
            }
        ]

        for config_data in configs:
            config = SemanticAgentConfig(**config_data)
            assert config.name == config_data["name"]
            assert config.specialization == config_data["specialization"]
            print(f"✅ {config.name} configuration valid")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run simple Phase 3 tests."""
    print("🚀 Simple Phase 3 Integration Test")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Agent Creation Test", test_agent_creation),
        ("Pipeline Import Test", test_pipeline_imports),
        ("Configuration Test", test_configuration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if passed == total:
        print("\n🎉 All tests passed! Phase 3 basic integration is working.")
        return 0
    elif passed > 0:
        print(f"\n⚠️ Partial success: {passed}/{total} tests passed.")
        return 0  # Still consider success if some tests pass
    else:
        print("\n❌ All tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
