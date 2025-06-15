#!/usr/bin/env python3
"""
Test runner for the Advanced Agent Orchestration System.
This script runs comprehensive tests for all orchestration components.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.advanced_planning import AdvancedPlanningEngine, Condition
from src.agents.advanced_reasoning import AdvancedReasoningEngine
from src.agents.meta_reasoning import MetaReasoningEngine
from src.agents.reflection_systems import AdvancedReflectionEngine
from src.core.orchestration_main import OrchestrationCoordinator
from src.memory.memory_persistence import MemoryDatabase


async def test_advanced_reasoning():
    """Test the Advanced Reasoning Engine."""
    print("🧠 Testing Advanced Reasoning Engine...")

    # Mock model for testing
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = '{"step_type": "inference", "content": "Test reasoning step", "confidence": 85, "evidence": {"test": "evidence"}, "alternatives": ["alt1"], "dependencies": [], "should_backtrack": false}'
            return MockResponse()

    model = MockModel()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            engine = AdvancedReasoningEngine(model, db)

            # Test starting reasoning chain
            chain_id = await engine.start_reasoning_chain(
                goal="Test reasoning goal",
                initial_context={"test": "context"}
            )

            print(f"   ✅ Created reasoning chain: {chain_id}")

            # Test continuing reasoning
            step = await engine.continue_reasoning(chain_id)
            print(f"   ✅ Added reasoning step: {step.step_type.value}")

            # Test causal analysis
            causal_result = await engine.analyze_causal_relationships(
                scenario="Test scenario",
                context={"factor": "value"}
            )
            print("   ✅ Causal analysis completed")

            # Test counterfactual exploration
            counterfactual_result = await engine.explore_counterfactuals(
                situation="Test situation",
                facts={"fact1": "value1"}
            )
            print("   ✅ Counterfactual exploration completed")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 Advanced Reasoning Engine tests passed!\n")


async def test_meta_reasoning():
    """Test the Meta-Reasoning Engine."""
    print("🤔 Testing Meta-Reasoning Engine...")

    # Mock model for testing
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = '{"recommended_strategy": "chain_of_thought", "supporting_strategies": [], "rationale": "Best for analytical tasks", "expected_effectiveness": 85, "resource_requirements": 50}'
            return MockResponse()

    model = MockModel()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            reasoning_engine = AdvancedReasoningEngine(model, db)
            meta_engine = MetaReasoningEngine(model, db, reasoning_engine)

            # Test strategy selection
            strategy = await meta_engine.select_reasoning_strategy(
                problem="Complex analytical problem",
                problem_type="analytical"
            )

            print(f"   ✅ Strategy selected: {strategy['recommended_strategy']}")

            # Test performance monitoring
            mock_chain = type('MockChain', (), {
                'steps': [],
                'goal': 'Test goal'
            })()

            performance = await meta_engine.monitor_performance(mock_chain)
            print("   ✅ Performance monitoring completed")

            # Test error detection
            errors = await meta_engine.detect_errors(
                reasoning_steps=[{"step": "test"}],
                context={"test": "context"},
                goal="Test goal"
            )
            print("   ✅ Error detection completed")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 Meta-Reasoning Engine tests passed!\n")


async def test_advanced_planning():
    """Test the Advanced Planning Engine."""
    print("📋 Testing Advanced Planning Engine...")

    # Mock model for testing
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = '{"plan_actions": ["web_search", "analyze_data"], "action_details": {}, "state_progression": [], "plan_rationale": "Sequential execution plan", "estimated_cost": 5.0}'
            return MockResponse()

    model = MockModel()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            engine = AdvancedPlanningEngine(model, db)

            # Test STRIPS planning
            plan = await engine.create_strips_plan(
                goal="Complete research task",
                initial_state={"resources_available"},
                goal_conditions=[Condition("task_completed", ["research"])]
            )

            print(f"   ✅ STRIPS plan created with {len(plan.actions)} actions")

            # Test plan validation
            validation = engine.validate_plan(plan)
            print(f"   ✅ Plan validation: {'Valid' if validation['is_valid'] else 'Invalid'}")

            # Test temporal planning
            temporal_plan = await engine.create_temporal_plan(
                goal="Time-constrained task",
                available_actions=plan.actions,
                temporal_constraints=[],
                resource_constraints={}
            )
            print("   ✅ Temporal planning completed")

            # Test contingency planning
            contingency_plan = await engine.create_contingency_plan(
                main_plan=plan,
                risk_factors=[{"risk": "network_failure", "probability": 0.1}],
                failure_probabilities={"web_search": 0.05}
            )
            print("   ✅ Contingency planning completed")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 Advanced Planning Engine tests passed!\n")


async def test_reflection_system():
    """Test the Reflection System."""
    print("🪞 Testing Reflection System...")

    # Mock model for testing
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = '{"surface_observations": ["Good performance"], "analytical_insights": ["Strategy effective"], "critical_evaluation": ["Could improve speed"], "meta_cognitive_insights": ["Learning rate optimal"], "performance_patterns": ["Consistent accuracy"], "improvement_opportunities": ["Optimize memory usage"], "confidence_assessment": 80}'
            return MockResponse()

    model = MockModel()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            engine = AdvancedReflectionEngine(model, db)

            # Test reflection session
            session = await engine.trigger_reflection(
                trigger_event="Test completion",
                focus_areas=["performance", "strategy", "learning"]
            )

            print(f"   ✅ Reflection session created with {len(session.insights)} insights")
            print(f"   ✅ Focus areas: {', '.join(session.focus_areas)}")

            if session.insights:
                print(f"   ✅ Generated insights for: {[i.reflection_type.value for i in session.insights]}")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 Reflection System tests passed!\n")


async def test_orchestration_coordinator():
    """Test the Orchestration Coordinator."""
    print("🎭 Testing Orchestration Coordinator...")

    # Mock model for testing
    class MockModel:
        async def ainvoke(self, messages):
            class MockResponse:
                content = '{"recommended_strategy": "chain_of_thought", "rationale": "Best for this task", "expected_effectiveness": 85}'
            return MockResponse()

    model = MockModel()
    tools = []  # Empty tools for testing

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            coordinator = OrchestrationCoordinator(model, tools, db)

            # Test problem classification
            test_cases = [
                ("Analyze market trends", "analytical"),
                ("Create a project plan", "planning"),
                ("Search for information", "information_retrieval"),
                ("Compare options", "comparative")
            ]

            for request, expected in test_cases:
                result = coordinator._classify_problem_type(request)
                assert result == expected, f"Expected {expected}, got {result}"

            print("   ✅ Problem classification working correctly")

            # Test planning requirement detection
            planning_requests = [
                "Create a plan",
                "Organize workflow",
                "Develop strategy"
            ]

            for request in planning_requests:
                requires_planning = coordinator._requires_planning(request)
                assert requires_planning, f"Should require planning: {request}"

            print("   ✅ Planning requirement detection working correctly")

            # Test goal condition extraction
            conditions = coordinator._extract_goal_conditions("I need information about AI")
            print(f"   ✅ Goal condition extraction: {len(conditions)} conditions")

            # Test current state extraction
            state = coordinator._get_current_state()
            print(f"   ✅ Current state: {len(state)} predicates")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 Orchestration Coordinator tests passed!\n")


async def test_integration():
    """Test integration between all components."""
    print("🔗 Testing System Integration...")

    # Mock model for testing
    class MockModel:
        def __init__(self):
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1

            # Different responses based on call count
            responses = [
                '{"recommended_strategy": "chain_of_thought", "rationale": "Best strategy", "expected_effectiveness": 85}',
                '{"step_type": "inference", "content": "Reasoning step", "confidence": 80, "evidence": {}, "alternatives": [], "dependencies": [], "should_backtrack": false}',
                '{"plan_actions": ["action1"], "action_details": {}, "state_progression": [], "plan_rationale": "Simple plan", "estimated_cost": 2.0}',
                '{"performance_score": 75, "identified_issues": [], "error_patterns": [], "cognitive_load_assessment": 40, "recommendations": [], "attention_alerts": []}',
                '{"surface_observations": ["Test"], "analytical_insights": [], "critical_evaluation": [], "meta_cognitive_insights": [], "performance_patterns": [], "improvement_opportunities": [], "confidence_assessment": 70}'
            ]

            class MockResponse:
                content = responses[min(self.call_count - 1, len(responses) - 1)]

            return MockResponse()

    model = MockModel()
    tools = []

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db = MemoryDatabase(tmp_db.name)

        try:
            # Test that all components can be initialized together
            reasoning_engine = AdvancedReasoningEngine(model, db)
            planning_engine = AdvancedPlanningEngine(model, db)
            meta_engine = MetaReasoningEngine(model, db, reasoning_engine)
            reflection_engine = AdvancedReflectionEngine(model, db)
            coordinator = OrchestrationCoordinator(model, tools, db)

            print("   ✅ All components initialized successfully")

            # Test basic workflow
            chain_id = await reasoning_engine.start_reasoning_chain(
                goal="Integration test",
                initial_context={"test": "integration"}
            )

            strategy = await meta_engine.select_reasoning_strategy(
                problem="Integration test problem",
                problem_type="general"
            )

            session = await reflection_engine.trigger_reflection(
                trigger_event="Integration test",
                focus_areas=["performance"]
            )

            print("   ✅ Basic workflow completed successfully")
            print(f"   ✅ Model called {model.call_count} times")

        finally:
            os.unlink(tmp_db.name)

    print("   🎉 System Integration tests passed!\n")


async def run_all_tests():
    """Run all orchestration system tests."""
    print("🚀 Starting Advanced Agent Orchestration System Tests")
    print("=" * 60)

    start_time = time.time()

    try:
        await test_advanced_reasoning()
        await test_meta_reasoning()
        await test_advanced_planning()
        await test_reflection_system()
        await test_orchestration_coordinator()
        await test_integration()

        end_time = time.time()
        duration = end_time - start_time

        print("🎉 All tests passed successfully!")
        print(f"⏱️  Total test duration: {duration:.2f} seconds")
        print("\n✨ The Advanced Agent Orchestration System is ready for use!")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
