"""
Tests for the Advanced Agent Orchestration System.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic

from src.agents.advanced_planning import AdvancedPlanningEngine, Condition
from src.agents.advanced_reasoning import AdvancedReasoningEngine, ReasoningStepType
from src.agents.meta_reasoning import MetaReasoningEngine, MetaReasoningStrategy
from src.agents.reflection_systems import AdvancedReflectionEngine, ReflectionType
from src.core.orchestration_main import OrchestrationCoordinator
from src.memory.memory_persistence import MemoryDatabase

class TestAdvancedReasoningEngine(unittest.TestCase):
    """Test cases for the Advanced Reasoning Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=ChatAnthropic)
        self.db = MagicMock(spec=MemoryDatabase)
        self.engine = AdvancedReasoningEngine(self.model, self.db)

    @patch('src.agents.advanced_reasoning.uuid.uuid4')
    async def test_start_reasoning_chain(self, mock_uuid):
        """Test starting a new reasoning chain."""
        mock_uuid.return_value.hex = "test-chain-id"

        # Mock database save method
        self.db.save_reasoning_chain = AsyncMock()

        chain_id = await self.engine.start_reasoning_chain(
            goal="Test goal",
            initial_context={"test": "context"}
        )

        self.assertEqual(chain_id, "test-chain-id")
        self.assertIn(chain_id, self.engine.active_chains)
        self.db.save_reasoning_chain.assert_called_once()

    async def test_continue_reasoning(self):
        """Test continuing reasoning in a chain."""
        # Set up a reasoning chain
        chain_id = await self.engine.start_reasoning_chain(
            goal="Test goal",
            initial_context={"test": "context"}
        )

        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"step_type": "inference", "content": "Test reasoning step", "confidence": 80, "evidence": {}, "alternatives": [], "dependencies": [], "should_backtrack": false}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        # Mock database save method
        self.db.save_reasoning_step = AsyncMock()

        step = await self.engine.continue_reasoning(chain_id)

        self.assertEqual(step.step_type, ReasoningStepType.INFERENCE)
        self.assertEqual(step.content, "Test reasoning step")
        self.assertEqual(step.confidence, 0.8)
        self.db.save_reasoning_step.assert_called_once()

    async def test_analyze_causal_relationships(self):
        """Test causal relationship analysis."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"causal_links": [{"cause": "A", "effect": "B"}], "confidence": 0.8, "alternative_causes": [], "mechanism": "Direct causation"}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await self.engine.analyze_causal_relationships(
            scenario="Test scenario",
            context={"test": "context"}
        )

        self.assertIn("causal_links", result)
        self.assertEqual(result["confidence"], 0.8)

    async def test_explore_counterfactuals(self):
        """Test counterfactual exploration."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"scenarios": ["Scenario 1"], "outcomes": ["Outcome 1"], "probabilities": [0.7], "implications": "Test implications"}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await self.engine.explore_counterfactuals(
            situation="Test situation",
            facts={"fact1": "value1"}
        )

        self.assertIn("scenarios", result)
        self.assertIn("outcomes", result)
        self.assertIn("probabilities", result)

class TestMetaReasoningEngine(unittest.TestCase):
    """Test cases for the Meta-Reasoning Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=ChatAnthropic)
        self.db = MagicMock(spec=MemoryDatabase)
        self.reasoning_engine = MagicMock(spec=AdvancedReasoningEngine)
        self.engine = MetaReasoningEngine(self.model, self.db, self.reasoning_engine)

    async def test_select_reasoning_strategy(self):
        """Test reasoning strategy selection."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"recommended_strategy": "chain_of_thought", "supporting_strategies": [], "rationale": "Best for this problem", "expected_effectiveness": 85, "resource_requirements": 50}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        # Mock database save method
        self.db.save_meta_decision = AsyncMock()

        result = await self.engine.select_reasoning_strategy(
            problem="Test problem",
            problem_type="analytical"
        )

        self.assertEqual(result["recommended_strategy"], "chain_of_thought")
        self.assertEqual(result["expected_effectiveness"], 85)
        self.db.save_meta_decision.assert_called_once()

    async def test_monitor_performance(self):
        """Test performance monitoring."""
        # Create mock reasoning chain
        mock_chain = MagicMock()
        mock_chain.steps = []
        mock_chain.goal = "Test goal"

        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"performance_score": 75, "identified_issues": [], "error_patterns": [], "cognitive_load_assessment": 40, "recommendations": [], "attention_alerts": []}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await self.engine.monitor_performance(mock_chain)

        self.assertEqual(result["performance_score"], 75)
        self.assertEqual(result["cognitive_load_assessment"], 40)

    async def test_detect_errors(self):
        """Test error detection."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"errors_detected": ["Logic error"], "error_types": ["logical"], "severity_levels": [7], "correction_suggestions": ["Fix logic"], "confidence_impact": 0.2}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        # Mock database save method
        self.db.save_meta_decision = AsyncMock()

        result = await self.engine.detect_errors(
            reasoning_steps=[{"step": "test"}],
            context={"test": "context"},
            goal="Test goal"
        )

        self.assertIn("errors_detected", result)
        self.assertEqual(len(result["errors_detected"]), 1)

class TestAdvancedPlanningEngine(unittest.TestCase):
    """Test cases for the Advanced Planning Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=ChatAnthropic)
        self.db = MagicMock(spec=MemoryDatabase)
        self.engine = AdvancedPlanningEngine(self.model, self.db)

    async def test_create_strips_plan(self):
        """Test STRIPS plan creation."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"plan_actions": ["web_search", "analyze_data"], "action_details": {}, "state_progression": [], "plan_rationale": "Test plan", "estimated_cost": 5.0}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        # Mock database save method
        self.db.save_plan = AsyncMock()

        plan = await self.engine.create_strips_plan(
            goal="Test goal",
            initial_state={"start_state"},
            goal_conditions=[Condition("goal_achieved", ["test"])]
        )

        self.assertEqual(plan.goal, "Test goal")
        self.assertEqual(len(plan.actions), 2)
        self.db.save_plan.assert_called_once()

    async def test_create_temporal_plan(self):
        """Test temporal plan creation."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = '{"temporal_plan": [], "resource_schedule": {}, "critical_path": [], "parallel_opportunities": [], "timeline": "Test timeline"}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await self.engine.create_temporal_plan(
            goal="Test goal",
            available_actions=[],
            temporal_constraints=[],
            resource_constraints={}
        )

        self.assertIn("temporal_plan", result)
        self.assertIn("timeline", result)

    def test_validate_plan(self):
        """Test plan validation."""
        # Create a simple valid plan
        from src.agents.advanced_planning import Plan, Action, ActionType

        action = Action(
            action_id="test_action",
            name="test_action",
            action_type=ActionType.PRIMITIVE,
            parameters=[],
            preconditions=[],
            effects=[]
        )

        plan = Plan(
            plan_id="test_plan",
            goal="Test goal",
            actions=[action],
            initial_state=set(),
            goal_state=set()
        )

        validation = self.engine.validate_plan(plan)

        self.assertIn("is_valid", validation)
        self.assertIn("issues", validation)
        self.assertIn("warnings", validation)

class TestAdvancedReflectionEngine(unittest.TestCase):
    """Test cases for the Advanced Reflection Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=ChatAnthropic)
        self.db = MagicMock(spec=MemoryDatabase)
        self.engine = AdvancedReflectionEngine(self.model, self.db)

    async def test_trigger_reflection(self):
        """Test triggering a reflection session."""
        # Mock model responses for different reflection types
        mock_response = MagicMock()
        mock_response.content = '{"surface_observations": ["Test observation"], "analytical_insights": [], "critical_evaluation": [], "meta_cognitive_insights": [], "performance_patterns": [], "improvement_opportunities": [], "confidence_assessment": 75}'
        self.model.ainvoke = AsyncMock(return_value=mock_response)

        # Mock database save method
        self.db.save_reflection_session = AsyncMock()

        session = await self.engine.trigger_reflection(
            trigger_event="Test event",
            focus_areas=["performance"]
        )

        self.assertEqual(session.trigger_event, "Test event")
        self.assertEqual(session.focus_areas, ["performance"])
        self.assertGreater(len(session.insights), 0)
        self.db.save_reflection_session.assert_called_once()

class TestOrchestrationCoordinator(unittest.TestCase):
    """Test cases for the Orchestration Coordinator."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=ChatAnthropic)
        self.tools = []

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = MemoryDatabase(self.temp_db.name)

        self.coordinator = OrchestrationCoordinator(self.model, self.tools, self.db)

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)

    @patch('src.core.orchestration_main.create_specialized_sub_agents')
    async def test_process_request(self, mock_create_agents):
        """Test processing a request through the orchestration system."""
        # Mock sub-agents
        mock_create_agents.return_value = {}

        # Mock all the async methods
        self.coordinator.meta_reasoning_engine.select_reasoning_strategy = AsyncMock(
            return_value={
                "recommended_strategy": "chain_of_thought",
                "rationale": "Best strategy",
                "expected_effectiveness": 80
            }
        )

        self.coordinator.reasoning_engine.start_reasoning_chain = AsyncMock(
            return_value="test-chain-id"
        )

        self.coordinator.base_coordinator.process_request = AsyncMock(
            return_value="Test response"
        )

        self.coordinator.reasoning_engine.continue_reasoning = AsyncMock(
            return_value=MagicMock(confidence=0.9, content="Test reasoning")
        )

        self.coordinator.meta_reasoning_engine.monitor_performance = AsyncMock(
            return_value={"performance_score": 80}
        )

        self.coordinator.reflection_engine.trigger_reflection = AsyncMock(
            return_value=MagicMock(insights=[])
        )

        # Mock active chains
        self.coordinator.reasoning_engine.active_chains = {
            "test-chain-id": MagicMock(goal="Test goal")
        }

        response = await self.coordinator.process_request("Test request")

        self.assertEqual(response, "Test response")
        self.assertEqual(len(self.coordinator.orchestration_history), 1)

    def test_classify_problem_type(self):
        """Test problem type classification."""
        test_cases = [
            ("Analyze the market trends", "analytical"),
            ("Create a plan for the project", "planning"),
            ("Search for information about AI", "information_retrieval"),
            ("Compare different approaches", "comparative"),
            ("What is the weather today?", "general")
        ]

        for request, expected_type in test_cases:
            result = self.coordinator._classify_problem_type(request)
            self.assertEqual(result, expected_type)

    def test_requires_planning(self):
        """Test planning requirement detection."""
        planning_requests = [
            "Create a plan for the project",
            "Organize a multi-step workflow",
            "Develop a strategy for growth"
        ]

        non_planning_requests = [
            "What is the weather?",
            "Analyze this data",
            "Search for information"
        ]

        for request in planning_requests:
            self.assertTrue(self.coordinator._requires_planning(request))

        for request in non_planning_requests:
            self.assertFalse(self.coordinator._requires_planning(request))

if __name__ == "__main__":
    # Run async tests
    async def run_async_tests():
        """Run all async tests."""
        test_classes = [
            TestAdvancedReasoningEngine,
            TestMetaReasoningEngine,
            TestAdvancedPlanningEngine,
            TestAdvancedReflectionEngine,
            TestOrchestrationCoordinator
        ]

        for test_class in test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

            for test in suite:
                if hasattr(test, '_testMethodName'):
                    method = getattr(test, test._testMethodName)
                    if asyncio.iscoroutinefunction(method):
                        print(f"Running async test: {test_class.__name__}.{test._testMethodName}")
                        try:
                            await method()
                            print("✅ PASSED")
                        except Exception as e:
                            print(f"❌ FAILED: {e}")
                    else:
                        print(f"Running sync test: {test_class.__name__}.{test._testMethodName}")
                        try:
                            method()
                            print("✅ PASSED")
                        except Exception as e:
                            print(f"❌ FAILED: {e}")

    asyncio.run(run_async_tests())
