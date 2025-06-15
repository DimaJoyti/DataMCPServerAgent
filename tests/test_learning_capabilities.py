"""
Tests for learning capabilities module.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.learning_capabilities import FeedbackCollector, LearningAgent


class TestFeedbackCollector(unittest.TestCase):
    """Tests for FeedbackCollector class."""

    @patch('langchain_anthropic.ChatAnthropic')
    def setUp(self, mock_model):
        """Set up test environment."""
        self.mock_model = mock_model
        self.mock_model.ainvoke = AsyncMock()

        # Create mock database
        self.mock_db = MagicMock()
        self.mock_db.save_learning_feedback = MagicMock()

        self.collector = FeedbackCollector(self.mock_model, self.mock_db)

    async def test_collect_user_feedback(self):
        """Test collecting user feedback."""
        # Collect feedback
        await self.collector.collect_user_feedback(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Great response!",
            "search_agent"
        )

        # Check that feedback was saved to the database
        self.mock_db.save_learning_feedback.assert_called_once()
        args = self.mock_db.save_learning_feedback.call_args[0]

        # Check arguments
        self.assertEqual(args[0], "search_agent")
        self.assertEqual(args[1], "user_feedback")
        self.assertEqual(args[2]["request"], "What is the capital of France?")
        self.assertEqual(args[2]["response"], "The capital of France is Paris.")
        self.assertEqual(args[2]["feedback"], "Great response!")

    async def test_perform_self_evaluation_success(self):
        """Test performing self-evaluation successfully."""
        # Mock model response
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "completeness": 9,
    "accuracy": 10,
    "clarity": 8,
    "helpfulness": 9,
    "overall_score": 9,
    "strengths": ["Accurate information", "Direct answer"],
    "weaknesses": ["Could provide more context"],
    "improvement_suggestions": ["Add more historical context"]
}
```"""
        self.mock_model.ainvoke.return_value = mock_response

        # Perform self-evaluation
        evaluation = await self.collector.perform_self_evaluation(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "search_agent"
        )

        # Check that the model was called with the correct input
        self.mock_model.ainvoke.assert_called_once()

        # Check that the evaluation matches the expected output
        self.assertEqual(evaluation["completeness"], 9)
        self.assertEqual(evaluation["accuracy"], 10)
        self.assertEqual(evaluation["clarity"], 8)
        self.assertEqual(evaluation["helpfulness"], 9)
        self.assertEqual(evaluation["overall_score"], 9)
        self.assertEqual(evaluation["strengths"], ["Accurate information", "Direct answer"])
        self.assertEqual(evaluation["weaknesses"], ["Could provide more context"])
        self.assertEqual(evaluation["improvement_suggestions"], ["Add more historical context"])

        # Check that the evaluation was saved to the database
        self.mock_db.save_learning_feedback.assert_called_once()

    async def test_perform_self_evaluation_error(self):
        """Test performing self-evaluation with an error."""
        # Mock model response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"
        self.mock_model.ainvoke.return_value = mock_response

        # Perform self-evaluation
        evaluation = await self.collector.perform_self_evaluation(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "search_agent"
        )

        # Check that a default evaluation is returned
        self.assertEqual(evaluation["completeness"], 5)
        self.assertEqual(evaluation["accuracy"], 5)
        self.assertEqual(evaluation["clarity"], 5)
        self.assertEqual(evaluation["helpfulness"], 5)
        self.assertEqual(evaluation["overall_score"], 5)
        self.assertEqual(evaluation["strengths"], ["Unable to determine strengths due to evaluation error"])
        self.assertIn("Error in self-evaluation", evaluation["weaknesses"][0])
        self.assertEqual(evaluation["improvement_suggestions"], ["Improve self-evaluation parsing"])

        # Check that the default evaluation was saved to the database
        self.mock_db.save_learning_feedback.assert_called_once()

class TestLearningAgent(unittest.TestCase):
    """Tests for LearningAgent class."""

    @patch('langchain_anthropic.ChatAnthropic')
    def setUp(self, mock_model):
        """Set up test environment."""
        self.mock_model = mock_model
        self.mock_model.ainvoke = AsyncMock()

        # Create mock database
        self.mock_db = MagicMock()
        self.mock_db.get_learning_feedback = MagicMock()
        self.mock_db.save_learning_feedback = MagicMock()

        # Create mock feedback collector
        self.mock_collector = MagicMock()

        self.agent = LearningAgent("search_agent", self.mock_model, self.mock_db, self.mock_collector)

    async def test_learn_from_feedback_success(self):
        """Test learning from feedback successfully."""
        # Mock database response
        self.mock_db.get_learning_feedback.side_effect = [
            # User feedback
            [
                {
                    "feedback_data": {
                        "request": "What is the capital of France?",
                        "response": "The capital of France is Paris.",
                        "feedback": "Great response!"
                    }
                }
            ],
            # Self-evaluations
            [
                {
                    "feedback_data": {
                        "request": "What is the capital of France?",
                        "response": "The capital of France is Paris.",
                        "evaluation": {
                            "completeness": 9,
                            "accuracy": 10,
                            "clarity": 8,
                            "helpfulness": 9,
                            "overall_score": 9,
                            "strengths": ["Accurate information", "Direct answer"],
                            "weaknesses": ["Could provide more context"],
                            "improvement_suggestions": ["Add more historical context"]
                        }
                    }
                }
            ]
        ]

        # Mock model response
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "identified_patterns": ["Direct answers are appreciated"],
    "common_strengths": ["Accurate information", "Direct answers"],
    "common_weaknesses": ["Limited context"],
    "improvement_strategies": ["Add more historical context when answering questions"],
    "updated_guidelines": ["Provide direct answers first, then add context"]
}
```"""
        self.mock_model.ainvoke.return_value = mock_response

        # Learn from feedback
        insights = await self.agent.learn_from_feedback()

        # Check that the model was called with the correct input
        self.mock_model.ainvoke.assert_called_once()

        # Check that the insights match the expected output
        self.assertEqual(insights["identified_patterns"], ["Direct answers are appreciated"])
        self.assertEqual(insights["common_strengths"], ["Accurate information", "Direct answers"])
        self.assertEqual(insights["common_weaknesses"], ["Limited context"])
        self.assertEqual(insights["improvement_strategies"], ["Add more historical context when answering questions"])
        self.assertEqual(insights["updated_guidelines"], ["Provide direct answers first, then add context"])

        # Check that the insights were saved to the database
        self.mock_db.save_learning_feedback.assert_called_once()

    async def test_learn_from_feedback_no_feedback(self):
        """Test learning from feedback with no feedback available."""
        # Mock database response with no feedback
        self.mock_db.get_learning_feedback.return_value = []

        # Learn from feedback
        insights = await self.agent.learn_from_feedback()

        # Check that the model was not called
        self.mock_model.ainvoke.assert_not_called()

        # Check that default insights are returned
        self.assertEqual(insights["identified_patterns"], [])
        self.assertEqual(insights["common_strengths"], [])
        self.assertEqual(insights["common_weaknesses"], [])
        self.assertEqual(insights["improvement_strategies"], [])
        self.assertEqual(insights["updated_guidelines"], [])

    async def test_learn_from_feedback_error(self):
        """Test learning from feedback with an error."""
        # Mock database response
        self.mock_db.get_learning_feedback.side_effect = [
            # User feedback
            [
                {
                    "feedback_data": {
                        "request": "What is the capital of France?",
                        "response": "The capital of France is Paris.",
                        "feedback": "Great response!"
                    }
                }
            ],
            # Self-evaluations
            [
                {
                    "feedback_data": {
                        "request": "What is the capital of France?",
                        "response": "The capital of France is Paris.",
                        "evaluation": {
                            "completeness": 9,
                            "accuracy": 10,
                            "clarity": 8,
                            "helpfulness": 9,
                            "overall_score": 9,
                            "strengths": ["Accurate information", "Direct answer"],
                            "weaknesses": ["Could provide more context"],
                            "improvement_suggestions": ["Add more historical context"]
                        }
                    }
                }
            ]
        ]

        # Mock model response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"
        self.mock_model.ainvoke.return_value = mock_response

        # Learn from feedback
        insights = await self.agent.learn_from_feedback()

        # Check that default insights are returned
        self.assertEqual(insights["identified_patterns"], ["Error in learning process"])
        self.assertEqual(insights["common_strengths"], [])
        self.assertIn("Error in learning", insights["common_weaknesses"][0])
        self.assertEqual(insights["improvement_strategies"], ["Improve learning process"])
        self.assertEqual(insights["updated_guidelines"], [])

        # Check that the default insights were saved to the database
        self.mock_db.save_learning_feedback.assert_called_once()

    async def test_get_learning_insights(self):
        """Test getting learning insights."""
        # Mock database response
        self.mock_db.get_learning_feedback.return_value = [
            {
                "feedback_data": {
                    "identified_patterns": ["Direct answers are appreciated"],
                    "common_strengths": ["Accurate information", "Direct answers"],
                    "common_weaknesses": ["Limited context"],
                    "improvement_strategies": ["Add more historical context when answering questions"],
                    "updated_guidelines": ["Provide direct answers first, then add context"]
                }
            }
        ]

        # Get learning insights
        insights = await self.agent.get_learning_insights()

        # Check that the insights are formatted correctly
        self.assertIn("# Learning Insights for search_agent", insights)
        self.assertIn("## Identified Patterns", insights)
        self.assertIn("- Direct answers are appreciated", insights)
        self.assertIn("## Common Strengths", insights)
        self.assertIn("- Accurate information", insights)
        self.assertIn("- Direct answers", insights)
        self.assertIn("## Common Weaknesses", insights)
        self.assertIn("- Limited context", insights)
        self.assertIn("## Improvement Strategies", insights)
        self.assertIn("- Add more historical context when answering questions", insights)
        self.assertIn("## Updated Guidelines", insights)
        self.assertIn("- Provide direct answers first, then add context", insights)

    async def test_get_learning_insights_no_insights(self):
        """Test getting learning insights with no insights available."""
        # Mock database response with no insights
        self.mock_db.get_learning_feedback.return_value = []

        # Get learning insights
        insights = await self.agent.get_learning_insights()

        # Check that a default message is returned
        self.assertEqual(insights, "No learning insights available yet.")

if __name__ == '__main__':
    unittest.main()
