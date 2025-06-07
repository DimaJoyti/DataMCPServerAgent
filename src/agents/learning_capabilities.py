"""
Learning capabilities module for DataMCPServerAgent.
This module provides mechanisms for agents to learn from past interactions.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from src.memory.memory_persistence import MemoryDatabase

class FeedbackCollector:
    """Collector for user and self-evaluation feedback."""

    def __init__(self, model: ChatAnthropic, db: MemoryDatabase):
        """Initialize the feedback collector.

        Args:
            model: Language model to use
            db: Memory database for persistence
        """
        self.model = model
        self.db = db

        # Create the self-evaluation prompt
        self.self_eval_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a self-evaluation agent responsible for analyzing your own performance.
Your job is to critically evaluate your response to a user request and identify areas for improvement.

For each response, you should:
1. Analyze whether the response fully addressed the user's request
2. Identify any errors, omissions, or misunderstandings
3. Evaluate the clarity and helpfulness of the response
4. Suggest specific improvements for future responses
5. Rate your performance on a scale of 1-10

Respond with a JSON object containing:
- "completeness": Score from 1-10 on how completely the response addressed the request
- "accuracy": Score from 1-10 on the factual accuracy of the response
- "clarity": Score from 1-10 on how clear and understandable the response was
- "helpfulness": Score from 1-10 on how helpful the response was
- "overall_score": Overall performance score from 1-10
- "strengths": Array of strengths in the response
- "weaknesses": Array of weaknesses in the response
- "improvement_suggestions": Array of specific suggestions for improvement
"""),
            HumanMessage(content="""
User request: {request}
Agent response: {response}

Evaluate this response.
""")
        ])

    async def collect_user_feedback(
        self,
        request: str,
        response: str,
        feedback: str,
        agent_name: str
    ) -> None:
        """Collect and store user feedback.

        Args:
            request: Original user request
            response: Agent response
            feedback: User feedback
            agent_name: Name of the agent
        """
        feedback_data = {
            "request": request,
            "response": response,
            "feedback": feedback,
            "timestamp": time.time()
        }

        # Save the feedback to the database
        self.db.save_learning_feedback(agent_name, "user_feedback", feedback_data)

    async def perform_self_evaluation(
        self,
        request: str,
        response: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """Perform self-evaluation of an agent's response.

        Args:
            request: Original user request
            response: Agent response
            agent_name: Name of the agent

        Returns:
            Self-evaluation results
        """
        # Prepare the input for the self-evaluation prompt
        input_values = {
            "request": request,
            "response": response
        }

        # Get the self-evaluation from the model
        messages = self.self_eval_prompt.format_messages(**input_values)
        response_obj = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response_obj.content
            json_str = content.split("```json")[1].split("```")[0] if "```json" in content else content
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            evaluation = json.loads(json_str)

            # Save the evaluation to the database
            self.db.save_learning_feedback(
                agent_name,
                "self_evaluation",
                {
                    "request": request,
                    "response": response,
                    "evaluation": evaluation
                }
            )

            return evaluation
        except Exception as e:
            # If parsing fails, return a default evaluation
            default_eval = {
                "completeness": 5,
                "accuracy": 5,
                "clarity": 5,
                "helpfulness": 5,
                "overall_score": 5,
                "strengths": ["Unable to determine strengths due to evaluation error"],
                "weaknesses": [f"Error in self-evaluation: {str(e)}"],
                "improvement_suggestions": ["Improve self-evaluation parsing"]
            }

            # Save the default evaluation to the database
            self.db.save_learning_feedback(
                agent_name,
                "self_evaluation",
                {
                    "request": request,
                    "response": response,
                    "evaluation": default_eval,
                    "error": str(e)
                }
            )

            return default_eval

class LearningAgent:
    """Agent with learning capabilities."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        feedback_collector: FeedbackCollector
    ):
        """Initialize the learning agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Memory database for persistence
            feedback_collector: Feedback collector
        """
        self.name = name
        self.model = model
        self.db = db
        self.feedback_collector = feedback_collector

        # Create the learning prompt
        self.learning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a learning agent responsible for improving your performance based on feedback.
Your job is to analyze feedback from users and self-evaluations to identify patterns and areas for improvement.

Based on the feedback, you should:
1. Identify common strengths and weaknesses
2. Recognize patterns in user requests and responses
3. Develop strategies to address recurring issues
4. Generate specific recommendations for improvement
5. Create updated guidelines for future responses

Respond with a JSON object containing:
- "identified_patterns": Array of patterns identified in the feedback
- "common_strengths": Array of common strengths
- "common_weaknesses": Array of common weaknesses
- "improvement_strategies": Array of strategies for improvement
- "updated_guidelines": Array of updated guidelines for future responses
"""),
            HumanMessage(content="""
Recent feedback:
{feedback}

Generate learning insights and improvements.
""")
        ])

    async def learn_from_feedback(self) -> Dict[str, Any]:
        """Learn from collected feedback to improve future performance.

        Returns:
            Learning insights and improvements
        """
        # Get recent feedback from the database
        user_feedback = self.db.get_learning_feedback(self.name, "user_feedback")
        self_evaluations = self.db.get_learning_feedback(self.name, "self_evaluation")

        # Combine and format the feedback
        formatted_feedback = self._format_feedback(user_feedback, self_evaluations)

        # If there's no feedback, return a default response
        if not formatted_feedback:
            return {
                "identified_patterns": [],
                "common_strengths": [],
                "common_weaknesses": [],
                "improvement_strategies": [],
                "updated_guidelines": []
            }

        # Prepare the input for the learning prompt
        input_values = {
            "feedback": formatted_feedback
        }

        # Get the learning insights from the model
        messages = self.learning_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = content.split("```json")[1].split("```")[0] if "```json" in content else content
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            insights = json.loads(json_str)

            # Save the insights to the database
            self.db.save_learning_feedback(
                self.name,
                "learning_insights",
                insights
            )

            return insights
        except Exception as e:
            # If parsing fails, return a default response
            default_insights = {
                "identified_patterns": ["Error in learning process"],
                "common_strengths": [],
                "common_weaknesses": [f"Error in learning: {str(e)}"],
                "improvement_strategies": ["Improve learning process"],
                "updated_guidelines": []
            }

            # Save the default insights to the database
            self.db.save_learning_feedback(
                self.name,
                "learning_insights",
                {
                    "insights": default_insights,
                    "error": str(e)
                }
            )

            return default_insights

    def _format_feedback(
        self,
        user_feedback: List[Dict[str, Any]],
        self_evaluations: List[Dict[str, Any]]
    ) -> str:
        """Format feedback for the learning prompt.

        Args:
            user_feedback: List of user feedback entries
            self_evaluations: List of self-evaluation entries

        Returns:
            Formatted feedback
        """
        formatted = "## User Feedback\n\n"

        # Format user feedback
        if user_feedback:
            for i, feedback in enumerate(user_feedback[:5], 1):  # Limit to 5 most recent
                formatted += f"### User Feedback {i}\n"
                formatted += f"Request: {feedback['feedback_data']['request'][:100]}...\n"
                formatted += f"Response: {feedback['feedback_data']['response'][:100]}...\n"
                formatted += f"Feedback: {feedback['feedback_data']['feedback']}\n\n"
        else:
            formatted += "No user feedback available.\n\n"

        formatted += "## Self-Evaluations\n\n"

        # Format self-evaluations
        if self_evaluations:
            for i, eval_data in enumerate(self_evaluations[:5], 1):  # Limit to 5 most recent
                formatted += f"### Self-Evaluation {i}\n"
                formatted += f"Request: {eval_data['feedback_data']['request'][:100]}...\n"
                formatted += f"Response: {eval_data['feedback_data']['response'][:100]}...\n"

                evaluation = eval_data['feedback_data']['evaluation']
                formatted += f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10\n"

                if 'strengths' in evaluation:
                    formatted += "Strengths:\n"
                    for strength in evaluation['strengths']:
                        formatted += f"- {strength}\n"

                if 'weaknesses' in evaluation:
                    formatted += "Weaknesses:\n"
                    for weakness in evaluation['weaknesses']:
                        formatted += f"- {weakness}\n"

                if 'improvement_suggestions' in evaluation:
                    formatted += "Improvement Suggestions:\n"
                    for suggestion in evaluation['improvement_suggestions']:
                        formatted += f"- {suggestion}\n"

                formatted += "\n"
        else:
            formatted += "No self-evaluations available.\n\n"

        return formatted

    async def get_learning_insights(self) -> str:
        """Get a summary of learning insights.

        Returns:
            Summary of learning insights
        """
        # Get the most recent learning insights
        insights_list = self.db.get_learning_feedback(self.name, "learning_insights")

        if not insights_list:
            return "No learning insights available yet."

        # Get the most recent insights
        insights = insights_list[0]['feedback_data']

        # Format the insights
        formatted = f"# Learning Insights for {self.name}\n\n"

        if 'identified_patterns' in insights:
            formatted += "## Identified Patterns\n\n"
            for pattern in insights['identified_patterns']:
                formatted += f"- {pattern}\n"
            formatted += "\n"

        if 'common_strengths' in insights:
            formatted += "## Common Strengths\n\n"
            for strength in insights['common_strengths']:
                formatted += f"- {strength}\n"
            formatted += "\n"

        if 'common_weaknesses' in insights:
            formatted += "## Common Weaknesses\n\n"
            for weakness in insights['common_weaknesses']:
                formatted += f"- {weakness}\n"
            formatted += "\n"

        if 'improvement_strategies' in insights:
            formatted += "## Improvement Strategies\n\n"
            for strategy in insights['improvement_strategies']:
                formatted += f"- {strategy}\n"
            formatted += "\n"

        if 'updated_guidelines' in insights:
            formatted += "## Updated Guidelines\n\n"
            for guideline in insights['updated_guidelines']:
                formatted += f"- {guideline}\n"

        return formatted
