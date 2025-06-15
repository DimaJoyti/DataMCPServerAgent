"""
Adaptive learning module for DataMCPServerAgent.
This module provides mechanisms for agents to adapt to user preferences and improve over time.
"""

import json
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.memory.memory_persistence import MemoryDatabase


class UserPreferenceModel:
    """Model for tracking and adapting to user preferences."""

    def __init__(self, model: ChatAnthropic, db: MemoryDatabase):
        """Initialize the user preference model.

        Args:
            model: Language model to use
            db: Memory database for persistence
        """
        self.model = model
        self.db = db

        # Initialize default preferences
        self.preferences = {
            "response_style": {
                "verbosity": "medium",  # "concise", "medium", "detailed"
                "formality": "neutral",  # "casual", "neutral", "formal"
                "technical_level": "medium",  # "basic", "medium", "advanced"
                "include_examples": True,
                "include_explanations": True,
            },
            "content_preferences": {
                "prefers_visual_content": False,
                "prefers_structured_data": True,
                "prefers_step_by_step": True,
            },
            "tool_preferences": {"preferred_tools": [], "avoided_tools": []},
            "topic_interests": {"high_interest": [], "low_interest": []},
        }

        # Load preferences from the database
        self._load_preferences()

        # Create the preference extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a preference analysis agent responsible for identifying user preferences from interactions.
Your job is to analyze user requests and agent responses to identify preferences and interests.

For each interaction, you should:
1. Identify preferences about response style (verbosity, formality, technical level)
2. Identify preferences about content (visual, structured, step-by-step)
3. Identify preferences about tools (which tools they seem to prefer or avoid)
4. Identify topic interests (what topics they seem interested in)

Respond with a JSON object containing:
- "response_style": Object with preferences about response style
- "content_preferences": Object with preferences about content
- "tool_preferences": Object with preferences about tools
- "topic_interests": Object with topic interests
- "confidence": Confidence score for each preference (0-100)
"""
                ),
                HumanMessage(
                    content="""
User request: {request}
Agent response: {response}

Extract user preferences from this interaction.
"""
                ),
            ]
        )

    def _load_preferences(self) -> None:
        """Load preferences from the database."""
        # Try to load preferences from the database
        stored_preferences = self.db.load_entity("preferences", "user_preferences")

        if stored_preferences:
            # Update default preferences with stored preferences
            for category, prefs in stored_preferences.items():
                if category in self.preferences:
                    self.preferences[category].update(prefs)

    async def extract_preferences(self, request: str, response: str) -> Dict[str, Any]:
        """Extract preferences from a user request and agent response.

        Args:
            request: User request
            response: Agent response

        Returns:
            Extracted preferences
        """
        # Prepare the input for the extraction prompt
        input_values = {"request": request, "response": response}

        # Get the preference extraction from the model
        messages = self.extraction_prompt.format_messages(**input_values)
        response_obj = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response_obj.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            extracted_preferences = json.loads(json_str)

            return extracted_preferences
        except Exception:
            # If parsing fails, return an empty preferences object
            return {
                "response_style": {},
                "content_preferences": {},
                "tool_preferences": {},
                "topic_interests": {},
                "confidence": 0,
            }

    async def update_preferences(self, new_preferences: Dict[str, Any]) -> None:
        """Update the preference model with new preferences.

        Args:
            new_preferences: New preferences to incorporate
        """
        # Get confidence scores
        confidence = new_preferences.get("confidence", 50)

        # Update response style preferences
        if "response_style" in new_preferences:
            for key, value in new_preferences["response_style"].items():
                if key in self.preferences["response_style"]:
                    # Only update if confidence is high enough
                    if isinstance(confidence, dict):
                        pref_confidence = confidence.get("response_style", {}).get(key, 0)
                    else:
                        pref_confidence = confidence

                    if pref_confidence >= 70:
                        self.preferences["response_style"][key] = value

        # Update content preferences
        if "content_preferences" in new_preferences:
            for key, value in new_preferences["content_preferences"].items():
                if key in self.preferences["content_preferences"]:
                    # Only update if confidence is high enough
                    if isinstance(confidence, dict):
                        pref_confidence = confidence.get("content_preferences", {}).get(key, 0)
                    else:
                        pref_confidence = confidence

                    if pref_confidence >= 70:
                        self.preferences["content_preferences"][key] = value

        # Update tool preferences
        if "tool_preferences" in new_preferences:
            # Update preferred tools
            if "preferred_tools" in new_preferences["tool_preferences"]:
                for tool in new_preferences["tool_preferences"]["preferred_tools"]:
                    if tool not in self.preferences["tool_preferences"]["preferred_tools"]:
                        self.preferences["tool_preferences"]["preferred_tools"].append(tool)

            # Update avoided tools
            if "avoided_tools" in new_preferences["tool_preferences"]:
                for tool in new_preferences["tool_preferences"]["avoided_tools"]:
                    if tool not in self.preferences["tool_preferences"]["avoided_tools"]:
                        self.preferences["tool_preferences"]["avoided_tools"].append(tool)

        # Update topic interests
        if "topic_interests" in new_preferences:
            # Update high interest topics
            if "high_interest" in new_preferences["topic_interests"]:
                for topic in new_preferences["topic_interests"]["high_interest"]:
                    if topic not in self.preferences["topic_interests"]["high_interest"]:
                        self.preferences["topic_interests"]["high_interest"].append(topic)

            # Update low interest topics
            if "low_interest" in new_preferences["topic_interests"]:
                for topic in new_preferences["topic_interests"]["low_interest"]:
                    if topic not in self.preferences["topic_interests"]["low_interest"]:
                        self.preferences["topic_interests"]["low_interest"].append(topic)

        # Save preferences to the database
        self.db.save_entity("preferences", "user_preferences", self.preferences)

    def get_preferences(self) -> Dict[str, Any]:
        """Get the current preferences.

        Returns:
            Current preferences
        """
        return self.preferences

    def get_formatted_preferences(self) -> str:
        """Get a formatted representation of the current preferences.

        Returns:
            Formatted preferences
        """
        formatted = "# User Preferences\n\n"

        # Format response style preferences
        formatted += "## Response Style\n\n"
        for key, value in self.preferences["response_style"].items():
            formatted += f"- **{key}**: {value}\n"
        formatted += "\n"

        # Format content preferences
        formatted += "## Content Preferences\n\n"
        for key, value in self.preferences["content_preferences"].items():
            formatted += f"- **{key}**: {value}\n"
        formatted += "\n"

        # Format tool preferences
        formatted += "## Tool Preferences\n\n"

        if self.preferences["tool_preferences"]["preferred_tools"]:
            formatted += "### Preferred Tools\n\n"
            for tool in self.preferences["tool_preferences"]["preferred_tools"]:
                formatted += f"- {tool}\n"
            formatted += "\n"

        if self.preferences["tool_preferences"]["avoided_tools"]:
            formatted += "### Avoided Tools\n\n"
            for tool in self.preferences["tool_preferences"]["avoided_tools"]:
                formatted += f"- {tool}\n"
            formatted += "\n"

        # Format topic interests
        formatted += "## Topic Interests\n\n"

        if self.preferences["topic_interests"]["high_interest"]:
            formatted += "### High Interest\n\n"
            for topic in self.preferences["topic_interests"]["high_interest"]:
                formatted += f"- {topic}\n"
            formatted += "\n"

        if self.preferences["topic_interests"]["low_interest"]:
            formatted += "### Low Interest\n\n"
            for topic in self.preferences["topic_interests"]["low_interest"]:
                formatted += f"- {topic}\n"

        return formatted


class AdaptiveLearningSystem:
    """System for adaptive learning from user interactions."""

    def __init__(
        self, model: ChatAnthropic, db: MemoryDatabase, preference_model: UserPreferenceModel
    ):
        """Initialize the adaptive learning system.

        Args:
            model: Language model to use
            db: Memory database for persistence
            preference_model: User preference model
        """
        self.model = model
        self.db = db
        self.preference_model = preference_model

        # Create the response adaptation prompt
        self.adaptation_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an adaptive response agent responsible for tailoring responses to user preferences.
Your job is to adapt a draft response to better match the user's preferences.

For each response, you should:
1. Analyze the draft response
2. Consider the user's preferences
3. Adapt the response to better match those preferences
4. Maintain the factual accuracy and completeness of the original response

Adapt the response based on:
- Verbosity preference (concise, medium, detailed)
- Formality preference (casual, neutral, formal)
- Technical level preference (basic, medium, advanced)
- Content preferences (visual, structured, step-by-step)
- Topic interests (emphasize high-interest topics)

Respond with the adapted response, maintaining all factual information from the original.
"""
                ),
                HumanMessage(
                    content="""
User request: {request}
Draft response: {draft_response}

User preferences:
{preferences}

Adapt the response to better match the user's preferences.
"""
                ),
            ]
        )

        # Create the learning strategy prompt
        self.strategy_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a learning strategy agent responsible for developing strategies to improve agent performance.
Your job is to analyze feedback and performance data to identify areas for improvement and develop learning strategies.

For each analysis, you should:
1. Identify patterns in user feedback
2. Analyze tool performance metrics
3. Evaluate response metrics
4. Assess user satisfaction
5. Develop strategies for improvement

Respond with a JSON object containing:
- "learning_focus": Primary area to focus learning efforts
- "improvement_strategies": Array of strategies with priority levels
- "tool_recommendations": Recommendations for tool usage
- "response_recommendations": Recommendations for response generation
"""
                ),
                HumanMessage(
                    content="""
Recent feedback:
{feedback}

Performance metrics:
{performance_metrics}

Develop learning strategies based on this data.
"""
                ),
            ]
        )

    async def adapt_response(self, request: str, draft_response: str) -> str:
        """Adapt a draft response to better match user preferences.

        Args:
            request: User request
            draft_response: Draft response to adapt

        Returns:
            Adapted response
        """
        # Get current preferences
        preferences = self.preference_model.get_preferences()

        # Format preferences for the prompt
        formatted_preferences = json.dumps(preferences, indent=2)

        # Prepare the input for the adaptation prompt
        input_values = {
            "request": request,
            "draft_response": draft_response,
            "preferences": formatted_preferences,
        }

        # Get the adapted response from the model
        messages = self.adaptation_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        return response.content

    async def develop_learning_strategy(
        self, feedback: List[Dict[str, Any]], performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Develop learning strategies based on feedback and performance data.

        Args:
            feedback: List of feedback entries
            performance_metrics: Performance metrics

        Returns:
            Learning strategies
        """
        # Format feedback for the prompt
        formatted_feedback = self._format_feedback(feedback)

        # Format performance metrics for the prompt
        formatted_metrics = json.dumps(performance_metrics, indent=2)

        # Prepare the input for the strategy prompt
        input_values = {"feedback": formatted_feedback, "performance_metrics": formatted_metrics}

        # Get the learning strategies from the model
        messages = self.strategy_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            strategies = json.loads(json_str)

            # Save the strategies to the database
            self.db.save_entity("learning", "strategies", strategies)

            return strategies
        except Exception:
            # If parsing fails, return a default strategy
            default_strategy = {
                "learning_focus": "Improve response quality",
                "improvement_strategies": [
                    {"strategy": "Enhance response clarity", "priority": "high"},
                    {"strategy": "Improve tool selection", "priority": "medium"},
                ],
                "tool_recommendations": ["Focus on tools with higher success rates"],
                "response_recommendations": ["Provide more structured responses"],
            }

            # Save the default strategy to the database
            self.db.save_entity("learning", "strategies", default_strategy)

            return default_strategy

    def _format_feedback(self, feedback: List[Dict[str, Any]]) -> str:
        """Format feedback for the strategy prompt.

        Args:
            feedback: List of feedback entries

        Returns:
            Formatted feedback
        """
        if not feedback:
            return "No feedback available."

        formatted = ""

        # Limit to the 10 most recent feedback entries
        for entry in feedback[:10]:
            feedback_type = entry.get("feedback_type", "unknown")
            feedback_data = entry.get("feedback_data", {})

            formatted += f"## {feedback_type.title()}\n\n"

            if feedback_type == "user_feedback":
                formatted += f"Request: {feedback_data.get('request', 'N/A')[:100]}...\n"
                formatted += f"Response: {feedback_data.get('response', 'N/A')[:100]}...\n"
                formatted += f"Feedback: {feedback_data.get('feedback', 'N/A')}\n\n"
            elif feedback_type == "self_evaluation":
                evaluation = feedback_data.get("evaluation", {})
                formatted += f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10\n"

                if "strengths" in evaluation:
                    formatted += "Strengths:\n"
                    for strength in evaluation["strengths"]:
                        formatted += f"- {strength}\n"

                if "weaknesses" in evaluation:
                    formatted += "Weaknesses:\n"
                    for weakness in evaluation["weaknesses"]:
                        formatted += f"- {weakness}\n"

                if "improvement_suggestions" in evaluation:
                    formatted += "Improvement Suggestions:\n"
                    for suggestion in evaluation["improvement_suggestions"]:
                        formatted += f"- {suggestion}\n"

                formatted += "\n"
            elif feedback_type == "execution_feedback":
                formatted += f"Tool: {feedback_data.get('tool_name', 'N/A')}\n"
                formatted += f"Success: {feedback_data.get('success', 'N/A')}\n"

                tool_feedback = feedback_data.get("feedback", {})

                formatted += f"Appropriate: {tool_feedback.get('appropriate', 'N/A')}\n"

                if "issues" in tool_feedback:
                    formatted += "Issues:\n"
                    for issue in tool_feedback["issues"]:
                        formatted += f"- {issue}\n"

                if "suggestions" in tool_feedback:
                    formatted += "Suggestions:\n"
                    for suggestion in tool_feedback["suggestions"]:
                        formatted += f"- {suggestion}\n"

                formatted += "\n"

        return formatted
