"""
Reinforcement Learning integration for the Research Assistant.
This module implements reinforcement learning capabilities for the Research Assistant
to improve tool selection, source quality, and overall research quality over time.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.enhanced_research_assistant import EnhancedResearchAssistant
from src.memory.research_memory_persistence import ResearchMemoryDatabase

class ResearchRewardSystem:
    """Reward system for the Research Assistant."""

    def __init__(self, memory_db: ResearchMemoryDatabase):
        """Initialize the research reward system.

        Args:
            memory_db: Memory database for persistence
        """
        self.memory_db = memory_db

    def calculate_reward(
        self, query: str, response: Dict[str, Any], feedback: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward for a research response.

        Args:
            query: Research query
            response: Research response
            feedback: Optional user feedback

        Returns:
            Tuple of (total_reward, reward_components)
        """
        reward_components = {}

        # Reward for source diversity
        source_types = set()
        for source in response.get("sources", []):
            if isinstance(source, dict):
                source_type = source.get("source_type", "unknown")
            else:
                source_type = "unknown"
            source_types.add(source_type)

        source_diversity_reward = min(len(source_types) * 0.2, 1.0)
        reward_components["source_diversity"] = source_diversity_reward

        # Reward for source count (up to a reasonable limit)
        source_count = len(response.get("sources", []))
        source_count_reward = min(source_count * 0.1, 1.0)
        reward_components["source_count"] = source_count_reward

        # Reward for tool usage diversity
        tool_count = len(response.get("tools_used", []))
        tool_diversity_reward = min(tool_count * 0.1, 1.0)
        reward_components["tool_diversity"] = tool_diversity_reward

        # Reward for summary length (within reasonable limits)
        summary_length = len(response.get("summary", ""))
        summary_length_reward = min(summary_length / 1000, 1.0)
        reward_components["summary_length"] = summary_length_reward

        # Reward for having a bibliography
        bibliography_reward = 1.0 if response.get("bibliography") else 0.0
        reward_components["bibliography"] = bibliography_reward

        # Reward for having visualizations
        visualization_count = len(response.get("visualizations", []))
        visualization_reward = min(visualization_count * 0.5, 1.0)
        reward_components["visualizations"] = visualization_reward

        # Reward for having tags
        tag_count = len(response.get("tags", []))
        tag_reward = min(tag_count * 0.1, 0.5)
        reward_components["tags"] = tag_reward

        # Reward based on user feedback (if available)
        feedback_reward = 0.0
        if feedback:
            feedback_lower = feedback.lower()
            if "excellent" in feedback_lower or "amazing" in feedback_lower:
                feedback_reward = 2.0
            elif "good" in feedback_lower or "helpful" in feedback_lower:
                feedback_reward = 1.0
            elif "okay" in feedback_lower or "average" in feedback_lower:
                feedback_reward = 0.5
            elif "poor" in feedback_lower or "bad" in feedback_lower:
                feedback_reward = -1.0
            elif "terrible" in feedback_lower or "useless" in feedback_lower:
                feedback_reward = -2.0

        reward_components["feedback"] = feedback_reward

        # Calculate total reward
        total_reward = sum(reward_components.values())

        # Save reward to database
        self.memory_db.save_agent_reward(
            agent_name="research_assistant",
            reward=total_reward,
            reward_components=reward_components,
        )

        return total_reward, reward_components

class ResearchRLAgent:
    """Reinforcement Learning agent for the Research Assistant."""

    def __init__(
        self,
        model: ChatAnthropic,
        memory_db: ResearchMemoryDatabase,
        reward_system: ResearchRewardSystem,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
    ):
        """Initialize the research RL agent.

        Args:
            model: Language model to use
            memory_db: Memory database for persistence
            reward_system: Reward system for calculating rewards
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            exploration_rate: Exploration rate for epsilon-greedy policy
        """
        self.model = model
        self.memory_db = memory_db
        self.reward_system = reward_system
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize Q-table
        self.q_table = self._load_q_table()

        # Initialize feedback prompt
        self.feedback_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an AI assistant that analyzes research results and provides feedback to improve future research.
Your task is to analyze the research query, response, and user feedback (if available) to identify strengths and weaknesses.

Follow these steps:
1. Analyze the research query to understand what was being asked
2. Analyze the research response to identify strengths and weaknesses
3. Consider user feedback if available
4. Provide specific recommendations for improvement

Your analysis should focus on:
- Source quality and diversity
- Tool selection and usage
- Summary comprehensiveness and clarity
- Citation and bibliography quality
- Visualization effectiveness
- Overall response quality

Your feedback will be used to improve future research responses.
"""
                ),
                HumanMessage(
                    content="""
Research Query: {query}
Research Response: {response}
User Feedback: {feedback}

Please analyze this research and provide feedback for improvement.
"""
                ),
            ]
        )

    def _load_q_table(self) -> Dict[str, Dict[str, float]]:
        """Load Q-table from the database.

        Returns:
            Q-table
        """
        q_table = self.memory_db.get_q_table("research_assistant")
        if not q_table:
            return {}
        return q_table

    def _save_q_table(self) -> None:
        """Save Q-table to the database."""
        self.memory_db.save_q_table("research_assistant", self.q_table)

    def get_state(self, query: str) -> str:
        """Get state representation for a query.

        Args:
            query: Research query

        Returns:
            State representation
        """
        # Extract key features from the query
        query_lower = query.lower()

        # Check for academic topics
        is_academic = any(
            term in query_lower
            for term in [
                "research",
                "study",
                "paper",
                "journal",
                "academic",
                "science",
                "scientific",
                "theory",
                "hypothesis",
                "experiment",
            ]
        )

        # Check for technical topics
        is_technical = any(
            term in query_lower
            for term in [
                "programming",
                "code",
                "software",
                "hardware",
                "technology",
                "computer",
                "algorithm",
                "data",
                "machine learning",
                "ai",
            ]
        )

        # Check for medical topics
        is_medical = any(
            term in query_lower
            for term in [
                "medical",
                "health",
                "disease",
                "treatment",
                "medicine",
                "doctor",
                "patient",
                "hospital",
                "symptom",
                "diagnosis",
            ]
        )

        # Check for historical topics
        is_historical = any(
            term in query_lower
            for term in [
                "history",
                "historical",
                "ancient",
                "century",
                "war",
                "civilization",
                "empire",
                "king",
                "queen",
                "president",
            ]
        )

        # Check for current events
        is_current = any(
            term in query_lower
            for term in [
                "news",
                "current",
                "recent",
                "today",
                "latest",
                "update",
                "development",
                "trend",
                "now",
                "this year",
            ]
        )

        # Create state representation
        state_parts = []
        if is_academic:
            state_parts.append("academic")
        if is_technical:
            state_parts.append("technical")
        if is_medical:
            state_parts.append("medical")
        if is_historical:
            state_parts.append("historical")
        if is_current:
            state_parts.append("current")

        if not state_parts:
            state_parts.append("general")

        return "_".join(state_parts)

    def get_action(self, state: str) -> List[str]:
        """Get action (tool selection) for a state using epsilon-greedy policy.

        Args:
            state: State representation

        Returns:
            List of selected tools
        """
        # Initialize state in Q-table if not exists
        if state not in self.q_table:
            self.q_table[state] = {}

        # Get all available tools
        available_tools = [
            "search_tool",
            "wiki_tool",
            "google_scholar_tool",
            "pubmed_tool",
            "arxiv_tool",
            "google_books_tool",
            "open_library_tool",
        ]

        # Epsilon-greedy policy
        if np.random.random() < self.exploration_rate:
            # Exploration: randomly select 2-4 tools
            num_tools = np.random.randint(2, 5)
            selected_tools = np.random.choice(
                available_tools, size=num_tools, replace=False
            ).tolist()
        else:
            # Exploitation: select tools with highest Q-values
            tool_q_values = {}
            for tool in available_tools:
                action = f"use_{tool}"
                tool_q_values[tool] = self.q_table[state].get(action, 0.0)

            # Sort tools by Q-value
            sorted_tools = sorted(
                tool_q_values.items(), key=lambda x: x[1], reverse=True
            )

            # Select top 3 tools
            selected_tools = [tool for tool, _ in sorted_tools[:3]]

        return selected_tools

    def update_q_table(
        self,
        state: str,
        actions: List[str],
        reward: float,
        next_state: Optional[str] = None,
    ) -> None:
        """Update Q-table based on state, actions, reward, and next state.

        Args:
            state: Current state
            actions: Actions taken
            reward: Reward received
            next_state: Next state (optional)
        """
        # Initialize state in Q-table if not exists
        if state not in self.q_table:
            self.q_table[state] = {}

        # Update Q-values for each action
        for tool in actions:
            action = f"use_{tool}"

            # Initialize action in Q-table if not exists
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0

            # Calculate max Q-value for next state
            max_next_q = 0.0
            if next_state:
                if next_state in self.q_table:
                    next_actions = self.q_table[next_state]
                    if next_actions:
                        max_next_q = max(next_actions.values())

            # Update Q-value using Q-learning formula
            current_q = self.q_table[state][action]
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            self.q_table[state][action] = new_q

        # Save Q-table to database
        self._save_q_table()

    async def get_feedback(
        self, query: str, response: Dict[str, Any], user_feedback: Optional[str] = None
    ) -> str:
        """Get feedback on research response.

        Args:
            query: Research query
            response: Research response
            user_feedback: Optional user feedback

        Returns:
            Feedback
        """
        # Format inputs for the feedback prompt
        input_values = {
            "query": query,
            "response": json.dumps(response, indent=2),
            "feedback": user_feedback or "No user feedback available",
        }

        # Generate feedback using the model
        feedback_message = self.feedback_prompt.format_messages(**input_values)
        feedback_response = self.model.invoke(feedback_message)

        return feedback_response.content

    async def learn_from_interaction(
        self,
        query: str,
        response: Dict[str, Any],
        tools_used: List[str],
        user_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Learn from interaction with user.

        Args:
            query: Research query
            response: Research response
            tools_used: Tools used in the research
            user_feedback: Optional user feedback

        Returns:
            Learning results
        """
        # Get state representation
        state = self.get_state(query)

        # Calculate reward
        reward, reward_components = self.reward_system.calculate_reward(
            query, response, user_feedback
        )

        # Update Q-table
        self.update_q_table(state, tools_used, reward)

        # Get feedback
        feedback = await self.get_feedback(query, response, user_feedback)

        # Save feedback to database
        self.memory_db.save_learning_feedback(
            agent_name="research_assistant",
            feedback_type="self_evaluation",
            feedback_data={
                "query": query,
                "state": state,
                "tools_used": tools_used,
                "reward": reward,
                "reward_components": reward_components,
                "feedback": feedback,
            },
        )

        if user_feedback:
            self.memory_db.save_learning_feedback(
                agent_name="research_assistant",
                feedback_type="user_feedback",
                feedback_data={
                    "query": query,
                    "feedback": user_feedback,
                },
            )

        # Return learning results
        return {
            "state": state,
            "tools_used": tools_used,
            "reward": reward,
            "reward_components": reward_components,
            "feedback": feedback,
        }

class RLEnhancedResearchAssistant(EnhancedResearchAssistant):
    """Research Assistant with Reinforcement Learning capabilities."""

    def __init__(
        self,
        model: Optional[ChatAnthropic] = None,
        db_path: str = "research_memory.db",
        tools: Optional[List] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
    ):
        """Initialize the RL-enhanced research assistant.

        Args:
            model: Language model to use
            db_path: Path to the research memory database
            tools: List of tools to use
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            exploration_rate: Exploration rate for epsilon-greedy policy
        """
        super().__init__(model, db_path, tools)

        # Initialize reward system
        self.reward_system = ResearchRewardSystem(self.memory_db)

        # Initialize RL agent
        self.rl_agent = ResearchRLAgent(
            model=self.model,
            memory_db=self.memory_db,
            reward_system=self.reward_system,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
        )

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the research assistant with RL-based tool selection.

        Args:
            inputs: Input parameters including query, project_id, and citation_format

        Returns:
            Research results
        """
        query = inputs.get("query", "").lower()

        # Get state representation
        state = self.rl_agent.get_state(query)

        # Get RL-recommended tools
        recommended_tools = self.rl_agent.get_action(state)

        # Override tool selection with RL recommendations
        # This is a simple approach; in a more sophisticated system,
        # you might combine RL recommendations with the tool selector's recommendations
        tools_used = []
        for tool_name in recommended_tools:
            for tool in self.tools:
                if tool.name == tool_name:
                    tools_used.append(tool_name)
                    break

        # Store the tools used for learning
        inputs["rl_tools_used"] = tools_used

        # Call the parent invoke method
        result = await super().invoke(inputs)

        # Learn from the interaction (in the background)
        response_data = json.loads(result.get("output", "{}"))
        asyncio.create_task(
            self.rl_agent.learn_from_interaction(
                query=query,
                response=response_data,
                tools_used=tools_used,
                user_feedback=None,  # No user feedback available at this point
            )
        )

        return result

    async def update_from_feedback(
        self, query: str, response: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Update the RL agent from user feedback.

        Args:
            query: Research query
            response: Research response
            feedback: User feedback

        Returns:
            Learning results
        """
        # Get tools used from the response
        tools_used = response.get("tools_used", [])

        # Learn from the interaction with user feedback
        learning_results = await self.rl_agent.learn_from_interaction(
            query=query,
            response=response,
            tools_used=tools_used,
            user_feedback=feedback,
        )

        return learning_results
