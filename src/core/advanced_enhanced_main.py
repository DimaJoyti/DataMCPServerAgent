"""
Advanced enhanced main entry point for DataMCPServerAgent with context-aware memory,
adaptive learning, and sophisticated tool selection.
"""

import asyncio
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.adaptive_learning import AdaptiveLearningSystem, UserPreferenceModel
from src.agents.enhanced_agent_architecture import (
    EnhancedCoordinatorAgent,
    create_enhanced_agent_architecture,
)
from src.agents.learning_capabilities import FeedbackCollector
from src.memory.context_aware_memory import ContextManager, MemoryRetriever
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker
from src.utils.error_handlers import format_error_for_user

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)


async def load_all_tools(session: ClientSession) -> List[BaseTool]:
    """Load both standard MCP tools and custom Bright Data tools.

    Args:
        session: An initialized MCP ClientSession

    Returns:
        A combined list of standard and custom tools
    """
    # Load standard MCP tools
    standard_tools = await load_mcp_tools(session)

    # Load custom Bright Data tools
    bright_data_toolkit = BrightDataToolkit(session)
    custom_tools = await bright_data_toolkit.create_custom_tools()

    # Combine tools, with custom tools taking precedence if there are name conflicts
    tool_dict = {tool.name: tool for tool in standard_tools}

    # Add custom tools, potentially overriding standard tools with the same name
    for tool in custom_tools:
        tool_dict[tool.name] = tool

    return list(tool_dict.values())


class AdvancedEnhancedAgent:
    """Advanced enhanced agent with context-aware memory and adaptive learning."""

    def __init__(
        self,
        coordinator: EnhancedCoordinatorAgent,
        memory_db: MemoryDatabase,
        context_manager: ContextManager,
        adaptive_learning: AdaptiveLearningSystem,
        preference_model: UserPreferenceModel,
    ):
        """Initialize the advanced enhanced agent.

        Args:
            coordinator: Enhanced coordinator agent
            memory_db: Memory database
            context_manager: Context manager
            adaptive_learning: Adaptive learning system
            preference_model: User preference model
        """
        self.coordinator = coordinator
        self.memory_db = memory_db
        self.context_manager = context_manager
        self.adaptive_learning = adaptive_learning
        self.preference_model = preference_model

        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "successful_responses": 0,
            "errors": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "feedback_received": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
        }

        self.last_request = ""
        self.last_response = ""
        self.last_context = {}

    async def process_request(self, request: str) -> str:
        """Process a user request with context-aware memory and adaptive learning.

        Args:
            request: User request

        Returns:
            Response to the user
        """
        start_time = time.time()
        self.last_request = request

        try:
            # Update context based on the request
            self.last_context = await self.context_manager.update_context(request)

            # Process the request through the coordinator
            draft_response = await self.coordinator.process_request(request)

            # Adapt the response based on user preferences
            adapted_response = await self.adaptive_learning.adapt_response(request, draft_response)

            # Extract preferences from the interaction
            preferences = await self.preference_model.extract_preferences(request, adapted_response)

            # Update the preference model
            await self.preference_model.update_preferences(preferences)

            # Update metrics
            self.metrics["requests_processed"] += 1
            self.metrics["successful_responses"] += 1
            response_time = time.time() - start_time
            self.metrics["total_response_time"] += response_time
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["requests_processed"]
            )

            # Save the response
            self.last_response = adapted_response

            return adapted_response
        except Exception as e:
            error_message = format_error_for_user(e)

            # Update metrics
            self.metrics["requests_processed"] += 1
            self.metrics["errors"] += 1

            # Save the error response
            self.last_response = f"An error occurred: {error_message}"

            return self.last_response

    async def collect_feedback(self, feedback: str) -> None:
        """Collect and process user feedback.

        Args:
            feedback: User feedback
        """
        if not self.last_request or not self.last_response:
            return

        # Update metrics
        self.metrics["feedback_received"] += 1

        # Determine if feedback is positive or negative (simple heuristic)
        positive_words = [
            "good",
            "great",
            "excellent",
            "helpful",
            "thanks",
            "thank",
            "perfect",
            "awesome",
        ]
        negative_words = [
            "bad",
            "poor",
            "unhelpful",
            "wrong",
            "incorrect",
            "error",
            "mistake",
            "not",
        ]

        feedback_lower = feedback.lower()
        is_positive = any(word in feedback_lower for word in positive_words)
        is_negative = any(word in feedback_lower for word in negative_words)

        if is_positive:
            self.metrics["positive_feedback"] += 1
        if is_negative:
            self.metrics["negative_feedback"] += 1

        # Collect feedback through the coordinator
        await self.coordinator.collect_user_feedback(
            self.last_request, self.last_response, feedback
        )

    async def learn(self) -> Dict[str, Any]:
        """Trigger learning from collected feedback and performance data.

        Returns:
            Learning strategies
        """
        # Get recent feedback
        feedback = self.memory_db.get_learning_feedback(None, None)

        # Prepare performance metrics
        performance_metrics = {
            "tool_performance": {},
            "response_metrics": {
                "avg_response_time": self.metrics["avg_response_time"],
                "success_rate": (
                    self.metrics["successful_responses"]
                    / max(1, self.metrics["requests_processed"])
                )
                * 100,
            },
            "user_satisfaction": {
                "feedback_rate": (
                    self.metrics["feedback_received"] / max(1, self.metrics["requests_processed"])
                )
                * 100,
                "positive_rate": (
                    self.metrics["positive_feedback"] / max(1, self.metrics["feedback_received"])
                )
                * 100,
                "negative_rate": (
                    self.metrics["negative_feedback"] / max(1, self.metrics["feedback_received"])
                )
                * 100,
            },
        }

        # Get tool performance metrics
        for tool_name in self.coordinator.tool_selector.tool_map:
            performance_metrics["tool_performance"][tool_name] = (
                self.coordinator.performance_tracker.get_performance(tool_name)
            )

        # Develop learning strategies
        strategies = await self.adaptive_learning.develop_learning_strategy(
            feedback, performance_metrics
        )

        return strategies

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics
        """
        return self.metrics

    def get_context(self) -> str:
        """Get the current context.

        Returns:
            Formatted context
        """
        return self.context_manager.get_formatted_context()

    def get_preferences(self) -> str:
        """Get the current user preferences.

        Returns:
            Formatted preferences
        """
        return self.preference_model.get_formatted_preferences()


async def create_advanced_enhanced_agent(
    model: ChatAnthropic, tools: List[BaseTool], db_path: str = "agent_memory.db"
) -> AdvancedEnhancedAgent:
    """Create an advanced enhanced agent with context-aware memory and adaptive learning.

    Args:
        model: Language model to use
        tools: List of available tools
        db_path: Path to the memory database

    Returns:
        Advanced enhanced agent
    """
    # Initialize memory database
    memory_db = MemoryDatabase(db_path)

    # Initialize tool performance tracker
    performance_tracker = ToolPerformanceTracker(memory_db)

    # Initialize enhanced tool selector
    tool_selector = EnhancedToolSelector(model, tools, memory_db, performance_tracker)

    # Initialize feedback collector
    feedback_collector = FeedbackCollector(model, memory_db)

    # Create enhanced coordinator agent
    coordinator = await create_enhanced_agent_architecture(model, tools, db_path)

    # Initialize memory retriever
    memory_retriever = MemoryRetriever(model, memory_db)

    # Initialize context manager
    context_manager = ContextManager(memory_retriever)

    # Initialize user preference model
    preference_model = UserPreferenceModel(model, memory_db)

    # Initialize adaptive learning system
    adaptive_learning = AdaptiveLearningSystem(model, memory_db, preference_model)

    # Create advanced enhanced agent
    agent = AdvancedEnhancedAgent(
        coordinator, memory_db, context_manager, adaptive_learning, preference_model
    )

    return agent


async def chat_with_advanced_enhanced_agent():
    """Run the advanced enhanced agent with context-aware memory and adaptive learning."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing DataMCPServerAgent with advanced enhanced capabilities...")

            # Load all tools
            tools = await load_all_tools(session)

            # Create the advanced enhanced agent
            agent = await create_advanced_enhanced_agent(model, tools)

            print("DataMCPServerAgent initialized with advanced enhanced capabilities:")
            print("- Context-aware memory")
            print("- Adaptive learning")
            print("- User preference modeling")

            print("\nType 'exit' or 'quit' to end the chat.")
            print("Type 'context' to see the current context.")
            print("Type 'preferences' to see the current user preferences.")
            print("Type 'learn' to trigger learning from feedback.")
            print("Type 'metrics' to see performance metrics.")
            print("Type 'feedback <your feedback>' to provide feedback on the last response.")

            while True:
                user_input = input("\nYou: ")

                # Check for special commands
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                elif user_input.strip().lower() == "context":
                    context = agent.get_context()
                    print("\nCurrent Context:")
                    print(context)
                    continue

                elif user_input.strip().lower() == "preferences":
                    preferences = agent.get_preferences()
                    print("\nUser Preferences:")
                    print(preferences)
                    continue

                elif user_input.strip().lower() == "learn":
                    print("\nLearning from feedback and performance data...")
                    strategies = await agent.learn()
                    print("\nLearning Strategies:")
                    print(f"Focus Area: {strategies.get('learning_focus', 'None')}")

                    if "improvement_strategies" in strategies:
                        print("\nImprovement Strategies:")
                        for i, strategy in enumerate(strategies["improvement_strategies"], 1):
                            print(
                                f"{i}. {strategy.get('strategy', 'Unknown')} (Priority: {strategy.get('priority', 'medium')})"
                            )
                    continue

                elif user_input.strip().lower() == "metrics":
                    metrics = agent.get_metrics()
                    print("\nPerformance Metrics:")
                    print(f"Requests Processed: {metrics['requests_processed']}")
                    print(f"Successful Responses: {metrics['successful_responses']}")
                    print(f"Errors: {metrics['errors']}")
                    print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
                    print(f"Feedback Received: {metrics['feedback_received']}")

                    if metrics["feedback_received"] > 0:
                        positive_rate = (
                            metrics["positive_feedback"] / metrics["feedback_received"]
                        ) * 100
                        negative_rate = (
                            metrics["negative_feedback"] / metrics["feedback_received"]
                        ) * 100
                        print(f"Positive Feedback Rate: {positive_rate:.2f}%")
                        print(f"Negative Feedback Rate: {negative_rate:.2f}%")
                    continue

                elif user_input.strip().lower().startswith("feedback "):
                    feedback = user_input[9:].strip()
                    await agent.collect_feedback(feedback)
                    print("Thank you for your feedback! It will help me improve.")
                    continue

                # Process regular user input
                print("Processing your request...")

                response = await agent.process_request(user_input)
                print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(chat_with_advanced_enhanced_agent())
